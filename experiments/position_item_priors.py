"""Estimate item response lift from all train impressions at their logged depth."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores, resolve_ties
from ranking import DATA, KEYS, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from semantic_transition_rerank import top_hits


FOLDS = 8
POSITION_CUTS = np.array([1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233])
LENGTH_CUTS = np.array([30, 100])
CELL_SHAPE = (len(POSITION_CUTS) + 1, len(LENGTH_CUTS) + 1, 128)
CELL_COUNT = int(np.prod(CELL_SHAPE))
SIGNALS = ('overall', 'position', 'position_length_category')
SHRINK = (1, 3, 10)
WEIGHTS = (.025, .05, .10, .20, .35, .50)


def held_out_rates(shown, positive):
    n = (shown.sum(axis=0) - shown).reshape(-1, *CELL_SHAPE)
    y = (positive.sum(axis=0) - positive).reshape(-1, *CELL_SHAPE)
    common = y.sum(axis=(1, 2, 3), keepdims=True) / n.sum(axis=(1, 2, 3), keepdims=True)
    position = (y.sum(axis=(2, 3), keepdims=True) + 500 * common) / (n.sum(axis=(2, 3), keepdims=True) + 500)
    length = (y.sum(axis=3, keepdims=True) + 500 * position) / (n.sum(axis=3, keepdims=True) + 500)
    category = (y + 200 * length) / (n + 200)
    return np.stack([np.broadcast_to(p, y.shape).reshape(len(n), CELL_COUNT) for p in (common, position, category)])


def shards():
    files = sorted((DATA / 'train' / 'impressions').glob('part-*.parquet'))
    if len(files) != 512:
        raise ValueError(f'expected 512 train files, got {len(files)}')
    return files


def observations(path):
    table = pq.read_table(path, columns=['timestamp', 'item_id', 'category_id', 'pos', 'target']).combine_chunks()
    parts = [table[key].chunk(0) for key in ('item_id', 'category_id', 'pos', 'target')]
    offsets = parts[0].offsets.to_numpy()
    for part in parts[1:]:
        np.testing.assert_array_equal(part.offsets.to_numpy(), offsets)
    items, category, position, response = [part.flatten().to_numpy() for part in parts]
    counts = np.diff(offsets)
    if (np.any((position < 0) | (position > 300)) or np.any(category < 0) or np.any(counts < 1)
            or np.any((response != 0) & (response != 1))):
        raise ValueError(f'invalid candidate position, category, feed count or target in {path}')
    depth = np.searchsorted(POSITION_CUTS, position, side='right')
    length = np.repeat(np.searchsorted(LENGTH_CUTS, counts, side='left'), counts)
    cells = ((depth * CELL_SHAPE[1] + length) * CELL_SHAPE[2] + category).astype(np.int16)
    return items, cells, response, counts, table['timestamp'].to_numpy()


def evaluation_ids(cache):
    keys, earliest = [], []
    for split in ('val', 'test'):
        with np.load(cache / f'{split}.npz') as data:
            keys.append(data['item'])
            earliest.append(int(data['timestamp'].min()))
    items = np.unique(np.concatenate(keys))
    if len(items) == 0 or items[0] < 0:
        raise ValueError('expected nonnegative evaluation item IDs')
    return items, min(earliest)


def fit(cache, out):
    if out.exists():
        raise FileExistsError(out)
    items, eval_start = evaluation_ids(cache)
    index = np.full(int(items[-1]) + 1, -1, dtype=np.int32)
    index[items] = np.arange(len(items), dtype=np.int32)
    seen = np.zeros((FOLDS, CELL_COUNT), dtype=np.int64)
    response = np.zeros_like(seen)
    matched_events = []
    train_feeds, latest = 0, 0
    files = shards()
    for i, path in enumerate(files):
        ids, cell, labels, counts, timestamps = observations(path)
        fold = i % FOLDS
        train_feeds += len(counts)
        latest = max(latest, int(timestamps.max()))
        seen[fold] += np.bincount(cell, minlength=CELL_COUNT)
        response[fold] += np.bincount(cell[labels == 1], minlength=CELL_COUNT)
        known = (ids >= 0) & (ids < len(index))
        slots = np.full(len(ids), -1, dtype=np.int32)
        slots[known] = index[ids[known]]
        selected = slots >= 0
        matched_events.append((fold, slots[selected], cell[selected], labels[selected]))
        if (i + 1) % 128 == 0 or i + 1 == len(files):
            print(f'{i + 1}/{len(files)} train files, {seen.sum():,} candidates', flush=True)
    if latest >= eval_start:
        raise ValueError('train observations reach the evaluation period')
    rates = held_out_rates(seen, response)
    if not np.isfinite(rates).all() or np.any((rates <= 0) | (rates >= 1)):
        raise ValueError('held-out background probabilities must be finite and within (0, 1)')

    shown = np.zeros(len(items), dtype=np.int32)
    observed = np.zeros_like(shown)
    expected = np.zeros((len(SIGNALS), len(items)), dtype=np.float64)
    for fold, slots, cell, labels in matched_events:
        np.add.at(shown, slots, 1)
        np.add.at(observed, slots[labels == 1], 1)
        for model, totals in enumerate(expected):
            np.add.at(totals, slots, rates[model, fold, cell])
    if np.any(observed > shown) or not np.array_equal(expected[0] > 0, shown > 0):
        raise ValueError('invalid per-item response or expectation')
    report = {'train_files': len(files), 'train_feeds': train_feeds, 'train_candidates': int(seen.sum()),
              'train_positive': int(response.sum()), 'train_last_time': latest, 'first_evaluation_time': eval_start,
              'fold_train_candidates': seen.sum(axis=1).tolist(), 'fold_train_positives': response.sum(axis=1).tolist(),
              'evaluation_item_ids': len(items), 'items_with_train_impressions': int((shown > 0).sum()),
              'matched_train_impressions': int(shown.sum()), 'matched_train_positives': int(observed.sum()),
              'background_expected_positives_for_matched_items': dict(zip(SIGNALS, expected.sum(axis=1).tolist())),
              'background_cross_fit_expected_positives': dict(zip(SIGNALS, (rates * seen).sum(axis=(1, 2)).tolist()))}
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'items.npz', items=items, shown=shown, observed=observed, expected=expected, train_last_time=latest)
    (out / 'fit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def aligned_priors(meta, path):
    with np.load(path) as data:
        if int(data['train_last_time']) >= meta['timestamp'].min():
            raise ValueError('item priors must precede the scored split')
        ids = data['items']
        slots = np.searchsorted(ids, meta['item']).clip(max=len(ids) - 1)
        np.testing.assert_array_equal(ids[slots], meta['item'])
        return data['shown'][slots], data['observed'][slots], data['expected'][:, slots]


def item_effect(observed, expected, shrink):
    return np.clip(np.log((observed + shrink) / (expected + shrink)), -3., 3.)


def select(cache, priors, anchor_path, out):
    if out.exists():
        raise FileExistsError(out)
    meta = read_meta(cache / 'val.npz')
    counts, target, time, item = (meta[key] for key in ('counts', 'target', 'timestamp', 'item'))
    keys = prediction_frame(meta, np.zeros(len(item))).select(KEYS)
    anchor = read_scores(anchor_path, keys)
    norm = standardized_scores(anchor, counts)
    shown, observed, expected = aligned_priors(meta, priors)
    early = temporal_boundary(time, .4)
    inner = temporal_boundary(time[:early], .65)
    late_inner = early + temporal_boundary(time[early:], .5)
    if (inner, early - inner, late_inner - early, len(counts) - late_inner) != (3749, 2017, 4317, 4340):
        raise ValueError('unexpected validation split')
    early_rows = int(counts[:early].sum(dtype=np.int64))
    base_early = ndcg_per_feed(target[:early_rows], anchor[:early_rows].astype(np.float32), counts[:early])[:, 0]

    def order(effect, weight, is_early):
        rows = slice(None, early_rows if is_early else None)
        sizes = counts[:early] if is_early else counts
        if not weight:
            return anchor[rows].astype(np.float32)
        return resolve_ties(norm[rows] + weight * effect[rows], anchor[rows], item[rows], sizes)

    grid, finalists = [], []
    for signal, y_expected in zip(SIGNALS, expected, strict=True):
        entries = []
        for shrink in SHRINK:
            effect = item_effect(observed, y_expected, shrink)
            for weight in WEIGHTS:
                scores = order(effect, weight, True)
                d = ndcg_per_feed(target[:early_rows], scores, counts[:early])[:, 0] - base_early
                entries.append({'signal': signal, 'shrink': shrink, 'weight': weight, 'early_gain': float(d.mean()),
                                'early_gain_by_block': [float(d[:inner].mean()), float(d[inner:].mean())]})
        finalists.append(max(entries, key=lambda cfg: (min(cfg['early_gain_by_block']), cfg['early_gain'])))
        grid.extend(entries)
    best = max(finalists, key=lambda cfg: (min(cfg['early_gain_by_block']), cfg['early_gain']))
    chosen = best if min(best['early_gain_by_block']) > 0 else {
        **best, 'weight': 0., 'early_gain': 0., 'early_gain_by_block': [0., 0.]}

    anchor32 = anchor.astype(np.float32)
    base = ndcg_per_feed(target, anchor32, counts)
    base_hits = top_hits(target, anchor32, counts)
    blocks = {'early_first': (0, inner), 'early_second': (inner, early), 'late_first': (early, late_inner),
              'late_second': (late_inner, len(counts)), 'late': (early, len(counts)), 'all': (0, len(counts))}

    def summarize(metrics, hits):
        return {name: metric_summary(metrics, hits, *bounds) for name, bounds in blocks.items()}

    def evaluate(config):
        e = item_effect(observed, expected[SIGNALS.index(config['signal'])], config['shrink'])
        scores = order(e, config['weight'], False)
        metric = ndcg_per_feed(target, scores, counts)
        hits = top_hits(target, scores, counts)
        delta = metric - base
        return {'config': config, 'metrics': summarize(metric, hits), 'gain': summarize(delta, hits - base_hits),
                'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0])}, scores

    selected, score = evaluate(chosen)
    late_labels = target[early_rows:].astype(bool)
    coverage = {'late_candidate_fraction_with_impressions': float((shown[early_rows:] > 0).mean()),
                'late_positive_fraction_with_impressions': float((shown[early_rows:][late_labels] > 0).mean()),
                'late_positive_fraction_with_prior_response': float((observed[early_rows:][late_labels] > 0).mean())}
    with np.load(cache / 'test.npz') as data:
        test_items = data['item']
        test_time = data['timestamp']
    test_shown, _, _ = aligned_priors({'item': test_items, 'timestamp': test_time}, priors)
    coverage['test_candidate_fraction_with_impressions'] = float((test_shown > 0).mean())
    report = {'incumbent': summarize(base, base_hits), 'coverage': coverage,
              'selected': selected, 'finalists': [evaluate(cfg)[0] for cfg in finalists]}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out / 'early_search.json').write_text(json.dumps(grid, indent=2) + '\n', encoding='utf-8')
    if chosen['weight']:
        prediction_frame(meta, score).write_parquet(out / 'val_predictions.parquet')
    print(json.dumps({'selected': chosen, 'coverage': coverage, 'late_metrics': selected['metrics']['late'],
                      'late_gain': selected['gain']['late'], 'late_paired_interval': selected['late_paired_ndcg_95_interval']},
                     indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    training = commands.add_parser('fit')
    for field in ('cache', 'out'):
        training.add_argument(field, type=Path)
    validation = commands.add_parser('select')
    for field in ('cache', 'priors', 'anchor', 'out'):
        validation.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action == 'fit':
        fit(args.cache, args.out)
    else:
        select(args.cache, args.priors, args.anchor, args.out)
