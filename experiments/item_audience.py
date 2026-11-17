"""Measure a cross-user audience signal on randomized Avito feeds."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores, resolve_ties
from position_item_priors import CELL_COUNT, CELL_SHAPE, FOLDS, LENGTH_CUTS, POSITION_CUTS, held_out_rates
from ranking import DATA, KEYS, ndcg_per_feed, normalize, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import top10_hits, write_json


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache-intent-024'
ANCHOR = ROOT / 'nn-calibration-joint-024/val_predictions.parquet'
WEIGHTS = (.05, .10, .20, .35, .50, 1.)
SHRINK = (1, 4)


def paths(split):
    files = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    if len(files) != (512 if split == 'train' else 64):
        raise ValueError(f'incomplete {split} input')
    return files


def feed_users(split, feeds, timestamp):
    tables = [pq.read_table(p, columns=['feed_id', 'timestamp', 'user_id', 'user_embedding']) for p in paths(split)]
    ids = np.concatenate([t['feed_id'].to_numpy() for t in tables])
    ts = np.concatenate([t['timestamp'].to_numpy() for t in tables])
    users = np.concatenate([t['user_id'].to_numpy() for t in tables])
    vectors = np.concatenate([np.asarray(t['user_embedding'].to_pylist(), dtype=np.int8) for t in tables])
    order = np.lexsort((ids, ts))
    np.testing.assert_array_equal(ids[order], feeds)
    np.testing.assert_array_equal(ts[order], timestamp)
    return users[order], normalize(vectors[order])


def train_rows(path):
    columns = ['timestamp', 'user_id', 'user_embedding', 'item_id', 'pos', 'category_id', 'target']
    table = pq.read_table(path, columns=columns).combine_chunks()
    objects = [table[k].chunk(0) for k in ('item_id', 'pos', 'category_id', 'target')]
    offsets = objects[0].offsets.to_numpy()
    for obj in objects[1:]:
        np.testing.assert_array_equal(obj.offsets.to_numpy(), offsets)
    item, position, category, y = [obj.flatten().to_numpy() for obj in objects]
    counts = np.diff(offsets)
    if np.any((position < 0) | (position > 300)) or np.any((y != 0) & (y != 1)):
        raise ValueError(f'bad positions or labels in {path}')
    band = np.searchsorted(POSITION_CUTS, position, side='right')
    lengths = np.repeat(np.searchsorted(LENGTH_CUTS, counts, side='left'), counts)
    cell = ((band * CELL_SHAPE[1] + lengths) * CELL_SHAPE[2] + category).astype(np.int16)
    values = table['user_embedding'].chunk(0)
    if not np.all(np.diff(values.offsets.to_numpy()) == 128):
        raise ValueError('invalid user embedding size')
    user_vectors = values.flatten().to_numpy().reshape(-1, 128)
    return table['timestamp'].to_numpy(), table['user_id'].to_numpy(), user_vectors, item, y, offsets, cell


def lookup(keys, query):
    index = np.searchsorted(keys, query).clip(max=len(keys) - 1)
    return np.where(keys[index] == query, index, -1).astype(np.int32)


def build(out):
    meta = read_meta(CACHE / 'val.npz')
    with np.load(CACHE / 'test.npz') as test:
        test_meta = {name: test[name] for name in ('feed', 'counts', 'item', 'timestamp')}
    val_users, val_z = feed_users('val', meta['feed'], meta['timestamp'])
    test_users, test_z = feed_users('test', test_meta['feed'], test_meta['timestamp'])
    # Exclude all recognized evaluation users from source audiences, so gains
    # measure transfer across users and cannot be driven by self matching.
    excluded = np.unique(np.r_[val_users, test_users, 0])
    eval_items = np.unique(np.r_[meta['item'], test_meta['item']])
    with np.load(ROOT / 'position-item-priors-fit/items.npz') as prior:
        if not np.array_equal(prior['items'], eval_items):
            raise ValueError('prior keys differ from evaluation cache')
        eligible = eval_items[(prior['observed'] > 0) & (eval_items != 0)]
    positions = np.full(int(eligible[-1]) + 1, -1, dtype=np.int32)
    positions[eligible] = np.arange(len(eligible), dtype=np.int32)
    shown = np.zeros((FOLDS, CELL_COUNT), dtype=np.int64)
    positive = np.zeros_like(shown)
    latest, total_feeds, cohort_rows, cohort_positives = 0, 0, 0, 0
    collected = []
    start = time.perf_counter()
    for i, path in enumerate(paths('train')):
        timestamp, users, vec, items, y, offsets, cell = train_rows(path)
        fold = i % FOLDS
        latest = max(latest, int(timestamp.max()))
        total_feeds += len(users)
        shown[fold] += np.bincount(cell, minlength=CELL_COUNT)
        positive[fold] += np.bincount(cell[y == 1], minlength=CELL_COUNT)
        known = (items > 0) & (items < len(positions))
        known_rows = np.flatnonzero(known)
        slot = positions[items[known_rows]]
        matching = slot >= 0
        matching_rows = known_rows[matching]
        owner = np.searchsorted(offsets[1:], matching_rows, side='right')
        user_ok = ~np.isin(users, excluded) & np.any(vec != 0, axis=1)
        keep = user_ok[owner]
        rows = matching_rows[keep]
        collected.append((fold, slot[matching][keep], owner[keep].astype(np.int32),
                          y[rows].astype(bool), cell[rows], vec.copy()))
        cohort_rows += len(rows)
        cohort_positives += int(y[rows].sum())
        if (i + 1) % 128 == 0:
            print(f'{i+1}/512 train shards, {cohort_positives:,} matched responses', flush=True)
    if latest >= min(meta['timestamp'].min(), test_meta['timestamp'].min()):
        raise ValueError('audience train timestamps overlap evaluation')
    if int(shown.sum()) != 246745836 or int(positive.sum()) != 5446872:
        raise ValueError('train row counts differ from full dataset audit')
    rates = held_out_rates(shown, positive)[2].astype(np.float32)
    pos_sum = np.zeros((len(eligible), 128), dtype=np.float32)
    exp_sum = np.zeros_like(pos_sum)
    n_pos = np.zeros(len(eligible), dtype=np.int32)
    n_exp = np.zeros(len(eligible), dtype=np.float32)
    for fold, item, owner, label, cell, vec in collected:
        unit = normalize(vec)
        weights = rates[fold, cell]
        np.add.at(n_pos, item[label], 1)
        np.add.at(n_exp, item, weights)
        np.add.at(pos_sum, item[label], unit[owner[label]])
        np.add.at(exp_sum, item, unit[owner] * weights[:, None])
    if int(n_pos.sum()) != cohort_positives:
        raise ValueError('unexpected number of cohort responses')
    summary = {'train_files': 512, 'train_feeds': total_feeds, 'train_candidates': int(shown.sum()),
               'train_positives': int(positive.sum()), 'train_end': latest, 'excluded_user_ids': len(excluded),
               'matched_train_candidates': cohort_rows, 'matched_train_responses': cohort_positives,
               'items_with_response_vector': int((n_pos > 0).sum()), 'seconds': time.perf_counter() - start}
    for split, data, vectors in (('val', meta, val_z), ('test', test_meta, test_z)):
        slots = lookup(eligible, data['item'])
        recognized = (slots >= 0) & (n_pos[slots] > 0)
        hits = np.flatnonzero(recognized)
        feed_idx = np.repeat(np.arange(len(data['counts'])), data['counts'])
        responses = np.zeros(len(slots), dtype=np.int32)
        responses[hits] = n_pos[slots[hits]]
        raw, expected = np.zeros(len(slots), dtype=np.float32), np.zeros(len(slots), dtype=np.float32)
        for start in range(0, len(hits), 20000):
            rows = hits[start:start+20000]
            ids = slots[rows]
            user = vectors[feed_idx[rows]]
            raw[rows] = np.einsum('ij,ij->i', user, pos_sum[ids]) / n_pos[ids]
            expected[rows] = np.einsum('ij,ij->i', user, exp_sum[ids]) / n_exp[ids]
        np.savez(out / f'{split}_signals.npz', feed=data['feed'], item=data['item'], counts=data['counts'],
                 raw=raw, expected=expected, responses=responses)
        summary[f'{split}_candidate_coverage'] = float(recognized.mean())
    write_json(out / 'cohorts.json', summary)
    print(json.dumps(summary), flush=True)


def effect(raw, expected, responses, family, shrink):
    signal = (raw - expected) if family == 'conditional' else raw
    return signal * responses / (responses + shrink)


def assess(out):
    meta = read_meta(CACHE / 'val.npz')
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    anchor = read_scores(ANCHOR, keys).astype(np.float32)
    counts, target, ts = (meta[k] for k in ('counts', 'target', 'timestamp'))
    early = temporal_boundary(ts, .4)
    first = temporal_boundary(ts[:early], .65)
    middle = early + temporal_boundary(ts[early:], .5)
    if (first, early, middle, len(counts)) != (3749, 5766, 10083, 14423):
        raise ValueError('unexpected chronological split')
    cut = int(counts[:early].sum())
    with np.load(out / 'val_signals.npz') as signals:
        for name in ('feed', 'item', 'counts'):
            np.testing.assert_array_equal(signals[name], meta[name])
        raw, expected, responses = (signals[k] for k in ('raw', 'expected', 'responses'))
    for v in (raw, expected):
        if not np.isfinite(v).all():
            raise ValueError('nonfinite cohort signal')
    norm = standardized_scores(anchor, counts)
    reference = ndcg_per_feed(target[:cut], anchor[:cut], counts[:early])[:, 0]
    grid = []
    for family in ('conditional', 'positive_mean'):
        for shrink in SHRINK:
            adjustment = effect(raw[:cut], expected[:cut], responses[:cut], family, shrink)
            for weight in WEIGHTS:
                # A 0.1 cosine difference is one unit of the auxiliary score.
                score = resolve_ties(norm[:cut] + 10 * weight * adjustment, anchor[:cut], meta['item'][:cut], counts[:early])
                gain = ndcg_per_feed(target[:cut], score, counts[:early])[:, 0] - reference
                grid.append({'family': family, 'shrink': shrink, 'weight': weight,
                             'gain': float(gain.mean()), 'blocks': [float(gain[:first].mean()), float(gain[first:].mean())]})
    best = max(grid, key=lambda entry: (min(entry['blocks']), entry['gain']))
    chosen = best if min(best['blocks']) > 0 else {'family': best['family'], 'shrink': best['shrink'], 'weight': 0.}
    write_json(out / 'early_selection.json', {'selected': chosen, 'best_nonzero': best, 'grid': grid})
    adjustment = effect(raw, expected, responses, chosen['family'], chosen['shrink'])
    score = (resolve_ties(norm + 10 * chosen['weight'] * adjustment, anchor, meta['item'], counts)
             if chosen['weight'] else anchor.copy())
    base, result = (ndcg_per_feed(target, s, counts) for s in (anchor, score))
    base_hits, hits = (top10_hits(target, s, counts) for s in (anchor, score))
    blocks = {'late': (early, len(counts)), 'late_first': (early, middle), 'late_second': (middle, len(counts))}
    long = (counts[early:] >= 101) & (counts[early:] <= 300)
    delta = result - base
    hit_delta = hits - base_hits
    report = {'selection': chosen,
              'baseline': {k: metric_summary(base, base_hits, *v) for k, v in blocks.items()},
              'candidate': {k: metric_summary(result, hits, *v) for k, v in blocks.items()},
              'gain': {k: metric_summary(delta, hit_delta, *v) for k, v in blocks.items()},
              'long_late_gain': metric_summary(delta[early:][long], hit_delta[early:][long], 0, int(long.sum())),
              'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0]),
              'coverage': {str(n): {'late_candidates': float((responses[cut:] >= n).mean()),
                                    'late_positives': float((responses[cut:][target[cut:] == 1] >= n).mean()),
                                    'test_candidates': float((test_responses >= n).mean())}
                           for test_responses in [np.load(out / 'test_signals.npz')['responses']]
                           for n in (1, 2, 4, 8)}}
    path = out / 'val_predictions.parquet'
    prediction_frame(meta, score).write_parquet(path)
    np.testing.assert_array_equal(read_scores(path, keys), score)
    write_json(out / 'assessment.json', report)
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    build(args.out)
    assess(args.out)


if __name__ == '__main__':
    main()
