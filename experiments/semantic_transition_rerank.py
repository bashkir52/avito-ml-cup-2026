"""Evaluate grouped history transitions against the submitted ranker."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from neural_blend import read_meta, read_scores, resolve_ties, save_result
from ranking import DATA, KEYS, METRICS, audit_submission, ndcg_per_feed, prediction_frame, temporal_boundary
from semantic_transitions import (GROUPS, HISTORY_CATEGORY, HISTORY_INDEX, ITEM_CATEGORY, ITEM_INDEX,
                                  encode_table, read_indices, transition_signal, vector_table)


def transition_grids(y, exposure, global_y, global_exp):
    pop = (global_y + 1) / (global_y.sum() + len(global_y))
    rate = (global_y + .5) / (global_exp + 25)
    result = {}
    for name, expected in [('positive', y.sum(axis=1, keepdims=True) * pop), ('exposure', exposure * rate)]:
        # A pseudo-count toward the expected frequency shrinks rare pair effects.
        effect = np.log((y + expected + 20) / (2 * expected + 20)).clip(-2, 2).astype(np.float32)
        result[name] = (effect, (y >= 3) | (expected >= 10))
    return result


def graphs(path):
    result = {}
    with np.load(path) as data:
        for group in GROUPS:
            y = data[f'{group}_positive_edges']
            exposure = data[f'{group}_exposure_edges']
            global_y, global_exp = data[f'{group}_positive_total'], data[f'{group}_exposure_total']
            for name, grid in transition_grids(y, exposure, global_y, global_exp).items():
                result[f'{group}/{name}'] = grid
        return data['basis'], data['planes'], result


def aligned_features(meta, split, graph, selected=None):
    basis, planes, models = graphs(graph)
    if selected is not None:
        models = {selected: models[selected]}
    codes = encode_table(vector_table(split), basis, planes, split)
    n = len(meta['item'])
    features = {name: (np.empty(n, dtype=np.float32), np.empty(n, dtype=bool)) for name in models}
    offsets = np.r_[0, np.cumsum(meta['counts'], dtype=np.int64)]
    feed_lookup = {int(key): i for i, key in enumerate(meta['feed'])}
    seen = np.zeros(len(feed_lookup), dtype=np.int32)
    columns = ['feed_id', 'item_id', ITEM_INDEX, HISTORY_INDEX, ITEM_CATEGORY, HISTORY_CATEGORY]
    for path in sorted((DATA / split / 'impressions').glob('part-*.parquet')):
        table = pq.read_table(path, columns=columns).combine_chunks()
        raw_offsets, destinations, history = read_indices(table, codes)
        feed = np.array([feed_lookup[key] for key in table['feed_id'].to_numpy()], dtype=np.int64)
        np.add.at(seen, feed, 1)
        counts = np.diff(raw_offsets)
        np.testing.assert_array_equal(counts, meta['counts'][feed])
        rows = np.repeat(offsets[feed] - raw_offsets[:-1], counts) + np.arange(raw_offsets[-1])
        np.testing.assert_array_equal(meta['item'][rows], table['item_id'].chunk(0).flatten().to_numpy())
        for name, (effect, supported) in models.items():
            group = name.split('/')[0]
            signal, present = transition_signal(history[group], destinations[group], counts, effect, supported)
            anchors = raw_offsets[:-1]
            center = np.add.reduceat(signal, anchors) / np.maximum(np.add.reduceat(present, anchors), 1)
            scores, coverage = features[name]
            scores[rows] = np.where(present, signal - np.repeat(center, counts), 0)
            coverage[rows] = present
    if not np.all(seen == 1):
        raise ValueError('raw split does not cover each cache feed exactly once')
    return features


def top_hits(label, score, counts):
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    hits = []
    for a, b in zip(offsets[:-1], offsets[1:]):
        top = np.argsort(-score[a:b], kind='stable')[:10]
        hits.append(int(label[a:b][top].sum()))
    return np.array(hits, dtype=np.int16)


def rank(meta, score, extra, weight, scope):
    counts = meta['counts']
    correction = weight * extra
    if scope == 'long':
        correction *= np.repeat(counts > 100, counts)
    return resolve_ties(score + correction, score, meta['item'], counts)


def select(cache, graph, anchor, out):
    if out.exists():
        raise FileExistsError(out)
    meta = read_meta(cache / 'val.npz')
    y, counts, time = (meta[k] for k in ('target', 'counts', 'timestamp'))
    keys = prediction_frame(meta, np.zeros(len(y))).select(KEYS)
    baseline = read_scores(anchor, keys).astype(np.float32)
    base = ndcg_per_feed(y, baseline, counts)
    features = aligned_features(meta, 'val', graph)
    cutoff = temporal_boundary(time, .4)
    inner = temporal_boundary(time[:cutoff], .65)
    midpoint = cutoff + temporal_boundary(time[cutoff:], .5)
    early_rows = int(counts[:cutoff].sum())
    early_meta = {'item': meta['item'][:early_rows], 'counts': counts[:cutoff]}
    weights = (-.10, -.05, .025, .05, .10, .20, .40)

    def validation(name, weight, scope):
        scores = rank(early_meta, baseline[:early_rows], features[name][0][:early_rows], weight, scope)
        d = ndcg_per_feed(y[:early_rows], scores, counts[:cutoff])[:, 0] - base[:cutoff, 0]
        return {'signal': name, 'weight': weight, 'scope': scope,
                'early_halves_gain': [float(d[:inner].mean()), float(d[inner:].mean())],
                'early_gain': float(d.mean())}

    def robust(config):
        return min(config['early_halves_gain']), config['early_gain']

    scopes = ('all', 'long')
    grid = [validation(name, weight, scope) for name in features for scope in scopes for weight in weights]
    finalists = [max((entry for entry in grid if entry['signal'] == name and entry['scope'] == scope), key=robust)
                 for name in features for scope in scopes]
    best_nonzero = max(finalists, key=robust)
    chosen = best_nonzero if robust(best_nonzero)[0] > 0 else {
        'signal': best_nonzero['signal'], 'weight': 0., 'scope': best_nonzero['scope'],
        'early_halves_gain': [0., 0.], 'early_gain': 0.}
    original_hits = top_hits(y[early_rows:], baseline[early_rows:], counts[cutoff:])

    def evaluate(config):
        signal, coverage = features[config['signal']]
        score = rank(meta, baseline, signal, config['weight'], config['scope'])
        metric = ndcg_per_feed(y, score, counts)
        hits = top_hits(y[early_rows:], score[early_rows:], counts[cutoff:])
        delta = metric - base
        present = coverage[early_rows:]
        actual = y[early_rows:] == 1
        report = {**config, 'late_metrics': dict(zip(METRICS, metric[cutoff:].mean(axis=0).tolist())),
                  'late_gain': dict(zip(METRICS, delta[cutoff:].mean(axis=0).tolist())),
                  'late_halves_gain': [float(delta[cutoff:midpoint, 0].mean()), float(delta[midpoint:, 0].mean())],
                  'late_candidate_coverage': float(present.mean()), 'late_positive_coverage': float(present[actual].mean()),
                  'late_positive_hits_at_10': int(hits.sum()), 'late_feeds_with_more_positive_hits': int((hits > original_hits).sum()),
                  'late_feeds_with_fewer_positive_hits': int((hits < original_hits).sum())}
        return report, score, metric

    selected, score, metric = evaluate(chosen)
    out.mkdir(parents=True, exist_ok=False)
    report = {'anchor': str(anchor), 'graph': str(graph), 'early_sizes': [inner, cutoff-inner],
              'late_sizes': [midpoint-cutoff, len(counts)-midpoint], 'late_positive_count': int(y[early_rows:].sum()),
              'incumbent_late': dict(zip(METRICS, base[cutoff:].mean(axis=0).tolist())),
              'incumbent_positive_hits_at_10': int(original_hits.sum()), 'selected': selected,
              'best_nonzero_by_signal': [evaluate(config)[0] for config in finalists]}
    (out / 'diagnostics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (out / 'early_search.json').write_text(json.dumps(grid, indent=2)+'\n', encoding='utf-8')
    np.savez(out / 'val_graph_scores.npz', feed=meta['feed'], item=meta['item'], counts=counts,
             **{name.replace('/', '_'): values[0] for name, values in features.items()})
    save_result(out, selected, score, metric[:, 0], base[:, 0], meta, cutoff)


def predict(cache, graph, config, anchor, output):
    if output.exists():
        raise FileExistsError(output)
    selected = json.loads(config.read_text(encoding='utf-8'))['selected']
    name, weight = selected['signal'], selected['weight']
    meta = read_meta(cache / 'test.npz')
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    baseline = read_scores(anchor, keys).astype(np.float32)
    features = aligned_features(meta, 'test', graph, name)
    score = rank(meta, baseline, features[name][0], weight, selected['scope'])
    prediction_frame(meta, score).write_parquet(output)
    print(json.dumps(audit_submission(output), indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='command', required=True)
    for command in ('select', 'predict'):
        action = actions.add_parser(command)
        action.add_argument('cache', type=Path)
        action.add_argument('graph', type=Path)
        if command == 'predict':
            action.add_argument('config', type=Path)
        action.add_argument('anchor', type=Path)
        action.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.command == 'select':
        select(args.cache, args.graph, args.anchor, args.out)
    else:
        predict(args.cache, args.graph, args.config, args.anchor, args.out)
