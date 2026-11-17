"""Learn when to replace a top-30 candidate with a history model proposal."""
import argparse
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from depth_hard_ranker import per_feed_hits
from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores
from ranking import KEYS, NAMES, ndcg_per_feed, prediction_frame, temporal_boundary
from sequence_ranker import sha256, write_json


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache-intent-024/val.npz'
ANCHOR = ROOT / 'nn-calibration-joint-024/val_predictions.parquet'
HISTORY = ROOT / 'history-retrieval-096/shards-96/hard_val_predictions.parquet'
PENALTIES = (.01, .1, 1.)
THRESHOLDS = (.0, .005, .01, .02, .04)
LIMITS = (1, 2, 4)
POOL = 10
FEATURE_NAMES = ('anchor_in', 'anchor_out', 'history_in', 'history_out', 'log_rank_anchor_in',
                 'log_rank_history_in', 'log_rank_history_out', 'log_feed_size', 'log_history_length',
                 'anchor_gap', 'history_gap', 'history_gap_times_history_length')


def candidate_order(anchor, history, items):
    order = np.lexsort((items, -anchor))
    rank = np.empty(len(order), dtype=np.int32)
    rank[order] = np.arange(len(rank))
    history_order = np.lexsort((items, -history))
    proposed = history_order[rank[history_order] >= 30][:POOL]
    incumbent = order[30-POOL:30]
    discount = 1 / np.log2(rank + 2)
    gain = discount[incumbent][None, :] - discount[proposed][:, None]
    return anchor, proposed, incumbent, gain


def choose_swaps(quality, gain, limit, threshold):
    priorities = np.where(quality > threshold, quality * gain, -np.inf)
    promoted, demoted = [], []
    for _ in range(limit):
        i, j = np.unravel_index(np.argmax(priorities), priorities.shape)
        if not np.isfinite(priorities[i, j]):
            break
        promoted.append(i)
        demoted.append(j)
        priorities[i, :] = -np.inf
        priorities[:, j] = -np.inf
    return np.asarray(promoted, dtype=np.int64), np.asarray(demoted, dtype=np.int64)


def swap_scores(scores, promote, demote):
    result = scores.copy()
    result[promote], result[demote] = scores[demote], scores[promote]
    return result


def normalize(score):
    return ((score - score.mean()) / max(float(score.std()), 1e-7)).clip(-6, 6)


@dataclass
class Pools:
    feeds: np.ndarray
    proposals: np.ndarray
    incumbents: np.ndarray
    gain: np.ndarray
    x: np.ndarray

    def between(self, first, last):
        return (self.feeds >= first) & (self.feeds < last)


def make_pools(meta, anchor, history, hist_length):
    counts = meta['counts']
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    feeds = np.flatnonzero((counts >= 101) & (counts <= 300))
    proposal = np.empty((len(feeds), POOL), dtype=np.int64)
    incumbent = np.empty_like(proposal)
    gain = np.empty((len(feeds), POOL, POOL), dtype=np.float64)
    x = np.empty((*gain.shape, len(FEATURE_NAMES)), dtype=np.float32)
    for cell, feed in enumerate(feeds):
        start, stop = offsets[feed:feed+2]
        a, h, item = anchor[start:stop], history[start:stop], meta['item'][start:stop]
        if len(np.unique(a)) != len(a):
            raise ValueError('exact incumbent has ties within a feed')
        _, u, v, lift = candidate_order(a, h, item)
        proposal[cell], incumbent[cell], gain[cell] = start+u, start+v, lift
        za, zh = normalize(a), normalize(h)
        a_rank = np.empty(len(a), dtype=np.int32)
        h_rank = np.empty_like(a_rank)
        for score, rank in ((a, a_rank), (h, h_rank)):
            rank[np.lexsort((item, -score))] = np.arange(1, len(a) + 1)
        history_size = np.log1p(hist_length[feed])
        history_gap = zh[u, None] - zh[v]
        values = (za[u, None], za[v], zh[u, None], zh[v], np.log(a_rank[u, None]),
                  np.log(h_rank[u, None]), np.log(h_rank[v]), np.log(len(a)), history_size,
                  za[u, None] - za[v], history_gap, history_size * history_gap)
        for index, value in enumerate(values):
            x[cell, :, :, index] = value
    return Pools(feeds, proposal, incumbent, gain, x)


def fit(pools, target, prefix, penalty):
    use = pools.feeds < prefix
    x = pools.x[use].reshape(-1, len(FEATURE_NAMES)).astype(np.float64)
    y = (target[pools.proposals[use], None].astype(np.int8)
         - target[pools.incumbents[use]][:, None, :].astype(np.int8)).ravel()
    mean = x.mean(axis=0)
    scale = x.std(axis=0).clip(min=1e-7)
    x = (x - mean) / scale
    size = len(y)
    coefficient = np.linalg.solve((x.T @ x) / size + penalty * np.eye(x.shape[1]), (x.T @ y) / size)
    return {'feature_names': FEATURE_NAMES, 'mean': mean.tolist(), 'scale': scale.tolist(),
            'coefficients': coefficient.tolist(), 'intercept': float(y.mean()), 'penalty': penalty,
            'train_prefix': prefix, 'train_long_feeds': int(use.sum()), 'train_pairs': size,
            'beneficial_pairs': int((y > 0).sum()), 'harmful_pairs': int((y < 0).sum())}


def predict(x, model):
    return ((x - np.asarray(model['mean'])) / np.asarray(model['scale'])) @ np.asarray(model['coefficients']) + model['intercept']


def proposed_swaps(pools, use, quality, limit, threshold):
    first, second, feed, gain, predicted = [], [], [], [], []
    for cell, q in zip(np.flatnonzero(use), quality, strict=True):
        i, j = choose_swaps(q, pools.gain[cell], limit, threshold)
        first.extend(pools.proposals[cell, i])
        second.extend(pools.incumbents[cell, j])
        feed.extend(np.full(len(i), pools.feeds[cell], dtype=np.int64))
        predicted.extend(q[i, j])
        gain.extend(pools.gain[cell, i, j])
    return {'promote': np.asarray(first, dtype=np.int64), 'demote': np.asarray(second, dtype=np.int64),
            'feed': np.asarray(feed, dtype=np.int64), 'prediction': np.asarray(predicted), 'dcg_discount_gap': np.asarray(gain)}


def movement_stats(swaps, labels, positive_by_feed, first, last):
    promote, demote = swaps['promote'], swaps['demote']
    moved = labels[promote].astype(np.int8) - labels[demote].astype(np.int8)
    swap_feeds = swaps['feed']
    per_feed = np.zeros(last - first)
    rank = np.arange(positive_by_feed.max() + 1)
    ideal = np.r_[0., np.cumsum(1/np.log2(rank[1:]+1))]
    denominator = ideal[positive_by_feed[swap_feeds]].clip(min=1.)
    np.add.at(per_feed, swap_feeds-first, swaps['dcg_discount_gap'] * moved / denominator)
    return {'ndcg_gain': float(per_feed.mean()), 'swaps': len(moved), 'recovered': int((moved > 0).sum()),
            'displaced': int((moved < 0).sum()), 'net_top30': int(moved.sum()),
            'changed_feeds': len(np.unique(swap_feeds))}


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    meta = read_meta(CACHE)
    counts, labels, ts = (meta[k] for k in ('counts', 'target', 'timestamp'))
    early = temporal_boundary(ts, .4)
    second = temporal_boundary(ts[:early], .65)
    first = temporal_boundary(ts[:second], .5)
    mid = early + temporal_boundary(ts[early:], .5)
    if (first, second, early, mid, len(counts)) != (1870, 3749, 5766, 10083, 14423):
        raise ValueError('unexpected chronological validation blocks')
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    with np.load(CACHE) as source:
        hist_length = source['x'][offsets[:-1], NAMES.index('hist_length')]
    if not np.isfinite(hist_length).all() or np.any(hist_length < 0):
        raise ValueError('invalid history length feature')
    keys = prediction_frame(meta, np.zeros(len(labels))).select(KEYS)
    anchor = read_scores(ANCHOR, keys).astype(np.float32)
    history = read_scores(HISTORY, keys).astype(np.float32)
    pools = make_pools(meta, anchor, history, hist_length)
    n_positive = np.add.reduceat(labels.astype(np.int64), offsets[:-1])
    windows = ((first, second), (second, early))
    selection = []
    models = {}
    for penalty in PENALTIES:
        models[penalty] = [fit(pools, labels, left, penalty) for left, _ in windows]
        use = [pools.between(a, b) for a, b in windows]
        quality = [predict(pools.x[keep], model) for keep, model in zip(use, models[penalty], strict=True)]
        for threshold in THRESHOLDS:
            for limit in LIMITS:
                blocks = []
                for (a, b), keep, q in zip(windows, use, quality, strict=True):
                    swaps = proposed_swaps(pools, keep, q, limit, threshold)
                    blocks.append(movement_stats(swaps, labels, n_positive, a, b))
                selection.append({'penalty': penalty, 'threshold': threshold, 'max_swaps': limit, 'blocks': blocks})

    def key(entry):
        gains = [b['ndcg_gain'] for b in entry['blocks']]
        hits = [b['net_top30'] for b in entry['blocks']]
        return min(gains), np.average(gains, weights=(1879, 2017)), min(hits), -entry['max_swaps']

    eligible = [e for e in selection if all(b['ndcg_gain'] > 0 and b['net_top30'] > 0 for b in e['blocks'])]
    chosen = max(eligible, key=key) if eligible else {'max_swaps': 0}
    active = [e for e in selection if any(b['swaps'] for b in e['blocks'])]
    best_active = max(active, key=key) if active else None
    report = {'chosen': chosen, 'active_best': best_active, 'grid': selection, 'forward_models': models,
              'evaluation_windows': [{'prefix': a, 'evaluation': b-a} for a, b in windows],
              'eligible_configs': len(eligible), 'history_sha256': sha256(HISTORY), 'anchor_sha256': sha256(ANCHOR)}
    write_json(out / 'selection.json', report)
    print(json.dumps({k: report[k] for k in ('chosen', 'active_best', 'eligible_configs')}), flush=True)

    final_model = fit(pools, labels, early, chosen.get('penalty', best_active['penalty'] if best_active else .1))
    write_json(out / 'model.json', final_model)
    use_late = pools.between(early, len(counts))
    quality = predict(pools.x[use_late], final_model)
    swaps = proposed_swaps(pools, use_late, quality, chosen['max_swaps'], chosen.get('threshold', 0.))
    score = swap_scores(anchor, swaps['promote'], swaps['demote'])
    path = out / 'val_predictions.parquet'
    prediction_frame(meta, score).write_parquet(path)
    np.testing.assert_array_equal(read_scores(path, keys), score)
    np.savez(out / 'late_swaps.npz', **swaps)
    base, result = (ndcg_per_feed(labels, s, counts) for s in (anchor, score))
    late_delta = result - base
    blocks = {'late': np.arange(early, len(counts)), 'late_first': np.arange(early, mid),
              'late_second': np.arange(mid, len(counts)), 'late_long': pools.feeds[use_late]}
    metrics = {}
    for name, current, measure in (('anchor', anchor, base), ('proposed', score, result)):
        top10 = per_feed_hits(labels, current, counts, 10)
        top30 = per_feed_hits(labels, current, counts, 30)
        metrics[name] = {k: {**metric_summary(measure[feeds], top10[feeds], 0, len(feeds)),
                             'top30_positives': int(top30[feeds].sum())} for k, feeds in blocks.items()}
    movement = movement_stats(swaps, labels, n_positive, early, len(counts))
    np.testing.assert_allclose(movement['ndcg_gain'], late_delta[early:, 0].mean(), rtol=0, atol=1e-12)
    ci = paired_interval(late_delta[early:, 0])
    report = {'selection': chosen, 'metrics': metrics, 'movement': movement,
              'late_gain_by_block': {k: float(late_delta[feeds, 0].mean()) for k, feeds in blocks.items()},
              'late_gain_ci95': ci, 'val_prediction_sha256': sha256(path)}
    write_json(out / 'assessment.json', report)
    print(json.dumps({k: report[k] for k in ('movement', 'late_gain_by_block', 'late_gain_ci95')}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)


if __name__ == '__main__':
    main()
