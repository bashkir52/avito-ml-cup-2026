"""Test label-free cross-user exposure features from strictly earlier randomized feeds."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from depth_hard_ranker import per_feed_hits
from feed_width import metric_summary, paired_interval
from item_audience import feed_users
from neural_blend import read_meta, read_scores, resolve_ties
from ranking import KEYS, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import sha256, write_json


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache-intent-024'
ANCHOR = ROOT / 'nn-calibration-joint-024/val_predictions.parquet'
USER_FOLDS = 4
SHRINK = 4
WEIGHTS = (.025, .05, .10, .20, .35, .50)
MIN_EARLY_GAIN = .001


def exposure_features(timestamps, counts, items, users, vectors):
    if (len(timestamps) != len(users) or len(counts) != len(users) or len(vectors) != len(users)
            or np.any(np.diff(timestamps.astype(np.int64)) < 0) or np.any(counts < 1)
            or counts.sum(dtype=np.int64) != len(items) or np.any(~np.isfinite(vectors))):
        raise ValueError('invalid chronological input')
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    unique, reverse, frequency = np.unique(items, return_inverse=True, return_counts=True)
    recurring = (frequency > 1) & (unique > 0)
    compact = np.full(len(unique), -1, dtype=np.int32)
    compact[recurring] = np.arange(recurring.sum(), dtype=np.int32)
    item_index = compact[reverse]
    n = int(recurring.sum())
    width = vectors.shape[1]
    per_fold_count = np.zeros((USER_FOLDS, n), dtype=np.int32)
    all_count = np.zeros(n, dtype=np.int32)
    per_fold_vector = np.zeros((USER_FOLDS, n, width), dtype=np.float32)
    all_vector = np.zeros((n, width), dtype=np.float32)
    counts_at_feed = np.zeros(len(items), dtype=np.int32)
    affinity_at_feed = np.zeros(len(items), dtype=np.float32)
    user_folds = (users % USER_FOLDS).astype(np.int32)
    valid_user = (users > 0) & (np.any(vectors != 0, axis=1))
    background_vector = np.zeros((USER_FOLDS, width), dtype=np.float64)
    background_count = np.zeros(USER_FOLDS, dtype=np.int64)
    boundaries = np.r_[0, np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1, len(timestamps)]
    for first, last in zip(boundaries[:-1], boundaries[1:], strict=True):
        start, stop = offsets[[first, last]]
        feed = np.repeat(np.arange(first, last), counts[first:last])
        valid = (item_index[start:stop] >= 0) & valid_user[feed]
        rows = np.flatnonzero(valid) + start
        slot = item_index[rows]
        feed = feed[valid]
        fold = user_folds[feed]
        before = all_count[slot] - per_fold_count[fold, slot]
        counts_at_feed[rows] = before
        seen = before > 0
        if np.any(seen):
            which, same_fold, seen_count = slot[seen], fold[seen], before[seen]
            item_sum = all_vector[which] - per_fold_vector[same_fold, which]
            pool_sum = background_vector.sum(axis=0) - background_vector[same_fold]
            pool_count = background_count.sum() - background_count[same_fold]
            if np.any(pool_count <= 0):
                raise AssertionError('support precedes the first eligible other user')
            centered = item_sum - seen_count[:, None] * pool_sum / pool_count[:, None]
            affinity_at_feed[rows[seen]] = np.einsum('ij,ij->i', centered, vectors[feed[seen]]) / (seen_count + SHRINK)
        # Update only after every feed with the current timestamp has been scored.
        np.add.at(all_count, slot, 1)
        np.add.at(per_fold_count, (fold, slot), 1)
        np.add.at(all_vector, slot, vectors[feed])
        np.add.at(per_fold_vector, (fold, slot), vectors[feed])
        eligible_feeds = np.flatnonzero(valid_user[first:last]) + first
        np.add.at(background_vector, user_folds[eligible_feeds], vectors[eligible_feeds])
        np.add.at(background_count, user_folds[eligible_feeds], 1)
    return counts_at_feed, affinity_at_feed


def make(out, split):
    start = perf_counter()
    columns = ('feed', 'item', 'counts', 'timestamp')
    with np.load(CACHE / f'{split}.npz') as source:
        data = {key: source[key] for key in columns}
    users, unit = feed_users(split, data['feed'], data['timestamp'])
    support, affinity = exposure_features(data['timestamp'], data['counts'], data['item'], users, unit)
    if not np.isfinite(affinity).all() or np.any(np.abs(affinity) > 2.00001) or np.any(affinity[support == 0] != 0):
        raise ValueError('out-of-range or unsupported affinity')
    np.savez(out / f'{split}_signals.npz', feed=data['feed'], item=data['item'], counts=data['counts'],
             support=support, affinity=affinity)
    status = {'split': split, 'feeds': len(data['feed']), 'candidates': len(support),
              'known_user_feeds': int((users > 0).sum()), 'supports': {str(i): float((support >= i).mean()) for i in (1, 2, 4)},
              'seconds': perf_counter()-start}
    write_json(out / f'{split}_coverage.json', status)
    print(json.dumps(status), flush=True)


def normalized(signal, counts):
    return standardized_scores(signal.astype(np.float64), counts).clip(-3., 3.)


def assess(out):
    meta = read_meta(CACHE / 'val.npz')
    counts, y, ts, item = (meta[k] for k in ('counts', 'target', 'timestamp', 'item'))
    early = temporal_boundary(ts, .4)
    first = temporal_boundary(ts[:early], .65)
    mid = early + temporal_boundary(ts[early:], .5)
    if (first, early, mid, len(counts)) != (3749, 5766, 10083, 14423):
        raise ValueError('unexpected validation split')
    cut = int(counts[:early].sum(dtype=np.int64))
    keys = prediction_frame(meta, np.zeros(len(y), dtype=np.float32)).select(KEYS)
    incumbent = read_scores(ANCHOR, keys).astype(np.float32)
    with np.load(out / 'val_signals.npz') as features:
        for name in ('feed', 'item', 'counts'):
            np.testing.assert_array_equal(meta[name], features[name])
        support, affinity = features['support'], features['affinity']
    frequency = normalized(np.log1p(support), counts)
    audience = normalized(affinity, counts)
    signals = {'frequency': frequency, 'affinity': audience, 'mean': .5 * (frequency + audience)}
    z = standardized_scores(incumbent, counts)
    base = ndcg_per_feed(y[:cut], incumbent[:cut], counts[:early])[:, 0]
    search = []
    for name, signal in signals.items():
        for weight in WEIGHTS:
            scores = resolve_ties(z[:cut] + weight * signal[:cut], incumbent[:cut], item[:cut], counts[:early])
            delta = ndcg_per_feed(y[:cut], scores, counts[:early])[:, 0] - base
            search.append({'signal': name, 'weight': weight, 'early_gain': float(delta.mean()),
                           'blocks': [float(delta[:first].mean()), float(delta[first:].mean())]})
    key = lambda row: (min(row['blocks']), row['early_gain'])
    best = {name: max((r for r in search if r['signal'] == name), key=key) for name in signals}
    active_best = max(best.values(), key=key)
    selected = active_best if min(active_best['blocks']) > 0 else {'signal': 'submitted', 'weight': 0.}
    gate = min(active_best['blocks']) >= MIN_EARLY_GAIN
    write_json(out / 'selection.json', {'best_by_family': best, 'selected': selected, 'grid': search,
               'gate': MIN_EARLY_GAIN, 'gate_passed': gate, 'anchor_sha256': sha256(ANCHOR)})
    w = selected['weight']
    score = resolve_ties(z+w*signals[selected['signal']], incumbent, item, counts) if w else incumbent.copy()
    path = out / 'val_predictions.parquet'
    prediction_frame(meta, score).write_parquet(path)
    np.testing.assert_array_equal(read_scores(path, keys).astype(np.float32), score)
    base, result = (ndcg_per_feed(y, s, counts) for s in (incumbent, score))
    base_hits, hits = (per_feed_hits(y, s, counts, 10) for s in (incumbent, score))
    blocks = {'late': (early, len(counts)), 'late_first': (early, mid), 'late_second': (mid, len(counts))}
    delta = result - base
    with np.load(out / 'test_signals.npz') as test:
        coverage = {str(i): {'late_candidates': float((support[cut:] >= i).mean()),
                             'late_positives': float((support[cut:][y[cut:] == 1] >= i).mean()),
                             'test_candidates': float((test['support'] >= i).mean())} for i in (1, 2, 4)}
    report = {'selection': selected, 'early_gate_passed': gate, 'coverage': coverage,
              'incumbent': {name: metric_summary(base, base_hits, *ab) for name, ab in blocks.items()},
              'candidate': {name: metric_summary(result, hits, *ab) for name, ab in blocks.items()},
              'gain': {name: metric_summary(delta, hits-base_hits, *ab) for name, ab in blocks.items()},
              'paired_late_ndcg_95_ci': paired_interval(delta[early:, 0])}
    write_json(out / 'assessment.json', report)
    print(json.dumps({'best_by_family': best, 'selection': selected, 'late_gain': report['gain']['late'],
                      'early_gate_passed': gate, 'coverage': coverage}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=False)
    for split in ('val', 'test'):
        make(args.out, split)
    assess(args.out)


if __name__ == '__main__':
    main()
