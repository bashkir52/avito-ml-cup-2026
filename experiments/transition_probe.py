"""Probe exact history-to-positive item transitions from train on randomized validation feeds."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow.parquet as pq

from neural_blend import read_meta, read_scores, resolve_ties, save_result
from ranking import DATA, KEYS, ndcg_per_feed, prediction_frame, temporal_boundary


RECENT = 12
ITEM_BASE = 2**32


def recent_items(history):
    offsets = history.offsets.to_numpy()
    positions = offsets[:-1, None] + np.arange(RECENT)
    items = np.r_[history.values.to_numpy(), np.int32(0)]
    selected = items[np.where(positions < offsets[1:, None], positions, len(items)-1)]
    selected.sort(axis=1)
    selected[:, 1:] = np.where(selected[:, 1:] != selected[:, :-1], selected[:, 1:], 0)
    return selected


def fit(sources, destinations):
    edges, source_counts, destination_counts = Counter(), Counter(), Counter()
    source_mask = np.zeros(int(sources.max()) + 1, dtype=bool)
    target_mask = np.zeros(int(destinations.max()) + 1, dtype=bool)
    source_mask[sources[sources > 0]] = True
    target_mask[destinations[destinations > 0]] = True
    files = sorted((DATA / 'train/impressions').glob('part-*.parquet'))
    for index, path in enumerate(files, 1):
        table = pq.read_table(path, columns=['item_id', 'target', 'history__item_id']).combine_chunks()
        item = table['item_id'].chunk(0)
        positive = np.flatnonzero(table['target'].chunk(0).flatten().to_numpy())
        targets = item.flatten().to_numpy()[positive]
        feed = np.searchsorted(item.offsets.to_numpy()[1:], positive, side='right')
        history = recent_items(table['history__item_id'].chunk(0))[feed]
        source_ids = np.where((history >= 0) & (history < len(source_mask)), history, 0)
        target_ids = np.where((targets >= 0) & (targets < len(target_mask)), targets, 0)
        source_ok = source_mask[source_ids] & (history != targets[:, None])
        target_ok = target_mask[target_ids]
        selected = source_ok & target_ok[:, None]
        pair_grid = history.astype(np.int64) * ITEM_BASE + targets[:, None]
        edges.update(pair_grid[selected].tolist())
        source_counts.update(history[source_ok].tolist())
        destination_counts.update(targets[target_ok].tolist())
        if index % 128 == 0 or index == len(files):
            print(f'{index}/{len(files)} train files; {len(edges)} observed transitions', flush=True)
    keys = np.array(sorted(edges), dtype=np.int64)
    scores = np.array([np.log1p(edges[k] / np.sqrt((source_counts[k // ITEM_BASE] + 5)
                                                  * (destination_counts[k % ITEM_BASE] + 5))) for k in keys])
    return keys, scores


def validation_history():
    frames = []
    for path in sorted((DATA / 'val/impressions').glob('part-*.parquet')):
        table = pq.read_table(path, columns=['feed_id', 'history__item_id']).combine_chunks()
        frames.append(pl.DataFrame({'feed': table['feed_id'].to_numpy(),
                                    'history': recent_items(table['history__item_id'].chunk(0))}))
    frame = pl.concat(frames).sort('feed')
    return frame['feed'].to_numpy(), frame['history'].to_numpy()


def best_transition(history, items, counts, keys, scores):
    if not len(keys):
        return np.zeros(len(items), dtype=np.float32)
    best = np.zeros(len(items), dtype=np.float32)
    for column in history.T:
        pairs = np.repeat(column, counts).astype(np.int64) * ITEM_BASE + items
        index = np.searchsorted(keys, pairs).clip(max=len(keys)-1)
        np.maximum(best, np.where(keys[index] == pairs, scores[index], 0), out=best)
    return best


def run(cache, anchor, out):
    if out.exists():
        raise FileExistsError(out)
    meta = read_meta(cache / 'val.npz')
    feed, history = validation_history()
    order = np.searchsorted(feed, meta['feed'])
    np.testing.assert_array_equal(feed[order], meta['feed'])
    history = history[order]
    items, counts, y = (meta[key] for key in ('item', 'counts', 'target'))
    baseline = read_scores(anchor, prediction_frame(meta, np.zeros(len(items))).select(KEYS)).astype(np.float32)
    keys, scores = fit(np.unique(history), np.unique(items))
    transition = best_transition(history, items, counts, keys, scores)
    cutoff = temporal_boundary(meta['timestamp'], .4)
    row_cut = int(counts[:cutoff].sum())
    orig = ndcg_per_feed(y, baseline, counts)[:, 0]
    grid = (0., .05, .1, .25, .5, 1., 2.)
    early = [float(ndcg_per_feed(y[:row_cut], baseline[:row_cut] + w*transition[:row_cut], counts[:cutoff])[:, 0].mean())
             for w in grid]
    weight = grid[int(np.argmax(early))]
    score = resolve_ties(baseline + weight*transition, baseline, items, counts)
    metric = ndcg_per_feed(y, score, counts)[:, 0]
    late = slice(row_cut, None)
    is_covered = transition[late] > 0
    positive = y[late] == 1
    covered_feeds = np.logical_or.reduceat(is_covered, np.r_[0, np.cumsum(counts[cutoff:-1])])
    out.mkdir(parents=True, exist_ok=False)
    result = {'edges': len(keys), 'lookback_events': RECENT, 'weight': weight,
              'late_candidate_coverage': float(is_covered.mean()), 'late_positive_coverage': float(is_covered[positive].mean()),
              'late_feed_coverage': float(covered_feeds.mean()), 'early_grid': dict(zip(grid, early)),
              'anchor': str(anchor)}
    save_result(out, result, score, metric, orig, meta, cutoff)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('anchor', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    run(args.cache, args.anchor, args.out)
