"""Build ranker features from complete train days before the feed's timestamp."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from item_priors import DAY, DAYS, START
from ranking import DATA
from semantic_transition_rerank import transition_grids
from semantic_transitions import (GROUPS, HISTORY_CATEGORY, HISTORY_INDEX, ITEM_CATEGORY, ITEM_INDEX,
                                  encode_table, read_indices, sample_rows, transition_signal, vector_table)


GRAPH_KEYS = ('positive_edges', 'exposure_edges', 'positive_total', 'exposure_total')
RAW_COLUMNS = [ITEM_INDEX, HISTORY_INDEX, ITEM_CATEGORY, HISTORY_CATEGORY, 'timestamp']
FEATURES = [f'graph_{name.replace("-", "_")}_{kind}_{scale}' for name in GROUPS
            for kind in ('positive', 'exposure') for scale in ('raw', 'centered')]


def prior_days(timestamp, split):
    day = (timestamp.astype(np.int64) - START) // DAY
    if split == 'train':
        if np.any((day < 0) | (day >= DAYS)):
            raise ValueError('train timestamp is outside the supported calendar')
        return day
    if np.any(day < DAYS):
        raise ValueError('inference split overlaps the last train day')
    return np.full(len(day), DAYS, dtype=np.int64)


def add_daily_pairs(grid, history, dest, feed, days, weight):
    size = grid.shape[-1]
    source = history[feed]
    keep = (source >= 0) & (dest[:, None] >= 0)
    # Bin 0 is the empty history. Events from day d first appear in prefix d+1.
    offset = (np.asarray(days)[feed] + 1) * size**2 + dest
    index = offset[:, None] + source.astype(np.int32) * size
    np.add.at(grid.ravel(), index[keep], np.broadcast_to(weight[:, None], index.shape)[keep])


def add_daily_totals(total, dest, day, weights):
    valid = dest >= 0
    index = (day[valid] + 1) * total.shape[-1] + dest[valid]
    np.add.at(total.ravel(), index, weights[valid])


def accumulate(codes):
    stats = {key: (np.zeros((DAYS+1, n, n)), np.zeros((DAYS+1, n, n)),
                   np.zeros((DAYS+1, n)), np.zeros((DAYS+1, n))) for key, n in GROUPS.items()}
    paths = sorted((DATA / 'train/impressions').glob('part-*.parquet'))
    rng = np.random.default_rng(2026)
    daily_feeds = np.zeros(DAYS, dtype=np.int64)
    daily_positives = np.zeros(DAYS, dtype=np.int64)
    for i, file in enumerate(paths, 1):
        table = pq.read_table(file, columns=[*RAW_COLUMNS, 'target']).combine_chunks()
        offsets, dest, history = read_indices(table, codes)
        days = prior_days(table['timestamp'].to_numpy(), 'train')
        positive = np.flatnonzero(table['target'].chunk(0).flatten().to_numpy())
        pos_feed = np.searchsorted(offsets[1:], positive, side='right')
        sample, sample_feed, weights = sample_rows(offsets, 16, rng)
        unit = np.ones(len(positive))
        daily_feeds += np.bincount(days, minlength=DAYS)
        daily_positives += np.bincount(days[pos_feed], minlength=DAYS)
        for name, (pos, exp, pos_total, exp_total) in stats.items():
            d, h = dest[name], history[name]
            add_daily_pairs(pos, h, d[positive], pos_feed, days, unit)
            add_daily_pairs(exp, h, d[sample], sample_feed, days, weights)
            add_daily_totals(pos_total, d[positive], days[pos_feed], unit)
            add_daily_totals(exp_total, d[sample], days[sample_feed], weights)
        if i % 64 == 0 or i == len(paths):
            print(f'daily graphs: {i}/{len(paths)} train shards', flush=True)
    for matrices in stats.values():
        for a in matrices:
            np.cumsum(a, axis=0, out=a)
    return stats, {'train_shards': len(paths), 'daily_feeds': daily_feeds.tolist(),
                   'daily_positives': daily_positives.tolist(), 'first_day_epoch': START, 'days': DAYS}


def aligned_rows(raw_items, raw_offsets, cache_items, start):
    counts = np.diff(raw_offsets)
    slots = np.repeat(start - raw_offsets[:-1], counts) + np.arange(len(raw_items))
    expected = cache_items[slots]
    if np.array_equal(raw_items, expected):
        return slots
    parent = np.repeat(np.arange(len(start)), counts)
    raw_order = np.lexsort((raw_items, parent))
    cache_order = np.lexsort((expected, parent))
    if not np.array_equal(raw_items[raw_order], expected[cache_order]):
        raise ValueError('raw candidate IDs do not match feature-cache candidates')
    slots[raw_order] = slots[cache_order]
    return slots


def graph_features(cache, split, codes, stats, out):
    if out.exists():
        raise FileExistsError(out)
    with np.load(cache / f'{split}.npz') as data:
        keys = {key: data[key] for key in ('feed', 'item', 'counts', 'timestamp')}
        names = set(data['shards'].tolist()) if 'shards' in data else None
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    if names is not None:
        paths = [p for p in paths if p.name in names]
        if len(paths) != len(names):
            raise ValueError('feature cache references missing shards')
    wanted_days = np.unique(prior_days(keys['timestamp'], split))
    grids = {name: {day: transition_grids(*(a[day] for a in arrays)) for day in wanted_days}
             for name, arrays in stats.items()}
    features = np.full((len(keys['item']), len(FEATURES)), np.nan, dtype=np.float32)
    lookup = {int(key): i for i, key in enumerate(keys['feed'])}
    offsets = np.r_[0, np.cumsum(keys['counts'], dtype=np.int64)]
    seen = np.zeros(len(lookup), dtype=np.int32)
    for i, path in enumerate(paths, 1):
        table = pq.read_table(path, columns=[*RAW_COLUMNS, 'item_id', 'feed_id']).combine_chunks()
        raw_offsets, dest, histories = read_indices(table, codes)
        index = np.array([lookup[key] for key in table['feed_id'].to_numpy()], dtype=np.int64)
        np.add.at(seen, index, 1)
        days = prior_days(table['timestamp'].to_numpy(), split)
        counts = np.diff(raw_offsets)
        np.testing.assert_array_equal(counts, keys['counts'][index])
        np.testing.assert_array_equal(table['timestamp'].to_numpy(), keys['timestamp'][index])
        raw_items = table['item_id'].chunk(0).flatten().to_numpy()
        slots = aligned_rows(raw_items, raw_offsets, keys['item'], offsets[index])
        for day in np.unique(days):
            use = days == day
            candidate = np.repeat(use, counts)
            sizes = counts[use]
            row_start = np.r_[0, np.cumsum(sizes[:-1], dtype=np.int64)]
            rows = slots[candidate]
            column = 0
            for name in GROUPS:
                history, destination = histories[name][use], dest[name][candidate]
                for matrix, supported in grids[name][day].values():
                    raw, valid = transition_signal(history, destination, sizes, matrix, supported)
                    means = np.add.reduceat(raw, row_start) / np.maximum(np.add.reduceat(valid, row_start), 1)
                    features[rows[valid], column] = raw[valid]
                    features[rows[valid], column+1] = (raw - np.repeat(means, sizes))[valid]
                    column += 2
        if i % 8 == 0 or i == len(paths):
            print(f'{split} daily graph features: {i}/{len(paths)} shards', flush=True)
    if not np.all(seen == 1):
        raise ValueError('expected exactly one source feed per cached feed')
    np.savez(out, **keys, features=features, names=FEATURES)
    report = {'split': split, 'feeds': len(seen), 'rows': len(features),
              'prefix_lengths': wanted_days.tolist(), 'coverage': dict(zip(FEATURES, np.isfinite(features).mean(axis=0).tolist()))}
    print(json.dumps(report), flush=True)
    return report


def prepare(projection_path, cache, out):
    if out.exists():
        raise FileExistsError(out)
    with np.load(projection_path) as source:
        basis, planes = source['basis'], source['planes']
    codes = encode_table(vector_table('train'), basis, planes, 'train')
    stats, report = accumulate(codes)
    out.mkdir(parents=True, exist_ok=False)
    arrays = {f'{group}_{key}': value for group, values in stats.items() for key, value in zip(GRAPH_KEYS, values)}
    np.savez(out / 'daily_graph.npz', basis=basis, planes=planes, **arrays)
    train = graph_features(cache, 'train', codes, stats, out / 'train.npz')
    del codes
    codes = encode_table(vector_table('val'), basis, planes, 'val')
    val = graph_features(cache, 'val', codes, stats, out / 'val.npz')
    (out / 'metrics.json').write_text(json.dumps({**report, 'train': train, 'val': val}, indent=2)+'\n', encoding='utf-8')


def features_from_graph(graph, cache, split, output):
    with np.load(graph) as data:
        codes = encode_table(vector_table(split), data['basis'], data['planes'], split)
        stats = {group: tuple(data[f'{group}_{key}'] for key in GRAPH_KEYS) for group in GROUPS}
    graph_features(cache, split, codes, stats, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    choices = parser.add_subparsers(dest='action', required=True)
    for name in ('prepare', 'val', 'test'):
        command = choices.add_parser(name)
        command.add_argument('graph', type=Path)
        command.add_argument('cache', type=Path)
        command.add_argument('out', type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.graph, args.cache, args.out)
    else:
        features_from_graph(args.graph, args.cache, args.action, args.out)
