"""Build latest-day and length-matched random train cohorts from the same cache."""
import argparse
from datetime import date
import json
from pathlib import Path

import numpy as np

from item_priors import DAY, START
from ranking import COLS


TRAIN_START = date(2026, 9, 6)


def length_matched_random(counts, target_counts, rng):
    samples = []
    lengths, size = np.unique(target_counts, return_counts=True)
    for count, number in zip(lengths, size, strict=True):
        pool = np.flatnonzero(counts == count)
        if len(pool) < number:
            raise ValueError(f'not enough random feeds of size {count}')
        samples.append(rng.choice(pool, size=number, replace=False))
    return np.sort(np.concatenate(samples))


def select_cohorts(counts, days, first, stop, seed):
    recent = np.flatnonzero((days >= first) & (days < stop))
    if not len(recent):
        raise ValueError('the recent time window has no feeds')
    control = length_matched_random(counts, counts[recent], np.random.default_rng(seed))
    np.testing.assert_array_equal(np.bincount(counts[recent]), np.bincount(counts[control]))
    return recent, control


def subset_rows(counts, feed):
    original = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    sizes = counts[feed]
    compact = np.r_[0, np.cumsum(sizes[:-1], dtype=np.int64)]
    return np.repeat(original[feed] - compact, sizes) + np.arange(sizes.sum(dtype=np.int64))


def prepare(source, out, first, stop, seed):
    if out.exists():
        raise FileExistsError(out)
    with np.load(source) as data:
        counts, timestamp, feed = (data[k] for k in ('counts', 'timestamp', 'feed'))
        if np.any(counts < 1) or len(counts) != len(feed) or len(counts) != len(timestamp):
            raise ValueError('invalid train feed keys or sizes')
        if len(np.unique(feed)) != len(feed):
            raise ValueError('train feed IDs are not unique')
        days = (timestamp.astype(np.int64) - START) // DAY
        cohorts = select_cohorts(counts, days, (first - TRAIN_START).days, (stop - TRAIN_START).days, seed)
        candidates, labels, positions = (data[k] for k in ('item', 'target', 'position'))
        matrix = data['x']
        n = counts.sum(dtype=np.int64)
        if matrix.shape != (n, COLS['history-intent']) or any(a.shape != (n,) for a in (candidates, labels, positions)):
            raise ValueError('inconsistent candidate arrays in the source cache')
        shards, prepare_seed = data['shards'], data['prepare_seed']
    out.mkdir(parents=True, exist_ok=False)
    report = {'source': str(source), 'source_feeds': len(feed), 'source_rows': int(n), 'seed': seed,
              'first_date': first.isoformat(), 'end_date_exclusive': stop.isoformat(),
              'shared_feeds': len(np.intersect1d(*cohorts)), 'cohorts': {}}
    for name, selected in zip(('recent', 'random'), cohorts, strict=True):
        rows = subset_rows(counts, selected)
        target = labels[rows]
        np.savez(out / f'{name}.npz', x=matrix[rows], feed=feed[selected], counts=counts[selected],
                 timestamp=timestamp[selected], item=candidates[rows], target=target, position=positions[rows],
                 shards=shards, prepare_seed=prepare_seed)
        day_values, day_counts = np.unique(days[selected], return_counts=True)
        date_counts = {date.fromordinal(TRAIN_START.toordinal()+int(d)).isoformat(): int(c)
                       for d, c in zip(day_values, day_counts, strict=True)}
        report['cohorts'][name] = {'feeds': len(selected), 'rows': len(rows), 'positive_rows': int(target.sum()),
                                  'feed_sizes': np.bincount(counts[selected]).tolist(), 'feeds_by_date': date_counts}
        print(f'{name}: {len(selected)} feeds, {len(rows)} candidates, {int(target.sum())} positives', flush=True)
    (out / 'cohorts.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')


def write_recency_weights(source, out, half_life):
    if out.exists():
        raise FileExistsError(out)
    if not np.isfinite(half_life) or half_life <= 0:
        raise ValueError('the recency half-life must be positive and finite')
    with np.load(source) as data:
        keys = {key: data[key] for key in ('feed', 'item', 'counts')}
        time = data['timestamp'].astype(np.int64)
    ages = (time.max() - time) / DAY
    feed_weight = np.exp2(-ages/half_life)
    feed_weight /= np.average(feed_weight, weights=keys['counts'])
    weight = np.repeat(feed_weight.astype(np.float32), keys['counts'])
    if np.any(weight <= 0):
        raise ValueError('the half-life gives some candidates zero weight')
    np.savez(out, **keys, weight=weight)
    print(json.dumps({'half_life_days': half_life, 'min': float(weight.min()), 'max': float(weight.max()),
                      'effective_row_fraction': float(1 / np.square(weight).mean(dtype=np.float64))}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    cohorts = actions.add_parser('prepare')
    weights = actions.add_parser('weights')
    for command in (cohorts, weights):
        command.add_argument('source', type=Path)
        command.add_argument('out', type=Path)
    cohorts.add_argument('--first', type=date.fromisoformat, default=date(2026, 9, 18))
    cohorts.add_argument('--stop', type=date.fromisoformat, default=date(2026, 9, 20))
    cohorts.add_argument('--seed', type=int, default=2026)
    weights.add_argument('--half-life', type=float, default=4.)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.out, args.first, args.stop, args.seed)
    else:
        write_recency_weights(args.source, args.out, args.half_life)
