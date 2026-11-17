"""Write item response features from strictly earlier train days."""
import argparse
import json
from pathlib import Path

import numpy as np

from item_priors import DAY, DAYS, START
from position_item_priors import CELL_COUNT, FOLDS, held_out_rates, item_effect, observations, shards
from ranking import read_extra_features


FEATURES = ('item_logged_expected_log', 'item_logged_response_lift_3', 'item_logged_response_lift_10')
KEYS = ('feed', 'counts', 'item', 'timestamp')


def feed_days(timestamps):
    return (timestamps.astype(np.int64) - START) // DAY


def historical_rows(meta, index, observed, expected):
    rows = index[meta['item']]
    if np.any(rows < 0):
        raise ValueError('an item is missing from the registry')
    days = np.clip(feed_days(meta['timestamp']), 0, DAYS)
    row_days = np.repeat(days.astype(np.int8), meta['counts'])
    o, e = observed[row_days, rows], expected[row_days, rows]
    return np.column_stack((np.log1p(e), item_effect(o, e, 3), item_effect(o, e, 10))).astype(np.float32), row_days


def prepare(cache, out):
    if out.exists():
        raise FileExistsError(out)
    metadata = {}
    for split in ('train', 'val', 'test'):
        with np.load(cache / f'{split}.npz') as data:
            metadata[split] = {key: data[key] for key in KEYS}
    items = np.unique(np.concatenate([meta['item'] for meta in metadata.values()]))
    if len(items) == 0 or items[0] < 0:
        raise ValueError('expected nonnegative registry item IDs')
    index = np.full(int(items[-1]) + 1, -1, dtype=np.int32)
    index[items] = np.arange(len(items), dtype=np.int32)
    shown = np.zeros((DAYS, FOLDS, CELL_COUNT), dtype=np.int64)
    positive = np.zeros_like(shown)
    events = []
    files = shards()
    for i, path in enumerate(files):
        ids, cell, target, counts, timestamps = observations(path)
        days = feed_days(timestamps)
        if np.any((days < 0) | (days >= DAYS)):
            raise ValueError(f'train day outside the prefix window in {path}')
        row_days = np.repeat(days.astype(np.int8), counts)
        fold = i % FOLDS
        # The product uses int32 to preserve the flattened day and context index.
        group = row_days.astype(np.int32) * CELL_COUNT + cell
        shown[:, fold] += np.bincount(group, minlength=DAYS * CELL_COUNT).reshape(DAYS, CELL_COUNT)
        positive[:, fold] += np.bincount(group[target == 1], minlength=DAYS * CELL_COUNT).reshape(DAYS, CELL_COUNT)
        within_registry = (ids >= 0) & (ids < len(index))
        positions = np.full(len(ids), -1, dtype=np.int32)
        positions[within_registry] = index[ids[within_registry]]
        present = positions >= 0
        events.append((fold, positions[present], row_days[present], cell[present], target[present]))
        if (i + 1) % 128 == 0 or i + 1 == len(files):
            print(f'{i + 1}/{len(files)} train files for daily item priors', flush=True)
    cumulative_shown = np.cumsum(shown, axis=0)
    cumulative_positive = np.cumsum(positive, axis=0)
    rate = np.stack([held_out_rates(n, y)[2] for n, y in zip(cumulative_shown, cumulative_positive, strict=True)])
    if not np.isfinite(rate).all() or np.any((rate <= 0) | (rate >= 1)):
        raise ValueError('daily background has invalid held-out rates')

    observed = np.zeros((DAYS + 1, len(items)), dtype=np.int32)
    expected = np.zeros((DAYS + 1, len(items)), dtype=np.float32)
    observed_total, expected_total, n_matched = 0, 0., 0
    for fold, rows, days, cell, target in events:
        slot = (days.astype(np.int64) + 1) * len(items) + rows
        weights = rate[days, fold, cell]
        np.add.at(observed.ravel(), slot[target == 1], 1)
        np.add.at(expected.ravel(), slot, weights.astype(np.float32))
        observed_total += int(target.sum())
        expected_total += weights.sum(dtype=np.float64)
        n_matched += len(rows)
    del events
    np.cumsum(observed, axis=0, dtype=observed.dtype, out=observed)
    np.cumsum(expected, axis=0, dtype=expected.dtype, out=expected)
    np.testing.assert_equal(observed[-1].sum(dtype=np.int64), observed_total)
    np.testing.assert_allclose(expected[-1].sum(dtype=np.float64), expected_total, rtol=2e-5)
    out.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, meta in metadata.items():
        features, day = historical_rows(meta, index, observed, expected)
        if (split == 'train' and (not (day == 0).any() or np.any(features[day == 0]))) or (
                split != 'train' and np.any(day != DAYS)):
            raise ValueError(f'incorrect prefix boundary in {split}')
        path = out / f'{split}.npz'
        np.savez(path, **meta, names=FEATURES, features=features)
        persisted, names = read_extra_features(path, cache / f'{split}.npz')
        np.testing.assert_array_equal(persisted, features)
        assert names == list(FEATURES)
        splits[split] = {'rows': len(features), 'feeds': len(meta['feed']),
                         'feature_coverage': float((features[:, 0] > 0).mean()),
                         'rows_with_empty_day_prefix': int((day == 0).sum())}
    report = {'train_files': len(files), 'train_candidates': int(shown.sum()), 'train_positive': int(positive.sum()),
              'registry_items': len(items), 'matched_train_impressions': n_matched, 'matched_train_positives': observed_total,
              'expected_train_positives': float(expected_total), 'splits': splits}
    (out / 'prefix_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    prepare(args.cache, args.out)
