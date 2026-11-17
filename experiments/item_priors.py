"""Add item-level priors from complete, strictly earlier train days."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow.parquet as pq

from ranking import DATA


DAY = 86_400
START = int(datetime(2026, 9, 6, tzinfo=timezone(timedelta(hours=3))).timestamp())
DAYS = 15  # Includes the small Sep 20 boundary in the published train files.


def item_mapping(paths):
    items = []
    for path in paths:
        with np.load(path) as f:
            items.append(np.unique(f['item']))
    unique = np.unique(np.concatenate(items))
    if unique[0] < 0:
        raise ValueError('negative item ID in feature cache')
    mapping = np.full(int(unique[-1]) + 1, -1, dtype=np.int32)
    mapping[unique] = np.arange(len(unique), dtype=np.int32)
    return mapping, len(unique)


def accumulate(mapping, size):
    exposure = np.zeros((DAYS + 1, size), dtype=np.int32)
    positive = np.zeros_like(exposure)
    train_files = sorted((DATA / 'train' / 'impressions').glob('part-*.parquet'))
    x, y = exposure.ravel(), positive.ravel()
    for i, path in enumerate(train_files, 1):
        data = pq.read_table(path, columns=['timestamp', 'item_id', 'target']).combine_chunks()
        item = data['item_id'].chunk(0)
        values = item.flatten().to_numpy()
        valid = (values > 0) & (values < len(mapping))
        local = np.full(len(values), -1, dtype=np.int32)
        local[valid] = mapping[values[valid]]
        valid = local >= 0
        day = (data['timestamp'].to_numpy() - START) // DAY
        if np.any((day < 0) | (day >= DAYS)):
            raise ValueError('train timestamp outside supported days')
        candidate_day = np.repeat(day, np.diff(item.offsets.to_numpy()))
        index = (candidate_day[valid].astype(np.int64) + 1) * size + local[valid]
        labels = data['target'].chunk(0).flatten().to_numpy()[valid]
        np.add.at(x, index, 1)
        np.add.at(y, index[labels == 1], 1)
        if i == len(train_files) or i % 64 == 0:
            print(f'Aggregated {i}/{len(train_files)} train files', flush=True)
    np.cumsum(exposure, axis=0, dtype=np.int32, out=exposure)
    np.cumsum(positive, axis=0, dtype=np.int32, out=positive)
    return exposure, positive


def prior_features(data, mapping, exposure, positive):
    days = np.clip((data['timestamp'] - START) // DAY, 0, DAYS)
    index = mapping[data['item']]
    # Prefix row d includes days 0..d-1. Every candidate in a feed uses its day's prefix.
    end = np.repeat(days, data['counts'])
    start = np.maximum(end - 5, 0)
    x = exposure[end, index]
    y = positive[end, index]
    recent = x - exposure[start, index]
    return np.column_stack((np.log1p(x), np.log1p(y), (y + .4) / (x + 20),
                            np.log1p(recent))).astype(np.float32)


def write_augmented(source, output, mapping, exposure, positive):
    with np.load(source) as f:
        data = dict(f)
    features = prior_features(data, mapping, exposure, positive)
    original = data.pop('x')
    data['x'] = np.concatenate((original, features), axis=1)
    del original
    with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez(tmp, **data)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return {'rows': len(features), 'item_seen_fraction': float((features[:, 0] > 0).mean())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('train_file', type=Path)
    parser.add_argument('validation_test_cache', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    base = {'train': args.train_file, **{s: args.validation_test_cache / f'{s}.npz' for s in ('val', 'test')}}
    args.output.mkdir(parents=True, exist_ok=False)
    mapping, size = item_mapping(base.values())
    print(f'Aggregating histories for {size} item IDs', flush=True)
    exposure, positive = accumulate(mapping, size)
    report = {}
    for split, source in base.items():
        report[split] = write_augmented(source, args.output / f'{split}.npz', mapping, exposure, positive)
        print(f'{split}: {report[split]}', flush=True)
    (args.output / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
