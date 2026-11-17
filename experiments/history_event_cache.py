"""Collect individual history events for candidate-conditioned attention."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ranking import DATA, NAMES

ANCHORS = (8, 12, 16, 24, 32, 48, 64, 96)
SLOTS = 8 + len(ANCHORS)


def select_events(indices):
    available = np.flatnonzero(np.asarray(indices, dtype=np.int32) >= 0)
    recent = available[:8]
    chosen = list(recent)
    cursor = int(recent[-1]) + 1 if len(recent) else 0
    for anchor in ANCHORS:
        i = available.searchsorted(max(cursor, anchor))
        if i < len(available):
            chosen.append(int(available[i]))
            cursor = chosen[-1] + 1
    return np.asarray(chosen, dtype=np.int32)


def fetch_vectors(path, metadata, indices):
    source = np.memmap(path, dtype=metadata['dtype'], mode='r', shape=tuple(metadata['shape']),
                       order=metadata['order'])
    mask = indices >= 0
    needed, reverse = np.unique(indices[mask], return_inverse=True)
    if len(needed) and needed[-1] >= len(source):
        raise ValueError('history embedding index out of range')
    compact = np.empty((len(needed), source.shape[1]), dtype=source.dtype)
    chunk = 1 << 19
    for start in range(0, source.shape[0], chunk):
        end = min(start + chunk, source.shape[0])
        lo, hi = needed.searchsorted([start, end])
        if lo < hi:
            block = source[start:end].copy()
            compact[lo:hi] = block[needed[lo:hi] - start]
    values = np.zeros((*indices.shape, source.shape[1]), dtype=source.dtype)
    values[mask] = compact[reverse]
    return values


def build(split, base, output):
    with np.load(base) as f:
        feed = f['feed']
        names = set(f['shards'].tolist()) if 'shards' in f else None
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    if names is not None:
        paths = [p for p in paths if p.name in names]
        if len(paths) != len(names):
            raise ValueError('cached shards are missing')
    names = ['history__item_id__embedding_idx', 'history__event_type',
             'history__event_source', 'history__category_id']
    indices = np.full((len(feed), SLOTS), -1, dtype=np.int32)
    events = np.zeros((len(feed), SLOTS, 4), dtype=np.int16)
    events[..., 0] = -1
    lookup = {int(k): i for i, k in enumerate(feed)}
    seen = np.zeros(len(feed), dtype=np.int32)
    for number, path in enumerate(paths, 1):
        for batch in pq.ParquetFile(path).iter_batches(columns=['feed_id', *names], batch_size=256):
            ids = batch['feed_id'].to_numpy()
            arrays = [pc.list_slice(batch[name], 0, 128).to_pylist() for name in names]
            for row, arrays_for_feed in enumerate(zip(*arrays, strict=True)):
                i = lookup[ids[row]]
                seen[i] += 1
                positions = select_events(arrays_for_feed[0])
                n = len(positions)
                if n:
                    indices[i, :n] = np.asarray(arrays_for_feed[0])[positions]
                    events[i, :n, 0] = positions
                    events[i, :n, 1:] = np.column_stack(arrays_for_feed[1:])[positions]
        if number % 8 == 0 or number == len(paths):
            print(f'{split} history: {number}/{len(paths)} shards', flush=True)
    if not np.all(seen == 1):
        raise ValueError('history does not exactly cover the cached feeds')
    source = DATA / split / 'item_embeddings'
    metadata = json.loads((source / 'data.meta').read_text())
    vectors = fetch_vectors(source / 'data.memmap', metadata, indices)
    with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez(temp, feed=feed, event_vectors=vectors, events=events)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'split': split, 'feeds': len(feed), 'max_slots': SLOTS,
                      'mean_available_slots': float((indices >= 0).sum(axis=1).mean())}), flush=True)


def build_nearest(split, base, output):
    with np.load(base) as f:
        feed, item, counts = (f[k] for k in ('feed', 'item', 'counts'))
        ranks = f['x'][:, NAMES.index('intent_match_event_rank')].copy()
        names = set(f['shards'].tolist()) if 'shards' in f else None
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    if names is not None:
        paths = [p for p in paths if p.name in names]
        if len(paths) != len(names):
            raise ValueError('cached shards are missing')
    indices = np.full(len(item), -1, dtype=np.int32)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    lookup = {int(k): i for i, k in enumerate(feed)}
    seen = np.zeros(len(feed), dtype=np.int32)
    hist = 'history__item_id__embedding_idx'
    for n, path in enumerate(paths, 1):
        for batch in pq.ParquetFile(path).iter_batches(columns=['feed_id', hist], batch_size=256):
            for fid, values in zip(batch['feed_id'].to_numpy(), batch[hist].to_pylist(), strict=True):
                i = lookup[fid]
                seen[i] += 1
                segment = slice(offsets[i], offsets[i + 1])
                positions = ranks[segment]
                valid = np.isfinite(positions)
                chosen = positions[valid].astype(np.int32)
                if (np.any(chosen != positions[valid]) or np.any(chosen < 0) or np.any(chosen >= len(values))):
                    raise ValueError(f'invalid cached nearest-event position in feed {fid}')
                indices[segment][valid] = np.asarray(values, dtype=np.int32)[chosen]
        if n % 8 == 0 or n == len(paths):
            print(f'{split} nearest events: {n}/{len(paths)} shards', flush=True)
    if not np.all(seen == 1) or np.any(indices[np.isfinite(ranks)] < 0):
        raise ValueError('history keys or vector availability differ from the feature cache')
    source = DATA / split / 'item_embeddings'
    metadata = json.loads((source / 'data.meta').read_text())
    vectors = fetch_vectors(source / 'data.memmap', metadata, indices)
    with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as stream:
        temp = Path(stream.name)
    try:
        np.savez(temp, feed=feed, item=item, counts=counts, nearest_vectors=vectors)
        temp.replace(output)
    finally:
        temp.unlink(missing_ok=True)
    print(json.dumps({'split': split, 'candidates': len(item),
                      'available_nearest_event_fraction': float((indices >= 0).mean())}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['train', 'val', 'test'])
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--nearest', action='store_true')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    destination = args.out / f'{args.split}.npz'
    if destination.exists():
        raise FileExistsError(destination)
    prepare = build_nearest if args.nearest else build
    prepare(args.split, args.cache / f'{args.split}.npz', destination)


if __name__ == '__main__':
    main()
