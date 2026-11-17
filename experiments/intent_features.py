"""Append features about recent intent to the existing, aligned feature cache."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from ranking import COLS, DATA, INTENT, StagedVectors, normalize, rank_fraction, read_vectors


COLUMNS = ['feed_id', 'item_id', 'category_id', 'item_id__embedding_idx',
           'history__category_id', 'history__event_type', 'history__event_source',
           'history__item_id__embedding_idx']


def intent_features(row, table):
    vectors, present = read_vectors(table, row['item_id__embedding_idx'])
    units = normalize(vectors)
    history, available = read_vectors(table, row['history__item_id__embedding_idx'])
    cats = np.asarray(row['category_id'])
    n = len(cats)
    result = np.full((n, len(INTENT)), np.nan, dtype=np.float32)
    result[present, 22] = (units @ normalize(units.sum(axis=0)))[present]
    offsets = np.flatnonzero(available)
    if not len(offsets):
        return result

    categories = np.asarray(row['history__category_id'])[available]
    types = np.asarray(row['history__event_type'])[available]
    sources = np.asarray(row['history__event_source'])[available]
    sims = units @ normalize(history[available]).T
    for i in range(1, 11):
        mask = types == i
        if mask.any():
            result[:, i - 1] = sims[:, mask].max(axis=1)
    for col, limit in enumerate((4, 16, 64), 10):
        end = offsets.searchsorted(limit)
        if end:
            result[:, col] = sims[:, :end].max(axis=1)

    weights = np.exp2(-offsets.astype(np.float32) / 12)
    result[:, 13] = sims @ (weights / weights.sum())
    result[:, 14] = (sims - .08 * np.log1p(offsets)).max(axis=1)
    best = sims.argmax(axis=1)
    result[:, 15] = offsets[best]
    result[:, 16] = types[best]
    result[:, 17] = sources[best]
    result[:, 18] = cats == categories[best]

    match = cats[:, None] == categories[None, :]
    mass = match @ weights
    result[:, 19] = mass
    result[:, 20] = np.divide((sims * match) @ weights, mass, out=np.full(n, np.nan), where=mass > 0)
    result[:, 21] = (sims > .7).sum(axis=1)
    end = offsets.searchsorted(16)
    if end and present.any():
        result[present, 23] = rank_fraction(result[present, 11])
    result[~present, :19] = np.nan
    result[~present, 20:24] = np.nan
    return result


class SharedVectors:
    def __init__(self, staging):
        self.indices = np.load(staging / 'indices.npy', mmap_mode='r')
        self.vectors = np.load(staging / 'vectors.npy', mmap_mode='r')
        self.shape = self.vectors.shape

    def __getitem__(self, indices):
        return self.vectors[self.indices.searchsorted(indices)]


def init_worker(staging, base):
    global worker_vectors, worker_extra, worker_feeds, worker_items, worker_offsets
    worker_vectors = SharedVectors(staging)
    worker_extra = np.load(staging / 'extra.npy', mmap_mode='r+')
    with np.load(base) as f:
        worker_items = f['item']
        worker_feeds = {int(v): i for i, v in enumerate(f['feed'])}
        worker_offsets = np.r_[0, np.cumsum(f['counts'], dtype=np.int64)]


def process_shard(path):
    completed = []
    for batch in pq.ParquetFile(path).iter_batches(columns=COLUMNS, batch_size=128, use_threads=False):
        for row in batch.to_pylist():
            i = worker_feeds[row['feed_id']]
            sl = slice(worker_offsets[i], worker_offsets[i + 1])
            item = np.asarray(row['item_id'], dtype=np.int32)
            order = np.argsort(item)
            sorted_items = item[order]
            locations = sorted_items.searchsorted(worker_items[sl])
            if (len(item) != len(locations) or np.any(sorted_items[1:] == sorted_items[:-1])
                    or np.any(locations == len(item)) or not np.array_equal(sorted_items[locations], worker_items[sl])
                    or len(np.unique(locations)) != len(item)):
                raise ValueError(f"misaligned candidate IDs in feed {row['feed_id']}")
            worker_extra[sl] = intent_features(row, worker_vectors)[order[locations]]
            completed.append(i)
    return np.asarray(completed, dtype=np.int32)


def augment(split, base, output, workers):
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    with np.load(base) as f:
        expected_feeds, expected_rows = len(f['feed']), len(f['item'])
        if 'shards' in f:
            selected = set(f['shards'].tolist())
            paths = [p for p in paths if p.name in selected]
            if len(paths) != len(selected):
                raise ValueError('some cached shards are unavailable')
    tables = DATA / split / 'item_embeddings'
    meta = json.loads((tables / 'data.meta').read_text())
    with TemporaryDirectory(dir=output.parent) as directory:
        staging = Path(directory)
        vectors = StagedVectors(tables / 'data.memmap', meta, paths)
        np.save(staging / 'indices.npy', vectors.indices)
        np.save(staging / 'vectors.npy', vectors.values)
        del vectors
        mmap = np.lib.format.open_memmap(staging / 'extra.npy', mode='w+', dtype=np.float32,
                                         shape=(expected_rows, len(INTENT)))
        del mmap
        seen = np.zeros(expected_feeds, dtype=np.int32)
        completed = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(staging, base)) as pool:
            jobs = [pool.submit(process_shard, path) for path in paths]
            for i, job in enumerate(as_completed(jobs), 1):
                indices = job.result()
                np.add.at(seen, indices, 1)
                completed += len(indices)
                if i == len(jobs) or i % 4 == 0:
                    print(f'{split}: {i}/{len(jobs)} shards, {completed} feeds', flush=True)
        if not np.all(seen == 1):
            raise ValueError('source rows do not cover the cached feed IDs exactly once')

        with np.load(base) as data:
            original = data['x']
            if original.shape[1] != COLS['history-priors']:
                raise ValueError('expected base cache with 55 features')
            extra = np.load(staging / 'extra.npy', mmap_mode='r')
            combined = np.concatenate((original, extra), axis=1)
            del original, extra
            remaining = {k: data[k] for k in data.files if k != 'x'}
        tmp = staging / 'combined.npz'
        np.savez(tmp, x=combined, **remaining)
        tmp.replace(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['train', 'val', 'test'])
    parser.add_argument('base_cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f'{args.split}.npz'
    if output.exists():
        raise FileExistsError(output)
    augment(args.split, args.base_cache / f'{args.split}.npz', output, args.workers)


if __name__ == '__main__':
    main()
