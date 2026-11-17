"""Expand the intent training cache by appending complete, disjoint Parquet shards."""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pyarrow.parquet as pq

from intent_features import SharedVectors, intent_features
from item_priors import accumulate, prior_features
from ranking import COLS, DATA, StagedVectors, feed_features


def metadata_counts(path):
    file = pq.ParquetFile(path)
    column = next(i for i in range(len(file.schema)) if file.schema.column(i).path.startswith('item_id.'))
    candidates = sum(file.metadata.row_group(i).column(column).num_values
                     for i in range(file.metadata.num_row_groups))
    return file.metadata.num_rows, candidates


def init_worker(work):
    global vectors, mapping, exposure, positive, output
    vectors = SharedVectors(work)
    mapping = np.load(work / 'mapping.npy', mmap_mode='r')
    exposure = np.load(work / 'exposure.npy', mmap_mode='r')
    positive = np.load(work / 'positive.npy', mmap_mode='r')
    output = np.load(work / 'features.npy', mmap_mode='r+')


def make_shard(path, start, expected_rows, seed):
    rng = np.random.default_rng(seed)
    items, targets, positions, ids, timestamps, counts = [], [], [], [], [], []
    offset = start
    for batch in pq.ParquetFile(path).iter_batches(batch_size=128, use_threads=False):
        for row in batch.to_pylist():
            items_for_feed = np.asarray(row['item_id'], dtype=np.int32)
            count = len(items_for_feed)
            order = rng.permutation(count)
            base = feed_features(row, vectors)
            prior = prior_features({'item': items_for_feed, 'counts': [count],
                                    'timestamp': np.array([row['timestamp']], dtype=np.int32)},
                                   mapping, exposure, positive)
            intent = intent_features(row, vectors)
            end = offset + count
            if end > start + expected_rows:
                raise ValueError(f'{path.name} exceeded its expected candidate count')
            output[offset:end, :] = np.column_stack((base, prior, intent))[order]
            items.append(items_for_feed[order])
            targets.append(np.asarray(row['target'], dtype=np.int8)[order])
            positions.append(np.asarray(row['pos'], dtype=np.int16)[order])
            ids.append(row['feed_id'])
            timestamps.append(row['timestamp'])
            counts.append(count)
            offset = end
    if offset != start + expected_rows:
        raise ValueError(f'{path.name} has a different candidate count than its metadata')
    return {'item': np.concatenate(items), 'target': np.concatenate(targets),
            'position': np.concatenate(positions), 'feed': np.asarray(ids, dtype=np.int32),
            'timestamp': np.asarray(timestamps, dtype=np.int32), 'counts': np.asarray(counts, dtype=np.int32)}


def stage_prior_tables(paths, work):
    ids = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=['item_id'], batch_size=4096):
            ids.append(batch.column(0).flatten().to_numpy())
    items = np.unique(np.concatenate(ids))
    del ids
    mapping = np.full(int(items[-1]) + 1, -1, dtype=np.int32)
    mapping[items] = np.arange(len(items), dtype=np.int32)
    exposure, positive = accumulate(mapping, len(items))
    np.save(work / 'mapping.npy', mapping)
    np.save(work / 'exposure.npy', exposure)
    np.save(work / 'positive.npy', positive)


def expand(base, out, shards, workers, seed):
    columns = ('item', 'target', 'position', 'feed', 'timestamp', 'counts')
    with np.load(base) as f:
        orig = {name: f[name] for name in columns}
        existing = f['shards'].tolist()
    rows = len(orig['item'])
    feeds = len(orig['feed'])
    if (len(orig['target']) != rows or len(orig['position']) != rows
            or len(orig['timestamp']) != feeds or len(orig['counts']) != feeds
            or np.any(orig['counts'] <= 0) or orig['counts'].sum(dtype=np.int64) != rows):
        raise ValueError('base cache has inconsistent candidate or feed metadata')
    candidates = sorted((DATA / 'train' / 'impressions').glob('part-*.parquet'))
    existing_set = set(existing)
    available = [path for path in candidates if path.name not in existing_set]
    if len(existing) != len(existing_set) or len(candidates) - len(available) != len(existing):
        raise ValueError('cached shard list is inconsistent with available files')
    if not len(existing) < shards <= len(candidates):
        raise ValueError('requested size must exceed existing shard count and fit the train split')
    rng = np.random.default_rng(seed)
    additional = sorted(rng.choice(available, shards - len(existing), replace=False))
    counts = [metadata_counts(path) for path in additional]
    total_rows = rows + sum(p[1] for p in counts)
    print(f'Reusing {len(existing)} shards; adding {len(additional)} shards; {total_rows} candidates', flush=True)

    with TemporaryDirectory(dir=out.parent) as directory:
        work = Path(directory)
        stage_prior_tables(additional, work)
        embedding_dir = DATA / 'train' / 'item_embeddings'
        meta = json.loads((embedding_dir / 'data.meta').read_text())
        staged = StagedVectors(embedding_dir / 'data.memmap', meta, additional)
        np.save(work / 'indices.npy', staged.indices)
        np.save(work / 'vectors.npy', staged.values)
        del staged
        x = np.lib.format.open_memmap(work / 'features.npy', mode='w+', dtype=np.float32,
                                     shape=(total_rows, COLS['history-intent']))
        with np.load(base) as f:
            original_features = f['x']
            if original_features.shape != (rows, COLS['history-intent']) or original_features.dtype != np.float32:
                raise ValueError('base cache must contain aligned float32 intent features')
            x[:rows] = original_features
        del original_features
        x.flush()
        del x
        completed = {}
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker, initargs=(work,)) as executor:
            jobs = {}
            for i, (path, (_, n)) in enumerate(zip(additional, counts)):
                job = executor.submit(make_shard, path, rows, n, [seed, i])
                jobs[job] = i
                rows += n
            for job in as_completed(jobs):
                i = jobs[job]
                data = job.result()
                if len(data['feed']) != counts[i][0]:
                    raise ValueError(f'{additional[i].name} feed count differs from its metadata')
                completed[i] = data
                if len(completed) % 4 == 0 or len(completed) == len(jobs):
                    print(f'New shards complete: {len(completed)}/{len(jobs)}', flush=True)

        data = {k: np.concatenate((orig[k], *[completed[i][k] for i in range(len(additional))])) for k in columns}
        if (len(np.unique(data['feed'])) != len(data['feed']) or data['counts'].sum(dtype=np.int64) != total_rows
                or len(data['item']) != total_rows):
            raise ValueError('expanded cache contains inconsistent feed or candidate rows')
        x = np.load(work / 'features.npy', mmap_mode='r')
        tmp = work / 'train.npz'
        np.savez(tmp, x=x, **data, shards=[*existing, *[p.name for p in additional]], prepare_seed=seed)
        del x
        tmp.replace(out)
    print(json.dumps({'rows': total_rows, 'feeds': len(data['feed']), 'original_shards': len(existing),
                      'added_shards': [p.name for p in additional], 'seed': seed}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('base', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--shards', type=int, default=48)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    expand(args.base, args.output, args.shards, args.workers, args.seed)


if __name__ == '__main__':
    main()
