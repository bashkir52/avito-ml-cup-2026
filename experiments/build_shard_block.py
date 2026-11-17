"""Build the 79-feature intent cache for an explicit, disjoint block of train shards.

The shipped LightGBM is trained on 48 shards (`experiments/cache-intent-048`). This
script materialises the same feature set for a second, disjoint block of shards, so the
booster can be continued on 2x data while the first block is never loaded again.

Feature definitions, priors and intent columns are reused from `expand_train.py` and
`intent_features.py` unchanged; only the shard list is explicit instead of sampled, and
the result contains the new block alone.
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from expand_train import init_worker, make_shard, metadata_counts, stage_prior_tables
from ranking import COLS, DATA, StagedVectors


def build(names, out, workers, seed):
    candidates = sorted((DATA / 'train' / 'impressions').glob('part-*.parquet'))
    by_name = {path.name: path for path in candidates}
    missing = [name for name in names if name not in by_name]
    if missing or len(set(names)) != len(names):
        raise ValueError(f'invalid or duplicated shard names: {missing}')
    paths = [by_name[name] for name in names]
    counts = [metadata_counts(path) for path in paths]
    total_rows = sum(candidates_ for _, candidates_ in counts)
    print(f'Building {len(paths)} shards, {total_rows} candidates', flush=True)

    started = time.time()
    with TemporaryDirectory(dir=out.parent) as directory:
        work = Path(directory)
        stage_prior_tables(paths, work)
        print(f'Priors staged in {time.time() - started:.1f}s', flush=True)
        embedding_dir = DATA / 'train' / 'item_embeddings'
        meta = json.loads((embedding_dir / 'data.meta').read_text())
        staged = StagedVectors(embedding_dir / 'data.memmap', meta, paths)
        np.save(work / 'indices.npy', staged.indices)
        np.save(work / 'vectors.npy', staged.values)
        del staged
        print(f'Item vectors staged in {time.time() - started:.1f}s', flush=True)

        features = np.lib.format.open_memmap(work / 'features.npy', mode='w+', dtype=np.float32,
                                             shape=(total_rows, COLS['history-intent']))
        del features

        completed = {}
        rows = 0
        with ProcessPoolExecutor(max_workers=workers, initializer=init_worker,
                                 initargs=(work,)) as executor:
            jobs = {}
            for i, (path, (_, candidates_)) in enumerate(zip(paths, counts)):
                jobs[executor.submit(make_shard, path, rows, candidates_, [seed, i])] = i
                rows += candidates_
            for job in as_completed(jobs):
                i = jobs[job]
                data = job.result()
                if len(data['feed']) != counts[i][0]:
                    raise ValueError(f'{paths[i].name} feed count differs from its metadata')
                completed[i] = data
                if len(completed) % 4 == 0 or len(completed) == len(jobs):
                    print(f'Shards complete: {len(completed)}/{len(jobs)}', flush=True)

        data = {key: np.concatenate([completed[i][key] for i in range(len(paths))])
                for key in ('item', 'target', 'position', 'feed', 'timestamp', 'counts')}
        if (len(np.unique(data['feed'])) != len(data['feed'])
                or data['counts'].sum(dtype=np.int64) != total_rows
                or len(data['item']) != total_rows):
            raise ValueError('built block contains inconsistent feed or candidate rows')
        matrix = np.load(work / 'features.npy', mmap_mode='r')
        if matrix.shape != (total_rows, COLS['history-intent']) or matrix.dtype != np.float32:
            raise ValueError('feature matrix does not match the expected shape')
        tmp = work / 'train.npz'
        np.savez(tmp, x=matrix, **data, shards=[path.name for path in paths], prepare_seed=seed)
        del matrix
        tmp.replace(out / 'train.npz')

    report = {'shards': len(paths), 'rows': total_rows, 'feeds': len(data['feed']),
              'features': COLS['history-intent'], 'workers': workers, 'seed': seed,
              'seconds': round(time.time() - started, 1), 'shard_names': [p.name for p in paths]}
    (out / 'profile.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('shards', 'rows', 'feeds', 'seconds')}),
          flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('shard_list', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2027)
    args = parser.parse_args()
    if (args.out / 'train.npz').exists():
        raise FileExistsError(args.out / 'train.npz')
    args.out.mkdir(parents=True, exist_ok=True)
    build(json.loads(args.shard_list.read_text()), args.out, args.workers, args.seed)


if __name__ == '__main__':
    main()
