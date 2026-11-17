"""Read original vectors in the exact feed/candidate order of a feature cache."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import pyarrow.parquet as pq

from ranking import DATA, StagedVectors, normalize


def collect(split, base, output):
    with np.load(base) as file:
        feed, item, counts = (file[name] for name in ('feed', 'item', 'counts'))
        names = set(file['shards'].tolist()) if 'shards' in file else None
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    if names is not None:
        paths = [p for p in paths if p.name in names]
        if len(paths) != len(names):
            raise ValueError('cached shards are missing')
    directory = DATA / split / 'item_embeddings'
    info = json.loads((directory / 'data.meta').read_text())
    vec = StagedVectors(directory / 'data.memmap', info, paths)
    width = int(info['shape'][1])
    item_vectors = np.zeros((len(item), width), dtype=np.int8)
    user_vectors = np.zeros((len(feed), width), dtype=np.int8)
    recent = np.zeros((len(feed), width), dtype=np.float16)
    seen = np.zeros(len(feed), dtype=np.int32)
    lookup = {int(fid): i for i, fid in enumerate(feed)}
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    columns = ['feed_id', 'item_id', 'user_embedding', 'item_id__embedding_idx', 'history__item_id__embedding_idx']
    for i, path in enumerate(paths, 1):
        for batch in pq.ParquetFile(path).iter_batches(columns=columns, batch_size=256):
            for row in batch.to_pylist():
                f = lookup[row['feed_id']]
                seen[f] += 1
                sl = slice(offsets[f], offsets[f + 1])
                ids = np.asarray(row['item_id'], dtype=np.int32)
                cached_ids = item[sl]
                order = np.argsort(ids)
                sorted_items = ids[order]
                positions = sorted_items.searchsorted(cached_ids)
                if (len(positions) != len(ids) or np.any(positions == len(ids))
                        or len(np.unique(positions)) != len(ids)
                        or not np.array_equal(sorted_items[positions], cached_ids)):
                    raise ValueError('raw and cached candidate IDs differ')
                indices = np.asarray(row['item_id__embedding_idx'], dtype=np.int32)[order[positions]]
                present = indices >= 0
                item_vectors[sl][present] = vec[indices[present]]
                user_vectors[f] = row['user_embedding']
                history = np.asarray(row['history__item_id__embedding_idx'][:48], dtype=np.int32)
                matches = np.flatnonzero(history >= 0)
                if len(matches):
                    weight = np.exp2(-matches.astype(np.float32) / 12)
                    recent[f] = normalize(weight @ normalize(vec[history[matches]])).astype(np.float16)
        if i % 8 == 0 or i == len(paths):
            print(f'{split} vectors: {i}/{len(paths)} shards', flush=True)
    if not np.all(seen == 1):
        raise ValueError('source feed keys do not exactly cover the feature cache')
    with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as stream:
        tmp = Path(stream.name)
    try:
        np.savez(tmp, feed=feed, item=item, counts=counts, item_vectors=item_vectors,
                 user_vectors=user_vectors, recent_vectors=recent)
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('split', choices=['train', 'val', 'test'])
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    output = args.out / f'{args.split}.npz'
    if output.exists():
        raise FileExistsError(output)
    collect(args.split, args.cache / f'{args.split}.npz', output)


if __name__ == '__main__':
    main()
