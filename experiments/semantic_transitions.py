"""Learn history-to-candidate transitions over semantic item groups."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from ranking import DATA, normalize


LOOKBACK = 12
GROUPS = {'category': 54, 'semantic-256': 256, 'semantic-1024': 1024}
ITEM_INDEX = 'item_id__embedding_idx'
HISTORY_INDEX = 'history__item_id__embedding_idx'
ITEM_CATEGORY = 'category_id'
HISTORY_CATEGORY = 'history__category_id'


def vector_table(split):
    path = DATA / split / 'item_embeddings'
    meta = json.loads((path / 'data.meta').read_text(encoding='utf-8'))
    return np.memmap(path / 'data.memmap', mode='r', dtype=meta['dtype'],
                     shape=tuple(meta['shape']), order=meta['order'])


def projection(vectors, rng):
    indices = np.sort(rng.choice(len(vectors), size=min(150_000, len(vectors)), replace=False))
    sample = normalize(vectors[indices])
    sample = sample[np.any(sample != 0, axis=1)]
    _, pca = np.linalg.eigh(np.cov(sample.T))
    basis = np.ascontiguousarray(pca[:, -10:][:, ::-1], dtype=np.float32)
    planes = np.median(sample @ basis, axis=0).astype(np.float32)
    return basis, planes


def assign_codes(raw, basis, planes):
    unit = normalize(raw)
    bits = unit @ basis > planes
    groups = bits.astype(np.int16) @ (1 << np.arange(len(planes), dtype=np.int16))
    groups[~np.any(raw != 0, axis=1)] = -1
    return groups


def encode_table(vectors, basis, planes, split):
    codes = np.empty(len(vectors), dtype=np.int16)
    step = 1 << 19
    for block, start in enumerate(range(0, len(codes), step), 1):
        stop = min(start + step, len(codes))
        codes[start:stop] = assign_codes(vectors[start:stop], basis, planes)
        if block % 32 == 0 or stop == len(codes):
            print(f'{split} grouped embeddings: {stop}/{len(codes)}', flush=True)
    return codes


def recent_events(arr, limit=LOOKBACK):
    offsets = arr.offsets.to_numpy()
    indices = offsets[:-1, None] + np.arange(limit)
    raw = np.r_[arr.values.to_numpy(), -1]
    return raw[np.where(indices < offsets[1:, None], indices, len(raw) - 1)]


def sample_rows(offsets, budget, rng):
    counts = np.diff(offsets)
    sizes = np.minimum(budget, counts)
    step = counts / sizes
    slots = rng.random(len(counts))[:, None] + np.arange(budget)
    keep = np.arange(budget) < sizes[:, None]
    feed = np.repeat(np.arange(len(counts)), sizes)
    rows = offsets[feed] + (step[:, None] * slots)[keep].astype(np.int64)
    return rows, feed, step[feed]


def joint_counts(history, dest, feed, weight, groups):
    source = history[feed].astype(np.int32, copy=False)
    keep = (source >= 0) & (dest[:, None] >= 0)
    edges = source * groups + dest[:, None]
    w = np.broadcast_to(weight[:, None], edges.shape)[keep]
    return np.bincount(edges[keep], weights=w, minlength=groups**2).reshape(groups, groups)


def transition_signal(history, dest, counts, matrix, supported):
    value = np.zeros(len(dest), dtype=np.float32)
    weight = np.zeros_like(value)
    for lag, event in enumerate(history.T):
        src = np.repeat(event, counts)
        valid = (src >= 0) & (dest >= 0) & supported[src, dest]
        decay = 2 ** (-lag / 6)
        value[valid] += decay * matrix[src[valid], dest[valid]]
        weight[valid] += decay
    return np.divide(value, weight, out=value, where=weight > 0), weight > 0


def grouped_codes(codes, categories, indices):
    category = np.where(categories > 0, categories, -1).astype(np.int16)
    semantic = np.where(indices >= 0, codes[indices], -1).astype(np.int16)
    coarse = np.where(semantic >= 0, semantic & 255, -1).astype(np.int16)
    return {'category': category, 'semantic-256': coarse, 'semantic-1024': semantic}


def read_indices(table, codes):
    item = table[ITEM_INDEX].chunk(0)
    dest = grouped_codes(codes, table[ITEM_CATEGORY].chunk(0).flatten().to_numpy(), item.flatten().to_numpy())
    history = grouped_codes(codes, recent_events(table[HISTORY_CATEGORY].chunk(0)),
                            recent_events(table[HISTORY_INDEX].chunk(0)))
    return item.offsets.to_numpy(), dest, history


def fit(out):
    if out.exists():
        raise FileExistsError(out)
    rng = np.random.default_rng(2026)
    vectors = vector_table('train')
    basis, planes = projection(vectors, rng)
    codes = encode_table(vectors, basis, planes, 'train')
    del vectors
    statistics = {name: (np.zeros((n, n)), np.zeros((n, n)), np.zeros(n), np.zeros(n))
                  for name, n in GROUPS.items()}
    paths = sorted((DATA / 'train/impressions').glob('part-*.parquet'))
    columns = [ITEM_INDEX, HISTORY_INDEX, ITEM_CATEGORY, HISTORY_CATEGORY, 'target']
    total_feeds, total_positives, total_sampled = 0, 0, 0
    for i, path in enumerate(paths, 1):
        table = pq.read_table(path, columns=columns).combine_chunks()
        offsets, dest, history = read_indices(table, codes)
        positive = np.flatnonzero(table['target'].chunk(0).flatten().to_numpy())
        positive_feed = np.searchsorted(offsets[1:], positive, side='right')
        sample, sample_feed, weight = sample_rows(offsets, 16, rng)
        total_feeds += len(offsets) - 1
        total_positives += len(positive)
        total_sampled += len(sample)
        for name, n in GROUPS.items():
            d, h = dest[name], history[name]
            cases = d[positive]
            exposures = d[sample]
            positive_edges, exposure_edges, positive_total, exposure_total = statistics[name]
            positive_edges += joint_counts(h, cases, positive_feed, np.ones(len(positive)), n)
            exposure_edges += joint_counts(h, exposures, sample_feed, weight, n)
            positive_total += np.bincount(cases[cases >= 0], minlength=n)
            exposure_total += np.bincount(exposures[exposures >= 0], weight[exposures >= 0], minlength=n)
        if i % 64 == 0 or i == len(paths):
            print(f'{i}/{len(paths)} train shards; {total_positives} labeled positives', flush=True)
    out.mkdir(parents=True, exist_ok=False)
    keys = ('positive_edges', 'exposure_edges', 'positive_total', 'exposure_total')
    arrays = {f'{name}_{key}': value for name, values in statistics.items() for key, value in zip(keys, values)}
    np.savez(out / 'graph.npz', basis=basis, planes=planes, **arrays)
    summary = {'train_shards': len(paths), 'train_feeds': total_feeds, 'positive_rows': total_positives,
               'sampled_exposure_rows': total_sampled, 'exposure_samples_per_feed': 16, 'lookback': LOOKBACK,
               'group_counts': GROUPS, 'seed': 2026}
    (out / 'fit.json').write_text(json.dumps(summary, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    fit(parser.parse_args().out)
