"""Compare the coverage of two supervised train pools with equal optimizer budgets."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import numpy as np
import pyarrow.parquet as pq
from scipy.stats import rankdata
import torch

from neural_blend import read_scores
from ranking import DATA, KEYS, NUMERIC, audit_submission, normalize, prediction_frame
from sequence_ranker import assess, scores_for_blend, sha256, vector_table, write_json
from set_ranker import FeedRanker, feed_loss, fit_normalization


HISTORY_VECTORS = 'history__item_id__embedding_idx'
HISTORY_CATEGORY = 'history__category_id'
ITEM_VECTORS = 'item_id__embedding_idx'
HISTORY_LIMIT = 48
PREPARATION_VERSION = 2
ARMS = ('control', 'wide')
COVERAGE_DELTA = 'coverage_delta'
FEATURES = [*NUMERIC, 'category', *[f'{f}_{s}' for s in ('rank', 'z') for f in NUMERIC],
            'feed_size', 'category_count', 'user_norm', 'item_norm', 'item_missing', 'user_cos',
            'hist_length', 'hist_category_8', 'hist_category_32', 'history_cos_4', 'history_cos_16',
            'history_cos_48', 'recent_cos']


def lengths(column):
    return np.diff(column.offsets.to_numpy())


def dense(column, width, fill=0):
    offsets = column.offsets.to_numpy()
    steps = np.arange(width)
    present = steps < np.diff(offsets)[:, None]
    values = column.values.to_numpy()
    result = np.full(present.shape, fill, dtype=values.dtype)
    result[present] = values[(offsets[:-1, None] + steps)[present]]
    return result


def sample_candidates(labels, valid, limit, rng):
    negatives = valid & (labels == 0)
    scores = np.where(negatives, rng.random(valid.shape), np.inf)
    n = min(limit, valid.shape[1])
    chosen = np.argpartition(scores, n - 1, axis=1)[:, :n]
    sampled = np.zeros_like(valid)
    np.put_along_axis(sampled, chosen, True, axis=1)
    return (sampled & negatives) | (valid & (labels == 1))


def int8_vectors(table, indices):
    exists = indices >= 0
    result = np.zeros((*indices.shape, table.shape[1]), dtype=table.dtype)
    result[exists] = table[indices[exists]]
    return result


def standardized_features(values, valid, counts):
    widths = counts[:, None, None]
    mask = valid[:, :, None]
    mean = np.sum(values * mask, axis=1, keepdims=True, dtype=np.float64) / widths
    centered = values - mean
    sd = np.sqrt((np.square(centered) * mask).sum(axis=1, keepdims=True) / widths)
    return centered / np.maximum(sd, 1e-6)


def process_batch(batch, table, rng=None, negatives=12):
    n = lengths(batch['item_id'])
    width = int(n.max())
    valid = np.arange(width) < n[:, None]
    labels = dense(batch['target'], width) if 'target' in batch.schema.names else np.zeros(valid.shape, dtype=np.int8)
    kept = sample_candidates(labels, valid, negatives, rng) if rng is not None else valid
    row, _ = np.nonzero(kept)
    count = kept.sum(axis=1).astype(np.int32)
    cat = dense(batch['category_id'], width)
    categories = cat[kept]
    f = np.stack([dense(batch[name], width) for name in NUMERIC], axis=-1)
    x = [f[kept], categories[:, None]]
    ranks = (rankdata(np.where(valid[:, :, None], f, np.inf), axis=1) - .5) / n[:, None, None]
    x.extend([ranks[kept], standardized_features(f, valid, n)[kept]])

    slots = np.arange(len(n))[:, None] * 128 + cat
    category_count = np.bincount(slots[valid], minlength=len(n) * 128)
    user = dense(batch['user_embedding'], 128)
    idx = dense(batch[ITEM_VECTORS], width, -1)[kept]
    items = int8_vectors(table, idx)
    item_unit = normalize(items)
    user_unit = normalize(user)
    user_cos = (item_unit * user_unit[row]).sum(axis=1)

    hist_idx = dense(batch[HISTORY_VECTORS], HISTORY_LIMIT, -1)
    hist_cat = dense(batch[HISTORY_CATEGORY], HISTORY_LIMIT, -1)
    hist_unit = normalize(int8_vectors(table, hist_idx))
    decay = np.exp2(-np.arange(HISTORY_LIMIT, dtype=np.float32) / 12)
    recent = normalize(np.matmul(decay, hist_unit))
    recent_cos = (item_unit * recent[row]).sum(axis=1)
    matches = categories[:, None] == hist_cat[row]
    x.append(np.column_stack((n[row], category_count[slots[kept]], np.linalg.norm(user, axis=1)[row],
                              np.linalg.norm(items, axis=1), idx < 0, user_cos,
                              lengths(batch[HISTORY_VECTORS])[row],
                              matches[:, :8].sum(axis=1), matches[:, :32].sum(axis=1))))

    compact_positions = np.arange(len(row)) - np.repeat(np.cumsum(count) - count, count)
    padded_vectors = np.zeros((len(n), int(count.max()), table.shape[1]), dtype=np.float32)
    padded_vectors[row, compact_positions] = item_unit
    similarities = (padded_vectors @ hist_unit.transpose(0, 2, 1))[row, compact_positions]
    similarities = np.where(hist_idx[row] >= 0, similarities, -np.inf)
    for limit in (4, 16, 48):
        maximum = similarities[:, :limit].max(axis=1)
        maximum[~np.isfinite(maximum) | (idx < 0)] = np.nan
        x.append(maximum[:, None])
    x.append(recent_cos[:, None])
    target = labels[kept]
    if int(target.sum()) != int(labels[valid].sum()):
        raise ValueError('sampling lost positives')
    return {'x': np.column_stack(x).astype(np.float32), 'item_vectors': items, 'user_vectors': user,
            'recent_vectors': recent.astype(np.float16), 'item': dense(batch['item_id'], width)[kept],
            'target': target, 'feed': batch['feed_id'].to_numpy(), 'counts': count}


def collect_shard(path, table, negative_limit, rng):
    columns = ['feed_id', 'item_id', *NUMERIC, 'category_id', 'user_embedding', ITEM_VECTORS,
               HISTORY_CATEGORY, HISTORY_VECTORS]
    parquet = pq.ParquetFile(path)
    if 'target' in parquet.schema_arrow.names:
        columns.append('target')
    batches = [process_batch(batch, table, rng, negative_limit)
               for batch in parquet.iter_batches(columns=columns, batch_size=128, use_threads=False)]
    return {key: np.concatenate([b[key] for b in batches]) for key in batches[0]}


def write_shard(output, data):
    with NamedTemporaryFile(dir=output.parent, suffix='.npz', delete=False) as file:
        temporary = Path(file.name)
    try:
        np.savez(temporary, **data)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def available_paths(split):
    return sorted((DATA / split / 'impressions').glob('part-*.parquet'))


def prepare(args):
    start = time.perf_counter()
    args.out.mkdir(parents=True, exist_ok=True)
    paths = available_paths('train')
    with np.load(args.reference) as cached:
        narrow = set(cached['shards'].tolist())
    base = [path for path in paths if path.name in narrow]
    extra = [path for path in paths if path.name not in narrow]
    if len(base) != len(narrow) or not 1 <= args.shards <= len(paths) or args.negatives < 1:
        raise ValueError('unexpected train shard population or budget')
    shuffled = np.random.default_rng(args.seed).permutation(len(extra))
    selected = [*base, *(extra[i] for i in shuffled)][:args.shards]
    jobs = []
    for path in selected:
        output = args.out / (path.stem + '.npz')
        shard_number = int(path.name.split('-')[1])
        signature = [args.seed, args.negatives, shard_number]
        fresh = False
        if output.exists():
            with np.load(output) as file:
                if file['sampling'].tolist() != signature:
                    raise ValueError(f'cached shard {output} has different sampling parameters')
                fresh = ('version' in file and int(file['version']) == PREPARATION_VERSION
                         and file['features'].tolist() == FEATURES)
        jobs.append((path, output, signature, fresh))
    stage = np.array(vector_table('train')) if any(not job[-1] for job in jobs) else None
    print(f'train table ready after {time.perf_counter() - start:.1f}s', flush=True)

    def process(job):
        path, output, signature, fresh = job
        if not fresh:
            rng = np.random.default_rng([args.seed, signature[2]])
            data = collect_shard(path, stage, args.negatives, rng)
            write_shard(output, dict(data, features=FEATURES, sampling=signature, version=PREPARATION_VERSION))
            del data
        with np.load(output) as file:
            if file['features'].tolist() != FEATURES or file['sampling'].tolist() != signature:
                raise ValueError(f'cached shard {output} has different preparation parameters')
            target = file['target']
            return {'name': output.name, 'control': path.name in narrow, 'feeds': len(file['counts']),
                    'rows': len(target), 'positives': int(target.sum())}

    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        for i, summary in enumerate(executor.map(process, jobs), 1):
            summaries.append(summary)
            if i % 8 == 0 or i == len(selected):
                print(f'{i}/{len(selected)} shards, {sum(s["feeds"] for s in summaries):,} feeds, '
                      f'{time.perf_counter() - start:.1f}s', flush=True)
    write_json(args.out / 'prepare.json', {'shards': summaries, 'features': FEATURES, 'seed': args.seed,
               'negatives': args.negatives, 'workers': args.workers, 'version': PREPARATION_VERSION,
               'seconds': round(time.perf_counter() - start, 2)})


def match_coverage(shapes, control, rng):
    """Repeat the narrow pool to match the wide pool's per-feed budgets exactly."""
    _, group = np.unique(shapes, axis=0, return_inverse=True)
    order = np.argsort(group, kind='stable')
    boundaries = np.flatnonzero(np.diff(group[order])) + 1
    is_control = np.zeros(len(shapes), dtype=bool)
    is_control[control] = True
    wide, matched = [], []
    for indices in np.split(order, boundaries):
        pool = indices[is_control[indices]]
        if not len(pool):
            continue
        copies, extra = divmod(len(indices), len(pool))
        choices = [rng.permutation(pool) for _ in range(copies)]
        if extra:
            choices.append(rng.permutation(pool)[:extra])
        wide.append(indices)
        matched.extend(choices)
    wide, matched = np.concatenate(wide), np.concatenate(matched)
    np.testing.assert_array_equal(shapes[wide], shapes[matched])
    return wide, matched


class Arrays:
    def __init__(self, data):
        self.data = data
        counts = data['counts']
        self.offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
        if np.any(counts < 1) or self.offsets[-1] != len(data['x']) or data['x'].shape[1] != len(FEATURES):
            raise ValueError('invalid prepared feed sizes or feature matrix')

    def batch(self, ids):
        data = self.data
        counts = data['counts'][ids]
        positions = np.arange(counts.max())
        mask = positions < counts[:, None]
        rows = self.offsets[ids, None] + np.minimum(positions, counts[:, None] - 1)
        vectors = (data['item_vectors'][rows], data['user_vectors'][ids], data['recent_vectors'][ids])
        return (torch.from_numpy(data['x'][rows]), torch.from_numpy(mask),
                tuple(torch.from_numpy(vec) for vec in vectors), rows, mask)


def load_prepared(directory, prepared):
    keys = ('x', 'item', 'target', 'counts', 'feed', 'item_vectors', 'recent_vectors', 'user_vectors')
    feeds, rows = sum(shard['feeds'] for shard in prepared), sum(shard['rows'] for shard in prepared)
    arrays = {}
    control = []
    feed_offset = row_offset = 0
    for shard in prepared:
        with np.load(directory / shard['name']) as data:
            if data['features'].tolist() != FEATURES or int(data['version']) != PREPARATION_VERSION:
                raise ValueError('train feature schema differs from the model')
            for key in keys:
                values = data[key]
                per_feed = key in ('counts', 'feed', 'recent_vectors', 'user_vectors')
                length, offset = (feeds, feed_offset) if per_feed else (rows, row_offset)
                if key not in arrays:
                    arrays[key] = np.empty((length, *values.shape[1:]), dtype=values.dtype)
                arrays[key][offset:offset + len(values)] = values
        if shard['control']:
            control.append(np.arange(feed_offset, feed_offset + shard['feeds']))
        feed_offset += shard['feeds']
        row_offset += shard['rows']
    if len(np.unique(arrays['feed'])) != feeds:
        raise ValueError('train feed IDs must be unique')
    return Arrays(arrays), np.concatenate(control)


def model_for(mean, scale, width):
    return FeedRanker(mean, scale, width, use_context=False, use_vectors=True, feature_names=FEATURES)


@torch.inference_mode()
def score_eval(models, cache, split, batch_size):
    for model in models.values():
        model.eval()
    with np.load(cache / f'{split}.npz') as file:
        meta = {key: file[key] for key in ('feed', 'item', 'counts')}
        labels = file['target'] if split == 'val' else None
    offsets = np.r_[0, np.cumsum(meta['counts'], dtype=np.int64)]
    lookup = {int(feed): i for i, feed in enumerate(meta['feed'])}
    scores = {name: np.full(len(meta['item']), np.nan, dtype=np.float32) for name in models}
    seen = np.zeros(len(meta['feed']), dtype=bool)
    table = np.array(vector_table(split))
    for path in available_paths(split):
        raw = collect_shard(path, table, 0, None)
        shard = Arrays(raw)
        alignment = np.empty(len(raw['item']), dtype=np.int64)
        for i, fid in enumerate(raw['feed']):
            target = lookup[fid]
            if seen[target]:
                raise ValueError(f'duplicate eval feed {fid}')
            a, b = offsets[target:target + 2]
            c, d = shard.offsets[i:i + 2]
            np.testing.assert_array_equal(raw['item'][c:d], meta['item'][a:b])
            if labels is not None:
                np.testing.assert_array_equal(raw['target'][c:d], labels[a:b])
            alignment[c:d] = np.arange(a, b)
            seen[target] = True
        order = np.argsort(raw['counts'], kind='stable')
        for start in range(0, len(order), batch_size):
            x, mask, vectors, rows, keep = shard.batch(order[start:start + batch_size])
            for name, model in models.items():
                scores[name][alignment[rows[keep]]] = model(x, mask, vectors).numpy()[keep]
    if not seen.all() or any(not np.isfinite(s).all() for s in scores.values()):
        raise ValueError('evaluation scores do not cover all candidates or contain nonfinite values')
    print(f'{split}: scored and aligned {len(meta["item"]):,} candidates in {seen.sum():,} full feeds', flush=True)
    return meta, scores


def tune(args):
    report = json.loads((args.data / 'prepare.json').read_text(encoding='utf-8'))
    if (args.epochs < 1 or args.batch_size < 1 or report['features'] != FEATURES
            or report['version'] != PREPARATION_VERSION):
        raise ValueError('invalid optimizer budget or feature schema')
    args.out.mkdir(parents=True, exist_ok=False)
    for name in ARMS:
        (args.out / name).mkdir()
    train, control = load_prepared(args.data, report['shards'])
    data = train.data
    rng = np.random.default_rng(args.seed)
    counts = data['counts']
    positives = np.add.reduceat(data['target'], train.offsets[:-1], dtype=np.int32)
    lengths = data['x'][train.offsets[:-1], FEATURES.index('feed_size')].astype(np.int32)
    wide, paired = match_coverage(np.column_stack((lengths, counts, positives)), control, rng)
    if len(wide) == len(control):
        raise ValueError('train coverage is identical in the two arms')
    control_rows = train.offsets[control, None] + np.arange(counts.max())
    eligible_rows = np.arange(counts.max()) < counts[control, None]
    norm_sample = rng.choice(control_rows[eligible_rows], size=120_000, replace=True)
    mean, scale = fit_normalization(data['x'][norm_sample], rng)
    torch.manual_seed(args.seed)
    narrow_model = model_for(mean, scale, args.width)
    models = {'control': narrow_model, 'wide': deepcopy(narrow_model)}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.03)
                  for name, model in models.items()}
    index = rng.permutation(len(wide))
    epochs = []
    start = time.perf_counter()
    print(f'{len(control):,} distinct narrow feeds vs {len(wide):,} distinct wide feeds at identical budgets', flush=True)
    for epoch, part in enumerate(np.array_split(index, args.epochs)):
        lr = args.lr * (.3 + .7 * (1 + math.cos(math.pi * epoch / max(args.epochs - 1, 1))) / 2)
        for optimizer in optimizers.values():
            optimizer.param_groups[0]['lr'] = lr
        running = dict.fromkeys(ARMS, 0.)
        for first in range(0, len(part), args.batch_size):
            selection = part[first:first + args.batch_size]
            dropout = torch.get_rng_state()
            for name, ids in zip(ARMS, (paired[selection], wide[selection]), strict=True):
                model, optimizer = models[name], optimizers[name]
                x, mask, vectors, rows, _ = train.batch(ids)
                torch.set_rng_state(dropout)
                optimizer.zero_grad(set_to_none=True)
                loss = feed_loss(model(x, mask, vectors), torch.from_numpy(data['target'][rows]), mask, .25)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                running[name] += float(loss.detach()) * len(ids)
        feeds = len(part)
        entry = {'epoch': epoch + 1, 'feeds_per_arm': feeds,
                 'rows_per_arm': int(counts[wide[part]].sum()), 'positives_per_arm': int(positives[wide[part]].sum()),
                 'steps_per_arm': math.ceil(feeds / args.batch_size), 'lr': lr,
                 'loss': {name: value / feeds for name, value in running.items()},
                 'seconds': round(time.perf_counter() - start, 2)}
        print(json.dumps(entry), flush=True)
        epochs.append(entry)
        write_json(args.out / 'progress.json', epochs)
    for name, model in models.items():
        torch.save({'features': FEATURES, 'width': args.width, 'model': model.state_dict()}, args.out / name / 'model.pt')
    shard_ends = np.cumsum([shard['feeds'] for shard in report['shards']])
    fit = {'epochs': epochs, 'width': args.width, 'threads': args.threads, 'seed': args.seed,
           'batch_size': args.batch_size, 'lr': args.lr, 'prepared_shards': len(report['shards']),
           'control_shards': sum(s['control'] for s in report['shards']), 'available_feeds': len(counts),
           'used_shards': {name: len(np.unique(np.searchsorted(shard_ends, ids, side='right')))
                           for name, ids in zip(ARMS, (paired, wide), strict=True)},
           'unique_feeds': {'control': len(np.unique(paired)), 'wide': len(wide)},
           'omitted_wide_feeds': len(counts) - len(wide), 'max_negatives': report['negatives'],
           'feature_preparation_version': PREPARATION_VERSION,
           'data_manifest_sha256': sha256(args.data / 'prepare.json')}
    del train, data
    meta, scores = score_eval(models, args.cache, 'val', 96)
    for name, score in scores.items():
        prediction_frame(meta, score).write_parquet(args.out / name / 'val_predictions.parquet')
    fit['artifacts'] = {arm: {filename: sha256(args.out / arm / filename)
                             for filename in ('model.pt', 'val_predictions.parquet')} for arm in ARMS}
    write_json(args.out / 'tune.json', fit)


def assess_coverage(args):
    fit = json.loads((args.run_dir / 'tune.json').read_text(encoding='utf-8'))
    for arm, artifacts in fit['artifacts'].items():
        for filename, digest in artifacts.items():
            if sha256(args.run_dir / arm / filename) != digest:
                raise ValueError(f'{arm}/{filename} differs from the scored run')
    anchor_digest = sha256(args.anchor)
    test_anchor_digest = sha256(args.test_anchor) if args.test_anchor is not None else None
    assess(args, arms=ARMS, difference=COVERAGE_DELTA)
    path = args.out / 'assessment.json'
    report = json.loads(path.read_text(encoding='utf-8'))
    report.update(anchor_sha256=anchor_digest, test_anchor_sha256=test_anchor_digest,
                  val_prediction_sha256={arm: files['val_predictions.parquet']
                                         for arm, files in fit['artifacts'].items()})
    write_json(path, report)


def submit(args):
    report = json.loads(args.assessment.read_text(encoding='utf-8'))
    if not report['prepare_submission'] or args.output.exists():
        raise ValueError('submission requires a qualifying assessment and a new output path')
    if sha256(args.anchor) != report['test_anchor_sha256']:
        raise ValueError('test anchor differs from the assessed reference')
    choice = report['choice']
    arms = ARMS if choice['name'] == COVERAGE_DELTA else (choice['name'],)
    models = {}
    mean = np.zeros(len(FEATURES), dtype=np.float32)
    for arm in arms:
        path = args.run_dir / arm / 'model.pt'
        if sha256(path) != report['model_sha256'][arm]:
            raise ValueError('checkpoint differs from the assessed run')
        checkpoint = torch.load(path, weights_only=True, map_location='cpu')
        if checkpoint['features'] != FEATURES:
            raise ValueError('checkpoint has a different feature schema')
        model = model_for(mean, np.ones_like(mean), checkpoint['width'])
        model.load_state_dict(checkpoint['model'])
        models[arm] = model
    meta, scores = score_eval(models, args.cache, 'test', 96)
    keys = prediction_frame(meta, np.zeros(len(meta['item']), dtype=np.float32)).select(KEYS)
    scores['submitted'] = read_scores(args.anchor, keys)
    result = scores_for_blend(scores, meta, choice['name'], choice['weight'], ARMS, COVERAGE_DELTA)
    with NamedTemporaryFile(dir=args.output.parent, suffix='.parquet', delete=False) as file:
        temporary = Path(file.name)
    try:
        prediction_frame(meta, result).write_parquet(temporary)
        audit = audit_submission(temporary)
        if audit['rows'] != 2_904_806 or audit['feeds_with_ties']:
            raise ValueError(f'submission audit failed: {audit}')
        temporary.rename(args.output)
    finally:
        temporary.unlink(missing_ok=True)
    write_json(args.output.with_suffix('.audit.json'), audit)
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    data = commands.add_parser('prepare')
    data.add_argument('reference', type=Path)
    data.add_argument('out', type=Path)
    data.add_argument('--shards', type=int, default=192)
    data.add_argument('--negatives', type=int, default=12)
    data.add_argument('--seed', type=int, default=2026)
    data.add_argument('--workers', type=int, default=4)
    data.set_defaults(run=prepare)
    train = commands.add_parser('tune')
    train.add_argument('data', type=Path)
    train.add_argument('cache', type=Path)
    train.add_argument('out', type=Path)
    train.add_argument('--epochs', type=int, default=4)
    train.add_argument('--batch-size', type=int, default=192)
    train.add_argument('--width', type=int, default=48)
    train.add_argument('--lr', type=float, default=.0015)
    train.add_argument('--threads', type=int, default=6)
    train.add_argument('--seed', type=int, default=2026)
    train.set_defaults(run=tune)
    evaluation = commands.add_parser('assess')
    evaluation.add_argument('cache', type=Path)
    evaluation.add_argument('run_dir', type=Path)
    evaluation.add_argument('anchor', type=Path)
    evaluation.add_argument('out', type=Path)
    evaluation.add_argument('--test-anchor', type=Path)
    evaluation.set_defaults(run=assess_coverage, prior_pretrained=None)
    prediction = commands.add_parser('submit')
    for field in ('cache', 'run_dir', 'assessment', 'anchor', 'output'):
        prediction.add_argument(field, type=Path)
    prediction.add_argument('--threads', type=int, default=6)
    prediction.set_defaults(run=submit)
    args = parser.parse_args()
    if args.action in ('tune', 'submit'):
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
    args.run(args)


if __name__ == '__main__':
    main()
