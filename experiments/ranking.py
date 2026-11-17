"""Reproducible CPU ranking experiments on complete feeds."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import polars as pl
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'competition'
KEYS = ['feed_id', 'item_id']
METRICS = ('ndcg', 'ndcg_at_10', 'ndcg_at_30')
POSITION_FEATURE = 'logged_position'
NUMERIC = [f'f{i}' for i in range(1, 6)]
BASE = [*NUMERIC, 'category', 'user_dot', 'user_cos', 'item_missing']
CONTEXT = [f'{f}_{s}' for f in [*NUMERIC, 'user_cos'] for s in ('rank', 'z')]
CONTEXT += ['feed_size', 'category_count', 'user_norm', 'item_norm']
HISTORY = [f'hist_category_{n}' for n in (8, 32, 256)]
HISTORY += [f'hist_event_{n}_category' for n in range(1, 11)]
HISTORY += ['hist_length', 'hist_valid', 'hist_same_item', 'hist_first_category',
            'hist_cos_max', 'hist_cos_top3', 'hist_cos_mean', 'hist_cos_recent8',
            'hist_cos_recent32', 'hist_cos_category_max', 'hist_cos_category_mean',
            'hist_cos_early_late', 'hist_cos_max_rank']
NAMES = BASE + CONTEXT + HISTORY
COLS = {'basic': len(BASE), 'context': len(BASE + CONTEXT), 'history': len(NAMES)}
NAMES += ['item_log_exposure', 'item_log_positive', 'item_smoothed_rate', 'item_log_exposure_last5']
COLS['history-priors'] = len(NAMES)
INTENT = [f'intent_max_type_{i}' for i in range(1, 11)]
INTENT += ['intent_max_4', 'intent_max_16', 'intent_max_64', 'intent_decayed_mean',
           'intent_recency_adjusted_max', 'intent_match_event_rank', 'intent_match_type',
           'intent_match_source', 'intent_match_same_category', 'intent_category_recency',
           'intent_category_recent_cos', 'intent_strong_matches', 'intent_feed_center_cos',
           'intent_rank_recent_max']
NAMES += INTENT
COLS['history-intent'] = len(NAMES)


def normalize(vectors):
    vectors = np.asarray(vectors, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    return np.divide(vectors, norms, out=np.zeros_like(vectors), where=norms > 0)


def temporal_boundary(timestamps, fraction):
    boundaries = np.flatnonzero(timestamps[1:] != timestamps[:-1]) + 1
    if len(boundaries) == 0:
        raise ValueError('need at least two timestamps')
    return int(boundaries[np.argmin(abs(boundaries - len(timestamps) * fraction))])


def ndcg_per_feed(target, score, counts):
    if len(target) != len(score) or sum(counts) != len(score) or np.any(counts < 1):
        raise ValueError('scores, targets and feed sizes do not align')
    ends = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    discount = 1 / np.log2(np.arange(counts.max()) + 2)
    result = np.ones((len(counts), 3))
    for i, (start, stop) in enumerate(zip(ends[:-1], ends[1:])):
        labels = target[start:stop]
        positives = int(labels.sum())
        if positives:
            ranked = labels[np.lexsort((labels, -score[start:stop]))]
            for j, cutoff in enumerate((len(labels), 10, 30)):
                n = min(len(labels), cutoff)
                result[i, j] = ranked[:n] @ discount[:n] / discount[:min(n, positives)].sum()
    return result


def rank_fraction(values):
    sorted_values = np.sort(values)
    left = sorted_values.searchsorted(values, side='left')
    right = sorted_values.searchsorted(values, side='right')
    return (left + right) * (0.5 / len(values))


def read_vectors(table, indices):
    indices = np.asarray(indices, dtype=np.int32)
    present = indices >= 0
    vectors = np.zeros((len(indices), table.shape[1]), dtype=np.float32)
    vectors[present] = table[indices[present]]
    return vectors, present


class StagedVectors:
    """Read selected vectors in increasing file order before random per-feed access."""

    def __init__(self, filename, meta, paths):
        self.shape = tuple(meta['shape'])
        cols = ['item_id__embedding_idx', 'history__item_id__embedding_idx']
        indices = []
        for path in paths:
            for batch in pq.ParquetFile(path).iter_batches(columns=cols, batch_size=4096):
                for col in batch.columns:
                    a = col.flatten().to_numpy()
                    indices.append(a[a >= 0])
        self.indices = np.unique(np.concatenate(indices))
        del indices
        self.values = np.empty((len(self.indices), self.shape[1]), dtype=meta['dtype'])
        source = np.memmap(filename, dtype=meta['dtype'], mode='r', shape=self.shape, order=meta['order'])
        rows = 1 << 19
        for start in range(0, self.shape[0], rows):
            end = min(start + rows, self.shape[0])
            lo, hi = self.indices.searchsorted([start, end])
            if lo != hi:
                # Copy a contiguous slice to keep disk reads sequential, then gather in RAM.
                block = source[start:end].copy()
                self.values[lo:hi] = block[self.indices[lo:hi] - start]
        print(f'Staged {len(self.indices)} item vectors', flush=True)

    def __getitem__(self, indices):
        locations = self.indices.searchsorted(indices)
        return self.values[locations]


def feed_features(row, table):
    items = np.asarray(row['item_id'], dtype=np.int32)
    cats = np.asarray(row['category_id'], dtype=np.int32)
    user = np.asarray(row['user_embedding'], dtype=np.float32)
    vectors, present = read_vectors(table, row['item_id__embedding_idx'])
    unit = normalize(vectors)
    cosine = unit @ normalize(user)
    cols = [*[np.asarray(row[f], dtype=np.float32) for f in NUMERIC], cats,
            vectors @ user, cosine, ~present]
    for v in [*cols[:5], cosine]:
        cols.extend((rank_fraction(v), (v - v.mean()) / max(float(v.std()), 1e-6)))
    n = len(items)
    cols.extend((np.full(n, n), np.bincount(cats)[cats], np.full(n, np.linalg.norm(user)),
                 np.linalg.norm(vectors, axis=1)))

    hist_ids = np.asarray(row['history__item_id'], dtype=np.int32)
    hist_cats = np.asarray(row['history__category_id'], dtype=np.int32)
    types = np.asarray(row['history__event_type'], dtype=np.int32)
    size = max(int(cats.max()), int(hist_cats.max(initial=0))) + 1
    for k in (8, 32, 256):
        cols.append(np.bincount(hist_cats[:k], minlength=size)[cats])
    for k in range(1, 11):
        cols.append(np.bincount(hist_cats[types == k], minlength=size)[cats])
    history, valid = read_vectors(table, row['history__item_id__embedding_idx'])
    same_category = cats[:, None] == hist_cats
    first = np.where(same_category.any(axis=1), same_category.argmax(axis=1), 257) if len(history) else np.full(n, 257)
    cols.extend((np.full(n, len(history)), np.full(n, valid.sum()), np.isin(items, hist_ids), first))

    if valid.any():
        sim = unit @ normalize(history[valid]).T
        same_valid = same_category[:, valid]
        positions = np.flatnonzero(valid)
        maximum = sim.max(axis=1)
        k = min(3, sim.shape[1])
        top = np.partition(sim, -k, axis=1)[:, -k:].mean(axis=1)
        mean = sim.mean(axis=1)
        category_max = np.where(same_valid, sim, -np.inf).max(axis=1)
        category_max[~np.isfinite(category_max)] = np.nan
        count = same_valid.sum(axis=1)
        category_mean = np.divide((sim * same_valid).sum(axis=1), count,
                                  out=np.full(n, np.nan), where=count > 0)
        windows = [sim[:, positions < k].mean(axis=1) if (positions < k).any()
                   else np.full(n, np.nan) for k in (8, 32)]
        cols.extend((maximum, top, mean, *windows, category_max, category_mean,
                     windows[1] - mean, rank_fraction(maximum)))
        # Candidates with missing vectors are unavailable for semantic matching.
        for col in cols[-9:]:
            col[~present] = np.nan
    else:
        cols.extend(np.full(n, np.nan) for _ in range(9))
    return np.column_stack(cols).astype(np.float32)


def prepare(split, out, shards, seed, stage, include):
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    rng = np.random.default_rng(seed)
    if split == 'train':
        if include is None:
            paths = sorted(rng.choice(paths, min(shards, len(paths)), replace=False))
        else:
            requested = set(json.loads(include.read_text()))
            base = [path for path in paths if path.name in requested]
            remaining = [path for path in paths if path.name not in requested]
            if len(base) != len(requested) or not len(base) <= shards <= len(paths):
                raise ValueError('included shards must exist and fit within the requested train size')
            extra = rng.choice(remaining, shards - len(base), replace=False)
            paths = sorted([*base, *extra])
    tables = DATA / split / 'item_embeddings'
    meta = json.loads((tables / 'data.meta').read_text())
    filename = tables / 'data.memmap'
    if stage:
        vectors = StagedVectors(filename, meta, paths)
    else:
        vectors = np.memmap(filename, dtype=meta['dtype'], mode='r',
                            shape=tuple(meta['shape']), order=meta['order'])
    batches = []
    for i, path in enumerate(paths, 1):
        feeds = []
        for batch in pq.ParquetFile(path).iter_batches(batch_size=128):
            for row in batch.to_pylist():
                features = feed_features(row, vectors)
                item = np.asarray(row['item_id'], dtype=np.int32)
                target = np.asarray(row.get('target', np.zeros(len(item))), dtype=np.int8)
                if split == 'train':
                    order = rng.permutation(len(item))
                    features, item, target = features[order], item[order], target[order]
                    pos = np.asarray(row['pos'], dtype=np.int16)[order]
                else:
                    pos = None
                feeds.append((row['timestamp'], row['feed_id'], features, item, target, pos))
        batches.extend(feeds)
        if split == 'train' or i == len(paths) or i % 8 == 0:
            print(f'{split}: {i}/{len(paths)} shards, {len(batches)} feeds', flush=True)
    batches.sort(key=lambda feed: (feed[0], feed[1]))
    timestamp, feed, features, items, target, positions = zip(*batches)
    counts = np.asarray([len(a) for a in items], dtype=np.int32)
    with NamedTemporaryFile(dir=out, suffix='.npz', delete=False) as f:
        tmp = Path(f.name)
    try:
        np.savez(tmp, x=np.concatenate(features), target=np.concatenate(target),
                 item=np.concatenate(items), feed=np.asarray(feed, dtype=np.int32),
                 timestamp=np.asarray(timestamp, dtype=np.int32), counts=counts,
                 shards=[p.name for p in paths], prepare_seed=seed,
                 position=np.concatenate(positions) if split == 'train' else np.empty(0, dtype=np.int16))
        tmp.replace(out / f'{split}.npz')
    finally:
        tmp.unlink(missing_ok=True)


def prediction_frame(data, score):
    return pl.DataFrame({'feed_id': np.repeat(data['feed'], data['counts']),
                         'item_id': data['item'], 'score': score.astype(np.float32)})


def audit_submission(path):
    frame = pl.read_parquet(path)
    template = pl.read_parquet(DATA / 'sample_submission.parquet', columns=KEYS)
    if frame.schema != {'feed_id': pl.Int32, 'item_id': pl.Int32, 'score': pl.Float32}:
        raise ValueError(f'unexpected submission schema: {frame.schema}')
    ordered = frame.sort(KEYS)
    if not ordered.select(KEYS).equals(template.sort(KEYS)):
        raise ValueError('submission keys differ from the official template')
    scores = frame['score']
    if scores.null_count() or not scores.is_finite().all():
        raise ValueError('submission contains missing or nonfinite scores')
    tied = frame.group_by('feed_id').agg(pl.len().alias('n'), pl.col('score').n_unique().alias('unique'))
    with path.open('rb') as f:
        digest = hashlib.file_digest(f, 'sha256').hexdigest()
    return {'rows': frame.height, 'feeds': tied.height, 'sha256': digest,
            'score_min': scores.min(), 'score_max': scores.max(), 'score_median': scores.median(),
            'feeds_with_ties': tied.filter(pl.col('n') != pl.col('unique')).height,
            'constant_feeds': tied.filter(pl.col('unique') == 1).height}


def evaluate(data, score, boundary):
    per_feed = ndcg_per_feed(data['target'], score.astype(np.float32), data['counts'])
    return {name: dict(zip(METRICS, values.mean(axis=0).tolist())) for name, values in
            [('early', per_feed[:boundary]), ('late', per_feed[boundary:]), ('all', per_feed)]}


def train_positions(train_file, train_shards, seed):
    edges = [1, 2, 3, 5, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    with np.load(train_file) as f:
        if 'position' in f:
            pos = f['position']
            if len(pos) != len(f['item']) or np.any((pos < 0) | (pos > 300)):
                raise ValueError('invalid positions in train cache')
            return np.searchsorted(edges, pos, side='right').astype(np.int32)
    all_paths = sorted((DATA / 'train/impressions').glob('part-*.parquet'))
    rng = np.random.default_rng(seed)
    paths = sorted(rng.choice(all_paths, len(train_shards), replace=False))
    if [p.name for p in paths] != train_shards:
        raise ValueError('the selected files differ from the original train cache')
    feeds = []
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(columns=['timestamp', 'feed_id', 'item_id', 'pos']):
            for row in batch.to_pylist():
                order = rng.permutation(len(row['pos']))
                feeds.append((row['timestamp'], row['feed_id'], np.asarray(row['item_id'])[order],
                              np.asarray(row['pos'])[order]))
    feeds.sort(key=lambda row: (row[0], row[1]))
    with np.load(train_file) as f:
        np.testing.assert_array_equal(f['feed'], [row[1] for row in feeds])
        np.testing.assert_array_equal(f['item'], np.concatenate([row[2] for row in feeds]))
    return np.searchsorted(edges, np.concatenate([row[3] for row in feeds]), side='right').astype(np.int32)


def position_averaged_prediction(model, features, threads, num_iteration=None):
    """Approximate a uniform reference over logged ranks using five weighted midpoints."""
    edges = np.array([0, 8, 32, 96, 192, 301], dtype=np.int32)
    feature_matrix = np.empty((len(features), model.num_feature()), dtype=np.float32)
    feature_matrix[:, :-1] = features
    result = np.zeros(len(features), dtype=np.float64)
    for left, right in zip(edges[:-1], edges[1:]):
        feature_matrix[:, -1] = (left + right - 1) // 2
        result += (right - left) * model.predict(feature_matrix, num_iteration=num_iteration,
                                                num_threads=threads)
    return result / edges[-1]


def fitted_scores(model, x, threads):
    if model.feature_name()[-1] == POSITION_FEATURE:
        return position_averaged_prediction(model, x[:, :model.num_feature() - 1], threads)
    return model.predict(np.ascontiguousarray(x[:, :model.num_feature()]),
                         num_threads=threads, raw_score=True)


def read_sample_weights(path, train_file):
    with np.load(path) as data, np.load(train_file) as train:
        for name in ('feed', 'item', 'counts'):
            if not np.array_equal(data[name], train[name]):
                raise ValueError(f'sample weights do not match training keys in {name}')
        weights = data['weight'].astype(np.float32, copy=False)
        n = len(data['item'])
    if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError('sample weights must be finite and strictly positive for every training candidate')
    return weights


def read_extra_features(path, base_file):
    with np.load(path) as extra, np.load(base_file) as base:
        for name in ('feed', 'item', 'counts', 'timestamp'):
            if not np.array_equal(extra[name], base[name]):
                raise ValueError(f'extra features do not match base cache {name} keys')
        names = extra['names'].tolist()
        values = extra['features'].astype(np.float32, copy=False)
        rows = len(extra['item'])
    if (not isinstance(names, list) or not names
            or any(not isinstance(name, str) or not name.isidentifier() for name in names)
            or len(set(names)) != len(names) or set(names).intersection([*NAMES, POSITION_FEATURE])):
        raise ValueError('extra feature names must be unique identifiers distinct from built-in features')
    if values.shape != (rows, len(names)) or np.isinf(values).any():
        raise ValueError('extra feature matrix must have a column per name and no infinite values')
    return values, names


def train(cache, out, kind, rounds, seed, threads, objective, train_file, position_debias,
          position_standardize, feed_weight_power=0.0, sample_weights=None, extra_features=None):
    import lightgbm as lgb

    if out.exists():
        raise FileExistsError(out)
    train_file = train_file or cache / 'train.npz'
    with np.load(train_file) as f:
        train_x, train_y, train_counts = f['x'], f['target'], f['counts']
        if 'shards' in f:
            train_shards, prepare_seed = f['shards'].tolist(), int(f['prepare_seed'])
        else:
            train_shards = json.loads(train_file.with_name('train_shards.json').read_text())
            prepare_seed = None  # Cache from the first experiment, recorded in the journal.
    with np.load(cache / 'val.npz') as f:
        val = dict(f)
    n = COLS[kind]
    if train_x.shape[1] < n or val['x'].shape[1] < n:
        raise ValueError(f'{kind} needs a cache with at least {n} features')
    x = train_x if train_x.shape[1] == n else np.ascontiguousarray(train_x[:, :n])
    del train_x
    val_x = np.ascontiguousarray(val['x'][:, :n])
    boundary = temporal_boundary(val['timestamp'], .4)
    early_rows = int(val['counts'][:boundary].sum(dtype=np.int64))
    params = {'objective': objective, 'metric': 'None', 'learning_rate': .08,
              'num_leaves': 63, 'min_data_in_leaf': 160, 'lambda_l2': 5,
              'max_bin': 127, 'feature_fraction': .9, 'num_threads': threads,
              'force_col_wise': True, 'verbosity': -1, 'seed': seed}
    if objective == 'lambdarank':
        params['lambdarank_truncation_level'] = 40
    position = None
    if position_debias:
        if objective != 'lambdarank':
            raise ValueError('position correction requires lambdarank')
        position = train_positions(train_file, train_shards, prepare_seed if prepare_seed is not None else seed)
        params['lambdarank_position_bias_regularization'] = .1
    names = NAMES[:n]
    if extra_features is not None:
        extra_train, extra_names = read_extra_features(extra_features / 'train.npz', train_file)
        extra_val, val_names = read_extra_features(extra_features / 'val.npz', cache / 'val.npz')
        if extra_names != val_names:
            raise ValueError('extra train and val columns disagree')
        x = np.column_stack((x, extra_train))
        val_x = np.column_stack((val_x, extra_val))
        del extra_train, extra_val
        names = [*names, *extra_names]
    if position_standardize:
        if objective != 'binary' or position_debias:
            raise ValueError('reference positions require the binary objective')
        with np.load(train_file) as f:
            logged = f['position']
        if len(logged) != len(x) or np.any((logged < 0) | (logged > 300)):
            raise ValueError('invalid reference positions in train cache')
        x = np.column_stack((x, logged)).astype(np.float32, copy=False)
        names = [*names, POSITION_FEATURE]
    categories = [name for name in ('category', 'intent_match_type', 'intent_match_source') if name in names]
    if not 0 <= feed_weight_power <= 1 or (feed_weight_power and objective != 'binary'):
        raise ValueError('feed weight power must be in [0, 1] and used with the binary objective')
    weight = None
    if sample_weights is not None:
        if feed_weight_power or objective != 'binary':
            raise ValueError('sample weights require the binary objective with no feed weights')
        weight = read_sample_weights(sample_weights, train_file)
    if feed_weight_power:
        per_feed_weight = train_counts.astype(np.float64) ** -feed_weight_power
        per_feed_weight /= np.average(per_feed_weight, weights=train_counts)
        weight = np.repeat(per_feed_weight.astype(np.float32), train_counts)
    train_data = lgb.Dataset(x, label=train_y, group=train_counts, params=params, weight=weight,
                             position=position, feature_name=names, categorical_feature=categories)
    print(f'{kind}: {len(train_counts)} train feeds, {len(x)} rows, {len(names)} features', flush=True)
    train_data.construct()
    del x
    print(f'{out.name}: training matrix binned', flush=True)

    def report_progress(state):
        completed = state.iteration + 1
        if completed % 40 == 0 or completed == rounds:
            print(f'{out.name}: trained {completed}/{rounds} trees', flush=True)

    model = lgb.train(params, train_data, num_boost_round=rounds, callbacks=[report_progress])
    del train_data
    checkpoints = sorted(set(range(40, rounds + 1, 40)) | {rounds})
    val_scores = np.zeros(len(val_x), dtype=np.float64)
    early_results = {}
    previous = 0
    best_score, best_iter = float('-inf'), 0
    saved = None
    for current in checkpoints:
        if position_standardize:
            val_scores = position_averaged_prediction(model, val_x, threads, num_iteration=current)
        else:
            val_scores += model.predict(val_x, start_iteration=previous, num_iteration=current-previous,
                                        num_threads=threads, raw_score=True)
        previous = current
        early = ndcg_per_feed(val['target'][:early_rows], val_scores[:early_rows].astype(np.float32),
                              val['counts'][:boundary])[:, 0].mean()
        early_results[current] = float(early)
        print(f'{kind} trees={current}, early NDCG={early:.6f}', flush=True)
        if early > best_score:
            best_score, best_iter = early, current
            saved = val_scores.copy()
    direct = saved
    if not position_standardize:
        direct = model.predict(val_x, num_iteration=best_iter, num_threads=threads, raw_score=True)
        np.testing.assert_allclose(saved, direct, rtol=1e-10, atol=1e-10)
    quality = evaluate(val, direct, boundary)
    out.mkdir(parents=True, exist_ok=False)
    model.save_model(str(out / 'model.txt'), num_iteration=best_iter)
    prediction_frame(val, direct).write_parquet(out / 'val_predictions.parquet')
    report = {'kind': kind, 'params': params, 'train_feeds': len(train_counts),
              'train_rows': len(train_y), 'train_shards': train_shards, 'prepare_seed': prepare_seed,
              'position_debias': position_debias,
              'position_standardize': position_standardize,
              'feed_weight_power': feed_weight_power,
              'sample_weights': str(sample_weights) if sample_weights is not None else None,
              'extra_features': str(extra_features) if extra_features is not None else None,
              'early_feeds': boundary, 'late_feeds': len(val['counts']) - boundary,
              'late_start_timestamp': int(val['timestamp'][boundary]), 'rounds': rounds,
              'selected_trees': best_iter, 'early_search': early_results, 'metrics': quality,
              'importance_gain': dict(zip(names, model.feature_importance('gain', iteration=best_iter).tolist()))}
    (out / 'metrics.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(quality), flush=True)


def predict(cache, model_path, output, threads, extra_features=None):
    import lightgbm as lgb

    if output.exists():
        raise FileExistsError(output)
    model = lgb.Booster(model_file=str(model_path))
    with np.load(cache / 'test.npz') as f:
        data = dict(f)
    x = data.pop('x')
    model_names = model.feature_name()
    extra_names = [name for name in model_names if name not in NAMES and name != POSITION_FEATURE]
    if extra_names:
        if extra_features is None:
            raise ValueError('this model requires --extra-features with an aligned test.npz')
        extra, names = read_extra_features(extra_features / 'test.npz', cache / 'test.npz')
        if names != extra_names:
            raise ValueError('extra test columns disagree with the trained model')
        n_base = len(model_names) - len(extra_names) - (POSITION_FEATURE in model_names)
        x = np.column_stack((x[:, :n_base], extra))
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_frame(data, fitted_scores(model, x, threads)).write_parquet(output)
    print(json.dumps(audit_submission(output)), flush=True)


def standardized_scores(score, counts):
    starts = np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
    centered = score - np.repeat(np.add.reduceat(score, starts) / counts, counts)
    deviations = np.sqrt(np.add.reduceat(centered ** 2, starts) / counts)
    return centered / np.repeat(np.maximum(deviations, 1e-8), counts)


def blend(cache, out, runs, predict_test, threads, weights=None):
    weights = np.asarray(weights if weights is not None else np.ones(len(runs)), dtype=np.float64)
    if (weights.shape != (len(runs),) or not np.isfinite(weights).all()
            or np.any(weights < 0) or weights.sum() <= 0):
        raise ValueError('give one nonnegative finite weight per model, with a positive sum')
    split = 'test' if predict_test else 'val'
    with np.load(cache / f'{split}.npz') as f:
        data = dict(f)
    keys = prediction_frame(data, np.zeros(len(data['item']))).select(KEYS)
    total = np.zeros(len(keys), dtype=np.float64)
    for path, weight in zip(runs, weights, strict=True):
        if predict_test:
            import lightgbm as lgb
            model = lgb.Booster(model_file=str(path / 'model.txt'))
            score = fitted_scores(model, data['x'], threads).astype(np.float32)
        else:
            prediction = pl.read_parquet(path / 'val_predictions.parquet')
            if not prediction.select(KEYS).equals(keys):
                raise ValueError(f'misaligned candidate keys for {path}')
            score = prediction['score'].to_numpy()
        total += weight * standardized_scores(score.astype(np.float64), data['counts'])
    score = total / weights.sum()
    out.mkdir(parents=True, exist_ok=True)
    output = out / ('submission.parquet' if predict_test else 'val_predictions.parquet')
    if output.exists():
        raise FileExistsError(output)
    prediction_frame(data, score).write_parquet(output)
    if predict_test:
        print(json.dumps(audit_submission(output)), flush=True)
    else:
        boundary = temporal_boundary(data['timestamp'], .4)
        component_sizes = [json.loads((p / 'metrics.json').read_text())['train_feeds'] for p in runs]
        train_feeds = component_sizes[0] if len(set(component_sizes)) == 1 else component_sizes
        result = {'components': [str(p) for p in runs], 'train_feeds': train_feeds, 'selected_trees': None,
                  'weights': (weights / weights.sum()).tolist(),
                  'early_feeds': boundary, 'late_feeds': len(data['counts']) - boundary,
                  'late_start_timestamp': int(data['timestamp'][boundary]),
                  'metrics': evaluate(data, score, boundary)}
        (out / 'metrics.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(result['metrics']), flush=True)


def compare(paths):
    runs = []
    for path in paths:
        info = json.loads((path / 'metrics.json').read_text())
        runs.append({'run': path.name, 'train_feeds': info['train_feeds'],
                     'selected_trees': info['selected_trees'], 'metrics': info['metrics'],
                     'validation': [info[k] for k in ('early_feeds', 'late_feeds', 'late_start_timestamp')]})
    if any(run['validation'] != runs[0]['validation'] for run in runs):
        raise ValueError('runs used different validation splits')
    runs.sort(key=lambda run: run['metrics']['late']['ndcg'], reverse=True)
    print(json.dumps({'selection_metric': 'late.ndcg', 'winner': runs[0]['run'], 'runs': runs}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    prep = subs.add_parser('prepare')
    prep.add_argument('split', choices=['train', 'val', 'test'])
    prep.add_argument('cache', type=Path)
    prep.add_argument('--shards', type=int, default=6)
    prep.add_argument('--seed', type=int, default=2026)
    prep.add_argument('--stage-embeddings', action='store_true')
    prep.add_argument('--include-train-shards', type=Path)
    fit = subs.add_parser('train')
    fit.add_argument('cache', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('kind', choices=COLS)
    fit.add_argument('--rounds', type=int, default=160)
    fit.add_argument('--train-file', type=Path)
    fit.add_argument('--position-debias', action='store_true')
    fit.add_argument('--position-standardize', action='store_true')
    fit.add_argument('--feed-weight-power', type=float, default=0.0)
    fit.add_argument('--sample-weights', type=Path)
    fit.add_argument('--extra-features', type=Path)
    fit.add_argument('--seed', type=int, default=2026)
    fit.add_argument('--threads', type=int, default=8)
    fit.add_argument('--objective', choices=['lambdarank', 'rank_xendcg', 'binary'], default='lambdarank')
    inf = subs.add_parser('predict')
    inf.add_argument('cache', type=Path)
    inf.add_argument('model', type=Path)
    inf.add_argument('output', type=Path)
    inf.add_argument('--threads', type=int, default=8)
    inf.add_argument('--extra-features', type=Path)
    audit = subs.add_parser('audit')
    audit.add_argument('submission', type=Path)
    comparison = subs.add_parser('compare')
    comparison.add_argument('runs', type=Path, nargs='+')
    ensemble = subs.add_parser('blend')
    ensemble.add_argument('cache', type=Path)
    ensemble.add_argument('out', type=Path)
    ensemble.add_argument('runs', type=Path, nargs='+')
    ensemble.add_argument('--test', action='store_true')
    ensemble.add_argument('--threads', type=int, default=8)
    ensemble.add_argument('--weights', type=float, nargs='+')
    args = parser.parse_args()
    if args.action == 'prepare':
        args.cache.mkdir(parents=True, exist_ok=True)
        prepare(args.split, args.cache, args.shards, args.seed, args.stage_embeddings,
                args.include_train_shards)
    elif args.action == 'train':
        train(args.cache, args.out, args.kind, args.rounds, args.seed, args.threads, args.objective,
              args.train_file, args.position_debias, args.position_standardize, args.feed_weight_power,
              args.sample_weights, args.extra_features)
    elif args.action == 'predict':
        predict(args.cache, args.model, args.output, args.threads, args.extra_features)
    elif args.action == 'audit':
        print(json.dumps(audit_submission(args.submission), indent=2))
    elif args.action == 'compare':
        compare(args.runs)
    else:
        blend(args.cache, args.out, args.runs, args.test, args.threads, args.weights)


if __name__ == '__main__':
    main()
