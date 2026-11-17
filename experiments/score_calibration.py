"""Fit a small within-feed calibration on early val and evaluate on later feeds."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from neural_blend import read_scores, resolve_ties
from ranking import (KEYS, NAMES, audit_submission, ndcg_per_feed, prediction_frame,
                     rank_fraction, temporal_boundary)


SIGNALS = ('f1', 'f2', 'f3', 'f4', 'f5', 'user_cos', 'hist_cos_max', 'intent_max_16',
           'intent_category_recency', 'intent_match_same_category', 'item_log_exposure',
           'item_smoothed_rate')


def slices(counts):
    return np.r_[0, np.cumsum(counts, dtype=np.int64)]


def within_feed_z(values, counts, limit=5):
    starts = slices(counts)[:-1]
    finite = np.isfinite(values)
    valid_count = np.maximum(np.add.reduceat(finite.astype(np.int32), starts), 1)
    clean = np.where(finite, values, 0).astype(np.float64, copy=False)
    means = np.add.reduceat(clean, starts) / valid_count
    centered = np.where(finite, clean - np.repeat(means, counts), 0)
    variance = np.add.reduceat(centered * centered, starts) / valid_count
    divisor = np.repeat(np.maximum(np.sqrt(variance), 1e-7), counts)
    return np.clip(centered / divisor, -limit, limit)


def design(raw_scores, features, counts):
    reference, trees, network, *extra = raw_scores
    starts = slices(counts)
    percentile = np.concatenate([rank_fraction(reference[a:b]) for a, b in zip(starts[:-1], starts[1:])])
    baseline = np.log(percentile / (1-percentile))
    columns = [baseline, within_feed_z(trees, counts), within_feed_z(network, counts)]
    names = ['reference_log_rank', 'trees_z', 'network_z']
    for i, score in enumerate(extra, 1):
        columns.append(within_feed_z(score, counts))
        names.append(f'extra_model_{i}_z')
    numeric = {name: within_feed_z(features[:, NAMES.index(name)], counts) for name in SIGNALS}
    columns.extend(numeric.values())
    names.extend(numeric)
    for name in ('f1', 'f2', 'f3', 'user_cos', 'hist_cos_max'):
        for direction in (1, -1):
            v = within_feed_z(np.maximum(direction*numeric[name] - 1, 0), counts)
            columns.append(v)
            names.append(f'{name}_tail_{direction:+d}')
    return np.column_stack(columns), names


def grouped_objective(weights, x, normalized_y, counts, penalty, center):
    starts = slices(counts)[:-1]
    logits = x @ weights
    max_score = np.maximum.reduceat(logits, starts)
    shifted = logits - np.repeat(max_score, counts)
    masses = np.exp(shifted)
    partition = np.add.reduceat(masses, starts)
    p = masses / np.repeat(partition, counts)
    diff = weights - center
    loss = ((np.log(partition).sum() - np.dot(normalized_y, shifted)) / len(counts)
            + .5 * penalty * np.dot(diff, diff))
    grad = x.T @ (p - normalized_y) / len(counts) + penalty * diff
    return loss, grad


def fit(x, y, counts, penalty, center=None):
    counts_positive = np.add.reduceat(y, slices(counts)[:-1])
    if np.any(counts_positive < 1):
        raise ValueError('calibration needs one or more positive labels per feed')
    y = y.astype(np.float64) / np.repeat(counts_positive, counts)
    if center is None:
        center = np.zeros(x.shape[1])
        center[0] = .45
    result = minimize(grouped_objective, center, args=(x, y, counts, penalty, center),
                      method='L-BFGS-B', jac=True,
                      bounds=[(.05, 4.)] + [(None, None)]*(x.shape[1]-1),
                      options={'maxiter': 65, 'ftol': 1e-8, 'gtol': 1e-5})
    if not result.success:
        raise RuntimeError(f'calibration optimizer failed: {result.message}')
    return result.x


def negative_rank_sample(negative, mode):
    if mode == 'top':
        return negative[:32], np.ones(min(len(negative), 32))
    if len(negative) <= 32:
        return negative, np.ones(len(negative))
    boundaries = np.linspace(8, len(negative), 25, dtype=np.int64)
    width = np.diff(boundaries)
    center = (boundaries[:-1] + boundaries[1:] - 1) // 2
    return np.concatenate((negative[:8], negative[center])), np.r_[np.ones(8), width]


def make_pairs(matrix, target, counts, reference, mode='stratified'):
    """Weight positive-negative pairs by the NDCG cost of a swap in the reference order."""
    data, weights = [], []
    starts = slices(counts)
    discount = 1 / np.log2(np.arange(counts.max()) + 2)
    for first, last in zip(starts[:-1], starts[1:]):
        order = np.argsort(-reference[first:last], kind='stable')
        positive = np.flatnonzero(target[first + order] == 1)
        negative, multiplicity = negative_rank_sample(np.flatnonzero(target[first + order] == 0), mode)
        if not len(positive) or not len(negative):
            continue
        cost = np.abs(discount[positive, None] - discount[negative]) * multiplicity
        weights.append((cost / cost.sum()).ravel())
        p, n = matrix[first + order[positive]], matrix[first + order[negative]]
        data.append((p[:, None] - n).reshape(-1, matrix.shape[1]))
    if not data:
        raise ValueError('no positive-negative calibration pairs')
    return np.concatenate(data), np.concatenate(weights) / len(data)


def pairwise_objective(weights, pairs, mass, penalty, center):
    margin = pairs @ weights
    diff = weights - center
    loss = np.dot(mass, np.logaddexp(0, -margin)) + .5 * penalty * np.dot(diff, diff)
    grad = -(pairs.T @ (mass * expit(-margin))) + penalty * diff
    return loss, grad


def fit_pairwise(pairs, mass, penalty):
    center = np.zeros(pairs.shape[1])
    center[0] = .45
    result = minimize(pairwise_objective, center, args=(pairs, mass, penalty, center),
                      method='L-BFGS-B', jac=True,
                      bounds=[(.05, 4.)] + [(None, None)]*(pairs.shape[1]-1),
                      options={'maxiter': 65, 'ftol': 1e-8, 'gtol': 1e-5})
    if not result.success:
        raise RuntimeError(f'pairwise calibration optimizer failed: {result.message}')
    return result.x


def load_inputs(cache, paths, split):
    if len(paths) < 3:
        raise ValueError('supply the reference, trees and network score files, in that order')
    with np.load(cache / f'{split}.npz') as f:
        meta = {k: f[k] for k in ('feed', 'item', 'counts', 'timestamp', 'target')}
        features = f['x']
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    raw = [read_scores(p, keys) for p in paths]
    matrix, names = design(raw, features, meta['counts'])
    return meta, raw[0], matrix, names


def ordered_scores(matrix, weights, reference, item, counts):
    primary = (matrix @ weights).astype(np.float32)
    return resolve_ties(primary, reference, item, counts)


def mix_by_size(calibrated, reference, counts, item, weights):
    bins = np.searchsorted([30, 100], counts, side='left')
    weight = np.repeat(np.asarray(weights)[bins], counts)
    score = (weight*calibrated + (1-weight)*reference).astype(np.float32)
    return resolve_ties(score, reference, item, counts)


def pick_size_weights(pred, reference, y, item, counts):
    selected = []
    bins = np.searchsorted([30, 100], counts, side='left')
    for i in range(3):
        feed_mask = bins == i
        if not feed_mask.any():
            selected.append(0.)
            continue
        row_mask = np.repeat(feed_mask, counts)
        groups = counts[feed_mask]
        labels = y[row_mask]
        scores = []
        for alpha in (0., .5, 1.):
            value = mix_by_size(pred[row_mask], reference[row_mask], groups, item[row_mask], [alpha]*3)
            scores.append(float(ndcg_per_feed(labels, value, groups)[:, 0].mean()))
        selected.append((0., .5, 1.)[int(np.argmax(scores))])
    return selected


def smooth_rank_data(x, target, counts, reference):
    differences, multiplicity, comparisons, importance = [], [], [], []
    offsets = slices(counts)
    discount = 1 / np.log2(np.arange(counts.max()) + 2)
    for start, stop in zip(offsets[:-1], offsets[1:]):
        order = np.argsort(-reference[start:stop], kind='stable') + start
        positive = order[target[order] == 1]
        negative, mult = negative_rank_sample(order[target[order] == 0], 'stratified')
        if len(order) < 2 or not len(positive):
            continue
        ideal = discount[:len(positive)].sum()
        all_weight = np.r_[mult, np.ones(len(positive)-1)]
        for i, candidate in enumerate(positive):
            others = np.r_[negative, positive[:i], positive[i+1:]]
            differences.append(x[others] - x[candidate])
            comparisons.append(len(others))
            multiplicity.append(all_weight)
            importance.append(1 / (ideal * len(counts)))
    return (np.concatenate(differences), np.concatenate(multiplicity), np.asarray(comparisons),
            np.asarray(importance))


def smooth_ndcg_objective(w, data, temperature, l2, center):
    differences, mult, comparisons, importance = data
    p = expit((differences @ w) / temperature)
    rank = 1 + np.add.reduceat(p*mult, slices(comparisons)[:-1])
    logarithm = np.log1p(rank)
    diff = w - center
    loss = -np.log(2) * np.dot(importance, 1/logarithm) + .5 * l2 * np.dot(diff, diff)
    rank_grad = importance * np.log(2) / ((rank+1) * logarithm**2)
    grad = differences.T @ (np.repeat(rank_grad, comparisons) * mult * p * (1-p) / temperature)
    return loss, grad + l2 * diff


def fit_smooth(data, center, temperature, l2):
    result = minimize(smooth_ndcg_objective, center, args=(data, temperature, l2, center),
                      method='L-BFGS-B', jac=True,
                      bounds=[(.05, 4.)] + [(None, None)]*(len(center)-1),
                      options={'maxiter': 45, 'ftol': 1e-8, 'gtol': 1e-5})
    if not np.isfinite(result.x).all() or result.fun > smooth_ndcg_objective(center, data, temperature, l2, center)[0]:
        raise RuntimeError(f'smooth NDCG refinement failed: {result.message}')
    return result.x


def refine(args):
    initial = json.loads(args.initial.read_text())
    paths = [Path(p) for p in initial['val_sources']]
    meta, reference, matrix, names = load_inputs(args.cache, paths, 'val')
    counts, target, item = meta['counts'], meta['target'], meta['item']
    inner, early = initial['inner_train_feeds'], initial['early_feeds']
    cut, end = slices(counts)[[inner, early]]
    config = initial['chosen']
    x = np.ascontiguousarray(matrix[:, :config['width']])
    center = fit(x[:cut], target[:cut], counts[:inner], config['penalty'])
    pairs = smooth_rank_data(x[:cut], target[:cut], counts[:inner], reference[:cut])
    trials = []
    for temperature, l2 in [(None, None), (.25,.01), (.25,.07), (.55,.01), (.55,.07)]:
        w = center if temperature is None else fit_smooth(pairs, center, temperature, l2)
        score = ordered_scores(x[cut:end], w, reference[cut:end], item[cut:end], counts[inner:early])
        value = float(ndcg_per_feed(target[cut:end], score, counts[inner:early])[:,0].mean())
        trials.append({'temperature': temperature, 'l2': l2, 'early_holdout_ndcg': value})
        print(json.dumps(trials[-1]), flush=True)
    best = max(trials, key=lambda trial: trial['early_holdout_ndcg'])
    coefficients = np.array(list(initial['coefficients'].values()))
    if best['temperature'] is not None:
        pairs = smooth_rank_data(x[:end], target[:end], counts[:early], reference[:end])
        coefficients = fit_smooth(pairs, coefficients, best['temperature'], best['l2'])
    prediction = ordered_scores(x, coefficients, reference, item, counts)
    report = {**initial, 'objective': 'smooth_ndcg', 'initial_model': str(args.initial),
              'refinement': best, 'refinement_trials': trials,
              'coefficients': dict(zip(names[:len(coefficients)], coefficients.tolist(), strict=True))}
    save_result(args.out, meta, reference, prediction, early, report)


def save_result(out, meta, reference, prediction, early, report):
    counts, y = meta['counts'], meta['target']
    old = ndcg_per_feed(y, reference.astype(np.float32), counts)
    scores = ndcg_per_feed(y, prediction, counts)
    delta = scores[:, 0] - old[:, 0]
    late = delta[early:]
    rng = np.random.default_rng(2026)
    samples = np.concatenate([late[rng.integers(len(late), size=(100,len(late)))].mean(axis=1) for _ in range(20)])
    report['metrics'] = {name: {'ndcg': float(scores[sl,0].mean()), 'ndcg_at_10': float(scores[sl,1].mean()),
                               'ndcg_at_30': float(scores[sl,2].mean()), 'gain': float(delta[sl].mean())}
                         for name, sl in (('early', slice(None,early)), ('late',slice(early,None)), ('all', slice(None)))}
    report['late_gain_by_size'] = {f'{lo}-{hi}': float(late[(counts[early:]>=lo)&(counts[early:]<=hi)].mean())
                                   for lo,hi in ((5,30),(31,100),(101,300))}
    report['paired_feed_95_interval'] = np.quantile(samples,[.025,.975]).tolist()
    out.mkdir(parents=True, exist_ok=False)
    prediction_frame(meta, prediction).write_parquet(out / 'val_predictions.parquet')
    (out / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def select(args):
    meta, reference, matrix, names = load_inputs(args.cache, args.scores, 'val')
    early = temporal_boundary(meta['timestamp'], .4)
    inner = temporal_boundary(meta['timestamp'][:early], .65)
    counts, y, item = meta['counts'], meta['target'], meta['item']
    cut, end = slices(counts)[[inner, early]]
    old = ndcg_per_feed(y[cut:end], reference[cut:end].astype(np.float32), counts[inner:early])
    inner_pairs = (make_pairs(matrix[:cut], y[:cut], counts[:inner], reference[:cut], args.negative_sampling)
                   if args.pairwise else None)
    trials = []
    for width in (len(args.scores), len(args.scores)+len(SIGNALS), len(names)):
        x = np.ascontiguousarray(matrix[:, :width])
        pair_x = np.ascontiguousarray(inner_pairs[0][:, :width]) if inner_pairs is not None else None
        for penalty in (.012, .04, .15):
            coeff = (fit_pairwise(pair_x, inner_pairs[1], penalty) if inner_pairs is not None
                     else fit(x[:cut], y[:cut], counts[:inner], penalty))
            pred = ordered_scores(x[cut:end], coeff, reference[cut:end], item[cut:end], counts[inner:early])
            metric = ndcg_per_feed(y[cut:end], pred, counts[inner:early])[:, 0].mean()
            trials.append({'width': width, 'penalty': penalty, 'early_holdout_ndcg': float(metric)})
            print(json.dumps(trials[-1]), flush=True)
    chosen = max(trials, key=lambda config: config['early_holdout_ndcg'])
    x = np.ascontiguousarray(matrix[:, :chosen['width']])
    size_weights = None
    if args.size_calibration:
        if inner_pairs is not None:
            masked_pairs = np.ascontiguousarray(inner_pairs[0][:, :chosen['width']])
            inner_coeff = fit_pairwise(masked_pairs, inner_pairs[1], chosen['penalty'])
        else:
            inner_coeff = fit(x[:cut], y[:cut], counts[:inner], chosen['penalty'])
        tune = ordered_scores(x[cut:end], inner_coeff, reference[cut:end], item[cut:end], counts[inner:early])
        size_weights = pick_size_weights(tune, reference[cut:end], y[cut:end], item[cut:end], counts[inner:early])
    if args.pairwise:
        pairs, mass = make_pairs(x[:end], y[:end], counts[:early], reference[:end], args.negative_sampling)
        coeff = fit_pairwise(pairs, mass, chosen['penalty'])
    else:
        coeff = fit(x[:end], y[:end], counts[:early], chosen['penalty'])
    prediction = ordered_scores(x, coeff, reference, item, counts)
    if size_weights is not None:
        prediction = mix_by_size(prediction, reference, counts, item, size_weights)
    report = {'chosen': chosen, 'inner_train_feeds': inner, 'inner_holdout_feeds': early-inner,
              'objective': 'pairwise' if args.pairwise else 'listwise',
              'negative_sampling': args.negative_sampling if args.pairwise else None,
              'size_weights': size_weights,
              'early_feeds': early, 'late_feeds': len(counts)-early,
              'early_holdout_reference': float(old[:,0].mean()), 'val_sources': [str(p) for p in args.scores],
              'coefficients': dict(zip(names[:len(coeff)], coeff.tolist(), strict=True)), 'trials': trials}
    save_result(args.out, meta, reference, prediction, early, report)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    report = json.loads(args.model.read_text())
    meta, reference, x, names = load_inputs(args.cache, args.scores, 'test')
    weights = report['coefficients']
    if len(args.scores) != len(report['val_sources']):
        raise ValueError('the test must have the same number of model sources as validation')
    if list(weights) != names[:len(weights)]:
        raise ValueError('calibration weights do not match the test feature order')
    pred = ordered_scores(x[:, :len(weights)], np.array(list(weights.values())), reference,
                          meta['item'], meta['counts'])
    if report.get('size_weights') is not None:
        pred = mix_by_size(pred, reference, meta['counts'], meta['item'], report['size_weights'])
    prediction_frame(meta, pred).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def combine(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    config = json.loads(args.model.read_text())
    with np.load(args.cache / 'test.npz') as f:
        meta = {name: f[name] for name in ('feed', 'item', 'counts')}
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    scores = [read_scores(path, keys) for path in args.scores]
    weights = np.asarray(config['size_weights'], dtype=np.float64)
    if (weights.shape != (len(config['size_boundaries'])+1, len(scores)) or not np.isfinite(weights).all()
            or np.any(weights < 0) or not np.allclose(weights.sum(axis=1), 1)):
        raise ValueError('each feed-size bucket needs one normalized weight per model')
    bins = np.repeat(np.searchsorted(config['size_boundaries'], meta['counts'], side='left'), meta['counts'])
    score = sum(v * weights[bins,i] for i,v in enumerate(scores)).astype(np.float32)
    ordered = resolve_ties(score, scores[0], meta['item'], meta['counts'])
    prediction_frame(meta, ordered).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    choose = sub.add_parser('select')
    choose.add_argument('cache', type=Path)
    choose.add_argument('out', type=Path)
    choose.add_argument('--pairwise', action='store_true')
    choose.add_argument('--negative-sampling', choices=('stratified', 'top'), default='stratified')
    choose.add_argument('--size-calibration', action='store_true')
    improve = sub.add_parser('refine')
    improve.add_argument('cache', type=Path)
    improve.add_argument('initial', type=Path)
    improve.add_argument('out', type=Path)
    inf = sub.add_parser('predict')
    inf.add_argument('cache', type=Path)
    inf.add_argument('model', type=Path)
    inf.add_argument('output', type=Path)
    mixture = sub.add_parser('combine')
    mixture.add_argument('cache', type=Path)
    mixture.add_argument('model', type=Path)
    mixture.add_argument('output', type=Path)
    mixture.add_argument('scores', type=Path, nargs=3, help='Reference, listwise correction, pairwise correction')
    for command in (choose, inf):
        command.add_argument('scores', type=Path, nargs='+', help='Reference, tree ensemble, neural model and optional extra models')
    args = parser.parse_args()
    if args.action == 'select':
        select(args)
    elif args.action == 'refine':
        refine(args)
    elif args.action == 'combine':
        combine(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
