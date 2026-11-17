"""Reorder the incumbent top 30 on long feeds using forward-fold calibration."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from neural_blend import read_scores
from ranking import KEYS, NAMES, audit_submission, ndcg_per_feed, prediction_frame, temporal_boundary
from score_calibration import fit, slices, within_feed_z
from top_rank_fusion import combine, held_out_calibration


WIDTH = 30
MIN_FEED = 101
SIGNALS = ('f1', 'f2', 'f3', 'f4', 'f5', 'user_cos', 'hist_cos_max', 'intent_max_16',
           'intent_category_recency', 'intent_match_event_rank', 'item_log_exposure', 'item_smoothed_rate')
MODELS = ('lgb-ensemble-weighted-intent-048', 'nn-vectors-024', 'nn-nearest-024', 'nn-vectors-048',
          'nn-calibration-extra-024', 'nn-calibration-stratified-024')


def load_data(experiments, split):
    path = experiments / 'cache-intent-024' / f'{split}.npz'
    with np.load(path) as data:
        meta = {key: data[key] for key in ('feed', 'item', 'counts', 'timestamp', 'target')}
        features = data['x'][:, [NAMES.index(name) for name in SIGNALS]].copy()
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    current = experiments / 'nn-calibration-joint-024' / ('val_predictions.parquet' if split == 'val' else 'submission-reserve.parquet')
    previous = experiments / 'nn-final-blend-024' / ('val_predictions_resolved.parquet' if split == 'val' else 'submission-resolved.parquet')
    raw = []
    for model in MODELS:
        name = 'val_predictions.parquet' if split == 'val' else ('submission.parquet' if model.startswith('lgb') else 'test_predictions.parquet')
        raw.append(read_scores(experiments / model / name, keys))
    return meta, features, read_scores(current, keys).astype(np.float32), read_scores(previous, keys), raw


def select_head(reference, counts):
    starts = slices(counts)
    feeds = np.flatnonzero(counts >= MIN_FEED)
    rows = np.empty((len(feeds), WIDTH), dtype=np.int64)
    for j, i in enumerate(feeds):
        a, b = starts[i:i+2]
        rows[j] = a + np.argsort(-reference[a:b], kind='stable')[:WIDTH]
    return feeds, rows


def design(reference, counts, features, components):
    feeds, rows = select_head(reference, counts)
    groups = np.full(len(feeds), WIDTH)
    base = np.broadcast_to(-np.log(np.arange(1, WIDTH+1)), rows.shape).ravel()
    columns = [within_feed_z(base, groups)]
    names = ['incumbent_log_rank']
    for name, scores in zip(MODELS, components, strict=True):
        columns.append(within_feed_z(scores[rows].ravel(), groups))
        names.append(name)
    norm = np.column_stack(columns[1:]).reshape(len(feeds), WIDTH, len(components))
    ids = [np.argmax(scores[rows], axis=1) for scores in components]
    vote = sum(np.arange(WIDTH) == j[:, None] for j in ids)
    columns.extend((within_feed_z(norm.std(axis=-1).ravel(), groups), within_feed_z(vote.ravel(), groups)))
    names.extend(('model_disagreement', 'head_first_votes'))
    score_width = len(names)
    for i, name in enumerate(SIGNALS):
        columns.append(within_feed_z(features[rows, i].ravel(), groups))
        names.append(name)
    return feeds, rows, np.column_stack(columns).reshape(len(feeds), WIDTH, -1), names, score_width


def order_within_head(matrix, coeff, weight, locked=0, movement=0):
    if weight == 0:
        return np.broadcast_to(np.arange(WIDTH), matrix.shape[:2])
    groups = np.full(len(matrix), WIDTH)
    base = matrix[:, :, 0]
    calibrated = within_feed_z((matrix @ coeff).ravel(), groups).reshape(base.shape)
    score = (1-weight)*base + weight*calibrated
    if movement:
        size = WIDTH-locked
        rank = np.arange(size)
        discount = 1 / np.log2(rank + locked + 2)
        allowed = np.abs(rank[:, None] - rank) <= movement
        result = np.tile(np.arange(WIDTH), (len(matrix),1))
        for row, values in enumerate(score[:,locked:]):
            cost = np.where(allowed, -values[:, None]*discount, np.inf)
            source, destination = linear_sum_assignment(cost)
            result[row, destination + locked] = source + locked
        return result
    return np.argsort(-score, axis=1, kind='stable')


def rewrite_head(reference, rows, order):
    """Reassign only existing top-30 scores, preserving every score below the head."""
    score = reference.copy()
    score[np.take_along_axis(rows, order, axis=1)] = reference[rows]
    return score


def fit_head(matrix, labels, penalty):
    # A feed with no positive in the head has no pair to learn from for this task.
    keep = labels.sum(axis=1) > 0
    x = matrix[keep].reshape(-1, matrix.shape[-1])
    y = labels[keep].ravel()
    return fit(x, y, np.full(int(keep.sum()), WIDTH), penalty)


def head_ndcg_gain(labels, total_pos, order):
    if np.any(total_pos < 1):
        raise ValueError('expected positives in each full feed')
    discounts = 1 / np.log2(np.arange(max(WIDTH, int(total_pos.max()))) + 2)
    ideal = np.cumsum(discounts)[total_pos-1]
    ranked = np.take_along_axis(labels, order, axis=1)
    return ((ranked - labels) @ discounts[:WIDTH]) / ideal


def subsets(meta, start, stop):
    offsets = slices(meta['counts'])
    a, b = offsets[[start, stop]]
    return a, b, meta['counts'][start:stop]


def forward_examples(args, meta, signals, previous, sources, early):
    counts, time = meta['counts'], meta['timestamp']
    cuts = [temporal_boundary(time[:early], r) for r in (.24, .44, .65, .82)] + [early]
    folds = list(zip(cuts[:-1], cuts[1:]))
    config_paths = [args.experiments / model / 'metrics.json' for model in MODELS[-2:]]
    cache = args.experiments / 'cache-intent-024'
    fixed_weights = json.loads((args.experiments / 'nn-calibration-joint-024/metrics.json').read_text())['size_weights']
    a, b, sizes = subsets(meta, cuts[0], early)
    adjusted = [held_out_calibration(path, cache, folds) for path in config_paths]
    incumbents = combine([previous[a:b], *adjusted], sizes, meta['item'][a:b], fixed_weights, None)
    parts = [s[a:b] for s in sources[:-2]] + adjusted
    feeds, rows, matrix, names, limited = design(incumbents, sizes, signals[a:b], parts)
    totals = np.add.reduceat(meta['target'][a:b], slices(sizes)[:-1])[feeds]
    return feeds + cuts[0], matrix, meta['target'][a:b][rows], totals, names, limited, folds


def select(args):
    meta, signals, incumbent, previous, components = load_data(args.experiments, 'val')
    early = temporal_boundary(meta['timestamp'], .4)
    inner = temporal_boundary(meta['timestamp'][:early], .65)
    feeds, x, y, total, names, simple, folds = forward_examples(args, meta, signals, previous, components, early)
    train = feeds < inner
    tune = ~train
    locked, movement = (3, 5) if args.conservative else (0, 0)
    trial_weights = (.10, .25, .5) if args.conservative else (.25, .5, .75, 1.)
    trials = [{'width': simple, 'penalty': .06, 'weight': 0., 'inner_long_gain': 0.,
               'locked': locked, 'movement': movement}]
    for dim in (simple, len(names)):
        for penalty in (.015, .06, .2):
            coeff = fit_head(x[train, :, :dim], y[train], penalty)
            for weight in trial_weights:
                order = order_within_head(x[tune, :, :dim], coeff, weight, locked, movement)
                gain = head_ndcg_gain(y[tune], total[tune], order).mean()
                trials.append({'width': dim, 'penalty': penalty, 'weight': weight, 'inner_long_gain': float(gain),
                               'locked': locked, 'movement': movement})
    selected = max(trials, key=lambda trial: trial['inner_long_gain'])
    dim = selected['width']
    coeff = fit_head(x[:, :, :dim], y, selected['penalty'])
    full_feeds, rows, full_x, full_names, _ = design(incumbent, meta['counts'], signals, components)
    if names != full_names:
        raise ValueError('feature schema changes between fitting and scoring')
    late_mask = full_feeds >= early
    order = order_within_head(full_x[:, :, :dim], coeff, selected['weight'], locked, movement)
    score = rewrite_head(incumbent, rows, order)
    late_rows = slices(meta['counts'])[early]
    scores = ndcg_per_feed(meta['target'][late_rows:], score[late_rows:], meta['counts'][early:])
    base = ndcg_per_feed(meta['target'][late_rows:], incumbent[late_rows:], meta['counts'][early:])
    delta = scores[:, 0] - base[:, 0]
    long = meta['counts'][early:] >= MIN_FEED
    late_ts = meta['timestamp'][early:]
    half = temporal_boundary(late_ts, .5)
    rng = np.random.default_rng(2026)
    samples = np.concatenate([delta[rng.integers(len(delta), size=(100,len(delta)))].mean(axis=1) for _ in range(20)])
    report = {'selected': selected, 'coefficients': dict(zip(names[:dim], coeff.tolist(), strict=True)),
              'early_scores_in_sample': True,
              'forward_folds': folds, 'inner_fit_long_feeds': int(train.sum()), 'inner_tune_long_feeds': int(tune.sum()),
              'fit_long_feeds': len(feeds), 'fit_long_feeds_with_head_positive': int((y.sum(axis=1) > 0).sum()),
              'late_feeds': len(delta), 'late_long_feeds': int(late_mask.sum()),
              'late_ndcg': float(scores[:,0].mean()), 'base_late_ndcg': float(base[:,0].mean()),
              'late_gain': float(delta.mean()), 'late_long_gain': float(delta[long].mean()),
              'late_long_ndcg_at_10': float(scores[long,1].mean()), 'base_late_long_ndcg_at_10': float(base[long,1].mean()),
              'late_gains_by_half': [float(delta[:half].mean()),float(delta[half:].mean())],
              'paired_feed_95_interval': np.quantile(samples,[.025,.975]).tolist(),
              'late_long_feeds_reordered': int((order[late_mask] != np.arange(WIDTH)).any(axis=1).sum()),
              'trials': sorted(trials, key=lambda trial: -trial['inner_long_gain'])}
    args.out.mkdir(parents=True, exist_ok=False)
    prediction_frame(meta, score).write_parquet(args.out / 'val_predictions.parquet')
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({**report, 'trials': report['trials'][:5]}, indent=2), flush=True)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    cfg = json.loads(args.config.read_text())
    meta, signals, old, _, components = load_data(args.experiments, 'test')
    _, rows, x, names, _ = design(old, meta['counts'], signals, components)
    coeff = cfg['coefficients']
    if list(coeff) != names[:len(coeff)]:
        raise ValueError('calibration columns do not match the fitted schema')
    variant = cfg['selected']
    order = order_within_head(x[:, :, :len(coeff)], np.array(list(coeff.values())), variant['weight'],
                              variant.get('locked', 0), variant.get('movement', 0))
    score = rewrite_head(old, rows, order)
    prediction_frame(meta, score).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='mode', required=True)
    fit = sub.add_parser('select')
    fit.add_argument('experiments', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('--conservative', action='store_true')
    inf = sub.add_parser('predict')
    inf.add_argument('experiments', type=Path)
    inf.add_argument('config', type=Path)
    inf.add_argument('output', type=Path)
    args = p.parse_args()
    if args.mode == 'select':
        select(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
