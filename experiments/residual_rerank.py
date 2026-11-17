"""Learn nonlinear corrections on early randomized feeds using forward validation."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits
import torch
from torch import nn

from feed_width import metric_summary, paired_interval
from neural_blend import read_scores
from ranking import KEYS, NAMES, audit_submission, ndcg_per_feed, prediction_frame, temporal_boundary
from score_calibration import design, fit, slices
from sequence_ranker import scores_for_blend, selection_key, sha256, top10_hits, write_json
from set_ranker import feed_loss
from top_rank_fusion import combine, held_out_calibration


ARMS = ('linear', 'nonlinear')
DIFFERENCE = 'nonlinear_delta'
FAMILIES = (*ARMS, DIFFERENCE)
WEIGHTS = (0., .025, .05, .10, .20, .35, .50)
LINEAR_PENALTY = .03
WIDTH = 24
EPOCHS = 12
BATCH_SIZE = 96
SEED = 2026
TREE_PARAMS = {'learning_rate': .06, 'max_iter': 64, 'max_leaf_nodes': 11, 'max_depth': 4,
               'min_samples_leaf': 120, 'l2_regularization': 5., 'max_bins': 127,
               'early_stopping': False, 'random_state': SEED}
SOURCES = (
    ('nn-final-blend-024', 'val_predictions_resolved.parquet', 'submission-resolved.parquet'),
    ('lgb-ensemble-weighted-intent-048', 'val_predictions.parquet', 'submission.parquet'),
    ('nn-vectors-024', 'val_predictions.parquet', 'test_predictions.parquet'),
    ('nn-nearest-024', 'val_predictions.parquet', 'test_predictions.parquet'),
    ('nn-vectors-048', 'val_predictions.parquet', 'test_predictions.parquet'),
)
ANCHOR = ('nn-calibration-joint-024', 'val_predictions.parquet', 'submission-reserve.parquet')


def forward_windows(timestamps):
    early = temporal_boundary(timestamps, .4)
    middle = temporal_boundary(timestamps[:early], .65)
    start = temporal_boundary(timestamps[:middle], .5)
    return ((start, middle), (middle, early)), early


def confidence(z, counts, strongest_intent):
    offsets = slices(counts)
    votes = np.zeros(len(z), dtype=np.int8)
    context = np.empty((len(counts), 3), dtype=np.float32)
    model_count = z.shape[1]
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        size = b - a
        width = min(10, size)
        top = np.argpartition(-z[a:b], width - 1, axis=0)[:width]
        votes[a:b] = np.bincount(top.ravel(), minlength=size)
        agree = (votes[a:b] * (votes[a:b] - 1)).sum() / (width * model_count * (model_count - 1))
        best_two = np.sort(z[a:b], axis=0)[-2:]
        gap = (best_two[-1] - best_two[0]).mean()
        peak = np.nan_to_num(strongest_intent[a:b], nan=0).max()
        context[i] = (agree, gap, peak)
    return [z.std(axis=1), votes / model_count, z.max(axis=1), np.log(np.repeat(counts, counts) / 60),
            *np.repeat(context, counts, axis=0).T]


def load(experiments, split):
    cache = experiments / 'cache-intent-024'
    with np.load(cache / f'{split}.npz') as f:
        fields = ('feed', 'counts', 'item', 'timestamp') + (('target',) if split == 'val' else ())
        meta = {key: f[key] for key in fields}
        features = f['x']
    if np.any(meta['counts'] < 2) or np.any(np.diff(meta['timestamp']) < 0):
        raise ValueError('invalid feed sizes or temporal ordering')
    keys = prediction_frame(meta, np.zeros(len(meta['item']), dtype=np.float32)).select(KEYS)
    slot = 1 if split == 'val' else 2
    source_paths = [experiments / source[0] / source[slot] for source in SOURCES]
    raw = [read_scores(path, keys) for path in source_paths]
    x, names = design(raw, features, meta['counts'])
    extra = confidence(x[:, 1:5], meta['counts'], features[:, NAMES.index('intent_max_16')])
    x = np.column_stack((x, *extra)).astype(np.float32)
    names.extend(('model_disagreement', 'top10_vote_fraction', 'best_model_z', 'log_feed_size',
                  'feed_top10_model_overlap', 'feed_mean_top_score_gap', 'feed_peak_intent'))
    anchor_path = experiments / ANCHOR[0] / ANCHOR[slot]
    return meta, x, names, raw[0], read_scores(anchor_path, keys), source_paths, anchor_path


class ResidualRanker(nn.Module):
    def __init__(self, coefficient):
        super().__init__()
        self.register_buffer('coefficient', torch.as_tensor(coefficient, dtype=torch.float32))
        self.residual = nn.Sequential(nn.Linear(len(coefficient), WIDTH), nn.SiLU(), nn.Dropout(.15),
                                      nn.Linear(WIDTH, 1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    def delta(self, x):
        return torch.tanh(self.residual(x).squeeze(-1))

    def forward(self, x):
        return (x @ self.coefficient) + self.delta(x)


def packed_rows(counts, rng):
    ends = slices(counts)
    groups = []
    bins = np.searchsorted([16, 32, 64, 96, 128, 192, 256], counts)
    order = rng.permutation(len(counts))
    for b in np.unique(bins):
        feeds = order[bins[order] == b]
        groups.extend(feeds[i:i + BATCH_SIZE] for i in range(0, len(feeds), BATCH_SIZE))
    rng.shuffle(groups)
    for feeds in groups:
        size = counts[feeds]
        offset = np.arange(size.max())
        valid = offset < size[:, None]
        yield ends[feeds, None] + np.minimum(offset, size[:, None] - 1), valid


def train_models(x, y, counts, trees=False):
    start = time.perf_counter()
    coefficient = fit(np.ascontiguousarray(x, dtype=np.float64), y, counts, LINEAR_PENALTY)
    if trees:
        z = np.column_stack((x, x @ coefficient)).astype(np.float32)
        positive = np.add.reduceat(y, slices(counts)[:-1])
        negative = counts - positive
        weight = 1 / np.repeat(np.maximum(positive, 1), counts)
        weight = np.where(y == 1, weight, 1 / np.repeat(np.maximum(negative, 1), counts))
        weight /= weight.mean()
        tree = HistGradientBoostingClassifier(**TREE_PARAMS, monotonic_cst=[0]*x.shape[1] + [1])
        with threadpool_limits(limits=4, user_api='openmp'):
            tree.fit(z, y, sample_weight=weight)
        print(f'linear and tree fit on {len(counts):,} feeds, {len(y):,} rows, '
              f'{time.perf_counter() - start:.1f}s', flush=True)
        return coefficient, tree
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = ResidualRanker(coefficient)
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.03)
    for _ in range(EPOCHS):
        for rows, valid in packed_rows(counts, rng):
            features, label, mask = (torch.from_numpy(array) for array in (x[rows], y[rows], valid))
            delta = model.delta(features)
            scores = features @ model.coefficient + delta
            loss = feed_loss(scores, label, mask, pointwise_weight=0) + .04 * delta[mask].square().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
    print(f'linear and nonlinear fit on {len(counts):,} feeds, {len(y):,} rows, '
          f'{time.perf_counter() - start:.1f}s', flush=True)
    return model


@torch.inference_mode()
def predictions(model, x):
    if isinstance(model, tuple):
        coefficient, tree = model
        linear = x @ coefficient
        with threadpool_limits(limits=4, user_api='openmp'):
            nonlinear = tree.decision_function(np.column_stack((x, linear)).astype(np.float32))
        return {'linear': linear.astype(np.float32), 'nonlinear': nonlinear.astype(np.float32)}
    model.eval()
    linear, nonlinear = [], []
    for i in range(0, len(x), 65_536):
        block = torch.from_numpy(x[i:i + 65_536])
        base = block @ model.coefficient
        linear.append(base.numpy())
        nonlinear.append((base + model.delta(block)).numpy())
    return {'linear': np.concatenate(linear), 'nonlinear': np.concatenate(nonlinear)}


def forward_anchor(experiments, meta, reference, windows):
    counts = meta['counts']
    cuts = slices(counts)
    start, end = cuts[[windows[0][0], windows[-1][1]]]
    prior = [held_out_calibration(experiments / name / 'metrics.json', experiments / 'cache-intent-024', windows)
             for name in ('nn-calibration-extra-024', 'nn-calibration-stratified-024')]
    weights = json.loads((experiments / ANCHOR[0] / 'metrics.json').read_text(encoding='utf-8'))['size_weights']
    return combine([reference[start:end], *prior], counts[windows[0][0]:windows[-1][1]],
                   meta['item'][start:end], weights, None)


def forward_predictions(experiments, meta, x, reference, windows, trees):
    counts = meta['counts']
    cuts = slices(counts)
    anchor = forward_anchor(experiments, meta, reference, windows)
    parts = []
    for a, b in windows:
        first, last = cuts[[a, b]]
        model = train_models(x[:first], meta['target'][:first], counts[:a], trees)
        parts.append(predictions(model, x[first:last]))
    return {'submitted': anchor, **{name: np.concatenate([part[name] for part in parts]) for name in ARMS}}


def choose(meta, raw, target, first):
    counts = meta['counts']
    base = ndcg_per_feed(target, raw['submitted'].astype(np.float32), counts)[:, 0]
    selected, grid = {}, {}
    for family in FAMILIES:
        trials = []
        for weight in WEIGHTS:
            score = scores_for_blend(raw, meta, family, weight, ARMS, DIFFERENCE)
            delta = ndcg_per_feed(target, score, counts)[:, 0] - base
            trials.append({'name': family, 'weight': weight, 'early_gain': float(delta.mean()),
                           'early_gain_by_block': [float(delta[:first].mean()), float(delta[first:].mean())]})
        grid[family] = trials
        eligible = [entry for entry in trials if min(entry['early_gain_by_block']) > 0]
        selected[family] = max(eligible, key=selection_key, default=trials[0])
    return {'choice': max(selected.values(), key=selection_key), 'families': selected, 'grid': grid,
            'oof_anchor_ndcg': float(base.mean())}


def summarize(meta, scores, early):
    counts, target = meta['counts'], meta['target']
    middle = early + temporal_boundary(meta['timestamp'][early:], .5)
    blocks = {'late_first': (early, middle), 'late_second': (middle, len(counts)),
              'late': (early, len(counts))}
    long = (counts[early:] >= 101) & (counts[early:] <= 300)

    def metrics(ndcg, hits):
        return {**{name: metric_summary(ndcg, hits, *interval) for name, interval in blocks.items()},
                'late_long_101_300': metric_summary(ndcg[early:][long], hits[early:][long], 0, int(long.sum()))}

    evaluated = {name: (ndcg_per_feed(target, values, counts), top10_hits(target, values, counts))
                 for name, values in scores.items()}
    comparisons = [(name, 'submitted') for name in scores if name != 'submitted'] + [('nonlinear', 'linear')]
    contrasts = {}
    for name, baseline in comparisons:
        delta, hits = (a - b for a, b in zip(evaluated[name], evaluated[baseline], strict=True))
        contrasts[f'{name}_vs_{baseline}'] = {'gain': metrics(delta, hits),
            'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0])}
    return {name: metrics(*values) for name, values in evaluated.items()}, contrasts


def select(args):
    args.out.mkdir(parents=True, exist_ok=False)
    meta, x, names, reference, anchor, paths, anchor_path = load(args.experiments, 'val')
    windows, early = forward_windows(meta['timestamp'])
    start, middle = windows[0]
    counts = meta['counts']
    cuts = slices(counts)
    a, b = cuts[[start, early]]
    print(f'forward windows {windows}, next {len(counts) - early:,} feeds untouched', flush=True)
    oof = forward_predictions(args.experiments, meta, x, reference, windows, args.trees)
    choice = choose({'item': meta['item'][a:b], 'counts': counts[start:early]}, oof,
                    meta['target'][a:b], middle - start)
    write_json(args.out / 'early_selection.json', {**choice, 'windows': windows})
    print(json.dumps(choice['choice']), flush=True)

    model = train_models(x[:b], meta['target'][:b], counts[:early], args.trees)
    raw = {'submitted': anchor, **predictions(model, x)}
    scores = {name: score.astype(np.float32) for name, score in raw.items()}
    for name, cfg in choice['families'].items():
        scores[name + '_blend'] = scores_for_blend(raw, meta, name, cfg['weight'], ARMS, DIFFERENCE)
    metrics, contrasts = summarize(meta, scores, early)
    winner = choice['choice']
    result = contrasts[winner['name'] + '_blend_vs_submitted']
    gains = result['gain']
    halves = [gains[name] for name in ('late_first', 'late_second')]
    stable = winner['weight'] > 0 and min(g['ndcg'] for g in halves) > 0 and (
        gains['late']['ndcg'] >= .003 or
        (gains['late']['ndcg'] >= .001 and result['late_paired_ndcg_95_interval'][0] > 0
         and all(g['ndcg_at_10'] > 0 and g['top10_positives'] > 0 for g in halves)))
    for name, score in scores.items():
        if name != 'submitted':
            prediction_frame(meta, score).write_parquet(args.out / f'{name}_val_predictions.parquet')
    model_file = 'model.joblib' if args.trees else 'model.pt'
    if args.trees:
        joblib.dump({'model': model, 'features': names}, args.out / model_file)
    else:
        torch.save({'model': model.state_dict(), 'features': names}, args.out / model_file)
    test_paths = [args.experiments / directory / test_file for directory, _, test_file in (*SOURCES, ANCHOR)]
    write_json(args.out / 'metrics.json', {'choice': winner, 'prepare_submission': stable, 'windows': windows,
               'early_feeds': early, 'early_rows': int(b), 'late_feeds': len(counts) - early,
               'metrics': metrics, 'contrasts': contrasts, 'parameters': {'seed': SEED, 'epochs': EPOCHS,
               'width': WIDTH, 'linear_penalty': LINEAR_PENALTY, 'batch_size': BATCH_SIZE},
               'val_sources': [str(p) for p in paths], 'anchor': str(anchor_path),
               'val_source_sha256': {str(p): sha256(p) for p in [*paths, anchor_path]},
               'test_source_sha256': {str(p): sha256(p) for p in test_paths},
               'tree_parameters': TREE_PARAMS if args.trees else None, 'model_file': model_file,
               'model_sha256': sha256(args.out / model_file)})
    print(json.dumps({'late': {name: block['late'] for name, block in metrics.items()},
                      'prepare_submission': stable}), flush=True)


def submit(args):
    report = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
    if not report['prepare_submission'] or args.output.exists():
        raise ValueError('submission requires a qualifying assessment and a new output path')
    path = args.run / report['model_file']
    if sha256(path) != report['model_sha256']:
        raise ValueError('model differs from the assessed checkpoint')
    supplied = {str(args.experiments / directory / name): sha256(args.experiments / directory / name)
                for directory, _, name in (*SOURCES, ANCHOR)}
    if supplied != report['test_source_sha256']:
        raise ValueError('test predictions differ from the designated base models and anchor')
    meta, x, names, _, anchor, _, _ = load(args.experiments, 'test')
    trees = report['tree_parameters'] is not None
    checkpoint = joblib.load(path) if trees else torch.load(path, weights_only=True, map_location='cpu')
    if names != checkpoint['features']:
        raise ValueError('test features do not match the checkpoint')
    if trees:
        model = checkpoint['model']
    else:
        model = ResidualRanker(np.zeros(len(names), dtype=np.float32))
        model.load_state_dict(checkpoint['model'])
    choice = report['choice']
    raw = {'submitted': anchor, **predictions(model, x)}
    scores = scores_for_blend(raw, meta, choice['name'], choice['weight'], ARMS, DIFFERENCE)
    with NamedTemporaryFile(dir=args.output.parent, suffix='.parquet', delete=False) as file:
        temp = Path(file.name)
    try:
        prediction_frame(meta, scores).write_parquet(temp)
        audit = audit_submission(temp)
        if audit['rows'] != 2_904_806 or audit['feeds_with_ties']:
            raise ValueError(f'submission failed audit: {audit}')
        temp.rename(args.output)
    finally:
        temp.unlink(missing_ok=True)
    write_json(args.output.with_suffix('.audit.json'), audit)
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    training = commands.add_parser('select')
    training.add_argument('experiments', type=Path)
    training.add_argument('out', type=Path)
    training.add_argument('--trees', action='store_true')
    training.set_defaults(run_action=select)
    prediction = commands.add_parser('submit')
    prediction.add_argument('experiments', type=Path)
    prediction.add_argument('run', type=Path)
    prediction.add_argument('output', type=Path)
    prediction.set_defaults(run_action=submit)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.set_num_interop_threads(1)
    args.run_action(args)


if __name__ == '__main__':
    main()
