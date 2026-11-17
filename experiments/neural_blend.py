"""Select neural ensemble weights on early val and report performance on late val."""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import polars as pl

from ranking import (KEYS, audit_submission, ndcg_per_feed, prediction_frame,
                     rank_fraction, standardized_scores, temporal_boundary)


def read_meta(cache):
    with np.load(cache) as f:
        return {k: f[k] for k in ('feed', 'item', 'counts', 'target', 'timestamp')}


def read_scores(path, keys):
    values = pl.read_parquet(path)
    if not values.select(KEYS).equals(keys):
        raise ValueError(f'candidate keys differ from the cache: {path}')
    score = values['score'].to_numpy().astype(np.float64)
    if not np.isfinite(score).all():
        raise ValueError(f'scores must be finite: {path}')
    return score


def normalize_scores(score, counts, method):
    if method == 'zscore':
        return standardized_scores(score, counts)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    return np.concatenate([rank_fraction(score[start:end]) for start, end in zip(offsets[:-1], offsets[1:])])


def score_mix(arrays, weights):
    return sum(a * w for a, w in zip(arrays, weights, strict=True)).astype(np.float32)


def mix_from_config(arrays, counts, config):
    if 'size_weights' not in config:
        return score_mix(arrays, config['weights'])
    bins = np.repeat(np.searchsorted(config['size_boundaries'], counts, side='left'), counts)
    weights = np.asarray(config['size_weights'], dtype=np.float64)
    return sum(a * weights[bins, i] for i, a in enumerate(arrays)).astype(np.float32)


def resolve_ties(primary, fallback, item, counts):
    """Keep the strict primary order, using the fallback score only inside ties."""
    out = np.empty_like(primary, dtype=np.float32)
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    for first, last in zip(offsets[:-1], offsets[1:]):
        order = np.lexsort((item[first:last], -fallback[first:last], -primary[first:last]))
        out[first + order] = 1 - np.arange(last-first, dtype=np.float32) / (last-first)
    return out


def select(args):
    args.out.mkdir(parents=True, exist_ok=False)
    meta = read_meta(args.cache / 'val.npz')
    n = len(meta['item'])
    keys = prediction_frame(meta, np.zeros(n)).select(KEYS)
    groups = meta['counts']
    boundary = temporal_boundary(meta['timestamp'], .4)
    row_cut = int(groups[:boundary].sum(dtype=np.int64))
    models = [*args.baselines, *args.neural]
    raw = {m: read_scores(m / 'val_predictions.parquet', keys) for m in models}
    reference = (read_scores(args.reference_scores, keys) if args.reference_scores is not None
                 else raw[args.baselines[0]])
    if args.reference_test_scores is not None:
        if args.reference_scores is None or not args.reference_test_scores.is_file():
            raise ValueError('overriding the first anchor requires matching existing val and test files')
        raw[args.baselines[0]] = reference
    labels = meta['target']
    original = ndcg_per_feed(labels, reference.astype(np.float32), groups)[:, 0]

    outcomes = []
    for method in ('zscore', 'percentile'):
        normal = {model: normalize_scores(score, groups, method) for model, score in raw.items()}
        for baseline in args.baselines:
            best = (-np.inf, [baseline], [1.0])
            options = [([baseline], [1.0])]
            for size in range(1, len(args.neural) + 1):
                for combo in itertools.combinations(args.neural, size):
                    for alpha in (.15, .30, .45, .60, .75, 1.0):
                        options.append(([baseline, *combo], [1-alpha] + [alpha/size]*size))
            for candidates, weights in options:
                early = score_mix([normal[p][:row_cut] for p in candidates], weights)
                value = ndcg_per_feed(labels[:row_cut], early, groups[:boundary])[:, 0].mean()
                if value > best[0]:
                    best = (value, candidates, weights)
            _, candidates, weights = best
            scores = score_mix([normal[p] for p in candidates], weights)
            per_feed = ndcg_per_feed(labels, scores, groups)[:, 0]
            fixed_sources = [str(args.reference_test_scores.resolve())
                             if p == args.baselines[0] and args.reference_test_scores else None for p in candidates]
            outcomes.append(({'anchor': baseline.name, 'normalization': method, 'test_score_sources': fixed_sources,
                              'models': [str(p) for p in candidates], 'weights': weights,
                              'early': float(per_feed[:boundary].mean()), 'late': float(per_feed[boundary:].mean()),
                              'all': float(per_feed.mean())}, scores, per_feed))

    result, score, metric = max(outcomes, key=lambda record: record[0]['early'])
    result = {**result, 'compared_to': args.baselines[0].name,
              'reference_scores': str(args.reference_scores) if args.reference_scores is not None else None,
              'compared': [r for r, _, _ in outcomes]}
    save_result(args.out, result, score, metric, original, meta, boundary)


def save_result(out, config, score, metric, original, meta, boundary):
    groups = meta['counts']
    delta = metric - original
    d = delta[boundary:]
    rng = np.random.default_rng(2026)
    samples = np.concatenate([d[rng.integers(len(d), size=(100,len(d)))].mean(axis=1) for _ in range(20)])
    result = {**config, 'early': float(metric[:boundary].mean()), 'late': float(metric[boundary:].mean()),
              'all': float(metric.mean()),
              'early_gain': float(delta[:boundary].mean()), 'late_gain': float(d.mean()),
              'full_gain': float(delta.mean()), 'paired_feed_95_interval': np.quantile(samples,[.025,.975]).tolist(),
              'late_gain_by_size': {f'{lo}-{hi}': float(d[(groups[boundary:] >= lo) & (groups[boundary:] <= hi)].mean())
                                    for lo, hi in ((5,30), (31,100), (101,300))}}
    prediction_frame(meta, score).write_parquet(out / 'val_predictions.parquet')
    (out / 'comparison.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)


def adapt(args):
    config = json.loads(args.config.read_text())
    if not config['reference_scores']:
        raise ValueError('size adaptation needs the submitted comparator via --reference-scores')
    meta = read_meta(args.cache / 'val.npz')
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    groups, labels = meta['counts'], meta['target']
    boundary = temporal_boundary(meta['timestamp'], .4)
    row_cut = int(groups[:boundary].sum(dtype=np.int64))
    paths = [Path(p) / 'val_predictions.parquet' for p in config['models']]
    for i, source in enumerate(config.get('test_score_sources', [None]*len(paths))):
        if source is not None:
            paths[i] = Path(config['reference_scores'])
    arrays = [normalize_scores(read_scores(p, keys), groups, config['normalization']) for p in paths]
    base = np.asarray(config['weights'], dtype=np.float64)
    if len(base) < 2 or base[1:].sum() <= 0:
        raise ValueError('size adaptation needs a selected neural component')
    neural_weights = base[1:] / base[1:].sum()
    alpha = 1 - base[0]
    options = [alpha, max(alpha-.15, 0), min(alpha+.15, 1)]
    buckets = np.searchsorted([30, 100], groups[:boundary], side='left')
    for_bin = []
    for i in range(3):
        mask = buckets == i
        if not mask.any():
            for_bin.append(base.tolist())
            continue
        row_mask = np.repeat(mask, groups[:boundary])
        scores = [x[:row_cut][row_mask] for x in arrays]
        y, sizes = labels[:row_cut][row_mask], groups[:boundary][mask]
        grid = [[1-w, *(w*neural_weights)] for w in options]
        best = max(grid, key=lambda w: ndcg_per_feed(y, score_mix(scores, w), sizes)[:, 0].mean())
        for_bin.append(best)
    config = {k: v for k, v in config.items() if k not in ('compared',)}
    config.update(size_boundaries=[30, 100], size_weights=for_bin)
    score = mix_from_config(arrays, groups, config)
    metrics = ndcg_per_feed(labels, score, groups)[:, 0]
    reference = read_scores(Path(config['reference_scores']), keys).astype(np.float32)
    original = ndcg_per_feed(labels, reference, groups)[:, 0]
    args.out.mkdir(parents=True, exist_ok=False)
    save_result(args.out, config, score, metrics, original, meta, boundary)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    selected = json.loads(args.config.read_text())
    if len(args.scores) != len(selected['models']):
        raise ValueError('supply scores in the order of models listed in the comparison file')
    for path, expected in zip(args.scores, selected.get('test_score_sources', [None]*len(args.scores)), strict=True):
        if expected is not None and path.resolve() != Path(expected):
            raise ValueError(f'validation used a different test anchor: expected {expected}')
    meta = read_meta(args.cache / 'test.npz')
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    raw = [read_scores(path, keys) for path in args.scores]
    arrays = [normalize_scores(x, meta['counts'], selected['normalization']) for x in raw]
    score = mix_from_config(arrays, meta['counts'], selected)
    if args.resolve_ties:
        fallback = mix_from_config([standardized_scores(x, meta['counts']) for x in raw], meta['counts'], selected)
        score = resolve_ties(score, fallback, meta['item'], meta['counts'])
    prediction_frame(meta, score).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    fit = subs.add_parser('select')
    fit.add_argument('cache', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('--baselines', nargs='+', type=Path, required=True)
    fit.add_argument('--neural', nargs='+', type=Path, required=True)
    fit.add_argument('--reference-scores', type=Path)
    fit.add_argument('--reference-test-scores', type=Path)
    sized = subs.add_parser('adapt')
    sized.add_argument('cache', type=Path)
    sized.add_argument('config', type=Path)
    sized.add_argument('out', type=Path)
    inf = subs.add_parser('predict')
    inf.add_argument('cache', type=Path)
    inf.add_argument('config', type=Path)
    inf.add_argument('output', type=Path)
    inf.add_argument('scores', nargs='+', type=Path)
    inf.add_argument('--resolve-ties', action='store_true')
    args = parser.parse_args()
    if args.action == 'select':
        select(args)
    elif args.action == 'adapt':
        adapt(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
