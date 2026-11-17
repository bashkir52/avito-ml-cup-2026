"""Try top-weighted rank fusion using only saved predictions and early val."""
import argparse
import json
from pathlib import Path

import numpy as np

from neural_blend import read_scores, resolve_ties
from ranking import KEYS, audit_submission, ndcg_per_feed, prediction_frame, temporal_boundary
from score_calibration import fit, fit_pairwise, load_inputs, make_pairs, ordered_scores, slices


SIZES = (30, 100)


def size_bucket(counts):
    return np.searchsorted(SIZES, counts, side='left')


def transform(scores, counts, decay):
    if decay is None:
        return scores
    ranks = 1 + np.rint((1 - scores) * np.repeat(counts, counts))
    if np.any((ranks < 1) | (ranks > np.repeat(counts, counts))):
        raise ValueError('input expects rank scores in the format 1 - (rank-1)/feed_size')
    return 1 / (decay + ranks)


def combine(inputs, counts, item, weights, decay):
    bucket = np.repeat(size_bucket(counts), counts)
    values = np.asarray(weights, dtype=np.float64)[bucket]
    score = sum(transform(s, counts, decay) * values[:, i] for i, s in enumerate(inputs)).astype(np.float32)
    return resolve_ties(score, inputs[0], item, counts)


def held_out_calibration(config_path, cache, cuts):
    cfg = json.loads(config_path.read_text())
    meta, ref, design, _ = load_inputs(cache, [Path(p) for p in cfg['val_sources']], 'val')
    counts, y, item = meta['counts'], meta['target'], meta['item']
    offsets = slices(counts)
    chosen = cfg['chosen']
    matrix = np.ascontiguousarray(design[:, :chosen['width']])
    scores = []
    for begin, end in cuts:
        a, b = offsets[[begin, end]]
        if cfg.get('objective') == 'pairwise':
            pairs, weights = make_pairs(matrix[:a], y[:a], counts[:begin], ref[:a], cfg['negative_sampling'])
            coeff = fit_pairwise(pairs, weights, chosen['penalty'])
        else:
            coeff = fit(matrix[:a], y[:a], counts[:begin], chosen['penalty'])
        scores.append(ordered_scores(matrix[a:b], coeff, ref[a:b], item[a:b], counts[begin:end]))
    return np.concatenate(scores)


def metrics(meta, score, boundary):
    values = ndcg_per_feed(meta['target'], score.astype(np.float32), meta['counts'])[:, 0]
    return {'early': float(values[:boundary].mean()), 'late': float(values[boundary:].mean()),
            'all': float(values.mean())}, values


def select(args):
    with np.load(args.cache / 'val.npz') as f:
        meta = {k: f[k] for k in ('feed', 'item', 'counts', 'target', 'timestamp')}
    counts, item, y = meta['counts'], meta['item'], meta['target']
    keys = prediction_frame(meta, np.zeros(len(item))).select(KEYS)
    ref = read_scores(args.reference, keys)
    old_reserve = read_scores(args.reserve, keys)
    early = temporal_boundary(meta['timestamp'], .4)
    middle = temporal_boundary(meta['timestamp'][:early], .65)
    start = temporal_boundary(meta['timestamp'][:middle], .65)
    cuts = [(start, middle), (middle, early)]
    begin, end = slices(counts)[[start, early]]
    calibration = [held_out_calibration(p / 'metrics.json', args.cache, cuts) for p in args.calibrations]
    oof = [ref[begin:end], *calibration]
    full = [ref, *[read_scores(p / 'val_predictions.parquet', keys) for p in args.calibrations]]
    pool_sizes, labels, pool_ids = counts[start:early], y[begin:end], item[begin:end]
    bucket = size_bucket(pool_sizes)
    grid = [[i/4, j/4, (4-i-j)/4] for i in range(4,-1,-1) for j in range(4-i,-1,-1)]
    grid.append([1/3]*3)
    candidates = []
    for decay in (None, 8, 20, 60):
        normalized = [transform(s, pool_sizes, decay) for s in oof]
        weights = []
        oof_total = 0.
        for i in range(3):
            keep = bucket == i
            mask = np.repeat(keep, pool_sizes)
            groups, target = pool_sizes[keep], labels[mask]
            values = [v[mask] for v in normalized]
            best = (-np.inf, grid[0])
            for trial in grid:
                primary = sum(w*s for w,s in zip(trial, values, strict=True)).astype(np.float32)
                pred = resolve_ties(primary, oof[0][mask], pool_ids[mask], groups)
                value = ndcg_per_feed(target, pred, groups)[:, 0].sum()
                if value > best[0]:
                    best = value, trial
            weights.append(best[1])
            oof_total += best[0]
        candidate = {'decay': decay, 'size_weights': weights, 'early_oof_ndcg': float(oof_total / len(pool_sizes))}
        candidate['metrics'], _ = metrics(meta, combine(full, counts, item, weights, decay), early)
        candidates.append(candidate)
    selected = max(candidates, key=lambda c: c['early_oof_ndcg'])
    result = combine(full, counts, item, selected['size_weights'], selected['decay'])
    base, base_values = metrics(meta, ref, early)
    reserve, reserve_values = metrics(meta, old_reserve, early)
    _, values = metrics(meta, result, early)
    delta = values[early:] - base_values[early:]
    rng = np.random.default_rng(2026)
    boot = np.concatenate([delta[rng.integers(len(delta), size=(100,len(delta)))].mean(axis=1) for _ in range(20)])
    report = {'selected': selected, 'trials': candidates, 'inner_folds': cuts, 'compared_to': base, 'reserve': reserve,
              'late_gain': float(delta.mean()), 'late_gain_vs_reserve': float((values[early:]-reserve_values[early:]).mean()),
              'paired_feed_95_interval': np.quantile(boot,[.025,.975]).tolist(),
              'late_gain_by_size': {str(i): float(delta[size_bucket(counts[early:]) == i].mean()) for i in range(3)}}
    args.out.mkdir(parents=True, exist_ok=False)
    prediction_frame(meta, result).write_parquet(args.out / 'val_predictions.parquet')
    (args.out / 'metrics.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    with np.load(args.cache / 'test.npz') as f:
        meta = {k: f[k] for k in ('feed', 'item', 'counts')}
    chosen = json.loads(args.config.read_text())['selected']
    keys = prediction_frame(meta, np.zeros(len(meta['item']))).select(KEYS)
    score = combine([read_scores(p, keys) for p in args.scores], meta['counts'], meta['item'],
                    chosen['size_weights'], chosen['decay'])
    prediction_frame(meta, score).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    fit = commands.add_parser('select')
    fit.add_argument('cache', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('reference', type=Path)
    fit.add_argument('reserve', type=Path)
    fit.add_argument('calibrations', type=Path, nargs=2)
    infer = commands.add_parser('predict')
    infer.add_argument('cache', type=Path)
    infer.add_argument('config', type=Path)
    infer.add_argument('output', type=Path)
    infer.add_argument('scores', type=Path, nargs=3)
    args = parser.parse_args()
    if args.action == 'select':
        select(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
