"""Fit a partially pooled linear specialist for feeds with 101–300 items."""
import argparse
import json
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from feed_width import metric_summary, paired_interval
from ranking import audit_submission, ndcg_per_feed, prediction_frame, temporal_boundary
from residual_rerank import ANCHOR, LINEAR_PENALTY, SOURCES, forward_anchor, forward_windows, load
from score_calibration import fit, slices
from sequence_ranker import scores_for_blend, selection_key, sha256, top10_hits, write_json


BLEND_ARMS = ('global', 'specialist')
DIFFERENCE = 'specialist_delta'
FAMILIES = (*BLEND_ARMS, DIFFERENCE)
WEIGHTS = (0., .05, .10, .20, .35, .50, .75, 1.)
SPECIALIST_PENALTY = .10


def long_feeds(counts):
    return (counts >= 101) & (counts <= 300)


def blend(raw, meta, family, weight, head_union):
    score = scores_for_blend(raw, meta, family, weight, BLEND_ARMS, DIFFERENCE)
    if not head_union or not weight:
        return score
    anchor = raw['submitted']
    out = anchor.astype(np.float32, copy=True)
    offsets = slices(meta['counts'])
    for a, b in zip(offsets[:-1], offsets[1:]):
        old_order = np.argsort(-anchor[a:b], kind='stable')
        other_order = np.lexsort((meta['item'][a:b], -raw[family][a:b]))
        candidates = a + np.union1d(old_order[:head_union], other_order[:head_union])
        order = np.argsort(-score[candidates], kind='stable')
        out[candidates[order]] = np.sort(anchor[candidates])[::-1]
    return out


def apply_to_long_feeds(meta, raw, family, weight, head_union=0):
    score = raw['submitted'].astype(np.float32, copy=True)
    if not weight:
        return score
    counts = meta['counts']
    eligible = long_feeds(counts)
    if not eligible.any():
        return score
    rows = np.repeat(eligible, counts)
    partial = {'counts': counts[eligible], 'item': meta['item'][rows]}
    score[rows] = blend({name: values[rows] for name, values in raw.items()},
                        partial, family, weight, head_union)
    return score


def fit_models(x, labels, counts):
    x = np.ascontiguousarray(x, dtype=np.float64)
    eligible = long_feeds(counts)
    keep = np.repeat(eligible, counts)
    global_coeff = fit(x, labels, counts, LINEAR_PENALTY)
    specialized = fit(x[keep], labels[keep], counts[eligible], SPECIALIST_PENALTY, center=global_coeff)
    print(f'fit on {len(counts):,} total feeds and {eligible.sum():,} long feeds', flush=True)
    return {'global': global_coeff, 'specialist': specialized}


def scores_for_models(x, coefficients):
    return {name: x @ coeff for name, coeff in coefficients.items()}


def choose(meta, raw, target, first, head_union):
    counts = meta['counts']
    eligible = long_feeds(counts)
    rows = np.repeat(eligible, counts)
    target = target[rows]
    partial = {'counts': counts[eligible], 'item': meta['item'][rows]}
    raw = {name: values[rows] for name, values in raw.items()}
    cutoff = int(eligible[:first].sum())
    base = ndcg_per_feed(target, raw['submitted'].astype(np.float32), partial['counts'])[:, 0]
    finalists, grid = {}, {}
    for name in BLEND_ARMS if head_union else FAMILIES:
        trials = []
        for weight in WEIGHTS:
            score = blend(raw, partial, name, weight, head_union)
            gain = ndcg_per_feed(target, score, partial['counts'])[:, 0] - base
            trials.append({'name': name, 'weight': weight, 'early_gain': float(gain.mean()),
                           'early_gain_by_block': [float(gain[:cutoff].mean()), float(gain[cutoff:].mean())]})
        admissible = [trial for trial in trials if min(trial['early_gain_by_block']) > 0]
        finalists[name] = max(admissible, key=selection_key, default=trials[0])
        grid[name] = trials
    return {'choice': max(finalists.values(), key=selection_key), 'families': finalists,
            'long_feeds_by_window': [cutoff, len(base) - cutoff], 'grid': grid}


def assess(meta, scores, early):
    counts, labels = meta['counts'], meta['target']
    middle = early + temporal_boundary(meta['timestamp'][early:], .5)
    blocks = {'late_first': (early, middle), 'late_second': (middle, len(counts)),
              'late': (early, len(counts))}
    long = long_feeds(counts[early:])

    def summary(ndcg, hits):
        return {**{key: metric_summary(ndcg, hits, *bounds) for key, bounds in blocks.items()},
                'late_long_101_300': metric_summary(ndcg[early:][long], hits[early:][long], 0, int(long.sum()))}

    per_feed = {name: (ndcg_per_feed(labels, score, counts), top10_hits(labels, score, counts))
                for name, score in scores.items()}
    contrasts = {}
    pairs = [(name, 'submitted') for name in scores if name != 'submitted']
    pairs.append(('specialist_full_long', 'global_full_long'))
    for name, control in pairs:
        delta, hits = (a - b for a, b in zip(per_feed[name], per_feed[control], strict=True))
        contrasts[f'{name}_vs_{control}'] = {'gain': summary(delta, hits),
            'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0]),
            'late_long_paired_ndcg_95_interval': paired_interval(delta[early:][long, 0])}
    return {name: summary(*values) for name, values in per_feed.items()}, contrasts


def select(args):
    args.out.mkdir(parents=True, exist_ok=False)
    meta, x, names, reference, anchor, sources, anchor_path = load(args.experiments, 'val')
    windows, early = forward_windows(meta['timestamp'])
    counts, labels = meta['counts'], meta['target']
    offsets = slices(counts)
    start, middle = windows[0]
    a, b = offsets[[start, early]]
    held_out = [scores_for_models(x[offsets[first]:offsets[last]],
                  fit_models(x[:offsets[first]], labels[:offsets[first]], counts[:first]))
                for first, last in windows]
    raw = {name: np.concatenate([fold[name] for fold in held_out]) for name in BLEND_ARMS}
    raw['submitted'] = forward_anchor(args.experiments, meta, reference, windows)
    partial = {'counts': counts[start:early], 'item': meta['item'][a:b]}
    selection = choose(partial, raw, labels[a:b], middle - start, args.head_union)
    write_json(args.out / 'early_selection.json', {**selection, 'windows': windows, 'head_union': args.head_union})
    print(json.dumps(selection['choice']), flush=True)

    coefficients = fit_models(x[:b], labels[:b], counts[:early])
    raw = {'submitted': anchor, **scores_for_models(x, coefficients)}
    scores = {'submitted': anchor.astype(np.float32)}
    for name, config in selection['families'].items():
        scores[name + '_blend'] = apply_to_long_feeds(meta, raw, name, config['weight'], args.head_union)
    for name in BLEND_ARMS:
        scores[name + '_full_long'] = apply_to_long_feeds(meta, raw, name, 1., args.head_union)
    metrics, contrasts = assess(meta, scores, early)
    choice = selection['choice']
    result = contrasts[choice['name'] + '_blend_vs_submitted']
    gain = result['gain']
    halves = [gain[name] for name in ('late_first', 'late_second')]
    stable = choice['weight'] > 0 and min(g['ndcg'] for g in halves) > 0 and (
        gain['late']['ndcg'] >= .003 or
        (gain['late']['ndcg'] >= .001 and result['late_paired_ndcg_95_interval'][0] > 0
         and all(g['ndcg_at_10'] > 0 and g['top10_positives'] > 0 for g in halves)))
    for name, score in scores.items():
        if name != 'submitted':
            prediction_frame(meta, score).write_parquet(args.out / f'{name}_val_predictions.parquet')
    config = {'choice': choice, 'prepare_submission': stable, 'feature_names': names, 'head_union': args.head_union,
              'metrics': metrics, 'contrasts': contrasts, 'windows': windows, 'fit_feeds': early,
              'fit_long_feeds': int(long_feeds(counts[:early]).sum()),
              'fit_long_rows': int(counts[:early][long_feeds(counts[:early])].sum()),
              'coefficients': {name: values.tolist() for name, values in coefficients.items()},
              'val_source_sha256': {str(p): sha256(p) for p in [*sources, anchor_path]},
              'test_source_sha256': {str(args.experiments / directory / path): sha256(args.experiments / directory / path)
                 for directory, _, path in (*SOURCES, ANCHOR)}}
    write_json(args.out / 'metrics.json', config)
    print(json.dumps({'late': {name: block['late'] for name, block in metrics.items()},
                      'prepare_submission': stable}), flush=True)


def submit(args):
    config = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
    if not config['prepare_submission'] or args.output.exists():
        raise ValueError('submission requires a qualifying assessment and a new output path')
    meta, x, names, _, anchor, paths, anchor_path = load(args.experiments, 'test')
    digests = {str(p): sha256(p) for p in [*paths, anchor_path]}
    if config['test_source_sha256'] != digests or names != config['feature_names']:
        raise ValueError('test inputs differ from the assessed feature source')
    raw = {'submitted': anchor, **scores_for_models(x, {name: np.asarray(w) for name, w in config['coefficients'].items()})}
    choice = config['choice']
    score = apply_to_long_feeds(meta, raw, choice['name'], choice['weight'], config.get('head_union', 0))
    with NamedTemporaryFile(dir=args.output.parent, suffix='.parquet', delete=False) as stream:
        temp = Path(stream.name)
    try:
        prediction_frame(meta, score).write_parquet(temp)
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
    training.add_argument('--head-union', type=int, choices=(0, 20), default=0)
    training.set_defaults(run_action=select)
    prediction = commands.add_parser('submit')
    prediction.add_argument('experiments', type=Path)
    prediction.add_argument('run', type=Path)
    prediction.add_argument('output', type=Path)
    prediction.set_defaults(run_action=submit)
    args = parser.parse_args()
    args.run_action(args)


if __name__ == '__main__':
    main()
