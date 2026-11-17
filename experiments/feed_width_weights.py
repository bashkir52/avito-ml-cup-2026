"""Reweight a fixed random sample of train candidates to restore negative mass."""
import argparse
import json
from pathlib import Path

import numpy as np

from compare_graph_rankers import compare
from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores
from ranking import KEYS, ndcg_per_feed, prediction_frame, read_sample_weights, temporal_boundary
from semantic_transition_rerank import top_hits


def load_metadata(path):
    with np.load(path) as data:
        return {key: data[key] for key in ('feed', 'timestamp', 'counts', 'item', 'target', 'position')}


def prepare(source_file, sample_file, out):
    if out.exists():
        raise FileExistsError(out)
    source, sample = map(load_metadata, (source_file, sample_file))
    source_counts, sample_counts = source['counts'], sample['counts']
    for key in ('feed', 'timestamp'):
        np.testing.assert_array_equal(sample[key], source[key])
    for data in (source, sample):
        rows = data['counts'].sum(dtype=np.int64)
        if (data['counts'].shape != data['feed'].shape or np.any(data['counts'] < 1)
                or any(data[key].shape != (rows,) for key in ('item', 'target', 'position'))
                or np.any((data['target'] != 0) & (data['target'] != 1))):
            raise ValueError('feeds and candidate arrays must align with binary targets')
    source_positive, sample_positive = (data['target'] == 1 for data in (source, sample))
    source_starts, sample_starts = (np.r_[0, np.cumsum(counts[:-1], dtype=np.int64)]
                                    for counts in (source_counts, sample_counts))
    positives = np.add.reduceat(source_positive, source_starts, dtype=np.int64)
    np.testing.assert_array_equal(np.add.reduceat(sample_positive, sample_starts, dtype=np.int64), positives)
    for key in ('item', 'position'):
        np.testing.assert_array_equal(sample[key][sample_positive], source[key][source_positive])
    source_negative, sample_negative = source_counts - positives, sample_counts - positives
    if np.any(sample_negative > source_negative) or not np.array_equal(sample_negative == 0, source_negative == 0):
        raise ValueError('each feed with source negatives must have a valid nonzero sampling quota')
    full_rows, sampled_rows = int(source_counts.sum()), int(sample_counts.sum())
    negative_total = int(source_negative.sum())
    if negative_total == 0:
        raise ValueError('no negatives to reweight')

    global_multiplier = negative_total / int(sample_negative.sum())
    feed_multiplier = np.divide(source_negative, sample_negative, out=np.ones(len(positives)), where=sample_negative > 0)
    normalization = sampled_rows / full_rows
    out.mkdir(parents=True, exist_ok=False)
    summary = {}
    for name, multiplier in (('global', global_multiplier), ('per_feed', np.repeat(feed_multiplier, sample_counts))):
        expected = (np.where(sample_positive, 1., multiplier) * normalization).astype(np.float32)
        path = out / f'{name}.npz'
        np.savez(path, **{key: sample[key] for key in ('feed', 'item', 'counts')}, weight=expected)
        weights = read_sample_weights(path, sample_file)
        np.testing.assert_array_equal(weights, expected)
        total = weights.sum(dtype=np.float64)
        prevalence = weights[sample_positive].sum(dtype=np.float64) / total
        np.testing.assert_allclose(total, sampled_rows, rtol=1e-7)
        np.testing.assert_allclose(prevalence, positives.sum() / full_rows, rtol=1e-7)
        negative_mass = np.add.reduceat(np.where(sample_positive, 0., weights), sample_starts, dtype=np.float64)
        expected_mass = source_negative if name == 'per_feed' else sample_negative * global_multiplier
        np.testing.assert_allclose(negative_mass, expected_mass * normalization, rtol=1e-7)
        summary[name] = {'mean': float(total / sampled_rows), 'min': float(weights.min()), 'max': float(weights.max()),
                         'weighted_positive_fraction': float(prevalence),
                         'effective_row_fraction': float(total ** 2 / (sampled_rows * np.square(weights, dtype=np.float64).sum())),
                         'max_feed_negative_mass_error_before_normalization': float(np.abs(negative_mass / normalization - source_negative).max())}

    report = {'source': str(source_file), 'sample': str(sample_file), 'feeds': len(positives),
              'source_rows': full_rows, 'sample_rows': sampled_rows, 'positive_rows': int(positives.sum()),
              'sample_positive_fraction': float(positives.sum() / sampled_rows),
              'source_positive_fraction': float(positives.sum() / full_rows), 'negative_global_multiplier': global_multiplier,
              'negative_feed_multiplier_range': [float(feed_multiplier.min()), float(feed_multiplier.max())],
              'normalization': normalization, 'weights': summary}
    (out / 'weight_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def assess(cache, anchor, weights, control, global_run, per_feed_run, out):
    models = {'control': control, 'global': global_run, 'per_feed': per_feed_run}
    model_reports = {name: json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
                     for name, path in models.items()}
    allowed = model_reports['control']
    for name, report in model_reports.items():
        for field in ('kind', 'params', 'train_shards', 'prepare_seed', 'train_feeds', 'train_rows', 'rounds',
                      'position_debias', 'position_standardize', 'feed_weight_power', 'extra_features'):
            if report[field] != allowed[field]:
                raise ValueError(f'{name} differs from the control in {field}')
        if name == 'control':
            if report['sample_weights'] is not None:
                raise ValueError('control must use unit weights')
        elif report['sample_weights'] is None or Path(report['sample_weights']).resolve() != (weights / f'{name}.npz').resolve():
            raise ValueError(f'{name} model used the wrong sample weights')

    compare(cache, anchor, [(global_run, control), (per_feed_run, control)], out, 'weighting')
    meta = read_meta(cache / 'val.npz')
    counts, target, time = (meta[key] for key in ('counts', 'target', 'timestamp'))
    keys = prediction_frame(meta, np.zeros(len(target))).select(KEYS)
    scores = {name: read_scores(path / 'val_predictions.parquet', keys) for name, path in models.items()}
    scores['submitted'] = read_scores(anchor, keys)
    scores['blend'] = read_scores(out / 'val_predictions.parquet', keys)
    metrics = {name: (ndcg_per_feed(target, score.astype(np.float32), counts), top_hits(target, score, counts))
               for name, score in scores.items()}
    early = temporal_boundary(time, .4)
    late_inner = early + temporal_boundary(time[early:], .5)
    blocks = {'early': (0, early), 'late': (early, len(counts)), 'all': (0, len(counts)),
              'late_first': (early, late_inner), 'late_second': (late_inner, len(counts))}

    def summarize(values, hits):
        return {name: metric_summary(values, hits, *bounds) for name, bounds in blocks.items()}

    contrasts = {}
    for name, baseline in (('global', 'control'), ('per_feed', 'control'), ('per_feed', 'global'), ('blend', 'submitted')):
        values, hits = metrics[name]
        base_values, base_hits = metrics[baseline]
        delta = values - base_values
        contrasts[f'{name}_vs_{baseline}'] = {'gain': summarize(delta, hits - base_hits),
                                            'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0])}
    report = {'selected_trees': {name: data['selected_trees'] for name, data in model_reports.items()},
              'metrics': {name: summarize(*values) for name, values in metrics.items()}, 'contrasts': contrasts}
    (out / 'diagnostics.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    for name, metrics_by_block in report['metrics'].items():
        print(name, json.dumps(metrics_by_block['late']), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    sampling = commands.add_parser('prepare')
    for name in ('source', 'sample', 'out'):
        sampling.add_argument(name, type=Path)
    comparison = commands.add_parser('assess')
    for name in ('cache', 'anchor', 'weights', 'control', 'global_run', 'per_feed_run', 'out'):
        comparison.add_argument(name, type=Path)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.sample, args.out)
    else:
        assess(args.cache, args.anchor, args.weights, args.control, args.global_run, args.per_feed_run, args.out)


if __name__ == '__main__':
    main()
