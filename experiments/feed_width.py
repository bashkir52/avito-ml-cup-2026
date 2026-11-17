"""Compare negative sampling rules at a fixed train row budget."""
import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import numpy as np

from neural_blend import normalize_scores, read_meta, read_scores, resolve_ties, score_mix
from ranking import COLS, KEYS, METRICS, NAMES, ndcg_per_feed, prediction_frame, temporal_boundary
from semantic_transition_rerank import top_hits


BUDGET = 3_575_775
SEED = 2026
HARDNESS_FEATURES = ('user_cos_rank', 'intent_rank_recent_max')
WEIGHTS = (0., .025, .05, .10, .20, .35, .50)


def negative_quotas(available, budget, rng):
    minimum = (available > 0).astype(np.int64)
    spare = available - minimum
    remaining = budget - minimum.sum()
    capacity = spare.sum()
    if not 0 <= remaining <= capacity:
        raise ValueError('budget cannot retain the positives and one negative per eligible feed')
    if capacity == 0:
        return minimum
    allocated, remainder = np.divmod(spare * remaining, capacity)
    order = np.lexsort((rng.random(len(spare)), -remainder))
    allocated[order[:remaining - allocated.sum()]] += 1
    quota = minimum + allocated
    if quota.sum() != budget or np.any(quota > available):
        raise ValueError('negative allocation differs from budget or exceeds capacity')
    return quota


def select_rows(target, offsets, hardness, quotas, rng):
    counts = np.add.reduceat(target, offsets[:-1], dtype=np.int64) + quotas
    sampled_offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    rows = {name: np.empty(sampled_offsets[-1], dtype=np.int64) for name in ('random', 'hard')}
    hard_slots = 0
    for i, (start, stop) in enumerate(zip(offsets[:-1], offsets[1:])):
        labels = target[start:stop]
        positive = np.flatnonzero(labels)
        negative = np.flatnonzero(labels == 0)
        priorities = rng.random(len(negative))
        random_order = np.argsort(priorities, kind='stable')
        quota = quotas[i]
        hard_count = quota // 2
        if hard_count:
            hard_order = np.lexsort((priorities, -hardness[start + negative]))
            hard_part = hard_order[:hard_count]
            free = np.ones(len(negative), dtype=bool)
            free[hard_part] = False
            random_part = random_order[free[random_order]][:quota - hard_count]
            hard = np.r_[hard_part, random_part]
        else:
            hard = random_order[:quota]
        hard_slots += hard_count
        sl = slice(sampled_offsets[i], sampled_offsets[i + 1])
        for name, take in (('random', random_order[:quota]), ('hard', hard)):
            rows[name][sl] = start + np.sort(np.r_[positive, negative[take]])
    return rows, counts, int(hard_slots)


def write_cohort(path, source, features, rows, counts, offsets, budget):
    if len(rows) != budget or np.any(np.diff(rows) <= 0):
        raise ValueError('sample must contain exactly the budget in distinct source rows')
    groups = np.searchsorted(offsets[1:], rows, side='right')
    np.testing.assert_array_equal(np.bincount(groups, minlength=len(counts)), counts)
    target = source['target'][rows]
    np.testing.assert_array_equal(rows[target == 1], np.flatnonzero(source['target']))
    expected = {key: source[key] for key in ('feed', 'timestamp', 'shards', 'prepare_seed')}
    expected.update(counts=counts.astype(source['counts'].dtype), target=target,
                    item=source['item'][rows], position=source['position'][rows])
    np.savez(path, x=features[rows], **expected)
    with np.load(path) as saved:
        for key, values in expected.items():
            np.testing.assert_array_equal(saved[key], values)
        matrix = saved['x']
        if matrix.shape != (budget, COLS['history-intent']) or matrix.dtype != features.dtype:
            raise ValueError('written features have the wrong shape or dtype')
        for start in range(0, budget, 100_000):
            sl = slice(start, start + 100_000)
            np.testing.assert_array_equal(matrix[sl].view(np.uint32), features[rows[sl]].view(np.uint32))


def prepare(source, control, out, budget, seed):
    if out.exists():
        raise FileExistsError(out)
    with np.load(source) as data:
        meta = {key: data[key] for key in data.files if key != 'x'}
    with np.load(control) as data:
        control_counts, control_feeds = data['counts'], data['feed']
        control_positive = int(data['target'].sum())
    counts, feed, time, target = (meta[key] for key in ('counts', 'feed', 'timestamp', 'target'))
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    if (np.any(counts < 1) or len(counts) != len(time) or len(counts) != len(np.unique(feed))
            or any(meta[key].shape != (offsets[-1],) for key in ('target', 'item', 'position'))
            or np.any((target != 0) & (target != 1))):
        raise ValueError('inconsistent source feeds, candidate arrays or binary labels')
    if control_counts.sum(dtype=np.int64) != budget or not np.isin(control_feeds, feed).all():
        raise ValueError('expected the whole-feed control at the same budget from the same source')
    positive_counts = np.add.reduceat(target, offsets[:-1], dtype=np.int64)
    positives = int(positive_counts.sum())
    rng = np.random.default_rng(seed)
    quota = negative_quotas(counts - positive_counts, budget - positives, rng)

    with TemporaryDirectory(dir=out.parent) as tmp:
        staging = Path(tmp)
        with ZipFile(source) as archive:
            x_path = archive.extract('x.npy', tmp)
        features = np.load(x_path, mmap_mode='r')
        try:
            if features.shape != (offsets[-1], COLS['history-intent']) or features.dtype != np.float32:
                raise ValueError('expected float32 history-intent features from full feeds')
            hardness = np.zeros(len(target), dtype=np.float32)
            for name in HARDNESS_FEATURES:
                rank = features[:, NAMES.index(name)]
                hardness += np.where(np.isfinite(rank), rank, .5) / len(HARDNESS_FEATURES)
            del rank
            rows, sampled_counts, hard_slots = select_rows(target, offsets, hardness, quota, rng)
            if not np.array_equal(sampled_counts, positive_counts + quota):
                raise ValueError('sampled feed counts differ from the common quotas')
            means = {}
            for name, selected in rows.items():
                write_cohort(staging / f'{name}.npz', meta, features, selected, sampled_counts, offsets, budget)
                negative_rows = selected[target[selected] == 0]
                means[name] = float(hardness[negative_rows].mean(dtype=np.float64))
                print(f'{name}: {len(feed):,} feeds, {len(selected):,} rows, {positives:,} positives; audited', flush=True)
        finally:
            del features
        report = {'source': str(source), 'seed': seed, 'source_rows': len(target), 'feeds': len(feed),
                  'rows_per_variant': budget, 'positive_rows_per_variant': positives,
                  'negative_rows_per_variant': budget - positives,
                  'negative_quota_histogram': np.bincount(quota).tolist(),
                  'sampled_feed_size_histogram': np.bincount(sampled_counts).tolist(),
                  'hard_slots': hard_slots, 'hardness_features': HARDNESS_FEATURES,
                  'negative_hardness_means': means,
                  'shared_negative_rows': int(len(np.intersect1d(rows['random'], rows['hard'], assume_unique=True)) - positives),
                  'control': {'path': str(control), 'feeds': len(control_feeds), 'rows': budget,
                              'positive_rows': control_positive},
                  'validated_metadata': ['feed', 'counts', 'item', 'target', 'timestamp', 'position', 'shards', 'prepare_seed'],
                  'validated_bitwise_features': COLS['history-intent']}
        (staging / 'cohorts.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        out.mkdir(parents=True, exist_ok=False)
        for name in ('random.npz', 'hard.npz', 'cohorts.json'):
            (staging / name).replace(out / name)


def metric_summary(values, hits, first, last):
    return {**dict(zip(METRICS, values[first:last].mean(axis=0).tolist())),
            'top10_positives': int(hits[first:last].sum())}


def paired_interval(delta):
    rng = np.random.default_rng(SEED)
    samples = np.concatenate([delta[rng.integers(len(delta), size=(100, len(delta)))].mean(axis=1)
                              for _ in range(20)])
    return np.quantile(samples, [.025, .975]).tolist()


def assess(cache, anchor, cohorts, out, runs):
    if out.exists():
        raise FileExistsError(out)
    cohort_report = json.loads((cohorts / 'cohorts.json').read_text(encoding='utf-8'))
    reports = {name: json.loads((path / 'metrics.json').read_text(encoding='utf-8')) for name, path in runs.items()}
    for name in ('random', 'hard'):
        control = reports['whole_feed']
        for field in ('kind', 'params', 'train_shards', 'train_rows', 'rounds', 'position_debias',
                      'position_standardize', 'feed_weight_power', 'sample_weights', 'extra_features'):
            if reports[name][field] != control[field]:
                raise ValueError(f'{name} and the whole-feed control differ in {field}')
        if (reports[name]['train_feeds'] != cohort_report['feeds']
                or reports[name]['train_rows'] != cohort_report['rows_per_variant']):
            raise ValueError(f'{name} model has the wrong training budget')
    if reports['whole_feed']['train_feeds'] != cohort_report['control']['feeds']:
        raise ValueError('whole-feed model and audited control have different feed counts')

    meta = read_meta(cache / 'val.npz')
    counts, target, time, item = (meta[key] for key in ('counts', 'target', 'timestamp', 'item'))
    keys = prediction_frame(meta, np.zeros(len(item))).select(KEYS)
    raw = {name: read_scores(run / 'val_predictions.parquet', keys) for name, run in runs.items()}
    raw['submitted'] = read_scores(anchor, keys)
    early = temporal_boundary(time, .4)
    early_inner = temporal_boundary(time[:early], .65)
    late_inner = early + temporal_boundary(time[early:], .5)
    early_rows = int(counts[:early].sum(dtype=np.int64))
    if (early_inner, early - early_inner, len(counts) - early) != (3749, 2017, 8657):
        raise ValueError('unexpected val split')
    normal = {name: normalize_scores(raw[name], counts, 'zscore') for name in ('submitted', 'random', 'hard')}

    def blend(name, weight, row_stop, feed_stop):
        base = raw['submitted'][:row_stop]
        if weight == 0:
            return base.astype(np.float32)
        mixed = score_mix([normal['submitted'][:row_stop], normal[name][:row_stop]], (1 - weight, weight))
        return resolve_ties(mixed, base, item[:row_stop], counts[:feed_stop])

    early_base = ndcg_per_feed(target[:early_rows], raw['submitted'][:early_rows].astype(np.float32), counts[:early])[:, 0]
    search, chosen = {}, {}
    for name in ('random', 'hard'):
        entries = []
        for weight in WEIGHTS:
            score = blend(name, weight, early_rows, early)
            delta = ndcg_per_feed(target[:early_rows], score, counts[:early])[:, 0] - early_base
            entries.append({'weight': weight, 'early_gain': float(delta.mean()),
                            'early_gain_by_block': [float(delta[:early_inner].mean()),
                                                    float(delta[early_inner:].mean())]})
        search[name] = entries
        eligible = [record for record in entries[1:] if min(record['early_gain_by_block']) > 0]
        chosen[name] = max(eligible, key=lambda x: (min(x['early_gain_by_block']), x['early_gain']), default=entries[0])

    evaluations, predictions = {}, {}
    for name, score in raw.items():
        score = score.astype(np.float32)
        evaluations[name] = ndcg_per_feed(target, score, counts), top_hits(target, score, counts)
    for name, config in chosen.items():
        score = blend(name, config['weight'], len(item), len(counts))
        predictions[name] = score
        evaluations[f'{name}_blend'] = ndcg_per_feed(target, score, counts), top_hits(target, score, counts)

    blocks = {'early_first': (0, early_inner), 'early_second': (early_inner, early), 'early': (0, early),
              'late_first': (early, late_inner), 'late_second': (late_inner, len(counts)),
              'late': (early, len(counts)), 'all': (0, len(counts))}

    def summarize(metric, hits):
        return {name: metric_summary(metric, hits, *bounds) for name, bounds in blocks.items()}

    def contrast(variant, control):
        metric, hits = evaluations[variant]
        old_metric, old_hits = evaluations[control]
        delta = metric - old_metric
        return {'control': control, 'gain': summarize(delta, hits - old_hits),
                'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0])}

    report = {'anchor': str(anchor), 'runs': {name: str(path) for name, path in runs.items()},
              'early_sizes': [early_inner, early - early_inner],
              'late_sizes': [late_inner - early, len(counts) - late_inner],
              'late_start_timestamp': int(time[early]),
              'selected_trees': {name: run['selected_trees'] for name, run in reports.items()},
              'metrics': {name: summarize(*values) for name, values in evaluations.items()},
              'contrasts': {'coverage': contrast('random', 'whole_feed'),
                            'hard_sampling': contrast('hard', 'random'),
                            'hard_vs_whole_feed': contrast('hard', 'whole_feed'),
                            **{name + '_vs_submitted': contrast(name, 'submitted') for name in ('random', 'hard')}},
              'blends': {name: {**config, **contrast(f'{name}_blend', 'submitted')}
                         for name, config in chosen.items()}}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'assessment.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    (out / 'early_search.json').write_text(json.dumps(search, indent=2) + '\n', encoding='utf-8')
    for name, score in predictions.items():
        if chosen[name]['weight']:
            prediction_frame(meta, score).write_parquet(out / f'{name}_blend_val_predictions.parquet')
    for name, metrics in report['metrics'].items():
        print(name, json.dumps(metrics['late']), flush=True)
    for name, blend_result in report['blends'].items():
        print(name, json.dumps({'weight': blend_result['weight'], 'late_gain': blend_result['gain']['late'],
                                'half_ndcg_gains': [blend_result['gain'][block]['ndcg']
                                                    for block in ('late_first', 'late_second')]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    sampling = commands.add_parser('prepare')
    sampling.add_argument('source', type=Path)
    sampling.add_argument('control', type=Path)
    sampling.add_argument('out', type=Path)
    sampling.add_argument('--budget', type=int, default=BUDGET)
    sampling.add_argument('--seed', type=int, default=SEED)
    comparison = commands.add_parser('assess')
    comparison.add_argument('cache', type=Path)
    comparison.add_argument('anchor', type=Path)
    comparison.add_argument('cohorts', type=Path)
    comparison.add_argument('out', type=Path)
    for name in ('random', 'hard', 'whole-feed', 'full'):
        comparison.add_argument(f'--{name}', type=Path, required=True)
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.source, args.control, args.out, args.budget, args.seed)
    else:
        runs = {name: getattr(args, name) for name in ('random', 'hard', 'whole_feed', 'full')}
        assess(args.cache, args.anchor, args.cohorts, args.out, runs)


if __name__ == '__main__':
    main()
