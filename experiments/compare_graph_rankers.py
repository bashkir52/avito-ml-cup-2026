"""Compare matched rankers against their controls and the submitted ensemble."""
import argparse
import json
from pathlib import Path

import numpy as np

from feed_width import paired_interval
from neural_blend import read_meta, read_scores, resolve_ties, save_result
from ranking import KEYS, METRICS, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from semantic_transition_rerank import top_hits


def compare(cache, anchor, pairs, out, variant_label='graph'):
    if out.exists():
        raise FileExistsError(out)
    meta = read_meta(cache / 'val.npz')
    counts, target, timestamp = (meta[k] for k in ('counts', 'target', 'timestamp'))
    keys = prediction_frame(meta, np.zeros(len(target))).select(KEYS)
    baseline = read_scores(anchor, keys)
    base = ndcg_per_feed(target, baseline.astype(np.float32), counts)
    baseline_norm = standardized_scores(baseline, counts)
    end = temporal_boundary(timestamp, .4)
    middle = temporal_boundary(timestamp[:end], .65)
    late_middle = end + temporal_boundary(timestamp[end:], .5)
    row_end = int(counts[:end].sum())

    def late_metrics(metric):
        return dict(zip(METRICS, metric[end:].mean(axis=0).tolist()))

    signals, standalone, sources, normalized = {}, {}, {}, []
    for graph, control in pairs:
        name = graph.name
        if name == 'joint' or name in standalone:
            raise ValueError('graph model directory names must be unique')
        graph_run, control_run = [json.loads((path / 'metrics.json').read_text(encoding='utf-8'))
                                  for path in (graph, control)]
        fields = ('kind', 'train_shards', 'train_feeds', 'train_rows', 'rounds')
        expected_params = control_run['params']
        if variant_label == 'objective':
            fields += ('prepare_seed', 'position_debias', 'position_standardize',
                       'feed_weight_power', 'sample_weights', 'extra_features')
            objective = graph_run['params']['objective']
            if expected_params['objective'] != 'binary' or objective not in ('rank_xendcg', 'lambdarank'):
                raise ValueError('objective comparison needs a ranking objective and a binary control')
            expected_params = {**expected_params, 'objective': objective}
            if objective == 'lambdarank':
                expected_params['lambdarank_truncation_level'] = 40
        if graph_run['params'] != expected_params:
            raise ValueError(f'ranker and control differ in params: {graph}, {control}')
        for field in fields:
            if graph_run[field] != control_run[field]:
                raise ValueError(f'ranker and control differ in {field}: {graph}, {control}')
        graph_scores, control_scores = [read_scores(path / 'val_predictions.parquet', keys) for path in (graph, control)]
        graph_metric, control_metric = [ndcg_per_feed(target, score.astype(np.float32), counts)
                                        for score in (graph_scores, control_scores)]
        delta = graph_metric - control_metric
        standalone[name] = {'control': control.name, f'late_{variant_label}': late_metrics(graph_metric),
                            'late_control': late_metrics(control_metric), 'late_gain': late_metrics(delta),
                            'late_halves_gain': [float(delta[a:b, 0].mean()) for a, b in
                                                 [(end, late_middle), (late_middle, len(counts))]]}
        if variant_label == 'objective':
            late_sizes = counts[end:]
            hits, control_hits = [top_hits(target[row_end:], score[row_end:], late_sizes)
                                  for score in (graph_scores, control_scores)]
            by_size = {}
            for low, high in ((5, 30), (31, 100), (101, 300)):
                mask = (late_sizes >= low) & (late_sizes <= high)
                if mask.any():
                    by_size[f'{low}-{high}'] = float(delta[end:, 0][mask].mean())
            standalone[name].update(late_top10_positives=int(hits.sum()),
                                    late_control_top10_positives=int(control_hits.sum()),
                                    late_gain_by_size=by_size,
                                    late_paired_ndcg_95_interval=paired_interval(delta[end:, 0]))
        values = tuple(standardized_scores(score, counts) for score in (graph_scores, control_scores))
        signals[name] = values[0], values[0] - values[1]
        normalized.append(values)
        sources[name] = [str(graph), str(control)]
    if len(sources) > 1:
        all_graph, all_control = np.mean(normalized, axis=0)
        signals['joint'] = all_graph, all_graph - all_control

    def rank(config, early):
        rows = slice(None, row_end if early else None)
        sizes = counts[:end] if early else counts
        mode, weight = config['mode'], config['weight']
        component = signals[config['source']][0 if mode == 'blend' else 1]
        anchor_weight = 1-weight if mode == 'blend' else 1.
        score = anchor_weight * baseline_norm[rows] + weight * component[rows]
        return resolve_ties(score, baseline[rows], meta['item'][rows], sizes)

    def robust(config):
        return min(config['early_gain_by_block']), config['early_gain']

    finalists, grid = [], []
    for source in signals:
        for mode in ('blend', 'delta'):
            options = []
            for weight in (.025, .05, .10, .20, .35, .50):
                config = {'source': source, 'mode': mode, 'weight': weight}
                d = ndcg_per_feed(target[:row_end], rank(config, early=True), counts[:end])[:, 0] - base[:end, 0]
                options.append({**config, 'early_gain_by_block': [float(d[:middle].mean()), float(d[middle:].mean())],
                                'early_gain': float(d.mean())})
            finalists.append(max(options, key=robust))
            grid.extend(options)
    best = max(finalists, key=robust)
    if robust(best)[0] <= 0:
        best = {**best, 'weight': 0., 'early_gain_by_block': [0., 0.], 'early_gain': 0.}

    def evaluate(config):
        score = rank(config, early=False)
        metric = ndcg_per_feed(target, score, counts)
        d = metric - base
        info = {**config, 'late_metrics': late_metrics(metric), 'late_gain': late_metrics(d),
                'late_gain_by_half': [float(d[a:b, 0].mean()) for a, b in [(end, late_middle), (late_middle, len(counts))]],
                'late_top10_positives': int(top_hits(target[row_end:], score[row_end:], counts[end:]).sum())}
        return info, score, metric

    best, score, metric = evaluate(best)
    report = {'anchor': str(anchor), 'models': sources, 'early_block_sizes': [middle, end-middle],
              'late_half_sizes': [late_middle-end, len(counts)-late_middle], 'incumbent': late_metrics(base),
              'incumbent_late_top10_positives': int(top_hits(target[row_end:], baseline[row_end:], counts[end:]).sum()),
              'standalone': standalone, 'finalists': [evaluate(c)[0] for c in finalists], 'selected': best}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'assessment.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    (out / 'early_search.json').write_text(json.dumps(grid, indent=2)+'\n', encoding='utf-8')
    save_result(out, best, score, metric[:, 0], base[:, 0], meta, end)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('anchor', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--pair', action='append', type=Path, nargs=2, required=True, metavar=('VARIANT', 'CONTROL'))
    parser.add_argument('--variant-label', default='graph', choices=('graph', 'temporal', 'objective', 'position-prior'))
    args = parser.parse_args()
    compare(args.cache, args.anchor, args.pair, args.out, args.variant_label)
