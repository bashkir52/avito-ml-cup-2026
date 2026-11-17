"""Check whether imitating the logged order improves randomized-feed ranking."""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from threadpoolctl import threadpool_limits

from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores, resolve_ties
from ranking import KEYS, NAMES, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import top10_hits, write_json


ROOT = Path(__file__).resolve().parent
CACHE = ROOT / 'cache-intent-024'
ANCHOR = ROOT / 'nn-calibration-joint-024/val_predictions.parquet'
SEED = 2026
WEIGHTS = (.025, .05, .10, .20, .35, .50)
PARAMS = {'learning_rate': .10, 'max_iter': 80, 'max_leaf_nodes': 63, 'min_samples_leaf': 100,
          'l2_regularization': 5., 'max_bins': 127, 'early_stopping': False, 'random_state': SEED}


def row_indices(counts, ids):
    edges = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    return np.concatenate([np.arange(edges[i], edges[i+1]) for i in ids])


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    rng = np.random.default_rng(SEED)
    with np.load(CACHE / 'train.npz') as cache:
        counts, ts = cache['counts'], cache['timestamp']
        split = temporal_boundary(ts, .85)
        train_ids = np.sort(rng.choice(split, size=15000, replace=False))
        check_ids = np.sort(rng.choice(np.arange(split, len(counts)), size=3000, replace=False))
        train_rows = row_indices(counts, train_ids)
        check_rows = row_indices(counts, check_ids)
        x = cache['x']
        tx, hx = x[train_rows], x[check_rows]
        del x
        pos = cache['position']
        ty = np.log1p(pos[train_rows].astype(np.float64))
        hp = pos[check_rows]
    val = read_meta(CACHE / 'val.npz')
    with np.load(CACHE / 'val.npz') as cache:
        vx = cache['x']
    if tx.shape[1] != len(NAMES) or ts[train_ids].max() >= ts[check_ids].min() or ts.max() >= val['timestamp'].min():
        raise ValueError('unexpected source features or chronology')
    np.savez(out / 'sampling.npz', train_feeds=train_ids, chronological_holdout_feeds=check_ids)
    config = {'seed': SEED, 'params': PARAMS, 'train_feeds': len(train_ids), 'train_rows': len(tx),
              'logged_order_holdout_feeds': len(check_ids), 'logged_order_holdout_rows': len(hx),
              'train_timestamp_max': int(ts[train_ids].max()), 'order_holdout_timestamp_min': int(ts[check_ids].min()),
              'source_labels': 'log1p(pos); competition target is not read in training',
              'weights': WEIGHTS}
    write_json(out / 'config.json', config)
    raw = {}
    diagnostics = {}
    for name, width in (('f1_f5', 5), ('all79', len(NAMES))):
        start = time.perf_counter()
        model = HistGradientBoostingRegressor(**PARAMS).fit(tx[:, :width], ty)
        held = -model.predict(hx[:, :width])
        has_top10 = np.add.reduceat((hp < 10), np.r_[0, np.cumsum(counts[check_ids][:-1])]) > 0
        order_metric = ndcg_per_feed(hp < 10, held, counts[check_ids])
        diagnostics[name] = {'log_position_mae': float(np.abs(held + np.log1p(hp.astype(np.float64))).mean()),
                             'ndcg_for_logged_top10': float(order_metric[has_top10, 0].mean()),
                             'feeds_with_logged_top10': int(has_top10.sum()), 'seconds': time.perf_counter()-start}
        path = out / f'{name}.joblib'
        joblib.dump(model, path)
        score = -joblib.load(path).predict(vx[:, :width]).astype(np.float32)
        prediction_frame(val, score).write_parquet(out / f'{name}_val_predictions.parquet')
        raw[name] = score
        print(json.dumps({name: diagnostics[name]}), flush=True)
    write_json(out / 'order_imitation.json', diagnostics)
    counts, y, ts, items = (val[k] for k in ('counts', 'target', 'timestamp', 'item'))
    early = temporal_boundary(ts, .4)
    first = temporal_boundary(ts[:early], .65)
    mid = early + temporal_boundary(ts[early:], .5)
    if (first, early, mid, len(counts)) != (3749, 5766, 10083, 14423):
        raise ValueError('unexpected chronological split')
    cut = int(counts[:early].sum())
    keys = prediction_frame(val, np.zeros(len(items))).select(KEYS)
    anchor = read_scores(ANCHOR, keys).astype(np.float32)
    z = standardized_scores(anchor, counts)
    base = ndcg_per_feed(y[:cut], anchor[:cut], counts[:early])[:, 0]
    grid = []
    for name, score in raw.items():
        x = standardized_scores(score[:cut], counts[:early])
        for w in WEIGHTS:
            blended = resolve_ties((1-w)*z[:cut] + w*x, anchor[:cut], items[:cut], counts[:early])
            d = ndcg_per_feed(y[:cut], blended, counts[:early])[:, 0] - base
            grid.append({'name': name, 'weight': w, 'gain': float(d.mean()),
                         'block_gain': [float(d[:first].mean()), float(d[first:].mean())]})
    best = max(grid, key=lambda c: (min(c['block_gain']), c['gain']))
    selected = best if min(best['block_gain']) > 0 else {'name': 'submitted', 'weight': 0.}
    write_json(out / 'selection.json', {'best_nonzero': best, 'selected': selected, 'grid': grid})
    w = selected['weight']
    score = (resolve_ties((1-w)*z + w*standardized_scores(raw[selected['name']], counts), anchor, items, counts)
             if w else anchor.copy())
    prediction_frame(val, score).write_parquet(out / 'blend_val_predictions.parquet')
    base = ndcg_per_feed(y, anchor, counts)
    blocks = {'late': (early, len(counts)), 'late_first': (early, mid), 'late_second': (mid, len(counts))}
    results = {}
    for name, s in {**raw, 'submitted': anchor, 'blend': score}.items():
        metric, hits = ndcg_per_feed(y, s, counts), top10_hits(y, s, counts)
        results[name] = {key: metric_summary(metric, hits, *bounds) for key, bounds in blocks.items()}
        results[name]['late_ndcg_gain'] = float((metric-base)[early:, 0].mean())
        results[name]['late_95_ci'] = paired_interval((metric-base)[early:, 0])
    write_json(out / 'assessment.json', {'selection': selected, 'metrics': results})
    print(json.dumps({'selection': selected, 'best_nonzero': best, 'late': {n: m['late'] for n, m in results.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    args = parser.parse_args()
    with threadpool_limits(6, user_api='openmp'), threadpool_limits(1, user_api='blas'):
        run(args.out)


if __name__ == '__main__':
    main()
