"""Independently audit saved adaptation predictions, budgets and official metrics."""
import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl
import torch

from neural_blend import read_meta, read_scores
from ranking import KEYS, prediction_frame
from sequence_ranker import sha256, write_json


def independent_metrics(frame):
    """NumPy reference formula from baseline/calculate_metrics.py, no CatBoost dependency.

    Rebuild boundaries from the saved keys, sort ideal labels explicitly and
    compute each denominator independently of experiments/ranking.py.
    """
    frame = frame.sort('feed_id')
    targets = frame['target'].to_numpy()
    scores = frame['score'].to_numpy().astype(np.float64)
    groups = frame['feed_id'].to_numpy()
    bounds = np.r_[0, np.flatnonzero(groups[1:] != groups[:-1]) + 1, len(groups)]
    values = np.zeros((len(bounds) - 1, 3))
    for i, (a, b) in enumerate(zip(bounds[:-1], bounds[1:])):
        labels = targets[a:b]
        ranked = labels[np.lexsort((labels, -scores[a:b]))]
        ideal = np.sort(labels)[::-1]
        discounts = 1 / np.log2(np.arange(b - a) + 2)
        for j, cutoff in enumerate((None, 10, 30)):
            denominator = np.dot(ideal[:cutoff], discounts[:cutoff])
            values[i, j] = np.dot(ranked[:cutoff], discounts[:cutoff]) / denominator if denominator else 1.
    return {'rows': len(frame), 'groups': len(bounds) - 1, 'positives': int(targets.sum()),
            **dict(zip(('ndcg', 'ndcg_at_10', 'ndcg_at_30'), values.mean(axis=0).tolist()))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    meta = read_meta(root / 'cache-intent-024/val.npz')
    selection = json.loads((args.run / 'selection.json').read_text())
    assessment = json.loads((args.run / 'assessment.json').read_text())
    config = selection['selected']
    early, mid = 5766, 10083
    offsets = np.r_[0, np.cumsum(meta['counts'], dtype=np.int64)]
    keys = prediction_frame(meta, np.zeros(len(meta['item']), dtype=np.float32)).select(KEYS)
    evaluator_path = root.parent / 'baseline/calculate_metrics.py'
    official = {}
    for name in ('frozen', 'random_only', 'joint', 'submitted'):
        path = args.run / f'{name}_val_predictions.parquet'
        score = read_scores(path, keys)
        frame = pl.read_parquet(path).with_columns(pl.Series('target', meta['target']))
        if frame.select(KEYS).n_unique() != len(frame):
            raise AssertionError('duplicate candidate key')
        official[name] = {}
        for block, a, b in (('late', early, len(meta['counts'])), ('late_first', early, mid),
                            ('late_second', mid, len(meta['counts']))):
            measured = independent_metrics(frame.slice(int(offsets[a]), int(offsets[b] - offsets[a])))
            for metric in ('ndcg', 'ndcg_at_10', 'ndcg_at_30'):
                np.testing.assert_allclose(measured[metric], assessment['metrics'][name][block][metric],
                                           rtol=0, atol=1e-10)
            official[name][block] = measured
        if name in ('frozen', 'submitted'):
            reference = (root / 'sequence-finetune-048/pretrained/val_predictions.parquet'
                         if name == 'frozen' else root / 'nn-calibration-joint-024/val_predictions.parquet')
            np.testing.assert_array_equal(score, read_scores(reference, keys))
        assert sha256(path) == assessment['prediction_sha256'][name]
    chosen = max(selection['grid'], key=lambda e: (
        min(v['gain'] for v in e['arms']['joint']),
        np.average([v['gain'] for v in e['arms']['joint']], weights=[1879, 2017]),
        -e['steps'], -e['lr']))
    assert chosen == config
    schedules = []
    for folder in sorted(args.run.glob('forward-*')) + [args.run / 'final']:
        cfg = json.loads((folder / 'config.json').read_text())
        with np.load(folder / 'batch_schedule.npz') as data:
            assert np.all(data['random_ids'] < cfg['prefix_feeds'])
            assert np.all(data['random_ids'] >= 0)
            assert np.all(data['train_ids'] < cfg['train_available_feeds'])
            assert np.all(data['train_ids'] >= 0)
            assert data['random_batch_counts'].sum() == len(data['random_ids'])
            assert data['train_batch_counts'].sum() == len(data['train_ids'])
            assert len(data['random_batch_counts']) == len(data['train_batch_counts'])
            schedules.append({'folder': str(folder), 'steps': len(data['random_batch_counts']),
                              'max_random_feed_index': int(data['random_ids'].max()),
                              'prefix': cfg['prefix_feeds'], 'schedule_sha256': sha256(folder / 'batch_schedule.npz')})
    source = torch.load(root / 'sequence-finetune-048/pretrained/model.pt', weights_only=True)
    checkpoint_report = {}
    for name in ('random_only', 'joint'):
        path = args.run / f'final/step-{config["steps"]:03d}/{name}.pt'
        saved = torch.load(path, weights_only=True)
        changed = []
        for key, value in source['model'].items():
            if not torch.equal(saved['model'][key], value):
                assert key.startswith(('local.', 'output.')), key
                changed.append(key)
        assert any(key.startswith('local.') for key in changed)
        assert any(key.startswith('output.') for key in changed)
        assert saved['adaptation']['steps'] == config['steps']
        assert saved['adaptation']['lr'] == config['lr']
        checkpoint_report[name] = {'sha256': sha256(path), 'changed_parameters': changed,
                                   'depth_bias': saved['depth']['weight'].flatten().tolist()}
    report = {'official_evaluator': str(evaluator_path), 'official_evaluator_sha256': sha256(evaluator_path),
              'independent_numpy_reference_metrics': official,
              'catboost_check': 'not run: CatBoost is not installed in the local Python 3.12 environment',
              'all_prediction_pairs': len(meta['item']), 'schedules': schedules,
              'checkpoint_audit': checkpoint_report, 'selection_recomputed': True,
              'source_sha256': sha256(root / 'sequence-finetune-048/pretrained/model.pt'),
              'code_sha256': {p.name: sha256(p) for p in (root / 'randomized_adaptation.py', Path(__file__))}}
    write_json(args.run / 'independent_verification.json', report)
    print(json.dumps({'verified_pairs_per_branch': len(meta['item']), 'independent_numpy_reference_metrics_match': True,
                      'selection_and_schedules_match': True, 'changed_parameters': checkpoint_report}), flush=True)


if __name__ == '__main__':
    main()
