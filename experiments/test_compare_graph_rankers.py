from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from compare_graph_rankers import compare
from ranking import prediction_frame


class ObjectiveComparisonTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache = Path(temporary.name)
        counts = np.tile([5, 31, 101], 10)
        self.targets = np.concatenate([np.arange(n) == 0 for n in counts]).astype(np.int8)
        self.meta = dict(feed=np.arange(len(counts), dtype=np.int32), counts=counts,
                         item=np.concatenate([np.arange(n, dtype=np.int32) for n in counts]),
                         timestamp=np.arange(len(counts)), target=self.targets)
        np.savez(self.cache / 'val.npz', **self.meta)
        self.params = {'objective': 'binary', 'seed': 2026}
        self.report = {'kind': 'history-intent', 'train_shards': ['shard'], 'train_feeds': 30, 'train_rows': 1370,
                       'rounds': 160, 'prepare_seed': 2026, 'sample_weights': None, 'feed_weight_power': 0.,
                       'position_debias': False, 'position_standardize': False, 'extra_features': None}
        self.control = self.run_dir('control', self.params, self.meta['item'])
        self.anchor = self.control / 'val_predictions.parquet'

    def run_dir(self, name, params, scores, **changes):
        path = self.cache / name
        path.mkdir(exist_ok=True)
        (path / 'metrics.json').write_text(json.dumps({**self.report, **changes, 'params': params}), encoding='utf-8')
        prediction_frame(self.meta, scores).write_parquet(path / 'val_predictions.parquet')
        return path

    def test_objective_mode_accepts_only_the_intended_training_change(self):
        xe = self.run_dir('xe', {**self.params, 'objective': 'rank_xendcg'}, -self.meta['item'])
        pair = self.run_dir('pair', {**self.params, 'objective': 'lambdarank', 'lambdarank_truncation_level': 40},
                            -self.meta['item'])
        out = self.cache / 'out'
        with redirect_stdout(StringIO()):
            try:
                compare(self.cache, self.anchor, [(xe, self.control), (pair, self.control)], out, 'objective')
            except ValueError as error:
                self.fail(str(error))
        report = json.loads((out / 'assessment.json').read_text(encoding='utf-8'))
        for name in ('xe', 'pair'):
            self.assertEqual(report['standalone'][name]['late_objective']['ndcg'], 1.)

    def test_objective_mode_rejects_seed_budget_and_weight_changes(self):
        cases = [({'seed': 43}, {}, 'params'), ({}, {'train_rows': 1371}, 'train_rows'),
                 ({}, {'sample_weights': 'weights.npz'}, 'sample_weights')]
        for params, fields, message in cases:
            with self.subTest(field=message):
                variant = self.run_dir('xe', {**self.params, 'objective': 'rank_xendcg', **params},
                                       -self.meta['item'], **fields)
                with self.assertRaisesRegex(ValueError, message):
                    compare(self.cache, self.anchor, [(variant, self.control)], self.cache / 'out', 'objective')

    def test_default_comparison_rejects_different_objectives(self):
        variant = self.run_dir('xe', {**self.params, 'objective': 'rank_xendcg'}, -self.meta['item'])
        with self.assertRaisesRegex(ValueError, 'params'):
            compare(self.cache, self.anchor, [(variant, self.control)], self.cache / 'out')
