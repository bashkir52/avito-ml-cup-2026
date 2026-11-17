import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from ranking import (normalize, ndcg_per_feed, position_averaged_prediction, read_extra_features,
                     read_sample_weights, temporal_boundary)


class RankingTests(unittest.TestCase):
    def test_pessimistic_ties_and_macro_average(self):
        labels = np.array([1, 0, 0, 0, 0, 1])
        scores = np.zeros(6)
        metrics = ndcg_per_feed(labels, scores, np.array([2, 3, 1]))
        np.testing.assert_allclose(metrics[:, 0], [1 / np.log2(3), 1, 1])

    def test_tied_candidate_order_has_no_effect(self):
        scores = np.array([.1, .2, .2, .1] * 11)
        labels = np.array([1, 0, 1, 0] * 11)
        counts = np.array([len(labels)])
        np.testing.assert_equal(ndcg_per_feed(labels, scores, counts),
                                ndcg_per_feed(labels[::-1], scores[::-1], counts))

    def test_cutoffs_have_consistent_dcg_and_idcg(self):
        labels = np.zeros(32, dtype=np.int8)
        labels[[10, 30, 31]] = 1
        score = -np.arange(len(labels), dtype=np.float32)
        best = (1 / np.log2(np.array([2, 3, 4]))).sum()
        expected = [(1 / np.log2(np.array([12, 32, 33]))).sum() / best,
                    0, 1 / (np.log2(12) * best)]
        np.testing.assert_allclose(ndcg_per_feed(labels, score, np.array([32]))[0], expected)

    def test_zero_vectors_and_int8_conversion(self):
        result = normalize(np.array([[100, 100], [0, 0]], dtype=np.int8))
        np.testing.assert_allclose(result, [[2 ** -.5, 2 ** -.5], [0, 0]])

    def test_temporal_boundary_keeps_equal_timestamps_together(self):
        times = np.array([1, 1, 2, 2, 2, 3, 3, 4, 4])
        self.assertEqual(temporal_boundary(times, .4), 5)

    def test_position_averaging_applies_equal_exposures_to_every_candidate(self):
        class LinearModel:
            def num_feature(self):
                return 2

            def predict(self, x, **kwargs):
                return x[:, 0] + x[:, 1]

        actual = position_averaged_prediction(LinearModel(), np.array([[1], [5]]), threads=1)
        expected_position = np.dot([8, 24, 64, 96, 109], [3, 19, 63, 143, 246]) / 301
        np.testing.assert_allclose(actual, np.array([1, 5]) + expected_position)

    def test_sample_weights_require_the_exact_candidate_order(self):
        with TemporaryDirectory() as path:
            train, weights = Path(path) / 'train.npz', Path(path) / 'weights.npz'
            keys = dict(feed=[10, 11], item=[4, 6, 4], counts=[2, 1])
            np.savez(train, **keys)
            np.savez(weights, **keys, weight=[.5, 1, 1.5])
            np.testing.assert_array_equal(read_sample_weights(weights, train), [.5, 1, 1.5])
            for wrong_items, wrong_weights in (([6, 4, 4], [.5, 1, 1.5]), ([4, 6, 4], [.5, 0, 1.5])):
                np.savez(weights, feed=keys['feed'], item=wrong_items, counts=keys['counts'], weight=wrong_weights)
                with self.assertRaises(ValueError):
                    read_sample_weights(weights, train)

    def test_extra_features_allow_missing_values_but_reject_different_keys(self):
        with TemporaryDirectory() as path:
            cache, extra = Path(path) / 'base.npz', Path(path) / 'extra.npz'
            keys = dict(feed=[10, 11], item=[4, 6, 4], counts=[2, 1], timestamp=[11, 25])
            matrix = np.array([[.3], [np.nan], [-.1]], dtype=np.float32)
            np.savez(cache, **keys)
            np.savez(extra, **keys, features=matrix, names=['graph_score'])
            values, names = read_extra_features(extra, cache)
            np.testing.assert_array_equal(values, matrix)
            self.assertEqual(names, ['graph_score'])
            np.savez(extra, **{**keys, 'timestamp': [11, 24]}, features=matrix, names=['graph_score'])
            with self.assertRaisesRegex(ValueError, 'timestamp'):
                read_extra_features(extra, cache)


if __name__ == '__main__':
    unittest.main()
