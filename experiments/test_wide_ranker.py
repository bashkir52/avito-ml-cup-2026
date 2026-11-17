import unittest

import numpy as np

from wide_ranker import match_coverage, sample_candidates, standardized_features


class WideRankerTest(unittest.TestCase):
    def test_constant_values_center_to_zero_despite_float32_accumulation_error(self):
        values = np.full((1, 10, 1), np.float32(.9833298325538635))
        values[0, -1] = 0
        valid = np.arange(10)[None, :] < 9
        self.assertTrue(np.all(standardized_features(values, valid, np.array([9]))[valid] == 0))

    def test_sampling_keeps_all_positives_and_excludes_padding(self):
        labels = np.array([[1, 0, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0]])
        valid = np.arange(6) < np.array([[6], [2]])
        sampled = sample_candidates(labels, valid, 2, np.random.default_rng(2026))
        self.assertTrue(np.all(sampled[labels == 1]))
        self.assertFalse(np.any(sampled & ~valid))
        np.testing.assert_array_equal((sampled & (labels == 0)).sum(axis=1), [2, 1])

    def test_sampling_is_deterministic_and_handles_a_short_batch(self):
        labels = np.array([[0, 1, 0]])
        valid = np.ones_like(labels, dtype=bool)
        first = sample_candidates(labels, valid, 12, np.random.default_rng(2026))
        second = sample_candidates(labels, valid, 12, np.random.default_rng(2026))
        np.testing.assert_array_equal(first, second)
        self.assertTrue(first.all())

    def test_control_matches_every_exposure_and_visits_all_supported_control_feeds(self):
        shapes = np.array([[22, 13, 1], [22, 13, 1], [5, 5, 1], [22, 13, 1], [5, 5, 1], [6, 6, 1]])
        wide, controls = match_coverage(shapes, np.array([0, 1, 2]), np.random.default_rng(2026))
        np.testing.assert_array_equal(np.sort(wide), np.arange(5))
        np.testing.assert_array_equal(shapes[wide], shapes[controls])
        np.testing.assert_array_equal(np.unique(controls), [0, 1, 2])
