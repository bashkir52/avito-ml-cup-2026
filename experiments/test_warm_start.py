"""Unit tests for warm-start blend selection (no LightGBM needed)."""
import unittest

import numpy as np

import warm_start


class SelectWeightTest(unittest.TestCase):
    def test_smallest_weight_is_chosen_when_candidate_is_worse(self):
        counts = np.array([3] * 6000, dtype=np.int32)
        targets = np.tile([1, 0, 0], 6000)
        base = np.tile([3.0, 2.0, 1.0], 6000)      # positive already first
        candidate = np.tile([0.0, 1.0, 2.0], 6000)  # positive pushed down
        chosen, grid = warm_start.select_weight(base, candidate, counts, targets)
        self.assertEqual(len(grid), len(warm_start.WEIGHTS))
        self.assertLessEqual(chosen['minimum'], 0.0)
        self.assertEqual(chosen['weight'], warm_start.WEIGHTS[0])

    def test_weight_grows_when_candidate_improves_both_blocks(self):
        counts = np.array([3] * 6000, dtype=np.int32)
        targets = np.tile([1, 0, 0], 6000)
        base = np.zeros(len(targets))                      # every feed is a tie
        candidate = np.tile([5.0, 0.0, 0.0], 6000)         # positive moved to the top
        chosen, _ = warm_start.select_weight(base, candidate, counts, targets)
        self.assertGreater(chosen['minimum'], 0.0)
        self.assertGreater(chosen['weight'], 0.0)

    def test_bootstrap_interval_brackets_the_mean(self):
        counts = np.array([2] * 500, dtype=np.int32)
        targets = np.tile([1, 0], 500)
        base = np.tile([0.0, 1.0], 500)
        candidate = np.tile([1.0, 0.0], 500)
        result = warm_start.paired_bootstrap(base, candidate, counts, targets)
        self.assertGreater(result['delta'], 0)
        self.assertLessEqual(result['interval'][0], result['delta'])
        self.assertGreaterEqual(result['interval'][1], result['delta'])

    def test_standardize_is_centered_per_feed_and_survives_constant_feed(self):
        counts = np.array([2, 3], dtype=np.int32)
        score = np.array([1.0, 3.0, 5.0, 5.0, 5.0])
        result = warm_start.standardize(score, counts)
        self.assertAlmostEqual(result[:2].mean(), 0.0)
        self.assertTrue(np.all(result[2:] == 0.0))


if __name__ == '__main__':
    unittest.main()
