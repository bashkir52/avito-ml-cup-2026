import unittest

import numpy as np

from history_swaps import candidate_order, choose_swaps, swap_scores


class HistorySwapsTest(unittest.TestCase):
    def test_candidate_pool_and_ties_use_scores_and_ids_only(self):
        ids = np.arange(150)
        anchor = -ids.astype(float)
        history = ids.astype(float)
        score, candidate, incumbent, gain = candidate_order(anchor, history, ids)
        np.testing.assert_array_equal(candidate, np.arange(149, 139, -1))
        np.testing.assert_array_equal(incumbent, np.arange(20, 30))
        self.assertEqual(len(np.unique(score)), len(score))
        self.assertTrue(np.all(gain > 0))

    def test_threshold_disjointness_and_score_permutation(self):
        quality = np.array([[.30, .29, .0], [.4, .03, .0], [.0, -.01, .0]])
        promotion, demotion = choose_swaps(quality, np.ones_like(quality), 3, .05)
        np.testing.assert_array_equal(promotion, [1, 0])
        np.testing.assert_array_equal(demotion, [0, 1])
        score = np.array([8., 7., 6., 5., 4., 3., 2.], dtype=np.float32)
        changed = swap_scores(score, np.array([4, 3]), np.array([1, 2]))
        np.testing.assert_array_equal(changed, [8., 4., 5., 6., 7., 3., 2.])
        np.testing.assert_array_equal(np.sort(changed), np.sort(score))
        np.testing.assert_array_equal(score, [8., 7., 6., 5., 4., 3., 2.])


if __name__ == '__main__':
    unittest.main()
