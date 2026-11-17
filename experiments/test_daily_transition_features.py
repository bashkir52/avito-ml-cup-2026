import unittest

import numpy as np

from daily_transition_features import add_daily_pairs, aligned_rows, prior_days
from item_priors import DAY, START


class DailyGraphTest(unittest.TestCase):
    def test_complete_day_boundary_and_inference_prefix(self):
        timestamp = START + np.array([0, DAY-1, DAY, 14*DAY])
        np.testing.assert_array_equal(prior_days(timestamp, 'train'), [0, 0, 1, 14])
        np.testing.assert_array_equal(prior_days(START + np.array([16*DAY]), 'val'), [15])
        with self.assertRaises(ValueError):
            prior_days(timestamp, 'test')

    def test_day_prefix_excludes_same_day_positive_and_future_rows(self):
        graph = np.zeros((4, 3, 3))
        history = np.array([[0, -1], [0, -1], [1, -1]])
        add_daily_pairs(graph, history, np.array([2, 2, 2]), np.arange(3), [0, 1, 2], np.array([5., 7., 9.]))
        prefix = graph.cumsum(axis=0)
        self.assertEqual(prefix[0, 0, 2], 0)
        self.assertEqual(prefix[1, 0, 2], 5)
        self.assertEqual(prefix[2, 0, 2], 12)
        self.assertEqual(prefix[2, 1, 2], 0)
        self.assertEqual(prefix[3, 1, 2], 9)

    def test_alignment_accounts_for_train_candidate_shuffle_and_feed_order(self):
        raw_items = np.array([8, 6, 6, 9, 3])
        candidate_offsets = np.array([0, 2, 5])
        cache_items = np.array([9, 3, 6, 6, 8])
        actual = aligned_rows(raw_items, candidate_offsets, cache_items, np.array([3, 0]))
        np.testing.assert_array_equal(actual, [4, 3, 2, 0, 1])
        with self.assertRaises(ValueError):
            aligned_rows(raw_items, candidate_offsets, np.array([9, 3, 3, 6, 8]), np.array([3, 0]))
