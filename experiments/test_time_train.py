import unittest

import numpy as np

from time_train import length_matched_random, select_cohorts, subset_rows


class TimeTrainTest(unittest.TestCase):
    def test_recent_calendar_window_and_size_matched_random_control(self):
        counts = np.array([2, 1, 1, 5, 5, 1, 2, 1])
        day = np.array([2, 2, 3, 4, 5, 5, 6, 7])
        recent, random = select_cohorts(counts, day, 5, 7, 2026)
        np.testing.assert_array_equal(recent, [4, 5, 6])
        np.testing.assert_array_equal(np.sort(counts[recent]), np.sort(counts[random]))
        self.assertEqual(len(np.unique(random)), len(recent))
        np.testing.assert_array_equal(random, select_cohorts(counts, day, 5, 7, 2026)[1])

    def test_subset_rows_preserve_whole_feeds_and_their_candidate_order(self):
        counts = np.array([2, 3, 1, 4])
        np.testing.assert_array_equal(subset_rows(counts, np.array([0, 2, 3])), [0, 1, 5, 6, 7, 8, 9])

    def test_matching_checks_availability(self):
        with self.assertRaises(ValueError):
            length_matched_random(np.array([1, 2]), np.array([1, 2, 1]), np.random.default_rng(42))
