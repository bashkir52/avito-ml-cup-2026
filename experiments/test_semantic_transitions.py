import unittest

import numpy as np
import pyarrow as pa

from semantic_transitions import (assign_codes, joint_counts, recent_events, sample_rows, transition_signal)
from semantic_transition_rerank import rank


class SemanticTransitionTest(unittest.TestCase):
    def test_projection_respects_missing_vectors_and_is_scale_invariant(self):
        raw = np.array([[0, 0], [2, -1], [4, -2], [-1, 1]])
        actual = assign_codes(raw, np.eye(2), np.zeros(2))
        np.testing.assert_array_equal(actual, [-1, 1, 1, 2])

    def test_recent_events_preserve_order_and_pad_with_missing(self):
        source = pa.array([[4, -1, 3, 5], [], [9]], type=pa.list_(pa.int32()))
        np.testing.assert_array_equal(recent_events(source, 3), [[4, -1, 3], [-1, -1, -1], [9, -1, -1]])

    def test_sample_weights_conserve_feed_sizes(self):
        offsets = np.array([0, 2, 20, 21, 42])
        rows, feed, weights = sample_rows(offsets, 4, np.random.default_rng(3))
        self.assertEqual(len(rows), len(np.unique(rows)))
        np.testing.assert_array_equal(feed, np.searchsorted(offsets[1:], rows, side='right'))
        np.testing.assert_allclose(np.bincount(feed, weights), np.diff(offsets))

    def test_directed_counts_ignore_missing_codes_and_use_feed_boundaries(self):
        history = np.array([[0, 1, -1], [1, -1, -1], [-1, -1, -1]])
        destinations = np.array([2, -1, 0, 0])
        feed = np.array([0, 0, 1, 2])
        actual = joint_counts(history, destinations, feed, np.array([2., 10., 5., 9.]), 3)
        np.testing.assert_array_equal(actual, [[0, 0, 2], [5, 0, 2], [0, 0, 0]])

    def test_transition_signal_respects_direction_missing_and_support(self):
        history = np.array([[0, 1], [2, -1]])
        dest = np.array([1, -1, 1, 2])
        counts = np.array([2, 2])
        grid = np.array([[0, 1, 0], [5, 2, 0], [0, 3, 4]], dtype=np.float32)
        support = grid > 0
        signal, covered = transition_signal(history, dest, counts, grid, support)
        recency = 2**(-1/6)
        np.testing.assert_allclose(signal, [(1+2*recency)/(1+recency), 0, 3, 4])
        np.testing.assert_array_equal(covered, [True, False, True, True])

    def test_long_feed_variant_changes_order_only_above_100_candidates(self):
        counts = np.array([100, 101])
        meta = {'item': np.arange(counts.sum()), 'counts': counts}
        baseline = -np.arange(counts.sum(), dtype=np.float32)
        signal = -baseline
        score = rank(meta, baseline, signal, 2., 'long')
        self.assertTrue(np.all(np.diff(score[:100]) < 0))
        self.assertTrue(np.all(np.diff(score[100:]) > 0))
