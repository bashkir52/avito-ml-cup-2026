import unittest

import numpy as np

from long_feed_calibration import apply_to_long_feeds, long_feeds


class LongFeedCalibrationTest(unittest.TestCase):
    def test_only_feeds_in_explicit_range_are_modified(self):
        counts = np.array([3, 101, 301])
        meta = {'counts': counts, 'item': np.arange(counts.sum())}
        anchor = np.concatenate([1 - np.arange(n, dtype=np.float32) / n for n in counts])
        raw = {'submitted': anchor, 'global': 1 - anchor, 'specialist': 1 - anchor}
        self.assertTrue(np.array_equal(long_feeds(counts), [False, True, False]))
        np.testing.assert_array_equal(apply_to_long_feeds(meta, raw, 'specialist', 0.), anchor)
        scores = apply_to_long_feeds(meta, raw, 'specialist', 1.)
        np.testing.assert_array_equal(scores[:3], anchor[:3])
        np.testing.assert_array_equal(scores[104:], anchor[104:])
        self.assertEqual(scores[103], 1)
        self.assertEqual(len(np.unique(scores[3:104])), 101)

    def test_union_can_promote_a_candidate_outside_incumbent_head(self):
        n = 101
        anchor = 1 - np.arange(n, dtype=np.float32) / n
        meta = {'counts': np.array([n]), 'item': np.arange(n)}
        raw = {'submitted': anchor, 'global': 1 - anchor, 'specialist': 1 - anchor}
        scores = apply_to_long_feeds(meta, raw, 'global', 1., head_union=20)
        self.assertEqual(scores[-1], 1.)
        np.testing.assert_array_equal(scores[20:-20], anchor[20:-20])
        np.testing.assert_array_equal(np.sort(scores), np.sort(anchor))
