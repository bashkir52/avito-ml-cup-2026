import unittest

import numpy as np
import pyarrow as pa

from transition_probe import ITEM_BASE, best_transition, recent_items


class TransitionProbeTest(unittest.TestCase):
    def test_recent_events_are_truncated_before_item_deduplication(self):
        source = pa.array([[3, 3, *range(10, 21)], [], [4, 4, 5]], type=pa.list_(pa.int32()))
        history = recent_items(source)
        self.assertEqual(set(history[0]) - {0}, {3, *range(10, 20)})
        self.assertEqual(np.count_nonzero(history[0]), 11)
        self.assertEqual(np.count_nonzero(history[1]), 0)
        self.assertEqual(np.count_nonzero(history[2]), 2)

    def test_each_candidate_gets_the_strongest_directed_edge_in_its_own_feed(self):
        history = np.array([[1, 2], [4, 0]])
        items = np.array([7, 8, 7, 8], dtype=np.int32)
        keys = np.array([a * ITEM_BASE + b for a, b in [(1, 7), (2, 7), (4, 8), (8, 4)]])
        score = best_transition(history, items, np.array([2, 2]), keys, np.array([.1, .3, .2, .4]))
        np.testing.assert_allclose(score, [.3, 0, 0, .2])
