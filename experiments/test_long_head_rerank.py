import unittest

import numpy as np

from long_head_rerank import WIDTH, head_ndcg_gain, order_within_head, rewrite_head, select_head
from ranking import ndcg_per_feed


class LongHeadTests(unittest.TestCase):
    def test_only_the_top_30_of_feeds_with_at_least_101_items_move(self):
        counts = np.array([100, 101, 165])
        baseline = np.concatenate([1 - np.arange(n, dtype=np.float32) / n for n in counts])
        feeds, head = select_head(baseline, counts)
        np.testing.assert_array_equal(feeds, [1, 2])
        order = np.tile(np.arange(WIDTH-1,-1,-1), (len(head),1))
        score = rewrite_head(baseline, head, order)
        excluded = np.ones(len(score), dtype=bool)
        excluded[head.ravel()] = False
        np.testing.assert_array_equal(score[excluded], baseline[excluded])
        np.testing.assert_array_equal(score[head], baseline[head][:, ::-1])
        self.assertEqual(set(score[100:201]), set(baseline[100:201]))

    def test_head_delta_uses_the_full_feed_ideal_ranking(self):
        counts = np.array([101, 130])
        ref = np.concatenate([1 - np.arange(n, dtype=np.float32) / n for n in counts])
        _, head = select_head(ref, counts)
        y = np.zeros(len(ref), dtype=np.int8)
        y[[10, 35, 90, 101 + 50]] = 1
        order = np.tile(np.arange(WIDTH-1,-1,-1), (len(head),1))
        score = rewrite_head(ref, head, order)
        actual = ndcg_per_feed(y, score, counts)[:,0] - ndcg_per_feed(y, ref, counts)[:,0]
        target = head_ndcg_gain(y[head], np.array([3,1]), order)
        np.testing.assert_allclose(actual, target, atol=1e-12)

    def test_conservative_order_honors_both_limits(self):
        matrix = np.array([[list(-np.arange(WIDTH)), list(np.arange(WIDTH))]]).transpose(0,2,1)
        result = order_within_head(matrix, np.array([0.,1.]), 1., locked=3, movement=5)[0]
        np.testing.assert_array_equal(result[:3], [0,1,2])
        np.testing.assert_array_equal(np.sort(result), np.arange(WIDTH))
        self.assertLessEqual(int(np.abs(result-np.arange(WIDTH)).max()), 5)


if __name__ == '__main__':
    unittest.main()
