import unittest

import numpy as np

from top_rank_fusion import combine, transform


class TopRankFusionTests(unittest.TestCase):
    def test_reciprocal_weights_favor_earlier_ranks(self):
        scores = np.array([.5, 1, .25, .75], dtype=np.float32)
        observed = transform(scores, np.array([4]), 8)
        np.testing.assert_allclose(observed, 1 / np.array([11, 9, 12, 10]))
        self.assertGreater(observed[1] - observed[3], observed[0] - observed[2])

    def test_pure_reference_weights_preserve_reference_order(self):
        scores = np.array([.5, 1, .25, .75], dtype=np.float32)
        others = scores[::-1]
        result = combine([scores, others, others], np.array([4]), np.arange(4), [[1,0,0]]*3, 8)
        np.testing.assert_array_equal(scores, result)


if __name__ == '__main__':
    unittest.main()
