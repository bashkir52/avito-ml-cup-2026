import unittest

import numpy as np

from causal_exposure import exposure_features


class CausalExposureTest(unittest.TestCase):
    def test_excludes_current_time_and_same_users(self):
        timestamps = np.array([1, 1, 2, 3])
        user = np.array([1, 2, 1, 3])
        vector = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=np.float32)
        counts = np.array([2, 2, 2, 2])
        item = np.array([7, 8, 7, 9, 7, 8, 7, 8])
        support, affinity = exposure_features(timestamps, counts, item, user, vector)
        np.testing.assert_array_equal(support, [0, 0, 0, 0, 1, 0, 3, 2])
        np.testing.assert_allclose(affinity[:6], 0)
        np.testing.assert_allclose(affinity[6:], [0., -1./9], atol=1e-7)
        prefix = exposure_features(timestamps[:2], counts[:2], item[:4], user[:2], vector[:2])
        np.testing.assert_array_equal(prefix[0], support[:4])
        np.testing.assert_array_equal(prefix[1], affinity[:4])


if __name__ == '__main__':
    unittest.main()
