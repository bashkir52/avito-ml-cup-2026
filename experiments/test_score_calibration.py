import unittest

import numpy as np
from scipy.optimize import check_grad

from score_calibration import (grouped_objective, negative_rank_sample, pairwise_objective,
                               smooth_ndcg_objective, within_feed_z)


class CalibrationTests(unittest.TestCase):
    def test_grouped_softmax_loss_and_analytic_gradient(self):
        x = np.array([[0, 1], [1, 0], [1, 1], [1, -1], [0, 0]], dtype=np.float64)
        counts = np.array([2, 3])
        labels = np.array([0, 1, .5, .5, 0])
        center = np.zeros(2)
        loss, _ = grouped_objective(center, x, labels, counts, .1, center)
        self.assertAlmostEqual(loss, (np.log(2) + np.log(3)) / 2)

        def objective(w):
            return grouped_objective(w, x, labels, counts, .1, center)

        error = check_grad(lambda w: objective(w)[0], lambda w: objective(w)[1], [.1, -.4])
        self.assertLess(error, 1e-6)

    def test_signal_scaling_is_independent_in_each_feed(self):
        x = np.array([2, 4, np.nan, 12, 12, np.nan])
        z = within_feed_z(x, np.array([3, 3]))
        np.testing.assert_allclose(z, [-1, 1, 0, 0, 0, 0])

    def test_negative_sampling_preserves_mass_across_the_ranking(self):
        sample, mass = negative_rank_sample(np.arange(295), 'stratified')
        self.assertEqual(int(mass.sum()), 295)
        self.assertEqual(len(sample), 32)
        np.testing.assert_array_equal(sample[:8], np.arange(8))
        self.assertGreater(sample[-1], 285)

    def test_pairwise_gradients_favor_positive_margin(self):
        x = np.array([[2, -.3], [.4, 1.]])
        mass = np.array([.7, .3])
        center = np.zeros(2)
        loss, gradient = pairwise_objective(center, x, mass, .1, center)
        self.assertAlmostEqual(loss, np.log(2))
        self.assertLess(gradient[0], 0)
        objective = lambda w: pairwise_objective(w, x, mass, .1, center)
        self.assertLess(check_grad(lambda w: objective(w)[0], lambda w: objective(w)[1], [.1, -.4]), 1e-6)

    def test_smooth_ndcg_gradient_matches_finite_differences(self):
        differences = np.array([[-1, 1], [.5, .1], [.3, 2.]])
        data = (differences, np.array([1, 3, 1]), np.array([2, 1]), np.array([.75, .25]))
        center = np.zeros(2)
        objective = lambda w: smooth_ndcg_objective(w, data, .55, .1, center)
        self.assertLess(check_grad(lambda w: objective(w)[0], lambda w: objective(w)[1], [.1, -.4]), 1e-6)


if __name__ == '__main__':
    unittest.main()
