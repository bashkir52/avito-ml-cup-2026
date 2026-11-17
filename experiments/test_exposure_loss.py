import unittest

import torch

from set_ranker import ExposureCurve


class ExposureLossTest(unittest.TestCase):
    def test_probability_is_anchored_and_decreases_with_position(self):
        curve = ExposureCurve()
        position = torch.tensor([0, 1, 2, 4, 8, 16, 32, 64, 128, 256])
        observed = curve.log_probability(torch.zeros_like(position, dtype=torch.float32), position).exp()
        self.assertAlmostEqual(observed[0].item(), .5)
        self.assertTrue(torch.all(observed[1:] < observed[:-1]))

    def test_padded_rows_do_not_affect_loss_or_gradients(self):
        curve = ExposureCurve()
        score = torch.tensor([[2., -1., 90.]], requires_grad=True)
        label = torch.tensor([[1, 0, 1]])
        position = torch.tensor([[0, 8, 300]])
        mask = torch.tensor([[True, True, False]])
        loss = curve.loss(score, label, mask, position)
        self.assertTrue(torch.isfinite(loss))
        torch.testing.assert_close(loss, curve.loss(score[:, :2], label[:, :2], mask[:, :2], position[:, :2]))
        loss.backward()
        self.assertEqual(score.grad[0, 2].item(), 0)
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertTrue(torch.isfinite(curve.steps.grad).all())
