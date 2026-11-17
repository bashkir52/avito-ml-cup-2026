import unittest

import torch
from torch.nn.functional import softplus

from depth_hard_ranker import mine_pairs, hard_pair_loss


class DepthPairTest(unittest.TestCase):
    def test_pairs_require_a_valid_negative_at_comparable_visible_depth(self):
        teacher = torch.tensor([[0., 10., 9., 8., 7., 6., 5., 20., 30.]])
        label = torch.tensor([[1, 0, 0, 0, 0, 0, 0, 1, 0]])
        position = torch.tensor([[20, 40, 1, 12, 20, 16, 15, 15, 18]])
        mask = torch.tensor([[True, True, True, True, True, True, True, True, False]])
        batch, pos, neg, valid, missed = mine_pairs(teacher, label, mask, position, k=2, cutoff=3)
        self.assertEqual(batch.tolist(), [0, 0])
        self.assertEqual(pos.tolist(), [0, 7])
        self.assertEqual(neg[0, valid[0]].tolist(), [3, 4])
        self.assertEqual(neg[1, valid[1]].tolist(), [3, 6])
        self.assertEqual(missed.tolist(), [True, False])

    def test_weighted_loss_gradients_only_touch_selected_pairs(self):
        logits = torch.tensor([[0., 2., 4.], [1., 2., 3.]], requires_grad=True)
        pairs = (torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([[1], [2]]),
                 torch.tensor([[True], [False]]), torch.tensor([True, False]))
        loss = hard_pair_loss(logits, pairs)
        torch.testing.assert_close(loss, softplus(torch.tensor(2.)))
        loss.backward()
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertGreater(float(logits.grad[0, 1]), 0)
        torch.testing.assert_close(logits.grad[0, 2:], torch.zeros(1))
        torch.testing.assert_close(logits.grad[1], torch.zeros(3))


if __name__ == '__main__':
    unittest.main()
