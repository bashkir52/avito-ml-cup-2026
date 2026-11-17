import unittest

import torch

from full_attention_ranker import FullAttentionRanker, attention_loss
from ranking import NAMES
from set_ranker import feed_loss


class FullAttentionTest(unittest.TestCase):
    def test_float_masks_preserve_existing_objective_and_gradient(self):
        scores = torch.tensor([[2., -.4, -3., 8.], [.2, 1., .5, 1.4]], requires_grad=True)
        mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]], dtype=torch.bool)
        labels = torch.tensor([[1, 1, 0, 1], [0, 1, 1, 0]], dtype=torch.bool)
        expected = feed_loss(scores, labels, mask, .25)
        actual = attention_loss(scores, mask, (labels & mask).float(), (~labels & mask).float())
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(torch.autograd.grad(actual, scores)[0], torch.autograd.grad(expected, scores)[0])

    def test_candidate_order_and_padding_do_not_change_real_logits(self):
        torch.manual_seed(2026)
        model = FullAttentionRanker().eval()
        candidates = torch.randn(2, 7, 2 * len(NAMES) + 128)
        user = torch.randn(2, 1, 256)
        history = torch.randn(2, 12, 141)
        valid = torch.tensor([[True] * 5 + [False] * 2, [True] * 7])
        history_mask = torch.tensor([[False] * 12, [True] * 3 + [False] * 9])
        order = torch.tensor([2, 6, 4, 0, 1, 3, 5])
        with torch.no_grad():
            scores = model(candidates, user, history, valid, history_mask)
            modified = candidates.clone()
            modified[~valid] = 50
            modified_history = history.clone()
            modified_history[~history_mask] = -50
            transformed = model(modified[:, order], user, modified_history, valid[:, order], history_mask)
        self.assertTrue(torch.isfinite(scores[valid]).all())
        torch.testing.assert_close(transformed[valid[:, order]], scores[:, order][valid[:, order]], atol=2e-6, rtol=1e-6)


if __name__ == '__main__':
    unittest.main()
