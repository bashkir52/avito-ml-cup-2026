import unittest
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from ranking import NAMES
from sequence_ranker import SequenceRanker
from set_ranker import feed_loss
from randomized_adaptation import AdaptedRanker, anchor_penalty, checked_windows, load_adapted, save_model


class RandomizedAdaptationTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.base = SequenceRanker(np.zeros(len(NAMES)), np.ones(len(NAMES)), 8, 8).eval()
        self.args = (torch.randn(2, 5, len(NAMES)), torch.ones(2, 5, dtype=torch.bool),
                     (torch.randn(2, 5, 128), torch.randn(2, 128), torch.randn(2, 128)),
                     (torch.randn(2, 3, 128), torch.ones(2, 3, dtype=torch.long), torch.tensor([3, 2])))
        self.y = torch.tensor([[1, 0, 0, 0, 0], [0, 1, 0, 1, 0]])

    def test_initial_scores_and_depth_head_isolation(self):
        model = AdaptedRanker(deepcopy(self.base)).eval()
        expected = self.base(*self.args)
        initial = model(*self.args)
        # Changing requires_grad on the frozen backbone can change CPU rounding.
        torch.testing.assert_close(initial, expected, rtol=1e-6, atol=1e-7)
        pos = torch.tensor([[0, 2, 4, 128, 300]] * 2)
        torch.testing.assert_close(model.train_scores(*self.args, pos), expected)
        with torch.no_grad():
            model.depth.weight[:, 0] = torch.arange(9)
        torch.testing.assert_close(model.train_scores(*self.args, pos) - expected,
                                   torch.tensor([[0., 1., 2., 7., 8.]] * 2))
        torch.testing.assert_close(model(*self.args), initial, rtol=0, atol=0)

    def test_auxiliary_gradient_reaches_shared_hidden_but_not_random_head(self):
        model = AdaptedRanker(deepcopy(self.base)).eval()
        loss = feed_loss(model.train_scores(*self.args, torch.zeros(2, 5, dtype=torch.long)),
                         self.y, self.args[1], .25)
        loss.backward()
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.ranker.local.parameters()), 0)
        for name, p in model.ranker.named_parameters():
            if not name.startswith(('local.', 'output.')):
                self.assertFalse(p.requires_grad)
                self.assertIsNone(p.grad)
        for p in model.ranker.output.parameters():
            self.assertTrue(p.grad is None or torch.count_nonzero(p.grad) == 0)

    def test_anchor_and_zero_aux_weight_matched_update(self):
        models = [AdaptedRanker(deepcopy(self.base)).train() for _ in range(2)]
        origin = {n: p.detach().clone() for n, p in models[0].shared_parameters()}
        self.assertEqual(float(anchor_penalty(models[0], origin)), 0)
        for i, model in enumerate(models):
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4, weight_decay=0)
            torch.manual_seed(100)
            loss = feed_loss(model(*self.args), self.y, self.args[1], .25)
            if i:
                loss = loss + 0 * feed_loss(model.train_scores(*self.args, torch.zeros(2, 5, dtype=torch.long)), self.y, self.args[1], .25)
            loss.backward()
            opt.step()
        for (_, a), (_, b) in zip(models[0].shared_parameters(), models[1].shared_parameters()):
            torch.testing.assert_close(a, b, atol=0, rtol=0)
        self.assertGreater(float(anchor_penalty(models[0], origin)), 0)

    def test_forward_boundaries_are_checked(self):
        with self.assertRaises(ValueError):
            checked_windows(np.arange(100))

    def test_checkpoint_restores_freezing_and_exact_inference(self):
        model = AdaptedRanker(deepcopy(self.base)).eval()
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / 'model.pt'
            save_model(path, model, {'base_width': 8, 'seq_width': 8}, {'steps': 0})
            restored = load_adapted(path).eval()
        for (a_name, a), (b_name, b) in zip(model.named_parameters(), restored.named_parameters()):
            self.assertEqual(a_name, b_name)
            self.assertEqual(a.requires_grad, b.requires_grad)
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        with torch.inference_mode():
            torch.testing.assert_close(restored(*self.args), model(*self.args), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
