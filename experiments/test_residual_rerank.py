import unittest

import numpy as np
import torch

from residual_rerank import ResidualRanker, forward_windows


class ResidualRerankTest(unittest.TestCase):
    def test_residual_starts_at_linear_control_and_is_bounded(self):
        coefficient = np.array([.2, -.4, 1.], dtype=np.float32)
        model = ResidualRanker(coefficient)
        x = torch.tensor([[[1., 1., .2], [2., 1., .7]]])
        base = x @ torch.from_numpy(coefficient)
        torch.testing.assert_close(model(x), base)
        with torch.no_grad():
            model.residual[-1].bias.fill_(99.)
        self.assertLessEqual((model(x) - base).abs().max().item(), 1.00001)

    def test_windows_respect_time_ties(self):
        timestamps = np.repeat(np.arange(50), 3)
        windows, early = forward_windows(timestamps)
        self.assertEqual(early, 60)
        self.assertEqual(windows[0][1], windows[1][0])
        self.assertEqual(windows[-1][1], early)
        for first, last in windows:
            self.assertLess(timestamps[first-1], timestamps[first])
            self.assertLess(timestamps[last-1], timestamps[last])
