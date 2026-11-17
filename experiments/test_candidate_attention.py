from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import torch

from ranking import NAMES
from sequence_ranker import SequenceRanker, load_ranker
from set_ranker import CandidateGroupAttention


class CandidateAttentionTest(unittest.TestCase):
    def test_specific_attention_is_permutation_equivariant_and_ignores_padding(self):
        torch.manual_seed(2026)
        model = CandidateGroupAttention(12, 4, candidate_specific=True)
        x = torch.randn(1, 5, 12)
        mask = torch.tensor([[True, True, True, True, False]])
        self.assertTrue(torch.equal(model(x, mask), torch.zeros_like(x)))
        with torch.no_grad():
            model.output.weight.normal_()
        expected = model(x, mask)
        order = torch.tensor([4, 2, 0, 3, 1])
        torch.testing.assert_close(model(x[:, order], mask[:, order]), expected[:, order])
        x[:, 4] = float('nan')
        torch.testing.assert_close(model(x, mask), expected)

    def test_shared_control_uses_one_context_for_all_candidates_with_same_parameters(self):
        torch.manual_seed(2026)
        specific = CandidateGroupAttention(12, 4, candidate_specific=True)
        with torch.no_grad():
            specific.output.weight.normal_()
        shared = deepcopy(specific)
        shared.candidate_specific = False
        x = torch.randn(2, 6, 12)
        mask = torch.ones(2, 6, dtype=torch.bool)
        self.assertEqual(shared(x, mask).shape, (2, 1, 12))
        self.assertEqual(specific(x, mask).shape, (2, 6, 12))
        self.assertFalse(torch.allclose(specific(x, mask), shared(x, mask).expand(-1, 6, -1)))

    def test_restore_saved_checkpoint_preserves_candidate_specific_attention(self):
        mean = np.zeros(len(NAMES), dtype=np.float32)
        model = SequenceRanker(mean, np.ones_like(mean), 12, 8, candidate_slots=4, candidate_specific=True)
        with TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            torch.save({'model': model.state_dict(), 'base_width': 12, 'seq_width': 8,
                        'candidate_slots': 4, 'candidate_specific': True}, path)
            restored = load_ranker(path)
        self.assertTrue(restored.candidate_attention.candidate_specific)
        torch.testing.assert_close(restored.state_dict(), model.state_dict())
