import unittest

import numpy as np
import pyarrow as pa
import torch

from sequence_ranker import (SequenceEncoder, gather_vectors, history_window,
                             mask_duplicate_targets, novel_target_mask,
                             position_band_loss, read_history)


class SequenceRankerTest(unittest.TestCase):
    def test_arrow_slice_retains_parallel_fields(self):
        batch = pa.record_batch([[99, 10], [[1], [5, 3, 2]], [[7], [9, -1, 4]], [[8], [4, 6, 1]]],
                                names=['feed_id', 'history__item_id',
                                       'history__item_id__embedding_idx', 'history__event_type'])
        ids, indices, events, lengths = read_history(batch.slice(1), 2)
        np.testing.assert_array_equal(ids, [[5, 3]])
        np.testing.assert_array_equal(indices, [[9, -1]])
        np.testing.assert_array_equal(events, [[4, 6]])
        np.testing.assert_array_equal(lengths, [2])

    def test_context_is_older_and_fields_follow_chronology(self):
        indices = np.array([[6, 5, -1, 3], [8, 7, -1, -1], [-1, -1, -1, -1]])
        events = np.array([[3, 2, 4, 1], [4, 2, 0, 0], [0, 0, 0, 0]])
        idx, event, length = history_window(indices, events, np.array([4, 2, 0]),
                                             np.array([2, 1, 0]), 3)
        np.testing.assert_array_equal(idx, [[3, -1, -1], [7, -1, -1], [-1, -1, -1]])
        np.testing.assert_array_equal(event, [[2, 5, 0], [3, 0, 0], [0, 0, 0]])
        np.testing.assert_array_equal(length, [2, 1, 0])

    def test_novelty_checks_only_older_events_beyond_encoder_window(self):
        items = pa.array([[9], [4, 3, 4, 2, *range(10, 30), 2], [8, 0]]).slice(1)
        targets = np.array([[4, 3, 4, 2], [8, 0, 0, 0]])
        np.testing.assert_array_equal(novel_target_mask(items, targets),
                                      [[False, True, True, False], [True, True, True, True]])

    def test_duplicate_negatives_are_masked_but_own_target_survives(self):
        scores = mask_duplicate_targets(torch.zeros(3, 3), torch.tensor([4, 7, 4]))
        self.assertTrue(torch.equal(scores.diag(), torch.zeros(3)))
        self.assertTrue(torch.isneginf(scores[0, 2]))
        self.assertTrue(torch.isneginf(scores[2, 0]))
        self.assertEqual(scores[1, 2], 0)

    def test_missing_embedding_does_not_read_last_table_row(self):
        table = np.array([[4, 5], [7, 8]], dtype=np.int8)
        actual = gather_vectors(table, np.array([[1, -1, 0]]))
        self.assertTrue(torch.equal(actual, torch.tensor([[[7, 8], [0, 0], [4, 5]]], dtype=torch.int8)))

    def test_encoder_ignores_trailing_padding_and_handles_empty_history(self):
        torch.manual_seed(99)
        encoder = SequenceEncoder(8)
        vectors = torch.randint(-16, 16, (2, 3, 128), dtype=torch.int8)
        events = torch.tensor([[4, 3, 0], [0, 0, 0]])
        lengths = torch.tensor([2, 0])
        encoded = encoder.history(vectors, events, lengths)
        self.assertTrue(torch.allclose(encoded, encoder.history(vectors[:, :2], events[:, :2], lengths)))
        self.assertTrue(torch.equal(encoded[1], torch.zeros(8)))

    def test_position_band_loss_uses_only_comparable_candidates(self):
        score = torch.tensor([[2., 1., 50., 60., 100.], [3., 4., 5., 6., 7.]], requires_grad=True)
        label = torch.tensor([[1, 0, 1, 0, 0], [1, 0, 1, 0, 0]])
        position = torch.tensor([[8, 10, 32, 100, 10], [0, 4, 8, 99, 99]])
        mask = torch.tensor([[True, True, True, True, False], [True, True, True, False, False]])
        loss, eligible = position_band_loss(score, label, mask, position)
        self.assertTrue(torch.equal(eligible, torch.tensor([True, False])))
        torch.testing.assert_close(loss, torch.tensor([0.3132617, 0.]))
        loss.sum().backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertTrue(torch.equal(score.grad[0, 2:], torch.zeros(3)))
        self.assertTrue(torch.equal(score.grad[1], torch.zeros(5)))
