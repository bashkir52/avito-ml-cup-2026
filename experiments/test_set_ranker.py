import math
import unittest

import numpy as np
import torch

from ranking import NAMES
from history_event_cache import select_events
from neural_blend import mix_from_config, resolve_ties
from set_ranker import FeedRanker, Feeds, feed_loss


class SetRankerTests(unittest.TestCase):
    def test_scores_ignore_padding_and_follow_candidate_permutation(self):
        torch.manual_seed(2026)
        model = FeedRanker(np.zeros(len(NAMES)), np.ones(len(NAMES)), 12, True).eval()
        x = torch.randn(1, 5, len(NAMES))
        x[0, 0, 0] = torch.nan
        mask = torch.as_tensor([[True, True, True, False, False]])
        with torch.inference_mode():
            base = model(x, mask)[0, :3]
            self.assertTrue(torch.isfinite(base).all())
            x[0, 3:] = 1000
            torch.testing.assert_close(base, model(x, mask)[0, :3])
            permutation = torch.as_tensor([4, 1, 2, 3, 0])
            result = model(x[:, permutation], mask[:, permutation])[0, torch.as_tensor([4, 1, 2])]
            torch.testing.assert_close(base, result)

    def test_listwise_loss_excludes_padding(self):
        score = torch.tensor([[0.0, math.log(2), 100.0]], requires_grad=True)
        labels = torch.tensor([[0, 1, 0]])
        mask = torch.tensor([[True, True, False]])
        loss = feed_loss(score, labels, mask, 0)
        self.assertAlmostEqual(float(loss.detach()), -math.log(2 / 3), places=6)
        loss.backward()
        self.assertEqual(float(score.grad[0, 2]), 0)

    def test_pairwise_loss_compares_the_positive_to_valid_negatives(self):
        score = torch.tensor([[0.0, math.log(2), 100.0]], requires_grad=True)
        labels = torch.tensor([[0, 1, 0]])
        mask = torch.tensor([[True, True, False]])
        loss = feed_loss(score, labels, mask, 0, pairwise_weight=1)
        self.assertAlmostEqual(float(loss.detach()), -math.log(2/3) + math.log(1.5), places=6)
        loss.backward()
        self.assertEqual(float(score.grad[0, 2]), 0)

    def test_batches_cover_original_candidate_indices_once(self):
        feeds = Feeds.__new__(Feeds)
        feeds.counts = np.array([4, 8, 2, 11])
        feeds.offsets = np.r_[0, np.cumsum(feeds.counts)]
        feeds.x = np.arange(feeds.offsets[-1], dtype=np.float32)[:, None]
        rows = []
        for x, _, index, mask in feeds.batches(np.array([2, 1, 3, 0]), 2, np.random.default_rng(3)):
            np.testing.assert_equal(x.numpy()[mask, 0], index[mask])
            rows.extend(index[mask].tolist())
        self.assertEqual(sorted(rows), list(range(len(feeds.x))))

    def test_vector_ranker_trains_with_missing_embedding(self):
        torch.manual_seed(2026)
        model = FeedRanker(np.zeros(len(NAMES)), np.ones(len(NAMES)), 12, True, True)
        features = torch.zeros((2, 3, len(NAMES)))
        items = torch.randint(-100, 100, (2, 3, 128), dtype=torch.int8)
        items[0, 0] = 0
        summary = torch.randn((2, 128))
        summary[1] = 0
        mask = torch.tensor([[True, True, True], [True, True, False]])
        score = model(features, mask, (items, summary, summary))
        self.assertTrue(torch.isfinite(score).all())
        feed_loss(score, torch.tensor([[0, 1, 0], [1, 0, 0]]), mask, .25).backward()
        for layer in model.vector_projections:
            self.assertGreater(float(layer.weight.grad.abs().sum()), 0)

    def test_tie_breaker_keeps_unequal_scores_in_their_original_order(self):
        primary = np.array([.6, .4, .6, .06, .06], dtype=np.float32)
        fallback = np.array([1, 500, 2, 2, 2], dtype=np.float32)
        resolved = resolve_ties(primary, fallback, np.array([3, 4, 5, 11, 10]), np.array([3, 2]))
        self.assertGreater(resolved[2], resolved[0])
        self.assertGreater(resolved[0], resolved[1])
        self.assertGreater(resolved[4], resolved[3])

    def test_feed_size_weights_cover_boundaries_and_whole_feeds(self):
        counts = np.array([30, 31, 100, 101])
        a = np.ones(counts.sum())
        b = np.zeros_like(a)
        config = {'size_boundaries': [30, 100], 'size_weights': [[.2,.8],[.5,.5],[.9,.1]]}
        result = mix_from_config([a, b], counts, config)
        np.testing.assert_allclose(result, np.repeat([.2,.5,.5,.9], counts))

    def test_history_slots_keep_original_ranks_and_never_repeat_indices(self):
        indices = np.arange(128, dtype=np.int32)
        indices[[0, 2, 8, 16, 24, 48, 96]] = -1
        chosen = select_events(indices)
        np.testing.assert_array_equal(chosen[:8], [1, 3, 4, 5, 6, 7, 9, 10])
        self.assertTrue(np.all(np.diff(chosen) > 0))
        self.assertTrue(np.all(indices[chosen] >= 0))
        self.assertIn(97, chosen)

    def test_event_attention_ignores_padded_events_and_handles_empty_history(self):
        torch.manual_seed(2026)
        model = FeedRanker(np.zeros(len(NAMES)), np.ones(len(NAMES)), 12, True, True, True).eval()
        features = torch.zeros((2, 3, len(NAMES)))
        items = torch.randint(-100, 100, (2, 3, 128), dtype=torch.int8)
        summaries = torch.rand((2, 128))
        event_vectors = torch.randint(-100, 100, (2, 2, 128), dtype=torch.int8)
        events = torch.tensor([[[0, 1, 2, 3], [-1, 0, 0, 0]], [[-1, 0, 0, 0], [-1, 0, 0, 0]]])
        valid_candidates = torch.ones((2, 3), dtype=torch.bool)
        with torch.inference_mode():
            x = model(features, valid_candidates, (items, summaries, summaries, event_vectors, events))
            self.assertTrue(torch.isfinite(x).all())
            event_vectors[0, 1] = -100
            event_vectors[1] = 100
            y = model(features, valid_candidates, (items, summaries, summaries, event_vectors, events))
            torch.testing.assert_close(x, y)


if __name__ == '__main__':
    unittest.main()
