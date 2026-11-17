"""Build a submission from the Transformer blend already selected on early val."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from depth_hard_ranker import per_feed_hits
from feed_width import paired_interval
from full_attention_ranker import ANCHOR, FullAttentionRanker, load, score_prefix
from gpu_training_probe import gpu_device
from neural_blend import read_scores, resolve_ties
from ranking import KEYS, audit_submission, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import sha256, write_json


def aligned_scores(feeds, anchor_path):
    keys = prediction_frame(feeds.keys, np.zeros(len(feeds.x), dtype=np.float32)).select(KEYS)
    return read_scores(anchor_path, keys).astype(np.float32)


def blend(feeds, anchor, transformer, weight):
    a, b = (standardized_scores(s, feeds.counts) for s in (anchor, transformer))
    return resolve_ties((1 - weight) * a + weight * b, anchor, feeds.keys['item'], feeds.counts)


def evaluate(out, model, mean, scale, device, saved_early, selected):
    val = load('val')
    early = temporal_boundary(val.timestamp, .4)
    mid = early + temporal_boundary(val.timestamp[early:], .5)
    first = temporal_boundary(val.timestamp[:early], .65)
    if (first, early, mid, len(val.counts)) != (3749, 5766, 10083, 14423):
        raise ValueError('unexpected validation split')
    cut = val.offsets[early]
    prefix = {'feed': val.keys['feed'][:early], 'item': val.keys['item'][:cut], 'counts': val.counts[:early]}
    early_keys = prediction_frame(prefix, np.zeros(cut, dtype=np.float32)).select(KEYS)
    original = read_scores(saved_early, early_keys).astype(np.float32)
    restored = score_prefix(model, val, early, mean, scale, device)
    np.testing.assert_array_equal(restored, original)
    score = score_prefix(model, val, len(val.counts), mean, scale, device)
    anchor = aligned_scores(val, ANCHOR)
    count = val.counts[:early]
    # Validate the exact batch shapes used in early selection before late evaluation.
    early_a, early_b = (standardized_scores(s, count) for s in (anchor[:cut], restored))
    weight = selected['transformer_weight']
    early_mixture = resolve_ties((1-weight)*early_a + weight*early_b, anchor[:cut], prefix['item'], count)
    early_delta = (ndcg_per_feed(val.target[:cut], early_mixture, count)
                   - ndcg_per_feed(val.target[:cut], anchor[:cut], count))
    np.testing.assert_allclose([early_delta[:first, 0].mean(), early_delta[first:, 0].mean()],
                               selected['early_block_gain'], atol=1e-12, rtol=0)
    # Match early-scoring shapes exactly in the saved val predictions.
    score[:cut] = restored
    final = blend(val, anchor, score, weight)
    np.testing.assert_array_equal(final[:cut], early_mixture)
    prediction_frame(val.keys, final).write_parquet(out / 'val_predictions.parquet')
    base = ndcg_per_feed(val.target, anchor, val.counts)
    candidate = ndcg_per_feed(val.target, final, val.counts)
    base_hits, hits = (per_feed_hits(val.target, s, val.counts, 10) for s in (anchor, final))
    delta = candidate - base
    blocks = {'late': slice(early, None), 'late_first': slice(early, mid), 'late_second': slice(mid, None)}
    metrics = {'earliest_scores_reproduced': len(restored), 'early_exact': True,
               'locked_configuration': selected, 'late_ndcg_delta_ci95': paired_interval(delta[early:, 0])}
    for name, block in blocks.items():
        metrics[name] = {'anchor': base[block].mean(axis=0).tolist(), 'blend': candidate[block].mean(axis=0).tolist(),
                         'gains': delta[block].mean(axis=0).tolist(), 'anchor_top10': int(base_hits[block].sum()),
                         'blend_top10': int(hits[block].sum())}
    write_json(out / 'assessment.json', metrics)
    print(json.dumps({'late_metrics_order': ['ndcg', 'ndcg_at_10', 'ndcg_at_30'], **metrics}), flush=True)


def run(source, out, backend):
    selection = source / 'selection.json'
    selected = json.loads(selection.read_text(encoding='utf-8'))['best']
    folder = source / f"feeds-{selected['feeds']:05d}"
    checkpoint_path = folder / 'model.pt'
    test_anchor = ANCHOR.with_name('submission-reserve.parquet')
    weight = selected['transformer_weight']
    if not 0 < weight < 1:
        raise ValueError('expected a selected nonzero blend with the submitted ensemble')
    checkpoint = torch.load(checkpoint_path, weights_only=True, map_location='cpu')
    if checkpoint['training']['feeds'] != selected['feeds']:
        raise ValueError('checkpoint does not match the selected blend')
    device, adapter = gpu_device(backend)
    model = FullAttentionRanker().to(device)
    model.load_state_dict(checkpoint['model'])
    mean, scale = checkpoint['mean'], checkpoint['scale']
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'configuration.json', {'selection': selected, 'selection_sha256': sha256(selection),
        'checkpoint': str(checkpoint_path), 'checkpoint_sha256': sha256(checkpoint_path),
        'val_anchor_sha256': sha256(ANCHOR), 'test_anchor_sha256': sha256(test_anchor),
        'adapter': adapter.rstrip('\0'), 'backend': backend, 'normalization': 'within-feed z-score'})
    evaluate(out, model, mean, scale, device, folder / 'early_predictions.parquet', selected)

    test = load('test')
    score = blend(test, aligned_scores(test, test_anchor),
                  score_prefix(model, test, len(test.counts), mean, scale, device), weight)
    pending = out / 'submission.pending.parquet'
    prediction_frame(test.keys, score).write_parquet(pending)
    audit = audit_submission(pending)
    if audit['rows'] != 2_904_806 or audit['feeds'] != 28_595 or audit['feeds_with_ties'] != 0:
        raise ValueError('submission must contain all official pairs and strict order in every feed')
    path = out / 'submission-transformer10.parquet'
    pending.rename(path)
    write_json(out / 'submission_audit.json', audit)
    print(json.dumps({'submission': str(path), **audit}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--backend', choices=('directml', 'cuda'), required=True)
    args = parser.parse_args()
    torch.set_num_threads(6)
    run(args.source, args.out, args.backend)


if __name__ == '__main__':
    main()
