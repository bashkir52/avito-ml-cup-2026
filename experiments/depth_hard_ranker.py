"""Train on complete feeds with fixed-teacher hard negatives at comparable logged depth."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.nn import functional as F

from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores, resolve_ties
from randomized_adaptation import load_feeds, memory
from ranking import KEYS, audit_submission, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import load_ranker, score_sequences, sha256, write_json
from set_ranker import feed_loss


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'sequence-finetune-048/pretrained/model.pt'
ANCHOR = ROOT / 'nn-calibration-joint-024'
CACHE = ROOT / 'cache-intent-024'
ARMS = ('control', 'depth_hard')
WEIGHTS = (.025, .05, .1, .2, .35, .5)
STAGES = (8192, 36000, 108000, 216000, 295376)
BATCH_SIZE = 96
PAIR_WEIGHT = .5
LR = .0003


@torch.no_grad()
def mine_pairs(teacher, label, mask, position, k=4, cutoff=30):
    """Mine same-feed negatives shown at least as high and within a factor two of depth+2."""
    batch, positive = (label.bool() & mask).nonzero(as_tuple=True)
    observed = position[batch]
    p = observed.gather(1, positive[:, None])
    allowed = (mask[batch] & ~label[batch].bool() & (observed <= p)
               & (2 * (observed + 2) >= p + 2))
    scores = teacher[batch]
    reference_score = scores.gather(1, positive[:, None])
    # Pessimistic ties, matching the competition's NDCG convention.
    rank = (((scores > reference_score) | ((scores == reference_score) & ~label[batch].bool()))
            & mask[batch]).sum(dim=1)
    largest, negative = scores.masked_fill(~allowed, -torch.inf).topk(min(k, scores.shape[1]), dim=1)
    return batch, positive, negative, largest.isfinite(), rank >= cutoff


def hard_pair_loss(score, pairs):
    batch, positive, negative, valid, missed = pairs
    gap = score[batch[:, None], negative] - score[batch, positive, None]
    per_positive = (F.softplus(gap) * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
    active = valid.any(dim=1)
    loss_sum = score.new_zeros(len(score)).scatter_add_(0, batch, per_positive * (1 + missed))
    comparisons = score.new_zeros(len(score)).scatter_add_(0, batch, active.to(score.dtype))
    return (loss_sum / comparisons.clamp_min(1)).mean()


def verify_source(teacher, val, out):
    score = score_sequences(teacher, val, BATCH_SIZE)
    keys = prediction_frame(val.keys, score).select(KEYS)
    original = read_scores(SOURCE.parent / 'val_predictions.parquet', keys).astype(np.float32)
    np.testing.assert_array_equal(score, original)
    prediction_frame(val.keys, score).write_parquet(out / 'frozen_val_predictions.parquet')
    return score


def train(out, seed, threads):
    out.mkdir(parents=True, exist_ok=False)
    teacher = load_ranker(SOURCE).eval()
    val = load_feeds('val')
    verified = verify_source(teacher, val, out)
    print(f'Reproduced {len(verified):,} original predictions exactly', flush=True)
    train_feeds = load_feeds('train')
    if len(train_feeds.counts) != STAGES[-1] or train_feeds.timestamp.max() >= val.timestamp.min():
        raise ValueError('train cache size or timestamps differ from experiment specification')
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_feeds.counts))
    np.save(out / 'train_feed_indices.npy', order)
    models = {name: deepcopy(teacher).train() for name in ARMS}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=.03)
                  for name, model in models.items()}
    source_digest = sha256(SOURCE)
    source = torch.load(SOURCE, map_location='cpu', weights_only=True)
    config = {'seed': seed, 'threads': threads, 'source': str(SOURCE), 'source_sha256': source_digest,
              'batch_size_max': BATCH_SIZE, 'train_feeds': len(train_feeds.counts),
              'train_candidates': len(train_feeds.x), 'stages': STAGES, 'lr_max': LR, 'lr_min': LR * .3,
              'pointwise_weight': .25, 'pair_weight': PAIR_WEIGHT, 'weight_decay': .03,
              'gradient_clip': 5, 'mining_teacher_top_candidates': 4, 'missed_positive_weight': 2,
              'missed_positive_cutoff': 30, 'log_depth_offset': 2, 'max_depth_ratio': 2,
              'negative_no_deeper_than_positive': True, 'late_labels_used_for_training_or_selection': False}
    write_json(out / 'config.json', config)
    start = time.perf_counter()
    progress, previous = [], 0
    stats = {'steps': 0, 'eligible_feeds': 0, 'eligible_positives': 0, 'pairs': 0,
             'eligible_teacher_missed_positives': 0, 'teacher_missed_positives': 0, 'positives': 0, 'candidates': 0}
    for end in STAGES:
        accumulated = dict.fromkeys(('control', 'depth_hard', 'pair'), 0.)
        stage_start = time.perf_counter()
        seen = previous
        sample = order[previous:end]
        for x, mask, rows, _ in train_feeds.batches(sample, BATCH_SIZE, rng):
            y = torch.from_numpy(train_feeds.target[rows])
            position = torch.from_numpy(train_feeds.position[rows])
            vectors, history = train_feeds.vector_batch(rows), train_feeds.history_batch(rows)
            with torch.no_grad():
                scores = teacher(x, mask, vectors, history)
                pairs = mine_pairs(scores, y, mask, position)
                batch, _, _, valid, missed = pairs
                eligible = valid.any(dim=1)
                stats['eligible_feeds'] += int(batch[eligible].unique().numel())
                stats['eligible_positives'] += int(eligible.sum())
                stats['pairs'] += int(valid.sum())
                stats['eligible_teacher_missed_positives'] += int((eligible & missed).sum())
                stats['teacher_missed_positives'] += int(missed.sum())
                stats['positives'] += int(batch.numel())
                stats['candidates'] += int(mask.sum())
            lr = LR * (.3 + .7 * (1 + math.cos(math.pi * seen / STAGES[-1])) / 2)
            dropout_state = torch.get_rng_state()
            for name, model in models.items():
                torch.set_rng_state(dropout_state)
                model.train()
                optimizer = optimizers[name]
                optimizer.param_groups[0]['lr'] = lr
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, mask, vectors, history)
                loss = feed_loss(logits, y, mask, pointwise_weight=.25)
                accumulated[name] += float(loss.detach()) * len(x)
                if name == 'depth_hard':
                    pair_loss = hard_pair_loss(logits, pairs)
                    accumulated['pair'] += float(pair_loss.detach()) * len(x)
                    loss = loss + PAIR_WEIGHT * pair_loss
                if not torch.isfinite(loss):
                    raise RuntimeError('nonfinite training loss')
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
            stats['steps'] += 1
            seen += len(x)
        assert seen == end
        record = {'feeds': end, **stats, 'stage_seconds': time.perf_counter() - stage_start,
                  'cumulative_seconds': time.perf_counter() - start, 'last_lr': lr,
                  'stage_loss': {name: value / len(sample) for name, value in accumulated.items()},
                  'memory': memory()}
        print(json.dumps(record), flush=True)
        folder = out / f'feeds-{end}'
        folder.mkdir()
        for name, model in models.items():
            path = folder / f'{name}.pt'
            torch.save({**source, 'model': model.state_dict(), 'run_config': config, 'train_feeds_seen': end}, path)
            # Score the serialized weights through the same loader used for test.
            score = score_sequences(load_ranker(path), val, BATCH_SIZE)
            prediction_frame(val.keys, score).write_parquet(folder / f'{name}_val_predictions.parquet')
        progress.append(record)
        write_json(out / 'training.json', progress)
        previous = end
    assert len(np.unique(order)) == len(order)
    if stats['candidates'] != len(train_feeds.x) or stats['positives'] != int(train_feeds.target.sum()):
        raise ValueError('training did not visit all candidate rows exactly once')


def blend(anchor, candidate, item, counts, weight):
    if not weight:
        return anchor
    score = (1 - weight) * standardized_scores(anchor, counts) + weight * standardized_scores(candidate, counts)
    return resolve_ties(score, anchor, item, counts)


def per_feed_hits(labels, scores, counts, cutoff):
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    hits = np.empty(len(counts), dtype=np.int32)
    for i, (a, b) in enumerate(zip(offsets[:-1], offsets[1:])):
        order = np.lexsort((labels[a:b], -scores[a:b]))
        hits[i] = labels[a:b][order[:cutoff]].sum()
    return hits


def assess(out):
    if (out / 'assessment.json').exists() or (out / 'selection.json').exists():
        raise FileExistsError('the selection or assessment has already been recorded')
    val = read_meta(CACHE / 'val.npz')
    counts, labels, ts, items = (val[k] for k in ('counts', 'target', 'timestamp', 'item'))
    early = temporal_boundary(ts, .4)
    first = temporal_boundary(ts[:early], .65)
    mid = early + temporal_boundary(ts[early:], .5)
    if (first, early, mid, len(counts)) != (3749, 5766, 10083, 14423):
        raise ValueError('validation boundaries differ from expected timestamps')
    cut = int(counts[:early].sum())
    keys = prediction_frame(val, np.zeros(len(items))).select(KEYS)
    raw = {name: read_scores(path, keys).astype(np.float32) for name, path in {
        'submitted': ANCHOR / 'val_predictions.parquet', 'frozen': out / 'frozen_val_predictions.parquet'}.items()}
    for stage in STAGES:
        for name in ARMS:
            path = out / f'feeds-{stage}/{name}_val_predictions.parquet'
            raw[f'{name}_{stage}'] = read_scores(path, keys).astype(np.float32)
    early_base = {name: ndcg_per_feed(labels[:cut], raw[name][:cut], counts[:early])[:, 0]
                  for name in ('submitted', 'frozen')}

    def selection_gain(score, baseline):
        metric = ndcg_per_feed(labels[:cut], score, counts[:early])[:, 0]
        delta = metric - early_base[baseline]
        return {'early_gain': float(delta.mean()), 'block_gain': [float(delta[:first].mean()), float(delta[first:].mean())]}

    def criterion(config):
        return min(config['block_gain']), config['early_gain']

    stages = [{'feeds': stage, **{name: selection_gain(raw[f'{name}_{stage}'][:cut], 'frozen')
                                 for name in ARMS}} for stage in STAGES]
    common = max(stages, key=lambda row: criterion(row['depth_hard']))['feeds']
    standalone_by_arm = {name: max(stages, key=lambda row: criterion(row[name]))['feeds'] for name in ARMS}
    blend_grid = []
    for name in ['frozen', *(f'{arm}_{stage}' for arm in ARMS for stage in STAGES)]:
        for weight in WEIGHTS:
            score = blend(raw['submitted'][:cut], raw[name][:cut], items[:cut], counts[:early], weight)
            blend_grid.append({'model': name, 'weight': weight, **selection_gain(score, 'submitted')})
    best_blend = max(blend_grid, key=criterion)
    chosen = best_blend if min(best_blend['block_gain']) > 0 else {'model': 'submitted', 'weight': 0.}
    selection = {'common_feeds': common, 'standalone_by_arm': standalone_by_arm, 'stages': stages,
                 'blend_choice': chosen, 'best_nonzero_blend': best_blend, 'blend_grid': blend_grid}
    write_json(out / 'selection.json', selection)
    print(json.dumps({'common_feeds': common, 'standalone_by_arm': standalone_by_arm, 'blend_choice': chosen}), flush=True)

    scores = {name: raw[name] for name in ('frozen', 'submitted', *(f'{arm}_{common}' for arm in ARMS),
                                         *(f'{arm}_{stage}' for arm, stage in standalone_by_arm.items()))}
    scores['selected_blend'] = blend(raw['submitted'], raw[chosen['model']], items, counts, chosen['weight'])
    prediction_frame(val, scores['selected_blend']).write_parquet(out / 'selected_blend_val_predictions.parquet')
    blocks = {'late': np.arange(early, len(counts)), 'late_first': np.arange(early, mid),
              'late_second': np.arange(mid, len(counts)),
              'late_long': np.flatnonzero((np.arange(len(counts)) >= early) & (counts >= 101) & (counts <= 300))}
    offset = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    positives = np.add.reduceat(labels, offset[:-1])
    values = {name: (ndcg_per_feed(labels, s, counts), per_feed_hits(labels, s, counts, 10),
                     per_feed_hits(labels, s, counts, 30)) for name, s in scores.items()}

    def summarize(metric, hits10, hits30):
        answer = {}
        for name, ids in blocks.items():
            answer[name] = {**metric_summary(metric[ids], hits10[ids], 0, len(ids)),
                            'top30_positives': int(hits30[ids].sum()),
                            'recall_at_30_micro': float(hits30[ids].sum() / positives[ids].sum())}
        return answer

    metrics = {name: summarize(*value) for name, value in values.items()}
    comparisons = [(f'depth_hard_{common}', f'control_{common}'),
                   *[(name, 'submitted') for name in scores if name != 'submitted'],
                   *[(name, 'frozen') for name in scores if name.startswith(ARMS)]]
    contrasts = {}
    for name, ref in comparisons:
        delta = tuple(a - b for a, b in zip(values[name], values[ref], strict=True))
        contrasts[f'{name}_vs_{ref}'] = {'gain': summarize(*delta),
            'late_ndcg_95_interval': paired_interval(delta[0][early:, 0]),
            'long_recall30_micro_delta': contrasts_recall(delta[2], positives, blocks['late_long'])}
    report = {'metrics': metrics, 'contrasts': contrasts, 'common_feeds': common,
              'chosen_blend': chosen, 'block_counts': {k: len(v) for k, v in blocks.items()},
              'late_long_positives': int(positives[blocks['late_long']].sum())}
    write_json(out / 'assessment.json', report)
    print(json.dumps({'metrics_late': {k: v['late'] for k, v in metrics.items()},
                      'metrics_long': {k: v['late_long'] for k, v in metrics.items()}}), flush=True)


def contrasts_recall(delta, positives, ids):
    rng = np.random.default_rng(2026)
    samples = []
    for _ in range(20):
        idx = ids[rng.integers(len(ids), size=(100, len(ids)))]
        samples.append(delta[idx].sum(axis=1) / positives[idx].sum(axis=1))
    return np.quantile(np.concatenate(samples), [.025, .975]).tolist()


def submit(out, candidate):
    selection = json.loads((out / 'selection.json').read_text())
    chosen = selection['blend_choice']
    if not chosen['weight']:
        raise ValueError('early selection did not retain a nonzero blend')
    name = chosen['model']
    if name == 'frozen':
        model_path = SOURCE
    else:
        arm, seen = name.rsplit('_', 1)
        model_path = out / f'feeds-{seen}/{arm}.pt'
    test = load_feeds('test')
    score = score_sequences(load_ranker(model_path), test, BATCH_SIZE)
    keys = prediction_frame(test.keys, score).select(KEYS)
    anchor = read_scores(ANCHOR / 'submission-reserve.parquet', keys).astype(np.float32)
    final = blend(anchor, score, test.keys['item'], test.counts, chosen['weight'])
    prediction_frame(test.keys, final).write_parquet(candidate)
    audit = audit_submission(candidate)
    if audit['rows'] != 2904806 or audit['feeds_with_ties']:
        raise ValueError('submission has wrong pair count or tied scores')
    write_json(out / 'submission_audit.json', audit)
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('train', 'assess', 'submit'))
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, default=2027)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--submission', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == 'train':
        train(args.out, args.seed, args.threads)
    elif args.mode == 'assess':
        assess(args.out)
    else:
        if args.submission is None or args.submission.exists():
            parser.error('submit requires a new --submission path')
        submit(args.out, args.submission)


if __name__ == '__main__':
    main()
