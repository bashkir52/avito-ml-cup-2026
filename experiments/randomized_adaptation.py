"""Controlled hidden-layer adaptation from logged to randomized feed responses."""
import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import psutil
import torch
from torch import nn

from feed_width import metric_summary, paired_interval
from neural_blend import read_scores
from ranking import KEYS, ndcg_per_feed, prediction_frame, temporal_boundary
from sequence_ranker import (SequenceFeeds, load_ranker, score_sequences, sha256,
                             top10_hits, write_json)
from set_ranker import feed_loss

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'sequence-finetune-048/pretrained/model.pt'
ARMS = ('random_only', 'joint')
EDGES = (2, 4, 8, 16, 32, 64, 128, 256)
LRS = (3e-5, 1e-4)
STEPS = (100, 300)
BATCH = 64
ANCHOR = .1
AUX_LR = .001
AUX_WEIGHT = .5


class AdaptedRanker(nn.Module):
    def __init__(self, ranker):
        super().__init__()
        self.ranker = ranker
        for name, p in ranker.named_parameters():
            p.requires_grad_(name.startswith(('local.', 'output.')))
        self.train_head = deepcopy(ranker.output)
        self.depth = nn.Embedding(len(EDGES) + 1, 1)
        nn.init.zeros_(self.depth.weight)
        self.register_buffer('edges', torch.tensor(EDGES))

    def shared_parameters(self):
        return [(n, p) for n, p in self.ranker.named_parameters() if p.requires_grad]

    def forward(self, x, mask, vectors, history):
        return self.ranker(x, mask, vectors, history)

    def train_scores(self, x, mask, vectors, history, position):
        # Reuse the exact original forward; capture the shared hidden state and
        # remove its random-head contribution while retaining the frozen GRU term.
        captured = []
        handle = self.ranker.output.register_forward_hook(lambda module, inputs, output: captured.append((inputs[0], output)))
        try:
            original = self.ranker(x, mask, vectors, history)
        finally:
            handle.remove()
        hidden, base = captured[0]
        residual = original - base.squeeze(-1)
        bins = torch.bucketize(position.contiguous(), self.edges, right=True)
        return self.train_head(hidden).squeeze(-1) + residual + self.depth(bins).squeeze(-1)


def anchor_penalty(model, origin):
    return sum((p - origin[n]).square().sum() for n, p in model.shared_parameters())


def checked_windows(timestamps):
    if np.any(timestamps[1:] < timestamps[:-1]):
        raise ValueError('unsorted temporal feeds')
    early = temporal_boundary(timestamps, .4)
    middle = temporal_boundary(timestamps[:early], .65)
    start = temporal_boundary(timestamps[:middle], .5)
    late_mid = early + temporal_boundary(timestamps[early:], .5)
    if (start, middle, early, late_mid, len(timestamps)) != (1870, 3749, 5766, 10083, 14423):
        raise ValueError('unexpected temporal partition')
    for i in (start, middle, early, late_mid):
        if timestamps[i - 1] >= timestamps[i]:
            raise ValueError('equal timestamps straddle a temporal boundary')
    return ((start, middle), (middle, early)), early, late_mid


def memory():
    info = psutil.Process().memory_info()
    return {'rss_gib': info.rss / 2**30, 'peak_rss_gib': info.peak_wset / 2**30,
            'available_gib': psutil.virtual_memory().available / 2**30}


def load_feeds(split):
    suffix = '048' if split == 'train' else '024'
    return SequenceFeeds(ROOT / f'cache-intent-{suffix}/{split}.npz',
                         ROOT / f'embedding-cache-{suffix}/{split}.npz', split,
                         logged_positions=split == 'train')


def verify(out, base, val):
    start = time.perf_counter()
    # Use the original batch size so the CPU GEMM shape also matches.
    score = score_sequences(base, val, 96)
    keys = prediction_frame(val.keys, score).select(KEYS)
    saved = read_scores(SOURCE.parent / 'val_predictions.parquet', keys).astype(np.float32)
    np.testing.assert_array_equal(score, saved)
    prediction_frame(val.keys, score).write_parquet(out / 'frozen_val_predictions.parquet')
    report = {'source': str(SOURCE), 'source_sha256': sha256(SOURCE),
              'saved_predictions_sha256': sha256(SOURCE.parent / 'val_predictions.parquet'),
              'pairs': len(score), 'exact_equal': bool(np.array_equal(score, saved)),
              'max_abs_error': float(np.max(np.abs(score - saved))),
              'seconds': time.perf_counter() - start, 'memory': memory(),
              'torch': torch.__version__, 'features': val.x.shape[1]}
    write_json(out / 'reproduction.json', report)
    print(json.dumps({'reproduction': report}), flush=True)
    return score


def stream(feeds, ids, seed):
    rng = np.random.default_rng(seed)
    while True:
        yield from feeds.batches(rng.permutation(ids), BATCH, rng)


@torch.inference_mode()
def predict_range(model, val, first, stop):
    model.eval()
    offset, end = val.offsets[[first, stop]]
    result = np.empty(end - offset, dtype=np.float32)
    for x, mask, rows, keep in val.batches(np.arange(first, stop), 96):
        score = model(x, mask, val.vector_batch(rows), val.history_batch(rows))
        result[rows[keep] - offset] = score.numpy()[keep]
    if not np.isfinite(result).all():
        raise ValueError('nonfinite predictions')
    return result


def save_model(path, model, source, config):
    torch.save({'model': model.ranker.state_dict(), 'base_width': source['base_width'],
                'seq_width': source['seq_width'], 'auxiliary_head': model.train_head.state_dict(),
                'depth': model.depth.state_dict(), 'adaptation': config,
                'source_sha256': sha256(SOURCE)}, path)


def load_adapted(path):
    saved = torch.load(path, weights_only=True, map_location='cpu')
    # requires_grad flags are not part of state_dict. Restoring these matters
    # for exact CPU inference dispatch, even inside torch.inference_mode().
    model = AdaptedRanker(load_ranker(path))
    model.train_head.load_state_dict(saved['auxiliary_head'])
    model.depth.load_state_dict(saved['depth'])
    return model


def fit_pair(base, source, val, train, prefix, lr, checkpoints, seed, out, evaluate_stop=None):
    if prefix not in (1870, 3749, 5766):
        raise ValueError('training must use an authorized chronological prefix')
    out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(seed)
    initial = AdaptedRanker(deepcopy(base))
    models = {name: deepcopy(initial) for name in ARMS}
    origin = {n: p.detach().clone() for n, p in initial.shared_parameters()}
    optimizers = {name: torch.optim.AdamW([
        {'params': [p for _, p in model.shared_parameters()], 'lr': lr},
        {'params': [*model.train_head.parameters(), *model.depth.parameters()], 'lr': AUX_LR}],
        weight_decay=0) for name, model in models.items()}
    random_stream = stream(val, np.arange(prefix), seed)
    train_stream = stream(train, np.arange(len(train.counts)), seed + 10000)
    history = []
    start_time = time.perf_counter()
    random_ids, train_ids = [], []
    random_counts, train_counts = [], []
    hashes = {name: hashlib.sha256() for name in ('random_rows_labels', 'train_rows_labels')}
    config = {'seed': seed, 'prefix_feeds': prefix, 'lr': lr, 'batch_size': BATCH,
              'anchor_coefficient': ANCHOR, 'anchor_reduction': 'sum_squared_parameter_deviations',
              'auxiliary_weight': AUX_WEIGHT, 'auxiliary_lr': AUX_LR, 'position_edges': EDGES,
              'trainable_common_names': list(origin), 'common_parameters': sum(p.numel() for p in origin.values()),
              'auxiliary_parameters': sum(p.numel() for p in initial.train_head.parameters()) + initial.depth.weight.numel(),
              'weight_decay': 0, 'pointwise_weight': .25, 'gradient_clip': None,
              'train_available_feeds': len(train.counts), 'train_available_candidates': len(train.x)}
    write_json(out / 'config.json', config)
    for step in range(1, max(checkpoints) + 1):
        x, mask, rows, keep = next(random_stream)
        tx, tm, tr, tk = next(train_stream)
        ids = np.searchsorted(val.offsets, rows[:, 0], side='right') - 1
        tids = np.searchsorted(train.offsets, tr[:, 0], side='right') - 1
        if ids.max() >= prefix:
            raise AssertionError('future label access')
        random_ids.extend(ids.tolist()); train_ids.extend(tids.tolist())
        random_counts.append(len(ids)); train_counts.append(len(tids))
        y = torch.from_numpy(val.target[rows])
        ty = torch.from_numpy(train.target[tr])
        vectors, histories = val.vector_batch(rows), val.history_batch(rows)
        tv, th = train.vector_batch(tr), train.history_batch(tr)
        pos = torch.from_numpy(train.position[tr])
        for name, rr, kk, yy in (('random_rows_labels', rows, keep, val.target), ('train_rows_labels', tr, tk, train.target)):
            hashes[name].update(rr[kk].tobytes()); hashes[name].update(yy[rr[kk]].tobytes())
        entry = {'step': step}
        for name, model in models.items():
            model.train()
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            # Auxiliary RNG consumption cannot alter subsequent random masks.
            torch.manual_seed(seed * 100000 + step)
            random_loss = feed_loss(model(x, mask, vectors, histories), y, mask, .25)
            penalty = anchor_penalty(model, origin)
            loss = random_loss + ANCHOR * penalty
            aux = torch.tensor(0.)
            if name == 'joint':
                torch.manual_seed(seed * 100000 + step + 50000)
                aux = feed_loss(model.train_scores(tx, tm, tv, th, pos), ty, tm, .25)
                loss = loss + AUX_WEIGHT * aux
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            optimizer.step()
            entry[name] = {'random_loss': float(random_loss.detach()), 'train_loss': float(aux.detach()),
                           'anchor_sum': float(penalty.detach())}
        history.append(entry)
        if step % 50 == 0 or step in checkpoints:
            print(json.dumps({'prefix': prefix, 'lr': lr, 'step': step,
                              'seconds': round(time.perf_counter() - start_time, 2), **entry}), flush=True)
        if step in checkpoints:
            checkpoint_dir = out / f'step-{step:03d}'
            checkpoint_dir.mkdir()
            for name, model in models.items():
                save_model(checkpoint_dir / f'{name}.pt', model, source, {**config, 'steps': step})
                if evaluate_stop is not None:
                    prediction = predict_range(model, val, prefix, evaluate_stop)
                    np.save(checkpoint_dir / f'{name}_forward_scores.npy', prediction)
            write_json(checkpoint_dir / 'budget.json', {
                'steps_per_arm': step, 'random_feed_presentations_per_arm': len(random_ids),
                'train_feed_presentations_joint': len(train_ids),
                'unique_random_feeds': len(set(random_ids)), 'unique_train_feeds': len(set(train_ids)),
                'random_candidates': int(val.counts[random_ids].sum()),
                'train_candidates': int(train.counts[train_ids].sum()),
                'random_positives': int(np.add.reduceat(val.target, val.offsets[:-1])[random_ids].sum()),
                'train_positives': int(np.add.reduceat(train.target, train.offsets[:-1])[train_ids].sum()),
                'batch_hashes_shared_across_arms': {n: h.hexdigest() for n, h in hashes.items()}})
    # Frozen tensors must be unchanged, not merely excluded from the optimizer.
    for model in models.values():
        for n, p in model.ranker.state_dict().items():
            if n not in origin:
                torch.testing.assert_close(p, base.state_dict()[n], rtol=0, atol=0)
    np.savez(out / 'batch_schedule.npz', random_ids=np.array(random_ids), train_ids=np.array(train_ids),
             random_batch_counts=random_counts, train_batch_counts=train_counts)
    report = {'config': config, 'seconds': time.perf_counter() - start_time,
              'memory': memory(), 'frozen_tensors_unchanged': True, 'history': history}
    write_json(out / 'training.json', report)
    return models, report


def forward_metrics(val, score, start, end):
    a, b = val.offsets[[start, end]]
    return ndcg_per_feed(val.target[a:b], score, val.counts[start:end])[:, 0]


def assessment(out, val, predictions, early, mid):
    counts, labels = val.counts, val.target
    blocks = {'late_first': (early, mid), 'late_second': (mid, len(counts)),
              'late': (early, len(counts))}
    long = (counts[early:] >= 101) & (counts[early:] <= 300)
    values = {name: (ndcg_per_feed(labels, scores, counts), top10_hits(labels, scores, counts))
              for name, scores in predictions.items()}

    def summarize(ndcg, hits):
        return {**{name: metric_summary(ndcg, hits, *bounds) for name, bounds in blocks.items()},
                'late_long_101_300': metric_summary(ndcg[early:][long], hits[early:][long], 0, int(long.sum()))}

    metrics = {name: summarize(*v) for name, v in values.items()}
    contrasts = {}
    comparisons = [('joint', 'random_only'), ('random_only', 'frozen'), ('joint', 'frozen'),
                   ('frozen', 'submitted'), ('random_only', 'submitted'), ('joint', 'submitted')]
    for name, control in comparisons:
        delta, hits = (a - b for a, b in zip(values[name], values[control]))
        contrasts[f'{name}_vs_{control}'] = {'gain': summarize(delta, hits),
            'late_paired_95_intervals': {metric: paired_interval(delta[early:, i])
                                        for i, metric in enumerate(('ndcg', 'ndcg_at_10', 'ndcg_at_30'))},
            'long_paired_ndcg_at_10_95_interval': paired_interval(delta[early:, 1][long])}
    eligible = []
    for name in ARMS:
        contrast = contrasts[f'{name}_vs_submitted']
        gains = contrast['gain']
        halves = [gains[k] for k in ('late_first', 'late_second')]
        if min(h['ndcg'] for h in halves) > 0 and (gains['late']['ndcg'] >= .003 or (
            gains['late']['ndcg'] >= .001 and contrast['late_paired_95_intervals']['ndcg'][0] > 0
            and all(h['ndcg_at_10'] > 0 and h['top10_positives'] > 0 for h in halves))):
            eligible.append(name)
    report = {'metrics': metrics, 'contrasts': contrasts, 'second_seed_required_for': eligible,
              'block_sizes': {name: b - a for name, (a, b) in blocks.items()},
              'late_long_feeds': int(long.sum()), 'bootstrap': {'replicates': 2000, 'seed': 2026, 'unit': 'feed'},
              'early_predictions_are_in_sample': True,
              'prediction_sha256': {n: sha256(out / f'{n}_val_predictions.parquet') for n in predictions}}
    write_json(out / 'assessment.json', report)
    print(json.dumps({'late': {n: m['late'] for n, m in metrics.items()},
                      'second_seed_required_for': eligible}), flush=True)
    return report


def run(args):
    args.out.mkdir(parents=True, exist_ok=False)
    source = torch.load(SOURCE, weights_only=True, map_location='cpu')
    base = load_ranker(SOURCE).eval()
    val = load_feeds('val')
    windows, early, mid = checked_windows(val.timestamp)
    frozen = verify(args.out, base, val)
    if args.mode == 'verify':
        return
    train_start = time.perf_counter()
    train = load_feeds('train')
    if train.timestamp.max() >= val.timestamp.min():
        raise ValueError('train includes future timestamps')
    print(json.dumps({'loaded_train_seconds': time.perf_counter() - train_start, 'memory': memory()}), flush=True)
    _, pilot = fit_pair(base, source, val, train, 1870, LRS[1], (20,), args.seed, args.out / 'pilot')
    budget = {'pilot_seconds': pilot['seconds'], 'pilot_memory': pilot['memory'],
              'grid_lrs': LRS, 'grid_steps': STEPS, 'batch_size': BATCH, 'threads': args.threads,
              'estimated_training_seconds': pilot['seconds'] / 20 * (2 * len(LRS) * max(STEPS) + max(STEPS)),
              'selection': 'max(min(two joint-vs-frozen forward gains)), then weighted mean', 'blends': False}
    write_json(args.out / 'budget_lock.json', budget)
    print(json.dumps({'budget_locked': budget}), flush=True)
    if args.mode == 'pilot':
        return
    grid = []
    for lr in LRS:
        fold_scores = {}
        for i, (start, end) in enumerate(windows):
            folder = args.out / f'forward-{i+1}-lr-{lr:g}'
            fit_pair(base, source, val, train, start, lr, STEPS, args.seed, folder, end)
            reference = forward_metrics(val, frozen[val.offsets[start]:val.offsets[end]], start, end)
            for steps in STEPS:
                for name in ARMS:
                    p = np.load(folder / f'step-{steps:03d}/{name}_forward_scores.npy')
                    value = forward_metrics(val, p, start, end)
                    fold_scores[(steps, name, i)] = {'ndcg': float(value.mean()),
                                                    'frozen_ndcg': float(reference.mean()),
                                                    'gain': float((value - reference).mean())}
        for steps in STEPS:
            grid.append({'lr': lr, 'steps': steps, 'arms': {
                name: [fold_scores[(steps, name, i)] for i in range(2)] for name in ARMS}})
    def key(entry):
        gains = [v['gain'] for v in entry['arms']['joint']]
        return min(gains), np.average(gains, weights=[1879, 2017]), -entry['steps'], -entry['lr']
    selected = max(grid, key=key)
    selection = {'selected': selected, 'grid': grid, 'seed': args.seed,
                 'same_configuration_for_both_arms': True, 'blends': False,
                 'source_sha256': sha256(SOURCE), 'late_labels_used': False}
    write_json(args.out / 'selection.json', selection)
    print(json.dumps({'selection_locked': selection}), flush=True)
    models, _ = fit_pair(base, source, val, train, early, selected['lr'], (selected['steps'],),
                         args.seed, args.out / 'final')
    del train
    predictions = {'frozen': frozen}
    for name, model in models.items():
        predictions[name] = score_sequences(model, val, 96)
        prediction_frame(val.keys, predictions[name]).write_parquet(args.out / f'{name}_val_predictions.parquet')
        # Check disk checkpoint reconstruction for the entire validation set.
        disk = load_adapted(args.out / f'final/step-{selected["steps"]:03d}/{name}.pt')
        np.testing.assert_array_equal(score_sequences(disk, val, 96), predictions[name])
    keys = prediction_frame(val.keys, frozen).select(KEYS)
    predictions['submitted'] = read_scores(ROOT / 'nn-calibration-joint-024/val_predictions.parquet', keys).astype(np.float32)
    prediction_frame(val.keys, predictions['submitted']).write_parquet(args.out / 'submitted_val_predictions.parquet')
    assessment(args.out, val, predictions, early, mid)


def resume_assessment(args):
    if (args.out / 'assessment.json').exists():
        raise FileExistsError('late assessment already exists; use independent verification')
    selection = json.loads((args.out / 'selection.json').read_text())
    if sha256(SOURCE) != selection['source_sha256']:
        raise ValueError('source checkpoint changed')
    val = load_feeds('val')
    _, early, mid = checked_windows(val.timestamp)
    keys = prediction_frame(val.keys, np.zeros(len(val.x), dtype=np.float32)).select(KEYS)
    predictions = {'frozen': read_scores(args.out / 'frozen_val_predictions.parquet', keys).astype(np.float32)}
    roundtrip = {}
    for name in ARMS:
        path = args.out / f'final/step-{selection["selected"]["steps"]:03d}/{name}.pt'
        model = load_adapted(path)
        predictions[name] = score_sequences(model, val, 96)
        prediction_path = args.out / f'{name}_val_predictions.parquet'
        existed = prediction_path.exists()
        if existed:
            np.testing.assert_array_equal(predictions[name], read_scores(prediction_path, keys).astype(np.float32))
        else:
            prediction_frame(val.keys, predictions[name]).write_parquet(prediction_path)
        np.testing.assert_array_equal(predictions[name], score_sequences(load_adapted(path), val, 96))
        roundtrip[name] = {'pairs': len(predictions[name]), 'exact_equal_after_reload': True,
                           'matched_pre_failure_predictions': existed, 'checkpoint_sha256': sha256(path)}
    predictions['submitted'] = read_scores(ROOT / 'nn-calibration-joint-024/val_predictions.parquet', keys).astype(np.float32)
    prediction_frame(val.keys, predictions['submitted']).write_parquet(args.out / 'submitted_val_predictions.parquet')
    write_json(args.out / 'checkpoint_roundtrip.json', {'arms': roundtrip,
        'fix': 'restore requires_grad flags in adapted checkpoint loader', 'retrained': False,
        'selection_sha256': sha256(args.out / 'selection.json')})
    assessment(args.out, val, predictions, early, mid)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=('verify', 'pilot', 'run', 'assess-saved'))
    parser.add_argument('out', type=Path)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if args.mode == 'assess-saved':
        resume_assessment(args)
    else:
        run(args)


if __name__ == '__main__':
    main()
