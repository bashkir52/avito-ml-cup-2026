"""Bounded GPU pilot of full attention over a feed and its recent history."""
import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from gpu_training_probe import AttentionBlock, gpu_device
from neural_blend import read_scores, resolve_ties
from ranking import KEYS, NAMES, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import SequenceFeeds, sha256, write_json
from set_ranker import fit_normalization


ROOT = Path(__file__).resolve().parent
ANCHOR = ROOT / 'nn-calibration-joint-024/val_predictions.parquet'
SEED = 2026
BATCH = 32
WIDTH = 64
CHECKPOINTS = (12000, 36000, 72000)
WEIGHTS = (.10, .25, .50, 1.)
GATE = .001
TRAIN_LIMIT_SECONDS = 240
RESUME_LIMIT_SECONDS = 300
RESUME_LR = .0006


class FullAttentionRanker(nn.Module):
    def __init__(self):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(2 * len(NAMES) + 128, WIDTH), nn.LayerNorm(WIDTH), nn.SiLU())
        self.user = nn.Linear(256, WIDTH)
        self.history = nn.Linear(141, WIDTH)
        self.blocks = nn.ModuleList(AttentionBlock(WIDTH) for _ in range(2))
        self.head = nn.Sequential(nn.LayerNorm(WIDTH), nn.Linear(WIDTH, 1))

    def forward(self, candidate, user, history, candidate_mask, history_mask):
        tokens = torch.cat((self.user(user), self.history(history), self.candidate(candidate)), dim=1)
        full_mask = torch.cat((torch.ones_like(candidate_mask[:, :1]), history_mask, candidate_mask), dim=1)
        for block in self.blocks:
            tokens = block(tokens, full_mask)
        return self.head(tokens[:, 1+history.shape[1]:]).squeeze(-1)


def load(split):
    return SequenceFeeds(ROOT / f'cache-intent-024/{split}.npz', ROOT / f'embedding-cache-024/{split}.npz', split)


def attention_loss(score, mask, positive, negative):
    # DirectML's PyTorch build cannot backpropagate through float * bool.
    # These two masks arrive as float32, with identical values to feed_loss.
    pos_count = positive.sum(dim=1).clamp_min(1.)
    neg_count = negative.sum(dim=1).clamp_min(1.)
    log_prob = F.log_softmax(score.masked_fill(~mask, -1e4), dim=1)
    list_loss = -(log_prob * positive).sum(dim=1) / pos_count
    positive_loss = (F.softplus(-score) * positive).sum(dim=1) / pos_count
    negative_loss = (F.softplus(score) * negative).sum(dim=1) / neg_count
    return (list_loss + .125 * (positive_loss + negative_loss)).mean()


def inputs(feeds, batch, mean, scale, device):
    x, mask, rows, _ = batch
    finite = torch.isfinite(x)
    numeric = torch.where(finite, ((x - mean) / scale).clamp(-7, 7), 0.)
    item, user, recent = feeds.vector_batch(rows)
    hvec, event, length = feeds.history_batch(rows)
    history_steps = torch.arange(event.shape[1])[None, :]
    history_mask = history_steps < length[:, None]
    age = ((length[:, None] - history_steps - 1).clamp_min(0) / event.shape[1]).unsqueeze(-1)
    candidate = torch.cat((numeric, (~finite).float(), 8 * F.normalize(item.float(), dim=-1)), dim=-1)
    user = torch.cat((8 * F.normalize(user.float(), dim=-1), 8 * F.normalize(recent.float(), dim=-1)), dim=-1)
    event_vector = F.one_hot(event.long(), num_classes=12).float()
    history = torch.cat((8 * F.normalize(hvec.float(), dim=-1), event_vector, age), dim=-1)
    return tuple(t.to(device) for t in (candidate, user[:, None, :], history, mask, history_mask))


@torch.no_grad()
def score_prefix(model, val, n, mean, scale, device):
    model.eval()
    score = np.empty(val.offsets[n], dtype=np.float32)
    for batch in val.batches(np.arange(n), BATCH):
        _, _, rows, mask = batch
        score[rows[mask]] = model(*inputs(val, batch, mean, scale, device)).cpu().numpy()[mask]
    if not np.isfinite(score).all():
        raise ValueError('nonfinite predictions')
    return score


def report_checkpoint(out, model, val, early, first, anchor, incumbent, mean, scale, device, seen, stage,
                      optimizer):
    path = out / f'feeds-{seen:05d}'
    path.mkdir()
    state = {key: param.detach().cpu() for key, param in model.state_dict().items()}
    optimizer_state = optimizer.state_dict()
    optimizer_state['state'] = {
        param_id: {key: value.detach().cpu() if isinstance(value, torch.Tensor) else value
                   for key, value in moments.items()}
        for param_id, moments in optimizer_state['state'].items()
    }
    torch.save({'model': state, 'mean': mean, 'scale': scale, 'training': stage,
                'optimizer': optimizer_state}, path / 'model.pt')
    restored = FullAttentionRanker().to(device)
    restored.load_state_dict(torch.load(path / 'model.pt', weights_only=True)['model'])
    scores = score_prefix(restored, val, early, mean, scale, device)
    n = val.offsets[early]
    data = {'feed': val.keys['feed'][:early], 'counts': val.counts[:early], 'item': val.keys['item'][:n]}
    prediction_frame(data, scores).write_parquet(path / 'early_predictions.parquet')
    z0, z1 = (standardized_scores(s, data['counts']) for s in (anchor, scores))
    configs = []
    for weight in WEIGHTS:
        final = (scores if weight == 1 else
                 resolve_ties((1 - weight) * z0 + weight * z1, anchor, data['item'], data['counts']))
        current = ndcg_per_feed(val.target[:n], final, data['counts'])[:, 0]
        gain = current - incumbent
        blocks = [slice(None, first), slice(first, None)]
        configs.append({'feeds': seen, 'transformer_weight': weight, 'early_ndcg': float(current.mean()),
                        'early_block_gain': [float(gain[block].mean()) for block in blocks]})
    write_json(path / 'early_metrics.json', configs)
    return configs


def continuation_data(directory, training_count, rng):
    history = json.loads((directory / 'training.json').read_text(encoding='utf-8'))
    config = json.loads((directory / 'config.json').read_text(encoding='utf-8'))
    if config.get('resume_source') or config['available_train_feeds'] != training_count:
        raise ValueError('resume requires the original pilot on the same train cache')
    previous_pool = np.load(directory / 'train_feed_indices.npy')
    if (previous_pool.ndim != 1 or len(np.unique(previous_pool)) != len(previous_pool)
            or np.any(previous_pool < 0) or np.any(previous_pool >= training_count)):
        raise ValueError('invalid original training sample')
    unused = np.setdiff1d(np.arange(training_count), previous_pool, assume_unique=True)
    prior = history[-1]
    path = directory / f"feeds-{prior['feeds']:05d}/model.pt"
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    if checkpoint['training'] != prior:
        raise ValueError('original checkpoint and training log differ')
    return checkpoint, prior, path, rng.permutation(unused), len(previous_pool)


def train(out, backend, resume=None):
    device, adapter = gpu_device(backend)
    print(json.dumps({'gpu': adapter.rstrip('\0'), 'device': str(device), 'torch': torch.__version__}), flush=True)
    out.mkdir(parents=True, exist_ok=False)
    val = load('val')
    training = load('train')
    if training.timestamp.max() >= val.timestamp.min():
        raise ValueError('training is not strictly earlier than validation')
    early = temporal_boundary(val.timestamp, .4)
    first = temporal_boundary(val.timestamp[:early], .65)
    if (first, early) != (3749, 5766):
        raise ValueError('unexpected early split')
    early_rows = val.offsets[early]
    keys = prediction_frame(val.keys, np.zeros(len(val.x), dtype=np.float32)).select(KEYS)
    anchor = read_scores(ANCHOR, keys).astype(np.float32)[:early_rows]
    incumbent = ndcg_per_feed(val.target[:early_rows], anchor, val.counts[:early])[:, 0]
    torch.manual_seed(SEED)
    model = FullAttentionRanker().to(device)
    source_notes = {}
    initial = {'feeds': 0, 'candidates': 0, 'positives': 0, 'steps': 0}
    if resume is None:
        rng = np.random.default_rng(SEED)
        mean, scale = map(torch.from_numpy, fit_normalization(training.x, rng))
        order = rng.permutation(len(training.counts))[:CHECKPOINTS[-1]]
        checkpoints, limit, lr, seed = CHECKPOINTS, TRAIN_LIMIT_SECONDS, .0012, SEED
    else:
        seed = SEED + 1
        rng = np.random.default_rng(seed)
        checkpoint, initial, path, order, excluded = continuation_data(resume, len(training.counts), rng)
        mean, scale = checkpoint['mean'], checkpoint['scale']
        model.load_state_dict(checkpoint['model'])
        checkpoints = tuple(sorted({min(36000, len(order)), len(order)}))
        if checkpoints[-1] == 0:
            raise ValueError('no fresh train feeds available for continuation')
        limit, lr = RESUME_LIMIT_SECONDS, RESUME_LR
        source_notes = {'resume_source': str(path), 'resume_sha256': sha256(path),
                        'excluded_previous_pool_feeds': excluded, 'optimizer_reset': True,
                        'initial_training': initial, 'new_feeds_available': len(order)}
        prefix = prediction_frame({'feed': val.keys['feed'][:early], 'counts': val.counts[:early],
                                   'item': val.keys['item'][:early_rows]}, np.zeros(early_rows)).select(KEYS)
        original = read_scores(path.with_name('early_predictions.parquet'), prefix).astype(np.float32)
        restored = score_prefix(model, val, early, mean, scale, device)
        np.testing.assert_allclose(restored, original, rtol=0, atol=3e-6)
        source_notes['original_early_score_max_abs_error'] = float(np.abs(restored-original).max())
        print(json.dumps({'restored_pairs': len(original), **source_notes}), flush=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=.03, foreach=False)
    config = {'backend': backend, 'adapter': adapter.rstrip('\0'), 'device': str(device), 'torch': torch.__version__,
              'numpy': np.__version__, 'parameters': sum(p.numel() for p in model.parameters()), 'width': WIDTH,
              'layers': 2, 'heads': 4, 'batch_max': BATCH, 'seed': seed, 'lr': lr, 'weight_decay': .03,
              'checkpoints': checkpoints, 'max_training_seconds': limit, 'gate_each_block': GATE,
              'early_feeds': early, 'early_first_feeds': first, 'available_train_feeds': len(training.counts),
              'train_cache': str(ROOT / 'cache-intent-024/train.npz'), 'position_used': False,
              'weight_grid': WEIGHTS, 'pointwise_loss_weight': .25, **source_notes}
    write_json(out / 'config.json', config)
    np.save(out / 'train_feed_indices.npy', order)
    elapsed = 0.
    steps, seen, rows_seen, positives = (initial[k] for k in ('steps', 'feeds', 'candidates', 'positives'))
    search, progress = [], []
    for begin, end in zip((0, *checkpoints[:-1]), checkpoints):
        model.train()
        start = perf_counter()
        total_loss = 0.
        stage_feeds = 0
        for batch in training.batches(order[begin:end], BATCH, rng):
            _, _, rows, mask = batch
            labels = training.target[rows].astype(bool)
            positive = torch.from_numpy((labels & mask).astype(np.float32)).to(device)
            negative = torch.from_numpy((~labels & mask).astype(np.float32)).to(device)
            data = inputs(training, batch, mean, scale, device)
            optimizer.zero_grad(set_to_none=True)
            score = model(*data)
            loss = attention_loss(score, data[3], positive, negative)
            loss.backward()
            optimizer.step()
            value = float(loss.detach().cpu())
            if not np.isfinite(value):
                raise ValueError('nonfinite training loss')
            stage_feeds += len(rows)
            total_loss += len(rows) * value
            seen += len(rows)
            rows_seen += int(mask.sum())
            positives += int(training.target[rows[mask]].sum())
            steps += 1
            if steps % 200 == 0:
                print(json.dumps({'steps': steps, 'train_feeds': seen, 'new_feeds': seen - initial['feeds'],
                                  'loss_last': value,
                                  'train_seconds': elapsed + perf_counter() - start}), flush=True)
            if elapsed + perf_counter() - start >= limit:
                break
        elapsed += perf_counter() - start
        stage = {'feeds': seen, 'steps': steps, 'candidates': rows_seen, 'positives': positives,
                 'train_seconds': elapsed, 'stage_loss': total_loss / stage_feeds,
                 'added_feeds': seen - initial['feeds'], 'added_steps': steps - initial['steps'],
                 'added_candidates': rows_seen - initial['candidates'], 'added_positives': positives - initial['positives']}
        print(json.dumps(stage), flush=True)
        search.extend(report_checkpoint(out, model, val, early, first, anchor, incumbent, mean, scale, device,
                                        seen, stage, optimizer))
        progress.append(stage)
        write_json(out / 'training.json', progress)
        if elapsed >= limit:
            break
    key = lambda e: (min(e['early_block_gain']), e['early_ndcg'])
    chosen = max(search, key=key)
    gate = min(chosen['early_block_gain']) >= GATE
    report = {'best': chosen, 'gate_passed': gate, 'early_gate_each': GATE, 'grid': search,
              'incumbent_early_block_ndcg': [float(incumbent[:first].mean()), float(incumbent[first:].mean())],
              'late_labels_used': False}
    write_json(out / 'selection.json', report)
    print(json.dumps({k: report[k] for k in ('best', 'gate_passed', 'incumbent_early_block_ndcg')}), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('out', type=Path)
    p.add_argument('--backend', choices=('directml', 'cuda'), required=True)
    p.add_argument('--resume-from', type=Path, help='original pilot run whose final checkpoint is the starting point')
    args = p.parse_args()
    torch.set_num_threads(6)
    train(args.out, args.backend, args.resume_from)


if __name__ == '__main__':
    main()
