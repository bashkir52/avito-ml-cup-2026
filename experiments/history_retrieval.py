"""Train a separate retrieval model on masked actions in historical sequences."""
import argparse
from copy import deepcopy
import json
import math
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from depth_hard_ranker import per_feed_hits
from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores, resolve_ties
from ranking import KEYS, ndcg_per_feed, prediction_frame, standardized_scores, temporal_boundary
from sequence_ranker import (HISTORY, SequenceEncoder, gather_vectors, novel_target_mask,
                             paths_for, read_history, vector_table, write_json)


ROOT = Path(__file__).resolve().parent
FEATURES = ROOT / 'cache-intent-024/val.npz'
ANCHOR = ROOT / 'nn-calibration-joint-024/val_predictions.parquet'
LEGACY = ROOT / 'sequence-novel-pretrain-510/encoder.pt'
FIELDS = [*HISTORY, 'history__category_id', 'user_id']
WIDTH = 48
SLOTS = 24
MODES = 4
HIDDEN = 4
BATCH = 192
SEED = 2026
STAGES = (8, 32, 96)
ARMS = ('uniform', 'hard')
WEIGHTS = (.05, .1, .2, .35, .5)


def history_examples(batch, include):
    ids, index, types, lengths = read_history(batch, SLOTS + HIDDEN)
    categories = batch['history__category_id']
    offsets = categories.offsets.to_numpy()
    items = batch[HISTORY[0]]
    if not np.array_equal(np.diff(offsets), np.diff(items.offsets.to_numpy())):
        raise ValueError('unaligned category arrays')
    can_predict = ((ids[:, :HIDDEN] > 0) & (index[:, :HIDDEN] >= 0) & (lengths[:, None] > HIDDEN)
                   & novel_target_mask(items, ids[:, :HIDDEN]))
    has_context = (index[:, HIDDEN:] >= 0).any(axis=1)
    rows = np.flatnonzero(include & has_context & can_predict.any(axis=1))
    owner, step = np.nonzero(can_predict[rows])
    selected = rows[owner]
    item_lists = items.take(rows)
    offsets_ids = item_lists.offsets.to_numpy()
    offsets_i = np.arange(np.diff(offsets_ids).max(initial=0))
    valid_ids = offsets_i < np.diff(offsets_ids)[:, None]
    known = np.zeros(valid_ids.shape, dtype=ids.dtype)
    known[valid_ids] = item_lists.values.to_numpy()[(offsets_ids[:-1, None] + offsets_i)[valid_ids]]
    return {'user': batch['user_id'].to_numpy()[rows], 'known': known, 'target_owner': owner,
            'target_id': ids[selected, step], 'target_index': index[selected, step],
            'target_category': categories.values.to_numpy()[offsets[selected] + step],
            'context_index': index[rows, HIDDEN:], 'context_event': (types[rows, HIDDEN:] + 1).clip(0, 11)}


def candidate_masks(histories, users, item, owner):
    own = np.arange(len(users))[:, None] == owner
    known = np.empty(own.shape, dtype=bool)
    for i, history in enumerate(histories):
        ordered = np.sort(history)
        slots = np.searchsorted(ordered, item).clip(max=len(ordered) - 1)
        known[i] = ordered[slots] == item
    return own, own | (~known & (users[:, None] != users[owner]))


class HistoryRetriever(nn.Module):
    def __init__(self):
        super().__init__()
        self.candidate = nn.Sequential(nn.Linear(128, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.events = nn.Embedding(12, 8)
        self.history = nn.Sequential(nn.Linear(128 + 8 + 1, WIDTH), nn.SiLU(), nn.Linear(WIDTH, WIDTH))
        self.attention = nn.Linear(WIDTH, MODES)
        self.mode_age = nn.Parameter(torch.tensor([-2., -1., -.3, 0.]))
        self.register_buffer('age', torch.log1p(torch.arange(SLOTS, dtype=torch.float32)) / math.log(SLOTS))

    def encode_history(self, vectors, event, mask):
        age = self.age[None, :, None].expand(len(event), -1, -1)
        h = self.history(torch.cat((F.normalize(vectors.float(), dim=-1), self.events(event.long()), age), dim=-1))
        attention = self.attention(h) + age * self.mode_age
        weights = torch.softmax(attention.masked_fill(~mask[..., None], -1e4), dim=1) * mask[..., None]
        return F.normalize(weights.transpose(1, 2) @ h, dim=-1)

    def encode_item(self, vectors):
        return F.normalize(self.candidate(F.normalize(vectors.float(), dim=-1)), dim=-1)

    @staticmethod
    def score(query, item):
        # A candidate can match any of the four pooled interests.
        return torch.logsumexp(12 * (query @ item.transpose(-1, -2)), dim=1) - math.log(MODES)


@torch.no_grad()
def hard_candidates(target, target_category, owner, valid, own):
    unit = F.normalize(target.float(), dim=-1)
    similarity = unit @ unit.T
    similarity.masked_fill_(target_category[:, None] != target_category[None, :], -torch.inf)
    shape = (len(valid), len(owner))
    best = similarity.new_full(shape, -torch.inf)
    best.scatter_reduce_(0, owner[:, None].expand_as(similarity), similarity, reduce='amax')
    best.masked_fill_(~valid | own, -torch.inf)
    values, indices = best.topk(min(32, len(owner)), dim=1)
    return torch.zeros_like(valid).scatter_(1, indices, values.isfinite())


def prepare_batch(batch, table, excluded, holdout):
    users = batch['user_id'].to_numpy()
    split = ((users % 16 == 0) == holdout) & (users > 0) & ~np.isin(users, excluded)
    example = history_examples(batch, split)
    if len(example['user']) < 2:
        return None
    own, valid = candidate_masks(example['known'], example['user'], example['target_id'], example['target_owner'])
    ctx = gather_vectors(table, example['context_index'])
    target = gather_vectors(table, example['target_index'])
    own = torch.from_numpy(own)
    valid = torch.from_numpy(valid)
    hard = hard_candidates(target, torch.from_numpy(example['target_category']), torch.from_numpy(example['target_owner']),
                           valid, own)
    return (ctx, torch.from_numpy(example['context_event']), torch.from_numpy(example['context_index'] >= 0),
            target, own, valid, hard)


def objective(logits, positive, valid, hard=None):
    denom = logits if hard is None else logits + math.log(4) * hard
    return (torch.logsumexp(denom.masked_fill(~valid, -torch.inf), dim=1)
            - (logits * positive).sum(dim=1) / positive.sum(dim=1)).mean()


def user_ids(split):
    return np.concatenate([pq.read_table(p, columns=['user_id'])['user_id'].to_numpy() for p in paths_for(split)])


def validation_data():
    val = read_meta(FEATURES)
    ref = {int(f): i for i, f in enumerate(val['feed'])}
    indices = np.full((len(ref), SLOTS), -1, dtype=np.int32)
    events = np.zeros_like(indices, dtype=np.int8)
    lengths = np.full(len(ref), -1, dtype=np.int32)
    for path in paths_for('val'):
        for batch in pq.ParquetFile(path).iter_batches(columns=['feed_id', *HISTORY]):
            _, index, event, count = read_history(batch, SLOTS)
            slots = np.array([ref[f] for f in batch['feed_id'].to_numpy()])
            if np.any(lengths[slots] >= 0):
                raise ValueError('duplicate val feed')
            indices[slots], events[slots], lengths[slots] = index, event + 1, count
    if np.any(lengths < 0):
        raise ValueError('history does not cover val')
    table = vector_table('val')
    with np.load(ROOT / 'embedding-cache-024/val.npz') as cache:
        for key in ('feed', 'item', 'counts'):
            np.testing.assert_array_equal(val[key], cache[key])
        candidate_vectors = cache['item_vectors']
    return val, candidate_vectors, gather_vectors(table, indices), events, indices >= 0, lengths


@torch.inference_mode()
def score_val(model, dataset, legacy=False):
    val, candidate, vectors, events, mask, lengths = dataset
    if legacy:
        count = lengths.clip(max=12)
        steps = np.arange(12)
        source = (count[:, None] - 1 - steps).clip(min=0)
        present = steps < count[:, None]
        v = vectors[np.arange(len(count))[:, None], source] * torch.from_numpy(present[..., None])
        e = np.where(present, np.take_along_axis(events, source, axis=1), 0)
        query = torch.cat([model.history(v[a:a+384], torch.from_numpy(e[a:a+384]), torch.from_numpy(count[a:a+384]))
                           for a in range(0, len(count), 384)])
    else:
        query = torch.cat([model.encode_history(vectors[a:a+384], torch.from_numpy(events[a:a+384]),
                                                torch.from_numpy(mask[a:a+384]))
                           for a in range(0, len(events), 384)])
    offsets = np.r_[0, np.cumsum(val['counts'], dtype=np.int64)]
    score = np.empty(len(candidate), dtype=np.float32)
    for start in range(0, len(query), 192):
        stop = min(start + 192, len(query))
        a, b = offsets[[start, stop]]
        feed = np.repeat(np.arange(start, stop), val['counts'][start:stop])
        if legacy:
            item = model.item(torch.from_numpy(candidate[a:b]))
            score[a:b] = (query[feed] * item).sum(dim=1).numpy()
        else:
            item = model.encode_item(torch.from_numpy(candidate[a:b]))
            m = (query[feed] * item[:, None]).sum(dim=2)
            score[a:b] = (torch.logsumexp(12 * m, dim=1) - math.log(MODES)).numpy()
    if not np.isfinite(score).all():
        raise ValueError('nonfinite retrieval scores')
    return score


def train(out, stages, threads):
    out.mkdir(parents=True, exist_ok=False)
    sources = sorted(paths_for('train'))
    rng = np.random.default_rng(SEED)
    order = rng.permutation(len(sources))
    train_files, holdout_files = [sources[i] for i in order[4:]], [sources[i] for i in order[:4]]
    excluded = np.unique(np.r_[user_ids('val'), user_ids('test')])
    dataset = validation_data()
    torch.manual_seed(SEED)
    initial = HistoryRetriever()
    models = {name: deepcopy(initial) for name in ARMS}
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
                  for name, model in models.items()}
    checkpoint = torch.load(LEGACY, weights_only=True, map_location='cpu')
    old = SequenceEncoder(checkpoint['width']).eval()
    old.load_state_dict(checkpoint['model'])
    prediction_frame(dataset[0], score_val(old, dataset, legacy=True)).write_parquet(out / 'legacy_val_predictions.parquet')
    start = time.perf_counter()
    table = np.array(vector_table('train'))
    print(f'{table.nbytes:,} training vector bytes staged in {time.perf_counter()-start:.2f}s', flush=True)
    config = {'stages': stages, 'seed': SEED, 'threads': threads, 'batch_feeds': BATCH, 'history_slots': SLOTS,
              'hidden_slots': HIDDEN, 'modes': MODES, 'width': WIDTH, 'lr': .001, 'weight_decay': .01,
              'hard_negative_weight': 4, 'hard_same_category_limit': 32, 'excluded_eval_users': len(excluded),
              'training_user_partition': 'user_id > 0 and user_id % 16 != 0 and no val/test user_id',
              'train_files': [p.name for p in train_files[:stages[-1]]], 'holdout_files': [p.name for p in holdout_files]}
    write_json(out / 'config.json', config)
    # Keep a small fixed synthetic holdout in RAM. Its users and shards are disjoint.
    holdout = []
    for path in holdout_files:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH * 5, columns=FIELDS):
            entry = prepare_batch(batch, table, excluded, holdout=True)
            if entry is not None:
                holdout.append(entry)
    history, previous = [], 0
    totals = {'feeds': 0, 'contexts': 0, 'targets': 0, 'hard_negatives': 0, 'contexts_with_hard_negatives': 0, 'steps': 0}
    for start_file, end in zip((0, *stages[:-1]), stages):
        for path in train_files[start_file:end]:
            for batch in pq.ParquetFile(path).iter_batches(batch_size=BATCH, columns=FIELDS):
                totals['feeds'] += len(batch)
                entry = prepare_batch(batch, table, excluded, holdout=False)
                if entry is None:
                    continue
                ctx, event, mask, target, positive, valid, hard = entry
                for name, model in models.items():
                    optimizer = optimizers[name]
                    optimizer.zero_grad(set_to_none=True)
                    logits = model.score(model.encode_history(ctx, event, mask), model.encode_item(target))
                    loss = objective(logits, positive, valid, hard if name == 'hard' else None)
                    if not torch.isfinite(loss):
                        raise ValueError('nonfinite retrieval loss')
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                totals['steps'] += 1
                totals['contexts'] += len(ctx)
                totals['targets'] += len(target)
                totals['hard_negatives'] += int(hard.sum())
                totals['contexts_with_hard_negatives'] += int(hard.any(dim=1).sum())
        folder = out / f'shards-{end}'
        folder.mkdir()
        metrics = {}
        for name, model in models.items():
            with torch.inference_mode():
                found, queries, positives, recovered = 0, 0, 0, 0
                for ctx, event, mask, target, positive, valid, hard in holdout:
                    logits = model.score(model.encode_history(ctx, event, mask), model.encode_item(target))
                    # Compare held-out targets against just their same-category hard negatives.
                    eligible = hard.any(dim=1)
                    ordering = logits.masked_fill(~(positive | hard), -torch.inf).topk(min(5, logits.shape[1]), dim=1).indices
                    hit = positive.gather(1, ordering)[eligible]
                    found += int(hit[:, 0].sum())
                    recovered += int(hit.sum())
                    queries += int(eligible.sum())
                    positives += int(positive[eligible].sum())
                metrics[name] = {'queries': queries, 'positive_targets': positives, 'hit_at_1': found / queries,
                                 'recall_at_5': recovered / positives}
            torch.save({'model': model.state_dict(), 'config': config, 'shards_seen': end}, folder / f'{name}.pt')
            prediction_frame(dataset[0], score_val(model, dataset)).write_parquet(folder / f'{name}_val_predictions.parquet')
        report = {'shards': end, **totals, 'seconds': time.perf_counter()-start, 'synthetic_holdout': metrics}
        history.append(report)
        write_json(out / 'training.json', history)
        print(json.dumps(report), flush=True)


def assess(out):
    if (out / 'selection.json').exists():
        raise FileExistsError('selection is already locked')
    val = read_meta(FEATURES)
    counts, labels, ts, items = (val[k] for k in ('counts', 'target', 'timestamp', 'item'))
    early = temporal_boundary(ts, .4)
    first = temporal_boundary(ts[:early], .65)
    mid = early + temporal_boundary(ts[early:], .5)
    if (first, early, mid, len(counts)) != (3749, 5766, 10083, 14423):
        raise ValueError('unexpected chronological blocks')
    keys = prediction_frame(val, np.zeros(len(items))).select(KEYS)
    anchor = read_scores(ANCHOR, keys).astype(np.float32)
    sources = {'legacy': out / 'legacy_val_predictions.parquet'}
    config = json.loads((out / 'config.json').read_text())
    for stage in config['stages']:
        for name in ARMS:
            sources[f'{name}_{stage}'] = out / f'shards-{stage}/{name}_val_predictions.parquet'
    raw = {name: read_scores(p, keys).astype(np.float32) for name, p in sources.items()}
    cut = int(counts[:early].sum())
    lengths = counts[:early]
    y = labels[:cut]
    norm = standardized_scores(anchor, counts)
    early_base = ndcg_per_feed(y, anchor[:cut], lengths)[:, 0]
    grid = []
    for name, score in raw.items():
        z = standardized_scores(score[:cut], lengths)
        for weight in WEIGHTS:
            mixed = resolve_ties((1-weight)*norm[:cut] + weight*z, anchor[:cut], items[:cut], lengths)
            gain = ndcg_per_feed(y, mixed, lengths)[:, 0] - early_base
            grid.append({'model': name, 'weight': weight, 'block_gain': [float(gain[:first].mean()), float(gain[first:].mean())],
                         'early_gain': float(gain.mean())})
    key = lambda row: (min(row['block_gain']), row['early_gain'])
    best = {name: max((row for row in grid if row['model'].startswith(name)), key=key) for name in (*ARMS, 'legacy')}
    winner = max([best[name] for name in ARMS], key=key)
    chosen = winner if min(winner['block_gain']) > 0 else {'model': 'submitted', 'weight': 0.}
    write_json(out / 'selection.json', {'best_by_family': best, 'selected': chosen, 'grid': grid})

    scores = {'submitted': anchor, 'legacy': raw['legacy']}
    for arm in ARMS:
        name = best[arm]['model']
        scores[name] = raw[name]
    if chosen['weight']:
        w = chosen['weight']
        scores['blend'] = resolve_ties((1-w)*norm + w*standardized_scores(raw[chosen['model']], counts), anchor, items, counts)
    else:
        scores['blend'] = anchor
    prediction_frame(val, scores['blend']).write_parquet(out / 'blend_val_predictions.parquet')
    base = ndcg_per_feed(labels, anchor, counts)
    long = (counts >= 101) & (counts <= 300)
    offset = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    blocks = {'late_first': np.arange(early, mid), 'late_second': np.arange(mid, len(counts)),
              'late': np.arange(early, len(counts)), 'late_long': np.flatnonzero(long & (np.arange(len(counts)) >= early))}
    baseline_hits = per_feed_hits(labels, anchor, counts, 30)

    def misses_recovered(score, feeds):
        hits = 0
        for feed in feeds:
            a, b = offset[feed:feed+2]
            label = labels[a:b]
            original = np.lexsort((label, -anchor[a:b]))[:30]
            current = np.lexsort((label, -score[a:b]))[:30]
            hits += int(label[np.setdiff1d(current, original, assume_unique=True)].sum())
        return hits

    metrics = {}
    for name, score in scores.items():
        result = ndcg_per_feed(labels, score, counts)
        hits10 = per_feed_hits(labels, score, counts, 10)
        hits30 = per_feed_hits(labels, score, counts, 30)
        metrics[name] = {k: {**metric_summary(result[ids], hits10[ids], 0, len(ids)),
                             'ndcg_gain': float((result-base)[ids, 0].mean()), 'top30_positives': int(hits30[ids].sum()),
                             'top30_positive_change': int((hits30-baseline_hits)[ids].sum())} for k, ids in blocks.items()}
        metrics[name]['late_ndcg_paired_interval'] = paired_interval((result-base)[early:, 0])
        metrics[name]['late_long_positives_recovered_from_below_submitted_30'] = misses_recovered(score, blocks['late_long'])
    report = {'chosen': chosen, 'metrics': metrics, 'late_long_feeds': len(blocks['late_long'])}
    write_json(out / 'assessment.json', report)
    print(json.dumps({'selected': chosen, 'late': {n: m['late'] for n, m in metrics.items()}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--shards', type=int, choices=STAGES, default=STAGES[-1])
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    train(args.out, tuple(n for n in STAGES if n <= args.shards), args.threads)
    assess(args.out)


if __name__ == '__main__':
    main()
