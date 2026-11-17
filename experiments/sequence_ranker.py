"""Pretrain a GRU on next items and transfer it to supervised feed ranking."""
import argparse
from copy import deepcopy
from hashlib import file_digest
import json
import math
from tempfile import NamedTemporaryFile
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import nn
from torch.nn import functional as F

from feed_width import metric_summary, paired_interval
from history_event_cache import fetch_vectors
from neural_blend import read_meta, read_scores, resolve_ties
from ranking import (DATA, KEYS, NAMES, audit_submission, ndcg_per_feed, prediction_frame,
                     standardized_scores, temporal_boundary)
from set_ranker import FeedRanker, Feeds, feed_loss, fit_normalization


HISTORY = ('history__item_id', 'history__item_id__embedding_idx', 'history__event_type')
SLOTS = 12
TARGET_SLOTS = 4
ARMS = ('scratch', 'pretrained')
ATTENTION_ARMS = ('shared', 'attention')
ATTENTION_DIFFERENCE = 'attention_delta'
CANDIDATE_SLOTS = 8
WEIGHTS = (0., .025, .05, .10, .20, .35, .50)
POSITION_EDGES = (2, 4, 8, 16, 32, 64, 128, 256)
POSITION_LOSS_WEIGHT = .5


def write_json(path, report):
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')


def sha256(path):
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def paths_for(split):
    return sorted((DATA / split / 'impressions').glob('part-*.parquet'))


def vector_table(split):
    root = DATA / split / 'item_embeddings'
    meta = json.loads((root / 'data.meta').read_text(encoding='utf-8'))
    return np.memmap(root / 'data.memmap', dtype=meta['dtype'], mode='r',
                     shape=tuple(meta['shape']), order=meta['order'])


def read_history(batch, limit):
    columns = [batch[name] for name in HISTORY]
    offsets = [column.offsets.to_numpy() for column in columns]
    lengths = np.diff(offsets[0])
    if any(column.null_count or column.values.null_count for column in columns):
        raise ValueError('null in parallel history columns')
    if any(not np.array_equal(lengths, np.diff(o)) for o in offsets[1:]):
        raise ValueError('misaligned parallel history columns')
    step = np.arange(limit)
    mask = step < lengths[:, None]
    output = []
    for column, offset, fill in zip(columns, offsets, (0, -1, 0), strict=True):
        values = column.values.to_numpy()
        dense = np.full(mask.shape, fill, dtype=values.dtype)
        dense[mask] = values[(offset[:-1, None] + step)[mask]]
        output.append(dense)
    return *output, lengths.clip(max=limit)


def history_window(indices, events, lengths, start, slots):
    """Return the previous slots in chronological order with padding on the right."""
    used = np.minimum(lengths - start, slots)
    step = np.arange(slots)
    valid = step < used[:, None]
    source = np.maximum((start + used - 1)[:, None] - step, 0)
    ordered = np.take_along_axis(indices, source, axis=1)
    kinds = np.take_along_axis(events, source, axis=1)
    if np.any((kinds[valid] < 0) | (kinds[valid] > 10)):
        raise ValueError('unexpected history event type')
    return np.where(valid, ordered, -1), np.where(valid, kinds + 1, 0), used


def gather_vectors(table, indices):
    present = indices >= 0
    if np.any(indices < -1):
        raise ValueError('invalid negative embedding index')
    result = np.zeros((*indices.shape, table.shape[1]), dtype=table.dtype)
    result[present] = table[indices[present]]
    return torch.from_numpy(result)


def mask_duplicate_targets(scores, items):
    duplicated = items[:, None].eq(items[None, :])
    duplicated.fill_diagonal_(False)
    return scores.masked_fill(duplicated, -torch.inf)


class SequenceEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.items = nn.Linear(128, width, bias=False)
        self.events = nn.Embedding(12, 8, padding_idx=0)
        self.gru = nn.GRU(width + 8, width, batch_first=True)

    def item(self, vectors):
        return F.normalize(self.items(F.normalize(vectors.float(), dim=-1)), dim=-1)

    def history(self, vectors, events, lengths):
        features = torch.cat((self.item(vectors), self.events(events.long())), dim=-1)
        encoded, _ = self.gru(features)
        last = encoded[torch.arange(len(lengths)), (lengths - 1).clamp_min(0)]
        return F.normalize(last, dim=-1) * (lengths > 0).unsqueeze(-1)


def novel_target_mask(item_lists, target_ids):
    """Check target IDs against every older event, including beyond the GRU input."""
    offsets = item_lists.offsets.to_numpy()
    lengths = np.diff(offsets)
    steps = np.arange(lengths.max(initial=0))
    valid = steps < lengths[:, None]
    values = item_lists.values.to_numpy()
    ids = np.zeros(valid.shape, dtype=values.dtype)
    ids[valid] = values[(offsets[:-1, None] + steps)[valid]]
    return np.column_stack([
        ~np.any((ids[:, i + 1:] == target_ids[:, i, None]) & valid[:, i + 1:], axis=1)
        for i in range(target_ids.shape[1])
    ])


def next_item_batches(path, batch_size, rng, novel_targets=False):
    for batch in pq.ParquetFile(path).iter_batches(batch_size=batch_size, columns=list(HISTORY)):
        ids, idx, event, lengths = read_history(batch, SLOTS + TARGET_SLOTS)
        positions = np.arange(TARGET_SLOTS)
        eligible = ((idx[:, :TARGET_SLOTS] >= 0) & (ids[:, :TARGET_SLOTS] > 0)
                    & (positions + 1 < lengths[:, None]))
        if novel_targets:
            eligible &= novel_target_mask(batch[HISTORY[0]], ids[:, :TARGET_SLOTS])
        keep = np.flatnonzero(eligible.any(axis=1))
        if len(keep) < 2:
            yield len(lengths), None
            continue
        target = np.argmax(np.where(eligible[keep], rng.random((len(keep), TARGET_SLOTS)), -1), axis=1)
        context = history_window(idx[keep], event[keep], lengths[keep], target + 1, SLOTS)
        yield len(lengths), (ids[keep, target], idx[keep, target], *context)


def next_item_pass(model, table, paths, args, rng, optimizer=None):
    start = time.perf_counter()
    feeds = samples = steps = top1 = repeats = contexts = visible = 0
    loss_sum = 0.0
    stats = {}
    model.train(optimizer is not None)
    for n, path in enumerate(paths, 1):
        for total, example in next_item_batches(path, args.batch_size, rng, args.novel_targets):
            feeds += total
            if example is None:
                continue
            items, idx, older_idx, older_types, lengths = example
            vectors = gather_vectors(table, np.column_stack((idx, older_idx)))
            query = model.history(vectors[:, 1:], torch.from_numpy(older_types), torch.from_numpy(lengths))
            target = model.item(vectors[:, 0])
            scores = mask_duplicate_targets(10 * (query @ target.T), torch.from_numpy(items))
            label = torch.arange(len(items))
            loss = F.cross_entropy(scores, label)
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
            loss_sum += float(loss.detach()) * len(items)
            top1 += int((scores.detach().argmax(dim=1) == label).sum())
            repeats += int((older_idx == idx[:, None]).any(axis=1).sum())
            contexts += int(lengths.sum())
            visible += int((older_idx >= 0).sum())
            samples += len(items)
            steps += 1
        if n % 32 == 0 or n == len(paths):
            if samples == 0:
                raise ValueError('next-item pass found no eligible histories')
            elapsed = time.perf_counter() - start
            stats = {'shards': n, 'feeds': feeds, 'examples': samples, 'steps': steps,
                     'loss': loss_sum / samples, 'recall_at_1': top1 / samples,
                     'exact_target_index_in_context_fraction': repeats / samples,
                     'context_events_per_example': contexts / samples,
                     'context_vectors_per_example': visible / samples,
                     'seconds': round(elapsed, 2), 'examples_per_second': round(samples / elapsed, 1)}
            print(json.dumps(stats), flush=True)
    return stats


def pretrain(args):
    args.out.mkdir(parents=True, exist_ok=False)
    paths = paths_for('train')
    rng = np.random.default_rng(args.seed)
    paths = [paths[i] for i in rng.permutation(len(paths))]
    holdout, train = paths[:2], paths[2:2 + args.shards]
    if not train or len(holdout) != 2:
        raise ValueError('need training shards and two train-only diagnostic shards')
    table = vector_table('train')
    stage_seconds = 0.0
    if args.stage_table:
        start = time.perf_counter()
        table = np.array(table)
        stage_seconds = round(time.perf_counter() - start, 2)
        print(f'staged {table.nbytes:,} embedding bytes in {stage_seconds}s', flush=True)
    torch.manual_seed(args.seed)
    model = SequenceEncoder(args.width)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.01)
    with torch.no_grad():
        baseline = next_item_pass(model, table, holdout, args, np.random.default_rng(args.seed))
    training = next_item_pass(model, table, train, args, rng, optimizer)
    with torch.no_grad():
        measured = next_item_pass(model, table, holdout, args, np.random.default_rng(args.seed))
    comparison = None
    if args.compare_encoder is not None:
        checkpoint = torch.load(args.compare_encoder, weights_only=True, map_location='cpu')
        if checkpoint['width'] != args.width or checkpoint['slots'] != SLOTS:
            raise ValueError('comparison encoder has a different architecture')
        prior = SequenceEncoder(args.width)
        prior.load_state_dict(checkpoint['model'])
        with torch.no_grad():
            stats = next_item_pass(prior, table, holdout, args, np.random.default_rng(args.seed))
        comparison = {'path': str(args.compare_encoder), 'sha256': sha256(args.compare_encoder), 'holdout': stats}
    torch.save({'model': model.state_dict(), 'width': args.width, 'slots': SLOTS, 'seed': args.seed},
               args.out / 'encoder.pt')
    write_json(args.out / 'pretrain.json', {'seed': args.seed, 'width': args.width, 'lr': args.lr,
               'threads': args.threads, 'batch_size': args.batch_size, 'slots': SLOTS,
               'target_slots': TARGET_SLOTS, 'stage_seconds': stage_seconds,
               'novel_targets': args.novel_targets, 'comparison': comparison,
               'holdout_shards': [p.name for p in holdout],
               'train_shards': [p.name for p in train], 'before': baseline, 'after': measured,
               'train': training})


def aligned_histories(split, feature_path, feed_ids):
    with np.load(feature_path) as cache:
        wanted = set(cache['shards'].tolist()) if 'shards' in cache else None
    paths = [path for path in paths_for(split) if wanted is None or path.name in wanted]
    if wanted is not None and len(paths) != len(wanted):
        raise ValueError('cached source shards are missing')
    lookup = {int(feed): i for i, feed in enumerate(feed_ids)}
    indices = np.full((len(feed_ids), SLOTS), -1, dtype=np.int32)
    codes = np.zeros_like(indices, dtype=np.int8)
    lens = np.zeros(len(feed_ids), dtype=np.int32)
    seen = np.zeros(len(feed_ids), dtype=bool)
    for path in paths:
        for batch in pq.ParquetFile(path).iter_batches(batch_size=1024, columns=['feed_id', *HISTORY]):
            rows = np.fromiter((lookup[feed] for feed in batch['feed_id'].to_numpy()), dtype=np.int64)
            if seen[rows].any() or len(np.unique(rows)) != len(rows):
                raise ValueError(f'duplicate source feed in {path}')
            _, idx, events, lengths = read_history(batch, SLOTS)
            indices[rows], codes[rows], lens[rows] = history_window(
                idx, events, lengths, np.zeros_like(lengths), SLOTS)
            seen[rows] = True
    if not seen.all():
        raise ValueError('source history does not cover the full feed cache')
    root = DATA / split / 'item_embeddings'
    meta = json.loads((root / 'data.meta').read_text(encoding='utf-8'))
    vectors = fetch_vectors(root / 'data.memmap', meta, indices)
    print(f'{split} history: {len(feed_ids):,} aligned feeds, {np.count_nonzero(indices >= 0):,} vectors',
          flush=True)
    return vectors, codes, lens


class SequenceFeeds(Feeds):
    def __init__(self, feature_path, vectors_path, split, logged_positions=False):
        super().__init__(feature_path, labels=split != 'test', keys=True, embeddings=vectors_path,
                         logged_positions=logged_positions)
        self.histories = aligned_histories(split, feature_path, self.keys['feed'])

    def history_batch(self, rows):
        feeds = np.searchsorted(self.offsets, rows[:, 0], side='right') - 1
        return tuple(torch.from_numpy(values[feeds]) for values in self.histories)


class SequenceRanker(FeedRanker):
    def __init__(self, mean, scale, base_width, seq_width, candidate_slots=0, candidate_specific=False):
        # Start the sequence branch from the same RNG state as pretraining.
        sequence = SequenceEncoder(seq_width)
        super().__init__(mean, scale, base_width, use_context=True, use_vectors=True,
                         candidate_slots=candidate_slots, candidate_specific=candidate_specific)
        self.sequence = sequence
        self.strength = nn.Parameter(torch.tensor(.5))

    def forward(self, x, mask, vectors, history):
        base = super().forward(x, mask, vectors)
        query = self.sequence.history(*history)
        items = self.sequence.item(vectors[0])
        return base + self.strength * torch.bmm(items, query.unsqueeze(-1)).squeeze(-1)


@torch.inference_mode()
def score_sequences(model, feeds, batch_size):
    model.eval()
    scores = np.empty(len(feeds.x), dtype=np.float32)
    for x, mask, rows, keep in feeds.batches(np.arange(len(feeds.counts)), batch_size):
        scores[rows[keep]] = model(x, mask, feeds.vector_batch(rows), feeds.history_batch(rows)).numpy()[keep]
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite sequence predictions')
    return scores


def position_band_loss(score, label, mask, position):
    """Compare each positive to its position band inside the same feed."""
    bands = torch.bucketize(position.contiguous(), position.new_tensor(POSITION_EDGES), right=True)
    shape = (score.shape[0], len(POSITION_EDGES) + 1)
    positive = label.bool() & mask
    negative = mask & ~positive
    negative_count = bands.new_zeros(shape).scatter_add_(1, bands, negative.long())
    eligible = positive & (negative_count.gather(1, bands) > 0)
    masked_score = score.detach().masked_fill(~mask, -torch.inf)
    maximum = score.new_full(shape, -torch.inf).scatter_reduce_(1, bands, masked_score, reduce='amax')
    centered = (score - maximum.gather(1, bands)).masked_fill(~mask, 0)
    totals = score.new_zeros(shape).scatter_add_(1, bands, centered.exp() * mask)
    log_normalizer = totals.clamp_min(torch.finfo(score.dtype).tiny).log().gather(1, bands)
    eligible_count = eligible.sum(dim=1)
    losses = torch.where(eligible, log_normalizer - centered, 0).sum(dim=1) / eligible_count.clamp_min(1)
    return losses, eligible_count > 0


def tune(args):
    if args.position_conditioned and args.candidate_attention:
        raise ValueError('choose one controlled experiment per run')
    args.out.mkdir(parents=True, exist_ok=False)
    arms = ATTENTION_ARMS if args.candidate_attention else ARMS
    slots = CANDIDATE_SLOTS if args.candidate_attention else 0
    for name in arms:
        (args.out / name).mkdir()
    checkpoint = torch.load(args.encoder, weights_only=True, map_location='cpu')
    if checkpoint['slots'] != SLOTS or checkpoint['seed'] != args.seed:
        raise ValueError('pretrained checkpoint has a different history width or initial seed')
    train = SequenceFeeds(args.train_cache / 'train.npz', args.train_vectors / 'train.npz', 'train',
                          logged_positions=args.position_conditioned)
    rng = np.random.default_rng(args.seed)
    mean, scale = fit_normalization(train.x, rng)
    torch.manual_seed(args.seed)
    control = SequenceRanker(mean, scale, args.width, checkpoint['width'], candidate_slots=slots)
    models = {arms[0]: control, arms[1]: deepcopy(control)}
    models[arms[1]].sequence.load_state_dict(checkpoint['model'])
    if args.candidate_attention:
        control.sequence.load_state_dict(checkpoint['model'])
        models[arms[1]].candidate_attention.candidate_specific = True
    optimizers = {name: torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=.03)
                  for name, model in models.items()}
    n_feeds = len(train.counts)
    shuffled = rng.permutation(n_feeds)
    if args.feeds_per_epoch * args.epochs > n_feeds:
        raise ValueError('epochs require more distinct feeds than the train cache contains')
    start = time.perf_counter()
    epochs = []
    for epoch in range(args.epochs):
        selected = shuffled[epoch * args.feeds_per_epoch:(epoch + 1) * args.feeds_per_epoch]
        lr = args.lr * (.3 + .7 * (1 + math.cos(math.pi * epoch / max(args.epochs - 1, 1))) / 2)
        for name, model in models.items():
            model.train()
            optimizers[name].param_groups[0]['lr'] = lr
        running = dict.fromkeys(arms, 0.)
        band_loss = 0.
        band_feeds = 0
        steps = 0
        for x, mask, rows, _ in train.batches(selected, args.batch_size, rng):
            target = torch.from_numpy(train.target[rows])
            vectors = train.vector_batch(rows)
            history = train.history_batch(rows)
            position = torch.from_numpy(train.position[rows]) if args.position_conditioned else None
            dropout_state = torch.get_rng_state()
            for name, model in models.items():
                torch.set_rng_state(dropout_state)
                optimizer = optimizers[name]
                optimizer.zero_grad(set_to_none=True)
                logits = model(x, mask, vectors, history)
                loss = feed_loss(logits, target, mask, pointwise_weight=.25)
                if name == 'pretrained' and position is not None:
                    local, eligible = position_band_loss(logits, target, mask, position)
                    loss = loss + POSITION_LOSS_WEIGHT * local.mean()
                    band_loss += float(local.detach().sum())
                    band_feeds += int(eligible.sum())
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                running[name] += float(loss.detach()) * len(x)
            steps += 1
        entry = {'epoch': epoch + 1, 'lr': lr, 'steps_per_arm': steps, 'feeds': len(selected),
                 'candidates': int(train.counts[selected].sum()),
                 'positives': int(np.add.reduceat(train.target, train.offsets[:-1])[selected].sum()),
                 'loss': {name: loss / len(selected) for name, loss in running.items()},
                 'sequence_strength': {name: float(model.strength.detach()) for name, model in models.items()},
                 'seconds': round(time.perf_counter() - start, 2)}
        if args.position_conditioned:
            entry.update(position_band_loss=band_loss / len(selected), feeds_with_position_comparisons=band_feeds)
        epochs.append(entry)
        print(json.dumps(entry), flush=True)
        write_json(args.out / 'progress.json', epochs)
    for name, model in models.items():
        torch.save({'model': model.state_dict(), 'base_width': args.width, 'seq_width': checkpoint['width'],
                    'candidate_slots': slots, 'candidate_specific': name == 'attention'},
                   args.out / name / 'model.pt')
    del train
    val = SequenceFeeds(args.cache / 'val.npz', args.vectors / 'val.npz', 'val')
    for name, model in models.items():
        score = score_sequences(model, val, args.batch_size)
        prediction_frame(val.keys, score).write_parquet(args.out / name / 'val_predictions.parquet')
    write_json(args.out / 'tune.json', {'seed': args.seed, 'train_feeds_available': n_feeds,
               'epochs': epochs, 'batch_size': args.batch_size, 'threads': args.threads,
               'base_width': args.width, 'sequence_width': checkpoint['width'], 'lr': args.lr,
               'candidate_slots': slots, 'arms': arms,
               'position_conditioned': ({'edges': POSITION_EDGES, 'weight': POSITION_LOSS_WEIGHT}
                                        if args.position_conditioned else None),
               'encoder': str(args.encoder), 'train_cache': str(args.train_cache),
               'train_vectors': str(args.train_vectors), 'val_cache': str(args.cache),
               'val_vectors': str(args.vectors)})


def scores_for_blend(raw, meta, name, weight, arms=ARMS, difference='pretraining_delta'):
    anchor = raw['submitted']
    if weight == 0:
        return anchor.astype(np.float32)
    counts = meta['counts']
    def normalized(arm):
        return standardized_scores(raw[arm].astype(np.float64, copy=False), counts)

    base = normalized('submitted')
    new = normalized(arms[1] if name == difference else name)
    control = normalized(arms[0]) if name == difference else base
    mixed = (base + weight * (new - control)).astype(np.float32)
    return resolve_ties(mixed, anchor, meta['item'], counts)


def top10_hits(labels, scores, counts):
    offsets = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    hits = [labels[a:b][np.lexsort((labels[a:b], -scores[a:b]))[:10]].sum()
            for a, b in zip(offsets[:-1], offsets[1:])]
    return np.asarray(hits)


def assess(args, arms=ARMS, difference='pretraining_delta'):
    args.out.mkdir(parents=True, exist_ok=False)
    control_arm, variant_arm = arms
    meta = read_meta(args.cache / 'val.npz')
    time_order = meta['timestamp']
    if np.any(time_order[1:] < time_order[:-1]):
        raise ValueError('validation feeds must be sorted by timestamp')
    counts, labels = meta['counts'], meta['target']
    early = temporal_boundary(time_order, .4)
    first = temporal_boundary(time_order[:early], .65)
    late_mid = early + temporal_boundary(time_order[early:], .5)
    if (first, early - first, late_mid - early, len(counts) - late_mid) != (3749, 2017, 4317, 4340):
        raise ValueError('unexpected temporal split')
    cutoff = int(counts[:early].sum(dtype=np.int64))
    keys = prediction_frame(meta, np.zeros(len(labels), dtype=np.float32)).select(KEYS)
    raw = {'submitted': read_scores(args.anchor, keys),
           **{name: read_scores(args.run_dir / name / 'val_predictions.parquet', keys) for name in arms}}
    if args.prior_pretrained is not None:
        raw['prior_pretrained'] = read_scores(args.prior_pretrained, keys)
    early_raw = {name: value[:cutoff] for name, value in raw.items()}
    early_meta = {'item': meta['item'][:cutoff], 'counts': counts[:early]}
    early_anchor = ndcg_per_feed(labels[:cutoff], early_raw['submitted'].astype(np.float32), counts[:early])[:, 0]
    search, selected = {}, {}
    for name in (*arms, difference):
        entries = []
        for weight in WEIGHTS:
            score = scores_for_blend(early_raw, early_meta, name, weight, arms, difference)
            gain = ndcg_per_feed(labels[:cutoff], score, counts[:early])[:, 0] - early_anchor
            entries.append({'name': name, 'weight': weight, 'early_gain': float(gain.mean()),
                            'early_gain_by_block': [float(gain[:first].mean()), float(gain[first:].mean())]})
        search[name] = entries
        eligible = [entry for entry in entries if min(entry['early_gain_by_block']) > 0]
        selected[name] = max(eligible, key=selection_key, default=entries[0])
    choice = max(selected.values(), key=selection_key)
    matched_weight = selected[control_arm]['weight']
    write_json(args.out / 'early_selection.json', {'normalization': 'zscore', 'choice': choice,
               'selected_by_family': selected, 'grid': search, 'matched_weight': matched_weight})

    scores = {name: score.astype(np.float32) for name, score in raw.items()}
    for name, entry in selected.items():
        blended = scores_for_blend(raw, meta, name, entry['weight'], arms, difference)
        scores[name + '_blend'] = blended
        prediction_frame(meta, blended).write_parquet(args.out / f'{name}_blend_val_predictions.parquet')
    matched_name = f'{variant_arm}_blend_at_{control_arm}_weight'
    scores[matched_name] = scores_for_blend(raw, meta, variant_arm, matched_weight, arms, difference)
    prediction_frame(meta, scores[matched_name]).write_parquet(args.out / f'{matched_name}_val_predictions.parquet')
    comparisons = [(variant_arm, control_arm), (variant_arm + '_blend', control_arm + '_blend'),
                   (matched_name, control_arm + '_blend')]
    if args.prior_pretrained is not None:
        prior_blend = f'prior_pretrained_blend_at_{control_arm}_weight'
        scores[prior_blend] = scores_for_blend(raw, meta, 'prior_pretrained', matched_weight, arms, difference)
        prediction_frame(meta, scores[prior_blend]).write_parquet(args.out / f'{prior_blend}_val_predictions.parquet')
        comparisons.extend([(variant_arm, 'prior_pretrained'), (matched_name, prior_blend)])
    values = {name: (ndcg_per_feed(labels, score, counts), top10_hits(labels, score, counts))
              for name, score in scores.items()}
    blocks = {'early_first': (0, first), 'early_second': (first, early), 'early': (0, early),
              'late_first': (early, late_mid), 'late_second': (late_mid, len(counts)),
              'late': (early, len(counts)), 'all': (0, len(counts))}
    long = (counts[early:] >= 101) & (counts[early:] <= 300)

    def summarize(ndcg, hits):
        return {**{name: metric_summary(ndcg, hits, *bounds) for name, bounds in blocks.items()},
                'late_long_101_300': metric_summary(ndcg[early:][long], hits[early:][long], 0, int(long.sum()))}

    metrics = {name: summarize(*value) for name, value in values.items()}
    contrasts = {}
    comparisons.extend((name, 'submitted') for name in values if name != 'submitted')
    for name, control in comparisons:
        delta, hits = (a - b for a, b in zip(values[name], values[control], strict=True))
        contrasts[f'{name}_vs_{control}'] = {'gain': summarize(delta, hits),
            'late_paired_ndcg_95_interval': paired_interval(delta[early:, 0])}
    final = contrasts[choice['name'] + '_blend_vs_submitted']
    gain = final['gain']
    half = [gain[name] for name in ('late_first', 'late_second')]
    stable = choice['weight'] > 0 and min(row['ndcg'] for row in half) > 0 and (
        gain['late']['ndcg'] >= .003 or
        (gain['late']['ndcg'] >= .001 and final['late_paired_ndcg_95_interval'][0] > 0
         and all(row['ndcg_at_10'] > 0 and row['top10_positives'] > 0 for row in half)))
    report = {'choice': choice, 'prepare_submission': stable, 'normalization': 'zscore',
              'matched_weight': matched_weight,
              'anchor_val': str(args.anchor),
              'prior_pretrained_val': str(args.prior_pretrained) if args.prior_pretrained is not None else None,
              'model_sha256': {name: sha256(args.run_dir / name / 'model.pt') for name in arms},
              'block_sizes': {name: stop - start for name, (start, stop) in blocks.items()},
              'late_start_timestamp': int(time_order[early]), 'late_long_feeds': int(long.sum()),
              'metrics': metrics, 'contrasts': contrasts}
    write_json(args.out / 'assessment.json', report)
    print(json.dumps({'choice': choice, 'prepare_submission': stable,
                      'late': {name: value['late'] for name, value in metrics.items()}}), flush=True)


def selection_key(entry):
    return (min(entry['early_gain_by_block']), entry['early_gain'])


def assess_run(args):
    if args.candidate_attention:
        assess(args, arms=ATTENTION_ARMS, difference=ATTENTION_DIFFERENCE)
    else:
        assess(args)


def load_ranker(path):
    checkpoint = torch.load(path, weights_only=True, map_location='cpu')
    mean = np.zeros(len(NAMES), dtype=np.float32)
    model = SequenceRanker(mean, np.ones_like(mean), checkpoint['base_width'], checkpoint['seq_width'],
                           candidate_slots=checkpoint.get('candidate_slots', 0),
                           candidate_specific=checkpoint.get('candidate_specific', False))
    model.load_state_dict(checkpoint['model'])
    return model


def ablate_attention(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    model = load_ranker(args.model)
    attention = model.candidate_attention
    if attention is None or not attention.candidate_specific:
        raise ValueError('the ablation needs a checkpoint with candidate-specific attention')
    val = SequenceFeeds(args.cache / 'val.npz', args.vectors / 'val.npz', 'val')
    specific = score_sequences(model, val, args.batch_size)
    keys = prediction_frame(val.keys, specific).select(KEYS)
    np.testing.assert_array_equal(specific, read_scores(args.model.with_name('val_predictions.parquet'), keys))
    attention.candidate_specific = False
    shared = score_sequences(model, val, args.batch_size)
    early = temporal_boundary(val.timestamp, .4)
    counts, labels = val.counts[early:], val.target[val.offsets[early]:]
    one, two = specific[val.offsets[early]:], shared[val.offsets[early]:]
    gain = ndcg_per_feed(labels, one, counts) - ndcg_per_feed(labels, two, counts)
    cuts = np.r_[0, np.cumsum(counts, dtype=np.int64)]
    report = {'model_sha256': sha256(args.model), 'late_feeds': len(counts),
              'lost_ndcg_when_shared': gain.mean(axis=0).tolist(),
              'late_paired_ndcg_95_interval': paired_interval(gain[:, 0]),
              'mean_abs_score_change': float(np.abs(one - two).mean()),
              'late_top1_changed': sum(int(one[a:b].argmax() != two[a:b].argmax())
                                       for a, b in zip(cuts[:-1], cuts[1:]))}
    write_json(args.output, report)
    print(json.dumps(report), flush=True)


def submit(args):
    report = json.loads(args.assessment.read_text(encoding='utf-8'))
    if not report['prepare_submission']:
        raise ValueError('the early-selected blend did not meet the late validation criterion')
    if args.output.exists():
        raise FileExistsError(args.output)
    choice = report['choice']
    is_attention = choice['name'] in (*ATTENTION_ARMS, ATTENTION_DIFFERENCE)
    arms, difference = (ATTENTION_ARMS, ATTENTION_DIFFERENCE) if is_attention else (ARMS, 'pretraining_delta')
    required = arms if choice['name'] == difference else (choice['name'],)
    for name in required:
        if sha256(args.run_dir / name / 'model.pt') != report['model_sha256'][name]:
            raise ValueError(f'{name} checkpoint differs from the assessed run')
    test = SequenceFeeds(args.cache / 'test.npz', args.vectors / 'test.npz', 'test')
    keys = prediction_frame(test.keys, np.zeros(len(test.x), dtype=np.float32)).select(KEYS)
    raw = {'submitted': read_scores(args.anchor, keys)}
    for name in required:
        model = load_ranker(args.run_dir / name / 'model.pt')
        raw[name] = score_sequences(model, test, args.batch_size)
    score = scores_for_blend(raw, test.keys, choice['name'], choice['weight'], arms, difference)
    with NamedTemporaryFile(dir=args.output.parent, suffix='.parquet', delete=False) as stream:
        candidate = Path(stream.name)
    try:
        prediction_frame(test.keys, score).write_parquet(candidate)
        audit = audit_submission(candidate)
        if audit['rows'] != 2_904_806 or audit['feeds_with_ties']:
            raise ValueError(f'submission failed the required size and tie audit: {audit}')
        candidate.rename(args.output)
    finally:
        candidate.unlink(missing_ok=True)
    write_json(args.output.with_suffix('.audit.json'), audit)
    print(json.dumps(audit), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    pre = commands.add_parser('pretrain')
    pre.add_argument('out', type=Path)
    pre.add_argument('--seed', type=int, default=2026)
    pre.add_argument('--width', type=int, default=32)
    pre.add_argument('--shards', type=int, default=510)
    pre.add_argument('--batch-size', type=int, default=384)
    pre.add_argument('--threads', type=int, default=4)
    pre.add_argument('--lr', type=float, default=.002)
    pre.add_argument('--stage-table', action='store_true')
    pre.add_argument('--novel-targets', action='store_true')
    pre.add_argument('--compare-encoder', type=Path)
    pre.set_defaults(run=pretrain)
    fit = commands.add_parser('tune')
    fit.add_argument('cache', type=Path)
    fit.add_argument('vectors', type=Path)
    fit.add_argument('train_cache', type=Path)
    fit.add_argument('train_vectors', type=Path)
    fit.add_argument('encoder', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('--seed', type=int, default=2026)
    fit.add_argument('--width', type=int, default=48)
    fit.add_argument('--epochs', type=int, default=6)
    fit.add_argument('--feeds-per-epoch', type=int, default=36_000)
    fit.add_argument('--lr', type=float, default=.0015)
    fit.add_argument('--position-conditioned', action='store_true')
    fit.add_argument('--candidate-attention', action='store_true')
    fit.set_defaults(run=tune)
    val = commands.add_parser('assess')
    val.add_argument('cache', type=Path)
    val.add_argument('run_dir', type=Path)
    val.add_argument('anchor', type=Path)
    val.add_argument('out', type=Path)
    val.add_argument('--prior-pretrained', type=Path)
    val.add_argument('--candidate-attention', action='store_true')
    val.set_defaults(run=assess_run, threads=1)
    ablation = commands.add_parser('ablate-attention')
    ablation.add_argument('cache', type=Path)
    ablation.add_argument('vectors', type=Path)
    ablation.add_argument('model', type=Path)
    ablation.add_argument('output', type=Path)
    ablation.set_defaults(run=ablate_attention)
    test = commands.add_parser('submit')
    test.add_argument('cache', type=Path)
    test.add_argument('vectors', type=Path)
    test.add_argument('run_dir', type=Path)
    test.add_argument('assessment', type=Path)
    test.add_argument('anchor', type=Path)
    test.add_argument('output', type=Path)
    test.set_defaults(run=submit)
    for command in (fit, test, ablation):
        command.add_argument('--batch-size', type=int, default=96)
        command.add_argument('--threads', type=int, default=6)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    args.run(args)


if __name__ == '__main__':
    main()
