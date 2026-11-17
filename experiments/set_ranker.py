"""CPU neural ranker with shared context and a listwise loss within each feed."""
import argparse
import json
import math
from pathlib import Path
from shutil import copyfileobj
from tempfile import TemporaryDirectory
import time
from zipfile import ZipFile

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from ranking import (NAMES, audit_submission, evaluate, prediction_frame,
                     temporal_boundary)


CATEGORIES = {'category': (54, 8), 'intent_match_type': (11, 4), 'intent_match_source': (31, 4)}


class Feeds:
    def __init__(self, path, labels=True, keys=False, embeddings=None, events=None, mapped_features=None,
                 nearest=None, logged_positions=False):
        with np.load(path) as data:
            self.x = np.load(mapped_features, mmap_mode='r') if mapped_features is not None else data['x']
            self.counts = data['counts']
            self.target = data['target'] if labels else None
            self.timestamp = data['timestamp'] if labels else None
            self.position = data['position'] if logged_positions else None
            self.keys = {key: data[key] for key in ('feed', 'item', 'counts')} if keys or embeddings or events or nearest else None
        self.offsets = np.r_[0, np.cumsum(self.counts, dtype=np.int64)]
        if self.x.shape != (self.offsets[-1], len(NAMES)) or np.any(self.counts < 1):
            raise ValueError('expected aligned features and nonempty feed sizes')
        if labels and (len(self.target) != len(self.x) or len(self.timestamp) != len(self.counts)):
            raise ValueError('labels and timestamps must align with feeds')
        if logged_positions and (self.position.shape != (len(self.x),)
                                 or not np.issubdtype(self.position.dtype, np.integer)
                                 or np.any((self.position < 0) | (self.position > 300))):
            raise ValueError('invalid logged positions in training cache')
        self.vectors = None
        if embeddings is not None:
            with np.load(embeddings) as data:
                for name in self.keys:
                    if not np.array_equal(data[name], self.keys[name]):
                        raise ValueError(f'vector cache does not match feature cache {name} keys')
                self.vectors = tuple(data[k] for k in ('item_vectors', 'user_vectors', 'recent_vectors'))
            expected = [(len(self.x), 128), (len(self.counts), 128), (len(self.counts), 128)]
            if [v.shape for v in self.vectors] != expected:
                raise ValueError('unexpected embedding width or row count')
        self.events = None
        if events is not None:
            if self.vectors is None:
                raise ValueError('event attention requires the candidate and user vectors')
            with np.load(events) as data:
                if not np.array_equal(data['feed'], self.keys['feed']):
                    raise ValueError('event cache feed keys differ from feature cache')
                history, metadata = data['event_vectors'], data['events']
            if (metadata.ndim != 3 or metadata.shape[-1] != 4 or metadata.shape[0] != len(self.counts)
                    or history.shape != (*metadata.shape[:2], 128)):
                raise ValueError('invalid event cache shape')
            self.events = history, metadata
        self.nearest = None
        if nearest is not None:
            if self.vectors is None or self.events is not None:
                raise ValueError('nearest-event projection requires item vectors and excludes event attention')
            with np.load(nearest) as data:
                for name in self.keys:
                    if not np.array_equal(data[name], self.keys[name]):
                        raise ValueError(f'nearest events do not match keys in {name}')
                self.nearest = data['nearest_vectors']
            if self.nearest.shape != (len(self.x), 128):
                raise ValueError('unexpected nearest-event vector shape')

    def vector_batch(self, rows):
        if self.vectors is None:
            return None
        item, user, recent = self.vectors
        feeds = np.searchsorted(self.offsets, rows[:, 0], side='right') - 1
        batch = (item[rows], user[feeds], recent[feeds])
        if self.events is not None:
            batch += tuple(v[feeds] for v in self.events)
        if self.nearest is not None:
            batch += (self.nearest[rows],)
        return tuple(torch.from_numpy(v) for v in batch)

    def batches(self, ids, batch_size, rng=None):
        sizes = self.counts[ids]
        bins = np.searchsorted([8, 16, 32, 64, 96, 128, 192, 256], sizes)
        groups = []
        for i in np.unique(bins):
            selected = ids[bins == i]
            groups.extend(selected[j:j + batch_size] for j in range(0, len(selected), batch_size))
        if rng is not None:
            rng.shuffle(groups)
        for selected in groups:
            shape = self.counts[selected]
            radius = np.arange(shape.max())
            mask = radius < shape[:, None]
            rows = self.offsets[selected, None] + np.minimum(radius, shape[:, None] - 1)
            x = torch.from_numpy(self.x[rows])
            yield x, torch.from_numpy(mask), rows, mask


class HistoryAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.event_codes = nn.ModuleList(nn.Embedding(k, 4, padding_idx=0) for k in (12, 32, 55))
        self.query = nn.Linear(24, 24, bias=False)
        self.key_value = nn.Linear(37, 48)
        self.output = nn.Linear(48, width, bias=False)

    def forward(self, items, item_unit, vectors, metadata, item_projection):
        position = metadata[..., 0]
        keep = position >= 0
        event_unit = F.normalize(vectors.float(), dim=-1)
        embedded = [item_projection(8 * event_unit), torch.log1p(position.clamp_min(0).float()).unsqueeze(-1) / 5]
        for i, layer in enumerate(self.event_codes, 1):
            index = torch.where(keep, metadata[..., i].long() + 1, 0).clamp(0, layer.num_embeddings - 1)
            embedded.append(layer(index))
        k, v = self.key_value(torch.cat(embedded, dim=-1)).chunk(2, dim=-1)
        cosine = torch.bmm(item_unit, event_unit.transpose(1, 2))
        q = self.query(items)
        similarity = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(k.shape[-1]) + 2 * cosine
        mask = keep[:, None, :]
        attention = torch.softmax(similarity.masked_fill(~mask, -1e4), dim=-1) * mask
        context = torch.bmm(attention, v)
        return F.silu(self.output(torch.cat((context, context * items), dim=-1)))


class CandidateGroupAttention(nn.Module):
    """Pool a feed into learned groups and read them with one query or one per candidate."""

    def __init__(self, width, slots, candidate_specific=False):
        super().__init__()
        dimension = min(24, width)
        self.candidate_specific = candidate_specific
        self.scale = dimension ** -.5
        self.seeds = nn.Parameter(torch.randn(slots, dimension))
        self.keys = nn.Linear(width, dimension, bias=False)
        self.values = nn.Linear(width, dimension, bias=False)
        self.queries = nn.Linear(width, dimension, bias=False)
        self.group_keys = nn.Linear(dimension, dimension, bias=False)
        self.output = nn.Linear(dimension, width, bias=False)
        nn.init.zeros_(self.output.weight)

    def forward(self, x, mask):
        hidden = x.masked_fill(~mask.unsqueeze(-1), 0)
        affinity = (self.seeds @ self.keys(hidden).transpose(1, 2)) * self.scale
        keep = mask[:, None]
        weights = torch.softmax(affinity.masked_fill(~keep, -1e4), dim=-1) * keep
        groups = weights @ self.values(hidden)
        if not self.candidate_specific:
            hidden = hidden.sum(dim=1, keepdim=True) / mask.sum(dim=1)[:, None, None].clamp_min(1)
        logits = (self.queries(hidden) @ self.group_keys(groups).transpose(1, 2)) * self.scale
        return self.output(torch.softmax(logits, dim=-1) @ groups)


class FeedRanker(nn.Module):
    def __init__(self, mean, scale, width, use_context, use_vectors=False, use_events=False, use_nearest=False,
                 feature_names=None, candidate_slots=0, candidate_specific=False):
        super().__init__()
        self.use_context = use_context
        names = NAMES if feature_names is None else feature_names
        categories = {key: value for key, value in CATEGORIES.items() if key in names}
        self.category_indices = [names.index(k) for k in categories]
        self.register_buffer('numeric_indices', torch.tensor([i for i in range(len(names))
                                                               if i not in self.category_indices]))
        self.register_buffer('mean', torch.as_tensor(mean, dtype=torch.float32)[self.numeric_indices])
        self.register_buffer('scale', torch.as_tensor(scale, dtype=torch.float32)[self.numeric_indices])
        self.embeddings = nn.ModuleList(nn.Embedding(k + 1, d, padding_idx=0) for k, d in categories.values())
        self.input = nn.Linear(2 * len(self.numeric_indices) + sum(d for _, d in categories.values()), width)
        self.local = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Dropout(.08), nn.Linear(width, width))
        self.context = nn.Linear(2 * width, width) if use_context else None
        self.output = nn.Sequential(nn.SiLU(), nn.Dropout(.08), nn.Linear(width, 1))
        self.vector_projections = nn.ModuleList(nn.Linear(128, 24, bias=False) for _ in range(3)) if use_vectors else None
        self.vector_context = nn.Linear(72, width) if use_vectors else None
        if ((use_events or use_nearest) and not use_vectors) or (use_events and use_nearest):
            raise ValueError('one history branch can be used with the vector branch')
        self.history_attention = HistoryAttention(width) if use_events else None
        self.nearest_context = nn.Linear(48, width, bias=False) if use_nearest else None
        self.candidate_attention = (CandidateGroupAttention(width, candidate_slots, candidate_specific)
                                    if candidate_slots else None)

    def forward(self, x, mask, vectors=None):
        numeric = x.index_select(dim=-1, index=self.numeric_indices)
        valid = torch.isfinite(numeric)
        numeric = torch.where(valid, ((numeric - self.mean) / self.scale).clamp(-7, 7), 0)
        cats = []
        for i, layer in zip(self.category_indices, self.embeddings):
            index = (torch.nan_to_num(x[..., i], nan=-1).long() + 1).clamp(0, layer.num_embeddings - 1)
            cats.append(layer(index))
        h = F.silu(self.input(torch.cat([numeric, (~valid).float(), *cats], dim=-1)))
        if self.vector_projections is not None:
            item_unit, user_unit, recent_unit = (F.normalize(v.float(), dim=-1) for v in vectors[:3])
            item, user, recent = (layer(8 * v) for layer, v in
                                  zip(self.vector_projections, (item_unit, user_unit, recent_unit), strict=True))
            features = torch.cat((item, item * user[:, None], item * recent[:, None]), dim=-1)
            h = h + F.silu(self.vector_context(features))
            if self.history_attention is not None:
                h = h + self.history_attention(item, item_unit, *vectors[3:], self.vector_projections[0])
            if self.nearest_context is not None:
                nearest = self.vector_projections[0](8 * F.normalize(vectors[3].float(), dim=-1))
                h = h + F.silu(self.nearest_context(torch.cat((nearest, nearest * item), dim=-1)))
        h = h + self.local(h)
        if self.context is not None:
            keep = mask.unsqueeze(-1)
            average = (h * keep).sum(dim=1) / keep.sum(dim=1)
            maximum = h.masked_fill(~keep, -torch.inf).amax(dim=1)
            context = self.context(torch.cat((average, maximum), dim=-1))
            h = h + context[:, None, :]
        if self.candidate_attention is not None:
            h = h + self.candidate_attention(h, mask)
        return self.output(h).squeeze(-1)


def feed_loss(score, label, mask, pointwise_weight, pairwise_weight=0):
    pos = label.bool() & mask
    neg = ~pos & mask
    pos_count = pos.sum(dim=1)
    log_prob = F.log_softmax(score.masked_fill(~mask, -1e4), dim=1)
    list_loss = -(log_prob * pos).sum(dim=1) / pos_count.clamp_min(1)
    positive_loss = (F.softplus(-score) * pos).sum(dim=1) / pos_count.clamp_min(1)
    negative_loss = (F.softplus(score) * neg).sum(dim=1) / neg.sum(dim=1).clamp_min(1)
    loss = list_loss + pointwise_weight * .5 * (positive_loss + negative_loss)
    if pairwise_weight:
        k = min(8, score.shape[1])
        hard_scores, hard_indices = score.masked_fill(~neg, -1e4).topk(k, dim=1)
        hard_neg = neg.gather(dim=1, index=hard_indices)
        pair_mask = pos[:, :, None] & hard_neg[:, None, :]
        gap = hard_scores[:, None, :] - score[:, :, None]
        pair_loss = (F.softplus(gap) * pair_mask).sum(dim=(1, 2)) / pair_mask.sum(dim=(1, 2)).clamp_min(1)
        loss = loss + pairwise_weight * pair_loss
    return loss.mean()


class ExposureCurve(nn.Module):
    """Model an observed action as relevance times examination at the logged position."""

    def __init__(self):
        super().__init__()
        self.register_buffer('edges', torch.tensor([1, 2, 4, 8, 16, 32, 64, 128, 256]))
        self.steps = nn.Parameter(torch.full((len(self.edges),), math.log(math.expm1(.15))))

    def attenuation(self):
        return F.pad(F.softplus(self.steps).cumsum(dim=0), (1, 0))

    def log_probability(self, score, position):
        bins = torch.bucketize(position.contiguous(), self.edges, right=True)
        return F.logsigmoid(score) - self.attenuation()[bins]

    def loss(self, score, label, mask, position):
        log_positive = self.log_probability(score, position)
        log_negative = torch.log(-torch.expm1(log_positive.clamp_max(-1e-7)))
        action_loss = -torch.where(label.bool(), log_positive, log_negative).masked_select(mask).mean()
        list_loss = feed_loss(log_positive, label, mask, pointwise_weight=0)
        penalty = .01 * (F.softplus(self.steps) - .15).square().mean()
        return action_loss + .05 * list_loss + penalty


def fit_normalization(x, rng):
    sample = x[rng.integers(0, len(x), size=min(120_000, len(x)))].astype(np.float64)
    valid = np.isfinite(sample)
    count = valid.sum(axis=0).clip(min=1)
    mean = np.where(valid, sample, 0).sum(axis=0) / count
    var = np.square(np.where(valid, sample - mean, 0)).sum(axis=0) / count
    return mean.astype(np.float32), np.maximum(np.sqrt(var), 1e-4).astype(np.float32)


@torch.inference_mode()
def score_feeds(model, feeds, batch_size):
    model.eval()
    scores = np.empty(len(feeds.x), dtype=np.float32)
    ids = np.arange(len(feeds.counts))
    for x, mask, rows, keep in feeds.batches(ids, batch_size):
        scores[rows[keep]] = model(x, mask, feeds.vector_batch(rows)).numpy()[keep]
    if not np.isfinite(scores).all():
        raise ValueError('nonfinite scores from neural ranker')
    return scores


def train(args, mapped_features=None):
    args.out.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    vector_file = lambda split: (args.embeddings / f'{split}.npz' if args.embeddings else None)
    event_file = lambda split: (args.events / f'{split}.npz' if args.events else None)
    nearest_file = lambda split: (args.nearest / f'{split}.npz' if args.nearest else None)
    training_file = args.train_file or args.cache / 'train.npz'
    train_data = Feeds(training_file, embeddings=args.train_embeddings or vector_file('train'),
                       events=event_file('train'), mapped_features=mapped_features, nearest=nearest_file('train'),
                       logged_positions=args.exposure_model)
    val = Feeds(args.cache / 'val.npz', keys=True, embeddings=vector_file('val'),
                events=event_file('val'), nearest=nearest_file('val'))
    mean, scale = fit_normalization(train_data.x, rng)
    model = FeedRanker(mean, scale, args.width, not args.no_context, bool(args.embeddings), bool(args.events), bool(args.nearest))
    exposure = ExposureCurve() if args.exposure_model else None
    parameters = list(model.parameters())
    if exposure is not None:
        parameters.extend(exposure.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.lr, weight_decay=.03)
    start = time.monotonic()
    cutoff = temporal_boundary(val.timestamp, .4)
    best, options = -np.inf, []
    n_feeds = len(train_data.counts)
    per_epoch = min(args.feeds_per_epoch, n_feeds)
    shuffled = rng.permutation(n_feeds)
    visit = 0
    best_epoch = 0
    print(f'CPU set ranker: {n_feeds} train feeds; {per_epoch} sampled per epoch', flush=True)
    for epoch in range(1, args.epochs + 1):
        end = visit + per_epoch
        if end <= n_feeds:
            selected = shuffled[visit:end]
            visit = end
        else:
            tail = shuffled[visit:]
            shuffled = rng.permutation(n_feeds)
            visit = per_epoch - len(tail)
            selected = np.concatenate((tail, shuffled[:visit]))
        lr = args.lr * (.3 + .7 * (1 + math.cos(math.pi * (epoch - 1) / max(args.epochs - 1, 1))) / 2)
        optimizer.param_groups[0]['lr'] = lr
        model.train()
        loss_sum = 0.0
        total = 0
        for x, mask, rows, _ in train_data.batches(selected, args.batch_size, rng):
            target = torch.from_numpy(train_data.target[rows])
            optimizer.zero_grad(set_to_none=True)
            score = model(x, mask, train_data.vector_batch(rows))
            if exposure is None:
                loss = feed_loss(score, target, mask, args.pointwise_weight, args.pairwise_weight)
            else:
                loss = exposure.loss(score, target, mask, torch.from_numpy(train_data.position[rows]))
            loss.backward()
            nn.utils.clip_grad_norm_(parameters, 5)
            optimizer.step()
            loss_sum += float(loss.detach()) * len(x)
            total += len(x)
        scores = score_feeds(model, val, args.batch_size)
        metrics = evaluate({'target': val.target, 'counts': val.counts}, scores, cutoff)
        entry = {'epoch': epoch, 'loss': loss_sum / total, 'lr': lr, 'metrics': metrics,
                 'elapsed_seconds': round(time.monotonic()-start, 1)}
        if exposure is not None:
            entry['examination_probability'] = (-exposure.attenuation().detach()).exp().tolist()
        print(json.dumps(entry), flush=True)
        options.append(entry)
        if metrics['early']['ndcg'] > best:
            best, best_epoch = metrics['early']['ndcg'], epoch
            torch.save({'model': model.state_dict(), 'width': args.width, 'use_context': not args.no_context,
                        'use_vectors': bool(args.embeddings), 'use_events': bool(args.events),
                        'use_nearest': bool(args.nearest),
                        'names': NAMES}, args.out / 'model.pt')
            prediction_frame(val.keys, scores).write_parquet(args.out / 'val_predictions.parquet')
        result = {'epochs': options, 'selected_epoch': best_epoch,
                  'metrics': options[best_epoch-1]['metrics'], 'train_feeds': n_feeds,
                  'val_feeds': len(val.counts), 'early_feeds': cutoff,
                  'params': {'seed': args.seed, 'width': args.width, 'context': not args.no_context,
                             'vectors': bool(args.embeddings),
                             'event_attention': bool(args.events), 'train_file': str(training_file),
                             'nearest_history': bool(args.nearest),
                             'exposure_model': args.exposure_model,
                             'lr': args.lr, 'pointwise_weight': args.pointwise_weight,
                             'pairwise_weight': args.pairwise_weight,
                             'batch_size': args.batch_size, 'feeds_per_epoch': per_epoch,
                             'threads': args.threads, 'max_epochs': args.epochs}}
        (args.out / 'metrics.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        if epoch >= 5 and epoch - best_epoch >= args.patience:
            break


def predict(args):
    if args.output.exists():
        raise FileExistsError(args.output)
    checkpoint = torch.load(args.model, weights_only=True, map_location='cpu')
    if checkpoint['names'] != NAMES:
        raise ValueError('model expects a different feature schema')
    use_vectors = checkpoint.get('use_vectors', False)
    if use_vectors != bool(args.embeddings):
        raise ValueError('provide the vector cache only for a model trained on vectors')
    use_events = checkpoint.get('use_events', False)
    if use_events != bool(args.events):
        raise ValueError('provide an event cache for a model trained with event attention')
    use_nearest = checkpoint.get('use_nearest', False)
    if use_nearest != bool(args.nearest):
        raise ValueError('provide nearest-event vectors for a model trained with them')
    model = FeedRanker(np.zeros(len(NAMES)), np.ones(len(NAMES)), checkpoint['width'], checkpoint['use_context'],
                       use_vectors, use_events, use_nearest)
    model.load_state_dict(checkpoint['model'])
    test = Feeds(args.cache / 'test.npz', labels=False, keys=True,
                 embeddings=args.embeddings / 'test.npz' if args.embeddings else None,
                 events=args.events / 'test.npz' if args.events else None,
                 nearest=args.nearest / 'test.npz' if args.nearest else None)
    prediction_frame(test.keys, score_feeds(model, test, args.batch_size)).write_parquet(args.output)
    print(json.dumps(audit_submission(args.output)), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    fit = sub.add_parser('train')
    fit.add_argument('cache', type=Path)
    fit.add_argument('out', type=Path)
    fit.add_argument('--seed', type=int, default=2026)
    fit.add_argument('--epochs', type=int, default=10)
    fit.add_argument('--patience', type=int, default=3)
    fit.add_argument('--width', type=int, default=48)
    fit.add_argument('--feeds-per-epoch', type=int, default=36_000)
    fit.add_argument('--lr', type=float, default=.0015)
    fit.add_argument('--pointwise-weight', type=float, default=.25)
    fit.add_argument('--pairwise-weight', type=float, default=0)
    fit.add_argument('--exposure-model', action='store_true', help='fit a train-only logged-position observation model')
    fit.add_argument('--no-context', action='store_true')
    fit.add_argument('--train-file', type=Path)
    fit.add_argument('--train-embeddings', type=Path)
    fit.add_argument('--mmap-train', action='store_true')
    infer = sub.add_parser('predict')
    infer.add_argument('cache', type=Path)
    infer.add_argument('model', type=Path)
    infer.add_argument('output', type=Path)
    for command in (fit, infer):
        command.add_argument('--threads', type=int, default=6)
        command.add_argument('--batch-size', type=int, default=96)
        command.add_argument('--embeddings', type=Path)
        command.add_argument('--events', type=Path)
        command.add_argument('--nearest', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    if args.action == 'train':
        if args.train_embeddings and not args.embeddings:
            parser.error('--train-embeddings requires a validation embedding cache via --embeddings')
        if args.mmap_train:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            with TemporaryDirectory(dir=args.out.parent, ignore_cleanup_errors=True) as temp:
                mapped = Path(temp) / 'training_features.npy'
                with ZipFile(args.train_file or args.cache / 'train.npz') as archive:
                    with archive.open('x.npy') as source, mapped.open('wb') as target:
                        copyfileobj(source, target, 8 << 20)
                train(args, mapped_features=mapped)
        else:
            train(args)
    else:
        predict(args)


if __name__ == '__main__':
    main()
