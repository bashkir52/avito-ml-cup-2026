"""Transfer a user's earlier positive responses from the full train split."""
import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from feed_width import metric_summary, paired_interval
from neural_blend import read_meta, read_scores, resolve_ties
from ranking import DATA, KEYS, ndcg_per_feed, normalize, prediction_frame, read_vectors, standardized_scores, temporal_boundary
from semantic_transition_rerank import top_hits
from semantic_transitions import vector_table


ITEM_INDEX = 'item_id__embedding_idx'
HISTORY_INDEX = 'history__item_id__embedding_idx'
CATEGORIES = 128
NAMES = ('prior_positive', 'prior_positive_absent_from_history', 'category_lift', 'mean')
WEIGHTS = (.025, .05, .10, .20, .35, .50)


def shards(split):
    paths = sorted((DATA / split / 'impressions').glob('part-*.parquet'))
    expected = {'train': 512, 'val': 64, 'test': 64}[split]
    if len(paths) != expected:
        raise ValueError(f'expected {expected} complete files for {split}, found {len(paths)}')
    return paths


def users_for_split(split):
    return np.concatenate([pq.read_table(path, columns=['user_id'])['user_id'].to_numpy()
                           for path in shards(split)])


def fit(out):
    if out.exists():
        raise FileExistsError(out)
    val_users, test_users = map(users_for_split, ('val', 'test'))
    users = np.unique(np.r_[val_users[val_users > 0], test_users[test_users > 0]])
    if not len(users):
        raise ValueError('no positive user IDs in evaluation splits')
    files = shards('train')
    source_feeds, source_candidates = 0, 0
    feeds = np.zeros(len(users), dtype=np.int32)
    last_time = np.zeros(len(users), dtype=np.int32)
    exposure = np.zeros((len(users), CATEGORIES), dtype=np.int32)
    response = np.zeros_like(exposure)
    category_present = np.zeros(CATEGORIES, dtype=bool)
    positive_items, positive_indices, owners = [], [], []
    columns = ['user_id', 'timestamp', 'item_id', 'category_id', 'target', ITEM_INDEX]
    for i, path in enumerate(files, 1):
        table = pq.read_table(path, columns=columns).combine_chunks()
        source_feeds += len(table)
        source_candidates += len(table['item_id'].chunk(0).flatten())
        category = table['category_id'].chunk(0).flatten().to_numpy()
        category_present |= np.bincount(category, minlength=CATEGORIES) > 0
        train_users = table['user_id'].to_numpy()
        locations = np.searchsorted(users, train_users).clip(max=len(users) - 1)
        matched = np.flatnonzero(users[locations] == train_users)
        for owner, row in zip(locations[matched], table.take(matched).to_pylist(), strict=True):
            category = np.asarray(row['category_id'], dtype=np.int8)
            target = np.asarray(row['target'], dtype=np.int8)
            item = np.asarray(row['item_id'], dtype=np.int32)
            indices = np.asarray(row[ITEM_INDEX], dtype=np.int32)
            if not (len(item) == len(category) == len(indices) == len(target)) or np.any((target != 0) & (target != 1)):
                raise ValueError('train item, category, embedding and binary response arrays must align')
            positive = target == 1
            feeds[owner] += 1
            last_time[owner] = max(last_time[owner], row['timestamp'])
            exposure[owner] += np.bincount(category, minlength=CATEGORIES).astype(np.int32)
            response[owner] += np.bincount(category[positive], minlength=CATEGORIES).astype(np.int32)
            positive_items.append(item[positive])
            positive_indices.append(indices[positive])
            owners.append(np.full(positive.sum(), owner, dtype=np.int32))
        if i % 128 == 0 or i == len(files):
            print(f'{i}/{len(files)} train files; {feeds.sum():,} personal train feeds', flush=True)

    if not owners:
        raise ValueError('no users shared between train and evaluation splits')
    owner = np.concatenate(owners)
    order = np.argsort(owner, kind='stable')
    pos_items = np.concatenate(positive_items)[order]
    pos_indices = np.concatenate(positive_indices)[order]
    counts = np.bincount(owner, minlength=len(users))
    np.testing.assert_array_equal(counts, response.sum(axis=1))
    pos_vectors, _ = read_vectors(vector_table('train'), pos_indices)
    pos_vectors = normalize(pos_vectors)
    prior = (exposure + 1) / (exposure.sum(axis=1, keepdims=True) + category_present.sum())
    distribution = (response + 3 * prior) / (counts[:, None] + 3)
    lift = np.log(distribution / prior).astype(np.float32)
    lift[:, ~category_present] = 0
    present_users = users[feeds > 0]
    report = {'train_files': len(files), 'train_feeds_scanned': source_feeds,
              'train_candidates_scanned': source_candidates, 'categories': int(category_present.sum()),
              'users_with_train_profile': int(len(present_users)), 'matched_train_feeds': int(feeds.sum()),
              'matched_train_candidates': int(exposure.sum()), 'matched_train_positives': len(pos_items),
              'matched_train_positives_with_embedding': int(np.any(pos_vectors, axis=1).sum()),
              'split_coverage': {split: {'feeds': len(ids), 'feeds_with_zero_id': int((ids == 0).sum()),
                                         'covered_feeds': int(np.isin(ids, present_users).sum()),
                                         'covered_users': int(np.isin(np.unique(ids), present_users).sum())}
                                 for split, ids in (('val', val_users), ('test', test_users))}}
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'profiles.npz', users=users, feeds=feeds, last_time=last_time, counts=counts,
             items=pos_items, vectors=pos_vectors, lift=lift)
    (out / 'profile_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


def centered(values, mask):
    result = np.zeros(len(values), dtype=np.float32)
    active = values[mask]
    if len(active) and active.std() > 1e-6:
        result[mask] = (active - active.mean()) / active.std()
        return result, True
    return result, False


def feed_signals(categories, candidate_vectors, history_items, positive_items, positive_vectors, lift):
    vectors_present = np.any(candidate_vectors, axis=1)
    positives_present = np.any(positive_vectors, axis=1)
    novel = ~np.isin(positive_items, history_items)
    sim = candidate_vectors @ positive_vectors.T
    signals = []
    for mask in (positives_present, positives_present & novel):
        if mask.any():
            signals.append(centered(sim[:, mask].max(axis=1), vectors_present))
        else:
            signals.append((np.zeros(len(categories), dtype=np.float32), False))
    signals.append(centered(lift[categories], np.ones(len(categories), dtype=bool)))
    values = np.column_stack([value for value, _ in signals])
    active = np.array([available for _, available in signals])
    average = values.sum(axis=1) / max(int(active.sum()), 1)
    return np.column_stack((values, average)), np.r_[active, active.any()], novel


def aligned_signals(meta, split, profile_path):
    with np.load(profile_path) as archive:
        profiles = dict(archive)
    offsets = np.r_[0, np.cumsum(meta['counts'], dtype=np.int64)]
    positive_offsets = np.r_[0, np.cumsum(profiles['counts'], dtype=np.int64)]
    lookup = {int(key): i for i, key in enumerate(meta['feed'])}
    users = {int(key): i for i, key in enumerate(profiles['users'])}
    table = vector_table(split)
    seen = np.zeros(len(lookup), dtype=bool)
    covered = np.zeros(len(lookup), dtype=bool)
    active = np.zeros((len(lookup), len(NAMES)), dtype=bool)
    prior_count = np.zeros(len(lookup), dtype=np.int32)
    novel_count = np.zeros_like(prior_count)
    exact_match = np.zeros(len(meta['item']), dtype=bool)
    signals = np.zeros((len(meta['item']), len(NAMES)), dtype=np.float32)
    common_id_cosines = []
    columns = ['feed_id', 'user_id', 'timestamp', 'item_id', 'category_id', ITEM_INDEX, 'history__item_id', HISTORY_INDEX]
    for path in shards(split):
        for row in pq.read_table(path, columns=columns).to_pylist():
            i = lookup[row['feed_id']]
            if seen[i] or row['timestamp'] != meta['timestamp'][i]:
                raise ValueError('raw feed keys or timestamps do not align with the cache')
            seen[i] = True
            sl = slice(offsets[i], offsets[i + 1])
            np.testing.assert_array_equal(meta['item'][sl], row['item_id'])
            owner = users.get(row['user_id'])
            if owner is None or not profiles['feeds'][owner]:
                continue
            if profiles['last_time'][owner] >= row['timestamp']:
                raise ValueError('profile contains a response from the evaluation feed time or later')
            covered[i] = True
            positives = slice(positive_offsets[owner], positive_offsets[owner + 1])
            prior_items, prior_vectors = profiles['items'][positives], profiles['vectors'][positives]
            prior_count[i] = len(prior_items)
            exact_match[sl] = np.isin(row['item_id'], prior_items)
            category = np.asarray(row['category_id'], dtype=np.int16)
            candidate_vectors, _ = read_vectors(table, row[ITEM_INDEX])
            if len(category) != meta['counts'][i] or len(candidate_vectors) != len(category):
                raise ValueError('candidate category and vector arrays do not match the cache')
            history = np.asarray(row['history__item_id'], dtype=np.int32)
            block, active[i], novel = feed_signals(category, normalize(candidate_vectors), history,
                                                    prior_items, prior_vectors, profiles['lift'][owner])
            signals[sl] = block
            novel_count[i] = int(novel.sum())
            _, previous, current = np.intersect1d(prior_items, history, return_indices=True)
            if len(previous):
                raw_history = np.asarray(row[HISTORY_INDEX], dtype=np.int32)
                common, _ = read_vectors(table, raw_history[current])
                earlier, later = prior_vectors[previous], normalize(common)
                present = np.any(earlier, axis=1) & np.any(later, axis=1)
                common_id_cosines.extend(np.sum(earlier[present] * later[present], axis=1).tolist())
    if not seen.all() or not np.isfinite(signals).all():
        raise ValueError('profile signals must cover the aligned split and be finite')
    if common_id_cosines and np.mean(common_id_cosines) < .95:
        raise ValueError('shared item IDs indicate inconsistent train and evaluation embedding spaces')
    coverage = {'covered': covered, 'active': active, 'prior_count': prior_count, 'novel_count': novel_count,
                'exact_match': exact_match, 'same_id_cosines': common_id_cosines}
    return signals, coverage


def prediction_scores(meta, anchor, signals, name, weight, end=None):
    counts = meta['counts'][:end]
    stop = int(counts.sum(dtype=np.int64))
    baseline = anchor[:stop]
    if weight == 0:
        return baseline.astype(np.float32)
    scores = standardized_scores(baseline, counts) + weight * signals[:stop, NAMES.index(name)]
    return resolve_ties(scores, baseline, meta['item'][:stop], counts)


def select(cache, profiles, anchor_path, out):
    if out.exists():
        raise FileExistsError(out)
    meta = read_meta(cache / 'val.npz')
    counts, target, time = (meta[key] for key in ('counts', 'target', 'timestamp'))
    keys = prediction_frame(meta, np.zeros(len(target))).select(KEYS)
    anchor = read_scores(anchor_path, keys)
    signals, coverage = aligned_signals(meta, 'val', profiles)
    end = temporal_boundary(time, .4)
    middle = temporal_boundary(time[:end], .65)
    late_middle = end + temporal_boundary(time[end:], .5)
    if (middle, end - middle, late_middle - end, len(counts) - late_middle) != (3749, 2017, 4317, 4340):
        raise ValueError('unexpected validation split')
    stop = int(counts[:end].sum(dtype=np.int64))
    early_baseline = ndcg_per_feed(target[:stop], anchor[:stop].astype(np.float32), counts[:end])[:, 0]
    grid = []
    for name in NAMES:
        for weight in WEIGHTS:
            score = prediction_scores(meta, anchor, signals, name, weight, end)
            d = ndcg_per_feed(target[:stop], score, counts[:end])[:, 0] - early_baseline
            grid.append({'signal': name, 'weight': weight, 'early_gain': float(d.mean()),
                         'early_gain_by_block': [float(d[:middle].mean()), float(d[middle:].mean())]})
    best = max(grid, key=lambda row: (min(row['early_gain_by_block']), row['early_gain']))
    config = best if min(best['early_gain_by_block']) > 0 else {
        'signal': best['signal'], 'weight': 0., 'early_gain': 0., 'early_gain_by_block': [0., 0.]}
    score = prediction_scores(meta, anchor, signals, config['signal'], config['weight'])
    anchor = anchor.astype(np.float32)
    anchor_metric, chosen_metric = [ndcg_per_feed(target, x, counts) for x in (anchor, score)]
    anchor_top, chosen_top = [top_hits(target, x, counts) for x in (anchor, score)]
    delta = chosen_metric - anchor_metric
    blocks = {'early_first': (0, middle), 'early_second': (middle, end), 'late': (end, len(counts)),
              'late_first': (end, late_middle), 'late_second': (late_middle, len(counts)), 'all': (0, len(counts))}

    def summarize(metric, hits):
        return {name: metric_summary(metric, hits, *bounds) for name, bounds in blocks.items()}

    covered = coverage['covered'][end:]
    prior_count, novel_count = (coverage[name][end:] for name in ('prior_count', 'novel_count'))
    cosines = coverage['same_id_cosines']
    audit = {'late_feeds_with_profile': int(covered.sum()), 'late_feeds_with_novel_positive': int((novel_count > 0).sum()),
             'late_prior_positives_across_feeds': int(prior_count.sum()), 'late_novel_positives_across_feeds': int(novel_count.sum()),
             'late_exact_candidate_matches': int(coverage['exact_match'][stop:].sum()),
             'late_exact_matches_positive': int(target[stop:][coverage['exact_match'][stop:]].sum()),
             'late_active_feeds_by_signal': dict(zip(NAMES, coverage['active'][end:].sum(axis=0).tolist())),
             'validation_shared_item_embedding_pairs': len(cosines),
             'validation_shared_item_embedding_cosine_mean': float(np.mean(cosines)) if cosines else None}
    result = {'selected': config, 'best_nonzero_early': best, 'audit': audit,
              'incumbent': summarize(anchor_metric, anchor_top), 'selected_metrics': summarize(chosen_metric, chosen_top),
              'selected_gain': summarize(delta, chosen_top - anchor_top),
              'late_paired_ndcg_95_interval': paired_interval(delta[end:, 0]),
              'late_ndcg_gain_for_covered_feeds': float(delta[end:, 0][covered].mean()) if covered.any() else None}
    out.mkdir(parents=True, exist_ok=False)
    (out / 'assessment.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    (out / 'early_search.json').write_text(json.dumps(grid, indent=2) + '\n', encoding='utf-8')
    if config['weight']:
        prediction_frame(meta, score).write_parquet(out / 'val_predictions.parquet')
    print(json.dumps({'selected': config, 'audit': audit, 'late': result['selected_metrics']['late'],
                      'delta': result['selected_gain']['late'], 'late_interval': result['late_paired_ndcg_95_interval']},
                     indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest='action', required=True)
    profile = actions.add_parser('fit')
    profile.add_argument('out', type=Path)
    validation = actions.add_parser('select')
    for field in ('cache', 'profiles', 'anchor', 'out'):
        validation.add_argument(field, type=Path)
    args = parser.parse_args()
    if args.action == 'fit':
        fit(args.out)
    else:
        select(args.cache, args.profiles, args.anchor, args.out)
