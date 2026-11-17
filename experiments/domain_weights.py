"""Estimate tempered train-to-randomized-val density ratios without val targets."""
import argparse
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from scipy.stats import rankdata

from ranking import NAMES, temporal_boundary


KEYS = ('feed', 'item', 'counts')


def sample_candidates(counts, eligible_feeds, n, rng):
    rows = np.flatnonzero(np.repeat(eligible_feeds, counts))
    return rng.choice(rows, size=min(len(rows), n), replace=False)


def fit(cache, out, seed, threads):
    if out.exists():
        raise FileExistsError(out)
    with np.load(cache / 'train.npz') as data:
        keys = {key: data[key] for key in KEYS}
        x = data['x']
    with np.load(cache / 'val.npz') as data:
        timestamp = data['timestamp']
        early = temporal_boundary(timestamp, .4)
        fit_early = temporal_boundary(timestamp[:early], .65)
        val_counts = data['counts'][:early]
        val_x = data['x'][:int(val_counts.sum())].copy()

    # Item priors use different train/val time windows and can expose the split by construction.
    columns = [i for i, name in enumerate(NAMES) if not name.startswith('item_log_') and name != 'item_smoothed_rate']
    rng = np.random.default_rng(seed)
    counts = keys['counts']
    train_fit = rng.permutation(len(counts)) < int(len(counts) * .8)
    val_fit = np.arange(early) < fit_early

    def domain_examples(train_mask, val_mask, n):
        source = sample_candidates(counts, train_mask, n, rng)
        destination = sample_candidates(val_counts, val_mask, n, rng)
        n = min(len(source), len(destination))
        features = np.concatenate((x[np.ix_(source[:n], columns)], val_x[np.ix_(destination[:n], columns)]))
        return features, np.repeat(np.array([0, 1], dtype=np.uint8), n)

    features, domains = domain_examples(train_fit, val_fit, 120_000)
    names = [NAMES[i] for i in columns]
    categories = ['category', 'intent_match_type', 'intent_match_source']
    data = lgb.Dataset(features, label=domains, feature_name=names, categorical_feature=categories)
    params = {'objective': 'binary', 'learning_rate': .05, 'metric': 'None', 'max_depth': 4, 'num_leaves': 15,
              'min_data_in_leaf': 400, 'lambda_l2': 5, 'verbosity': -1, 'num_threads': threads, 'seed': seed}
    learner = lgb.train(params, data, num_boost_round=70)
    heldout, label = domain_examples(~train_fit, ~val_fit, 70_000)
    prediction = learner.predict(heldout, num_threads=threads)
    positives = int(label.sum())
    auc = (rankdata(prediction)[label == 1].sum() - positives*(positives+1)/2) / (positives*(len(label)-positives))
    importance = sorted(zip(names, learner.feature_importance('gain').tolist()), key=lambda pair: -pair[1])
    print(f'held-out domain AUC={auc:.4f}; top feature gains: {importance[:5]}', flush=True)

    weights = np.empty(len(x), dtype=np.float32)
    for left in range(0, len(x), 200_000):
        right = min(left + 200_000, len(x))
        log_ratio = learner.predict(x[left:right, columns], raw_score=True, num_threads=threads)
        weights[left:right] = np.exp(.5 * np.clip(log_ratio, -np.log(4), np.log(4)))
    weights /= weights.mean(dtype=np.float64)
    metrics = {'heldout_auc': float(auc), 'domain_fit_rows': len(domains), 'domain_holdout_rows': len(label),
               'train_fit_feeds': int(train_fit.sum()), 'val_fit_feeds': fit_early,
               'val_holdout_feeds': early-fit_early, 'params': params, 'trees': 70,
               'importance_gain': dict(importance), 'ratio_clip': [.25, 4.0], 'ratio_power': .5,
               'train_weight_quantiles': np.quantile(weights, [0, .05, .5, .95, 1]).tolist(),
               'effective_sample_fraction': float(1 / np.square(weights).mean(dtype=np.float64))}
    out.mkdir(parents=True, exist_ok=False)
    np.savez(out / 'train_weights.npz', **keys, weight=weights)
    (out / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n', encoding='utf-8')
    print(f"train weight quantiles: {metrics['train_weight_quantiles']}", flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path)
    parser.add_argument('out', type=Path)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    fit(args.cache, args.out, args.seed, args.threads)
