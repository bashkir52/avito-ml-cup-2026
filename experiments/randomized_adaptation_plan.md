# Randomized-feed adaptation implementation plan

Goal: test whether labeled train feeds preserve transferable representations during adaptation to randomized exposures.
Spec: user request of 2026-11-16; materials/HACKATHON.md and cycles 9, 11, 19, 21–27 of materials/EXPERIMENTS.md.
Architecture: reuse SequenceFeeds and SequenceRanker, all 79 features, existing embeddings and listwise + 0.25 binary loss. Start every fit from sequence-finetune-048/pretrained/model.pt (fixed sixth epoch). Unfreeze the local residual hidden block and output head; frozen context operations still propagate gradients to local. Freeze input, embedding projections, normalization, context weights, GRU and strength.

Protocol (written before new scores):
- Frozen checkpoint; random-only adaptation; joint adaptation with separate train head initialized from the original head, plus learned depth-bin bias (edges 2/4/8/16/32/64/128/256). Inference uses only the random head and never pos.
- Identical randomized batches, dropout RNG, updates, shared trainable parameters and anchor penalty for both adapted branches. Joint loss = random loss + 0.5 train loss + anchor penalty. Each step has one full randomized batch and one full train batch, no candidate subsampling.
- AdamW, weight_decay=0, anchor = 0.1 * sum of squared deviations of local and random-head parameters from checkpoint. Auxiliary head LR=0.001, no auxiliary anchor. Main LR grid 0.00003/0.0001, update checkpoints 100/300, batch size 64, seed 2026, six CPU threads. Confirm budget by 20-step pilot before starting grid; adjustments only for measured resources, never late quality.
- Forward folds: fit [0,1870), evaluate [1870,3749); fit [0,3749), evaluate [3749,5766). Select ONE shared configuration by largest minimum joint-vs-frozen full NDCG gain across the two blocks; tie by weighted mean, then fewer updates/lower LR. Also report random-only grid without giving it an unequal-budget main comparison.
- No blends in this experiment. The exact submitted ensemble is an external reference, never a fitting or selection target (its early calibration labels overlap the forward folds).
- Reset both branches to original checkpoint, fit all 5766 early feeds with the selected update count, then evaluate 8657 late feeds once. Save all predictions and checkpoints; report NDCG/full/10/30, halves 4317/4340, long feeds 101–300, top10 positives and paired 2000-resample bootstrap intervals.
- Second seed and submission only if an early-selected adapted branch beats submitted ensemble in both halves and either gains >=0.003, or >=0.001 with positive CI lower bound and positive @10 and top10-hit gains in both halves. No post-late tuning. Second seed must confirm stability with same config. Audit any generated submission against all 2904806 template pairs, finite scores and zero feed ties.

Tasks / ledger:
- [x] Verify source load and reproduce every saved validation score before adaptation; hash checkpoint and predictions.
- [x] Add meaningful tests for train-head isolation, gradient flow/frozen parameters, matched zero-aux updates, anchor, depth boundaries, chronological schedule isolation, checkpoint round trip.
- [x] Run 20-step pilot, record throughput and memory, freeze budget.
- [x] Run both forward folds, save grid and selection before final fit.
- [x] Fit final branches, evaluate once, independently verify saved artifacts and metrics, run complete tests.
- [x] Record results and commands in EXPERIMENTS.md; conditionally second seed/submission/audit.

Ruling: execute directly in the requested workspace (not a git repository); user has supplied and authorized the experimental design and full execution. No approval pause or branch operations required.

Completion ledger, 2026-11-16:
- Source and adapted checkpoint predictions reproduced bitwise on 1389971 pairs. Review discovered requires_grad is absent from state_dict; load_adapted now restores flags. The failed roundtrip assertion stopped execution before late metrics. Resumed from locked saved weights without retraining or retuning.
- Pilot 2.107s / 20 steps; peak loading RSS 20.87 GiB, working RSS 6.48 GiB. Budget unchanged. Selection lr=0.0001, steps=100.
- Both final branches saw every one of 5766 early feeds (6086 presentations); joint also 6400 distinct full train feeds. Batch size means up to64 feeds; nine random batches were partial, one had a single feed. Existing batching preserved.
- Joint late NDCG 0.3698439669, random-only 0.3699837160, frozen 0.3704108748, submitted 0.3729571873. Joint minus random-only -0.0001397491, paired95% [-0.0002721623,-0.0000370678]. No second-seed/submission trigger.
- 72 tests passed. Independent saved-key NumPy reference metrics, schedules, selection and unchanged frozen tensors passed. CatBoost dependency unavailable; its evaluator engine was not claimed as run.
- Existing best submission re-audited: 2904806 template pairs, finite scores, zero tied feeds. No new submission generated. Full record: materials/EXPERIMENTS.md cycle28.
