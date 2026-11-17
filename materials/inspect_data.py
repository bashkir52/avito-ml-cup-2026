"""Read-only data audit: all Parquet footers/scalar IDs, full val/test, 8 train shards.

Run from the project root with Python 3.11, numpy, pyarrow and polars installed:
    python materials/inspect_data.py
Writes only materials/data_profile.json. Does not train a model.
"""
from collections import Counter
from datetime import datetime, timezone, timedelta
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "competition"
MOSCOW = timezone(timedelta(hours=3))


def distribution(values):
    a = np.concatenate(values)
    return {"count": len(a), "min": int(a.min()), "mean": float(a.mean()),
            "p50": float(np.quantile(a, .5)), "p95": float(np.quantile(a, .95)),
            "max": int(a.max()), "sum": int(a.sum()), "zeros": int((a == 0).sum())}


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


report = {"checked_at_moscow": datetime.now(MOSCOW).isoformat(),
          "versions": {"numpy": np.__version__, "polars": pl.__version__, "pyarrow": pyarrow.__version__},
          "scope": "All footers, feed_id/user_id/timestamp; all val/test content; 8 evenly spaced full train shards.",
          "splits": {}}
ids, users, test_keys = {}, {}, []
for split in ("train", "val", "test"):
    folder = DATA / split
    paths = sorted((folder / "impressions").glob("part-*.parquet"))
    chosen = set(paths if split != "train" else [paths[i] for i in np.linspace(0, len(paths)-1, 8, dtype=int)])
    meta = json.loads((folder / "item_embeddings/data.meta").read_text())
    emb_path = folder / "item_embeddings/data.memmap"
    # Python integers avoid NumPy 1.x's platform-dependent int32 overflow on Windows.
    emb_size = math.prod(meta["shape"]) * np.dtype(meta["dtype"]).itemsize
    result = {"parquet_files": len(paths), "parquet_bytes": sum(p.stat().st_size for p in paths),
              "embedding_meta": meta, "embedding_bytes": emb_path.stat().st_size,
              "embedding_size_matches_meta": emb_path.stat().st_size == emb_size,
              "success_marker_present": (folder / "impressions/_SUCCESS").exists(),
              "profiled_files": [p.name for p in paths if p in chosen],
              "profile_scope": "sample: 8 of 512 train shards" if split == "train" else "complete split"}
    checks = Counter()
    counts, histories, positives = [], [], []
    scalars = []
    floats = {f"f{i}": {"count": 0, "nonfinite": 0, "min": float("inf"), "max": float("-inf"), "sum": 0.0} for i in range(1, 6)}
    codes = {c: Counter() for c in ("category_id", "history__category_id", "history__event_type", "history__event_source")}
    missing = Counter()
    rows_total = 0
    candidate_slots = 0
    initial_schema = pq.ParquetFile(paths[0]).schema_arrow
    result["schema"] = {f.name: str(f.type) for f in initial_schema}
    candidate_columns = ["item_id", "category_id", "item_id__embedding_idx", *floats]
    candidate_columns += [c for c in ("pos", "target") if c in initial_schema.names]
    history_columns = [c for c in initial_schema.names if c.startswith("history__")]
    for path in paths:
        pf = pq.ParquetFile(path)
        checks["schema_mismatched_files"] += int(not pf.schema_arrow.equals(initial_schema, check_metadata=False))
        rows_total += pf.metadata.num_rows
        leaf = [pf.schema.column(i).path for i in range(len(pf.schema))]
        item_column = next(i for i, name in enumerate(leaf) if name.startswith("item_id."))
        candidate_slots += sum(pf.metadata.row_group(i).column(item_column).num_values for i in range(pf.metadata.num_row_groups))
        scalar_table = pf.read(columns=["feed_id", "user_id", "timestamp"])
        scalars.append(np.column_stack([scalar_table[c].to_numpy() for c in scalar_table.column_names]))
        if path not in chosen:
            continue
        for batch in pf.iter_batches(batch_size=256):
            df = pl.from_arrow(batch)
            m = df["item_id"].list.len()
            h = df["history__item_id"].list.len()
            counts.append(m.to_numpy())
            histories.append(h.to_numpy())
            checks["feeds_under_5_candidates"] += int((m < 5).sum())
            checks["histories_over_256"] += int((h > 256).sum())
            checks["user_vectors_not_128"] += int((df["user_embedding"].list.len() != 128).sum())
            checks["feeds_with_duplicate_items"] += int((df["item_id"].list.n_unique() != m).sum())
            checks["outer_nulls"] += int(df.null_count().to_numpy().sum())
            for col in candidate_columns + history_columns + ["user_embedding"]:
                checks["inner_nulls"] += int(df.select(pl.col(col).list.eval(pl.element().is_null()).list.sum().sum()).item() or 0)
            for col in candidate_columns:
                checks["candidate_length_mismatches"] += int((df[col].list.len() != m).sum())
            for col in history_columns:
                checks["history_length_mismatches"] += int((df[col].list.len() != h).sum())
            for col in ("item_id__embedding_idx", "history__item_id__embedding_idx"):
                a = df[col].explode().drop_nulls().to_numpy()
                missing[col + "_missing"] += int((a == -1).sum())
                missing[col + "_total"] += len(a)
                checks["embedding_indices_out_of_range"] += int(((a < -1) | (a >= meta["shape"][0])).sum())
            for col, stats in floats.items():
                a = df[col].explode().to_numpy()
                finite = np.isfinite(a)
                stats["count"] += len(a)
                stats["nonfinite"] += int((~finite).sum())
                if finite.any():
                    valid = a[finite]
                    stats["min"] = min(stats["min"], float(valid.min()))
                    stats["max"] = max(stats["max"], float(valid.max()))
                    stats["sum"] += float(valid.sum(dtype=np.float64))
            for col in codes:
                values, freq = np.unique(df[col].explode().drop_nulls().to_numpy(), return_counts=True)
                codes[col].update({int(k): int(v) for k, v in zip(values, freq)})
            if "target" in df.columns:
                a = df["target"].explode().to_numpy()
                checks["nonbinary_targets"] += int(((a != 0) & (a != 1)).sum())
                p = df.select(pl.col("target").list.eval(pl.element().cast(pl.Int64)).list.sum()).to_series().to_numpy()
                positives.append(p)
                checks["feeds_without_positive"] += int((p == 0).sum())
            if "pos" in df.columns:
                a = df["pos"].explode().to_numpy()
                checks["positions_outside_0_300"] += int(((a < 0) | (a > 300)).sum())
            if split == "test":
                test_keys.append(df.select("feed_id", "item_id").explode("item_id"))
    scalar = np.concatenate(scalars)
    ids[split] = np.unique(scalar[:, 0])
    users[split] = np.unique(scalar[:, 1])
    result.update({"feeds_from_all_footers": rows_total, "candidate_leaf_values_from_all_footers": candidate_slots,
                   "unique_feed_ids_full_split": len(ids[split]), "unique_users_full_split": len(users[split]),
                   "duplicate_feed_ids_full_split": len(scalar) - len(ids[split]),
                   "timestamp_range_moscow": [datetime.fromtimestamp(int(t), MOSCOW).isoformat() for t in (scalar[:, 2].min(), scalar[:, 2].max())],
                   "candidates_per_feed_profile": distribution(counts), "history_length_profile": distribution(histories),
                   "checks_profile": dict(checks), "embedding_missing_profile": dict(missing),
                   "categorical_counts_profile": {k: dict(sorted(v.items())) for k, v in codes.items()}})
    for stats in floats.values():
        stats["mean"] = stats.pop("sum") / (stats["count"] - stats["nonfinite"])
    result["numeric_features_profile"] = floats
    if positives:
        result["positives_per_feed_profile"] = distribution(positives)
        result["positive_candidate_fraction_profile"] = result["positives_per_feed_profile"]["sum"] / result["candidates_per_feed_profile"]["sum"]
    embeddings = np.memmap(emb_path, dtype=meta["dtype"], mode="r", shape=tuple(meta["shape"]), order=meta["order"])
    vectors = embeddings[np.linspace(0, meta["shape"][0]-1, 128, dtype=int)].astype(np.float32)
    result["embedding_read_sample"] = {"rows": len(vectors), "min_norm": float(np.linalg.norm(vectors, axis=1).min()), "max_norm": float(np.linalg.norm(vectors, axis=1).max())}
    report["splits"][split] = result
    print(json.dumps({"split": split, "feeds": rows_total, "candidate_slots": candidate_slots, "profiled_feeds": result["candidates_per_feed_profile"]["count"], "checks": dict(checks)}), flush=True)

report["cross_split"] = {f"{a}_{b}": {"shared_feed_ids": len(np.intersect1d(ids[a], ids[b])), "shared_users": len(np.intersect1d(users[a], users[b]))} for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}
sample = pl.read_parquet(DATA / "sample_submission.parquet")
keys = pl.concat(test_keys).sort("feed_id", "item_id")
sample_keys = sample.select("feed_id", "item_id").sort("feed_id", "item_id")
report["submission_template"] = {"rows": len(sample), "schema": {k: str(v) for k, v in sample.schema.items()},
    "duplicate_pairs": len(sample) - sample.select("feed_id", "item_id").unique().height,
    "nulls": int(sample.null_count().to_numpy().sum()), "scores_finite": bool(sample["score"].is_finite().all()),
    "scores_all_zero": bool((sample["score"] == 0).all()), "pairs_exactly_match_test": keys.equals(sample_keys),
    "root_copy_sha256": sha256(ROOT / "sample_submission.parquet"),
    "competition_copy_sha256": sha256(DATA / "sample_submission.parquet")}
report["small_source_sha256"] = {str(p.relative_to(ROOT)): sha256(p) for p in sorted((ROOT / "materials/sources").glob("*"))}
out = ROOT / "materials/data_profile.json"
out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
print(f"Saved {out}", flush=True)
