"""Measure whether later train snapshots reveal new actions on earlier feed candidates."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pyarrow.parquet as pq

from ranking import DATA


HISTORY = ['history__item_id', 'history__event_type', 'history__event_source']
META = ['user_id', 'timestamp', 'feed_id']
ORIGINS = 16
HORIZON = 24 * 3600


def aligned_start(earlier, later):
    """Use exact four-event anchors with at least two distinct item IDs."""
    if len(earlier) < 4 or len({event[0] for event in earlier[:4]}) < 2:
        return None
    anchor = earlier[:4]
    matches = [i for i in range(len(later) - 3) if later[i:i+4] == anchor]
    if len(matches) != 1:
        return None
    start = matches[0]
    # Where both snapshots still have older events, require their full overlap.
    shared = min(len(later) - start, len(earlier))
    return start if earlier[:shared] == later[start:start+shared] else None


def snapshots(path, rows, cols):
    table = pq.read_table(path, columns=cols)
    return table.take(np.sort(np.unique(rows))).to_pylist()


def event_list(row):
    return list(zip(*(row[k] for k in HISTORY), strict=True))


def run(out):
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    files = sorted((DATA / 'train/impressions').glob('part-*.parquet'))
    if len(files) != 512:
        raise ValueError('expected 512 train shards')
    indices = np.sort(np.random.default_rng(2026).choice(len(files), ORIGINS, replace=False))
    basis = {int(i): pq.read_table(files[i], columns=META) for i in indices}
    origins = np.unique(np.concatenate([t['user_id'].to_numpy() for t in basis.values()]))
    origins = origins[origins > 0]
    selected = []
    for i, f in enumerate(files):
        table = basis[i] if i in basis else pq.read_table(f, columns=META)
        users = table['user_id'].to_numpy()
        keep = np.flatnonzero(np.isin(users, origins))
        selected.append(np.column_stack((users[keep], table['timestamp'].to_numpy()[keep],
                                       table['feed_id'].to_numpy()[keep], np.full(len(keep), i), keep)))
    events = np.concatenate(selected)
    events = events[np.lexsort((events[:, 2], events[:, 1], events[:, 0]))]
    previous, following = events[:-1], events[1:]
    duration = following[:, 1] - previous[:, 1]
    # Exclude users whose adjacent timestamp ordering is ambiguous.
    ambiguous_users = np.unique(previous[(previous[:, 0] == following[:, 0]) & (duration == 0), 0])
    keep = ((previous[:, 0] == following[:, 0]) & (duration > 0) & (duration <= HORIZON)
            & np.isin(previous[:, 3], indices) & ~np.isin(previous[:, 0], ambiguous_users))
    left, right = previous[keep], following[keep]
    report = {'train_files_scanned_for_feed_keys': len(files), 'origin_shards': [files[i].name for i in indices],
              'origin_feeds': sum(len(x) for x in basis.values()), 'origin_users_nonzero': len(origins),
              'paired_origins_with_next_snapshot_within_24h': len(left), 'timestamp_tie_users_excluded': len(ambiguous_users)}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    earlier, later = {}, {}
    for shard in sorted(set(left[:, 3])):
        rows = left[left[:, 3] == shard, 4]
        for row in snapshots(files[shard], rows, [*META, *HISTORY, 'item_id', 'pos', 'target']):
            earlier[row['feed_id']] = row
    for shard in sorted(set(right[:, 3])):
        rows = right[right[:, 3] == shard, 4]
        for row in snapshots(files[shard], rows, [*META, *HISTORY]):
            later[row['feed_id']] = row
    # Rows: 0-31, 32-127, 128+. Columns: positive and zero competition labels.
    total = np.zeros((3, 2), dtype=np.int64)
    overlap_counts = np.zeros_like(total)
    event_type_counts = np.zeros((12, 2), dtype=np.int64)
    aligned = with_fresh = with_matches = with_zero_match = 0
    align_minutes = []
    for a, b in zip(left, right, strict=True):
        past, future = earlier[a[2]], later[b[2]]
        if past['user_id'] != future['user_id'] or past['timestamp'] >= future['timestamp']:
            raise ValueError('misaligned snapshot pair')
        old, new = event_list(past), event_list(future)
        cut = aligned_start(old, new)
        if cut is None:
            continue
        aligned += 1
        with_fresh += cut > 0
        align_minutes.append(float((future['timestamp'] - past['timestamp']) / 60))
        old_ids = {event[0] for event in old}
        fresh = {}
        for item, event, _ in new[:cut]:
            if item > 0 and item not in old_ids and item not in fresh:
                fresh[item] = event
        labels = np.asarray(past['target'], dtype=np.int8)
        columns = 1 - labels
        depth = np.searchsorted([32, 128], past['pos'], side='right')
        np.add.at(total, (depth, columns), 1)
        match = np.asarray([item in fresh for item in past['item_id']])
        if match.any():
            with_matches += 1
            with_zero_match += bool((match & (labels == 0)).any())
            types = np.array([fresh[item] for item, known in zip(past['item_id'], match) if known])
            np.add.at(overlap_counts, (depth[match], columns[match]), 1)
            np.add.at(event_type_counts, (types, columns[match]), 1)
    report.update(aligned_snapshots=aligned, aligned_with_new_prefix=with_fresh, feeds_with_candidate_match=with_matches,
                  feeds_with_zero_target_candidate_match=with_zero_match,
                  median_minutes_between_aligned=round(float(np.median(align_minutes)), 2) if aligned else None,
                  depth_bands=['0-31', '32-127', '128+'], label_order=['positive', 'zero'],
                  candidates_by_depth_label=total.tolist(), new_history_matches_by_depth_label=overlap_counts.tolist(),
                  event_type_counts_by_label=event_type_counts.tolist(), seconds=round(time.perf_counter()-start, 2))
    (out / 'report.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    run(parser.parse_args().out)


if __name__ == '__main__':
    main()
