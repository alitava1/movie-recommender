"""Phase 12 sparse-history / cold-start feasibility calculations for authorized local data."""

import csv
from collections import Counter
from pathlib import Path

HISTORY_GROUPS = ("0", "1-4", "5-9", "10-20", ">20")


def history_group(count):
    if count <= 0:
        return "0"
    if count <= 4:
        return "1-4"
    if count <= 9:
        return "5-9"
    if count <= 20:
        return "10-20"
    return ">20"


def sparse_history_audit(ratings_csv, personality_csv, user_col="userid", id_col=0, rating_col=2, ts_col=3):
    """Compute the Section 8 feasibility metrics for a locally authorized linked dataset.

    Participants are NEVER double-counted: participant count is distinct users in the
    personality file; interaction counts are distinct rating rows per linked user.
    """
    with Path(personality_csv).open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        first_col = reader.fieldnames[0].strip()
        linked_users = {row[first_col].strip() for row in reader}

    counts = Counter()
    ts_min, ts_max = None, None
    distinct_items = set()
    rating_values = set()
    with Path(ratings_csv).open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        for row in reader:
            uid = row[id_col].strip()
            if uid not in linked_users:
                continue
            counts[uid] += 1
            item = row[1].strip()
            distinct_items.add(item)
            rating_values.add(row[rating_col].strip())
            ts = row[ts_col].strip()
            if ts:
                ts_min = ts if ts_min is None or ts < ts_min else ts_min
                ts_max = ts if ts_max is None or ts > ts_max else ts_max

    groups = Counter(history_group(c) for c in counts.values())
    return {
        "personality_participants": len(linked_users),
        "linked_participants_with_preferences": len(counts),
        "participants_without_any_rating": len(linked_users - set(counts)),
        "total_ratings_by_linked_users": sum(counts.values()),
        "distinct_rated_items": len(distinct_items),
        "distinct_rating_values": sorted(rating_values),
        "timestamp_min": ts_min,
        "timestamp_max": ts_max,
        "timestamps_available": ts_min is not None,
        "history_groups": {g: groups.get(g, 0) for g in HISTORY_GROUPS},
        "group_1_4": groups.get("1-4", 0),
        "group_5_9": groups.get("5-9", 0),
        "group_10_20": groups.get("10-20", 0),
        "group_over_20": groups.get(">20", 0),
        "mean_ratings_per_linked_user": round(sum(counts.values()) / max(1, len(counts)), 2),
        "median_group": max(groups.items(), key=lambda kv: kv[1])[0] if groups else None,
    }


def check_thresholds(audit, min_linked=300, min_sparse_group=100):
    """Screen against the Section 8 preliminary feasibility thresholds; returns dict of verdicts."""
    sparse_total = audit["group_1_4"] + audit["group_5_9"] + audit["group_10_20"]
    return {
        "linked_participants": audit["linked_participants_with_preferences"],
        "meets_min_linked": audit["linked_participants_with_preferences"] >= min_linked,
        "sparse_history_total_1_to_20": sparse_total,
        "meets_min_sparse_group": sparse_total >= min_sparse_group,
        "temporal_evaluation_possible": audit["timestamps_available"],
        "repeated_ratings_sufficient": audit["total_ratings_by_linked_users"] >= 10 * max(1, audit["linked_participants_with_preferences"]),
        "cold_start_users_available": audit["group_1_4"] > 0 or audit["group_5_9"] >= min_sparse_group // 4,
    }
