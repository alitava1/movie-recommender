"""Phase 13 dataset loader, reconciliation, and integrity audit (aggregates only)."""

import csv
import hashlib
from pathlib import Path

TRAITS = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
DATASET_DIR = Path(__file__).resolve().parents[2] / "artifacts" / "evaluation" / "phase12-dataset-feasibility" / "downloads" / "personality-isf2018"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_personality(path=None):
    """Load trait rows with first-occurrence de-duplication of duplicate participant ids.

    Returns (traits_raw, audit) where traits_raw maps userid -> {trait: score 1-7 with
    neuroticism = 8 - emotional_stability} and audit documents the 1834/1820 reconciliation.
    """
    path = Path(path) if path else DATASET_DIR / "personality-data.csv"
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = [c.strip() for c in next(reader)]
        rows = list(reader)
    idx = {name: i for i, name in enumerate(header)}

    seen = {}
    duplicate_ids = {}
    for row in rows:
        uid = row[idx["userid"]].strip()
        if not uid:
            continue
        if uid in seen:
            duplicate_ids.setdefault(uid, []).append(row)
            continue
        seen[uid] = row

    traits_raw = {}
    divergent = 0
    for uid, row in seen.items():
        es = float(row[idx["emotional_stability"]])
        traits_raw[uid] = {
            "openness": float(row[idx["openness"]]),
            "conscientiousness": float(row[idx["conscientiousness"]]),
            "extraversion": float(row[idx["extraversion"]]),
            "agreeableness": float(row[idx["agreeableness"]]),
            "neuroticism": 8.0 - es,  # reverse of emotional stability; documented direction fix
        }
    for uid, extra_rows in duplicate_ids.items():
        for row in extra_rows:
            es = float(row[idx["emotional_stability"]])
            check = {
                "openness": float(row[idx["openness"]]),
                "conscientiousness": float(row[idx["conscientiousness"]]),
                "extraversion": float(row[idx["extraversion"]]),
                "agreeableness": float(row[idx["agreeableness"]]),
                "neuroticism": 8.0 - es,
            }
            if check != traits_raw[uid]:
                divergent += 1

    audit = {
        "personality_rows_total": len(rows),
        "distinct_participants": len(seen),
        "duplicate_ids": len(duplicate_ids),
        "duplicate_rows": len(rows) - len(seen),
        "duplicate_trait_divergent": divergent,
        "resolution": "first occurrence kept; duplicates counted, never silently discarded",
        "reconciliation": "1834 rows = 1820 distinct participants + 14 duplicate rows across 7 ids (README counts rows, not participants)",
    }
    return traits_raw, audit


def load_ratings(path=None):
    """Load ratings keeping the MOST RECENT rating per (user, movie) pair.

    Returns (rows, users, audit) with rows = [(user_idx, movie_id, rating, tstamp)] sorted by
    (tstamp, movie_id) per user; user_idx is the rank of the hashed id in sorted order
    (0..1819) so no raw participant identifiers are needed downstream.
    """
    path = Path(path) if path else DATASET_DIR / "ratings.csv"
    with path.open(encoding="utf-8") as fh:
        reader = csv.reader(fh)
        next(reader)
        raw = [(r[0].strip(), r[1].strip(), float(r[2]), r[3].strip()) for r in reader]
    return _finalize_ratings(raw)


def latest_per_pair(raw_rows):
    """Preregistered duplicate rule: keep the most recent rating per (user, movie) pair."""
    latest = {}
    for uid, mid, rating, ts in raw_rows:
        key = (uid, mid)
        if key not in latest or ts > latest[key][0]:
            latest[key] = (ts, rating)
    return latest


def _finalize_ratings(raw):
    users = sorted({r[0] for r in raw})
    user_index = {uid: i for i, uid in enumerate(users)}

    latest = latest_per_pair(raw)
    duplicate_pairs = len(raw) - len(latest)

    rows = []
    for (uid, mid), (ts, rating) in latest.items():
        rows.append((user_index[uid], int(mid), rating, ts))
    rows.sort(key=lambda r: (r[3], r[1]))

    audit = {
        "rating_rows_total": len(raw),
        "rows_after_latest_per_pair": len(rows),
        "duplicate_user_movie_pairs_removed": duplicate_pairs,
        "distinct_raters": len(users),
        "distinct_movies": len({mid for _, mid, _, _ in rows}),
        "rating_values": sorted({r[2] for r in rows}),
        "timestamp_min": min(r[3] for r in rows),
        "timestamp_max": max(r[3] for r in rows),
        "user_identity_mapping": "hashed ids replaced by deterministic index (sorted order); ids never stored",
    }
    return rows, users, audit


def full_audit():
    traits_raw, p_audit = load_personality()
    rows, users, r_audit = load_ratings()
    user_index = {uid: i for i, uid in enumerate(users)}
    traits_by_idx = {user_index[uid]: traits for uid, traits in traits_raw.items()}
    rating_users = {r[0] for r in rows}
    trait_audit = {}
    for t in TRAITS:
        vals = [traits_by_idx[u][t] for u in sorted(traits_by_idx)]
        trait_audit[t] = {
            "min": min(vals), "max": max(vals), "missing": 0,
            "mean": round(sum(vals) / len(vals), 4),
            "distinct_values": len(set(vals)),
        }
    audit = {
        "checksums": {
            "personality_isf2018_zip": sha256_file(DATASET_DIR.parent / "personality-isf2018.zip"),
            "personality_data_csv": sha256_file(DATASET_DIR / "personality-data.csv"),
            "ratings_csv": sha256_file(DATASET_DIR / "ratings.csv"),
        },
        "personality": p_audit,
        "ratings": r_audit,
        "linkage": {
            "personality_users_with_ratings": len(set(traits_raw) & set(users)),
            "raters_without_personality": len(set(users) - set(traits_raw)),
            "personality_users_without_ratings": len(set(traits_raw) - set(users)),
            "verified_same_participants": set(traits_raw) == set(users),
        },
        "trait_ranges_raw_1_7": trait_audit,
        "experiment_population": len(traits_by_idx),
    }
    return traits_by_idx, rows, audit
