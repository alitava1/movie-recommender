"""Read-only MovieLens 1M dataset audit with SHA-256 checksums and completeness checks."""

import hashlib
import math
from pathlib import Path

from ..data import load_movies, load_ratings

CHUNK_SIZE = 1024 * 1024


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _quantile(sorted_vals, prob):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * prob
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def _distribution(counts_by_value):
    return {str(key): counts_by_value[key] for key in sorted(counts_by_value)}


def _size_stats(sizes):
    sizes = sorted(sizes)
    return {
        "min": sizes[0] if sizes else 0,
        "p25": _quantile(sizes, 0.25),
        "median": _quantile(sizes, 0.5),
        "mean": sum(sizes) / len(sizes) if sizes else 0.0,
        "p75": _quantile(sizes, 0.75),
        "p95": _quantile(sizes, 0.95),
        "max": sizes[-1] if sizes else 0,
    }


def audit_ml1m(dataset_dir):
    """Read-only audit of ratings.dat, movies.dat, and users.dat."""
    directory = Path(dataset_dir)
    checksums = {name: sha256_file(directory / name) for name in ("ratings.dat", "movies.dat", "users.dat")}

    ratings = load_ratings("ml-1m", directory / "ratings.dat")
    movies = load_movies("ml-1m", directory / "movies.dat")

    rating_counts = {}
    timestamps = []
    per_user = {}
    per_movie = {}
    for user_id, movie_id, rating, ts in ratings:
        rating_counts[rating] = rating_counts.get(rating, 0) + 1
        timestamps.append(ts)
        per_user[user_id] = per_user.get(user_id, 0) + 1
        per_movie[movie_id] = per_movie.get(movie_id, 0) + 1

    # users.dat is audited for completeness only (Gender/Age/Occupation not used by models).
    user_rows = 0
    user_malformed = 0
    genders = {}
    ages = {}
    with (directory / "users.dat").open("r", encoding="latin-1", newline="") as source:
        for line in source:
            parts = line.rstrip("\r\n").split("::")
            if len(parts) != 5 or not parts[0].strip():
                user_malformed += 1
                continue
            user_rows += 1
            genders[parts[1]] = genders.get(parts[1], 0) + 1
            ages[int(parts[2])] = ages.get(int(parts[2]), 0) + 1

    rated_movie_ids = {row[1] for row in ratings}
    eligible_details = []
    by_user_rows = {}
    for user_id, movie_id, rating, ts in ratings:
        by_user_rows.setdefault(user_id, []).append((ts, movie_id, rating))
    for rows in by_user_rows.values():
        rows.sort()
        train = len(rows) - 1
        last_pos = max((i for i, (_, _, r) in enumerate(rows) if r >= 4.0), default=-1)
        eligible_details.append((train, last_pos))

    eligible_1 = sum(1 for train, last_pos in eligible_details if last_pos >= 1)
    eligible_5 = sum(1 for train, last_pos in eligible_details if last_pos >= 5)

    return {
        "dataset": "ml-1m",
        "checksums_sha256": checksums,
        "ratings": {
            "total": len(ratings),
            "unique_users": len(per_user),
            "unique_rated_movies": len(rated_movie_ids),
            "rating_distribution": _distribution(rating_counts),
            "timestamp_min": min(timestamps),
            "timestamp_max": max(timestamps),
            "interactions_per_user": _size_stats(per_user.values()),
            "interactions_per_movie": _size_stats(per_movie.values()),
        },
        "movies": {
            "catalog_total": len(movies),
            "rated_catalog_total": len(rated_movie_ids),
            "missing_year": sum(1 for m in movies.values() if m.year is None),
            "year_min": min((m.year for m in movies.values() if m.year is not None), default=None),
            "year_max": max((m.year for m in movies.values() if m.year is not None), default=None),
            "empty_genres": sum(1 for m in movies.values() if not m.genres),
            "rated_movies_without_metadata": len(rated_movie_ids - movies.keys()),
        },
        "users_file": {
            "rows": user_rows,
            "malformed": user_malformed,
            "genders": _distribution(genders),
            "ages": _distribution(ages),
        },
        "temporal_eligibility": {
            "users_with_latest_positive": eligible_1,
            "users_eligible_min5_train": eligible_5,
            "protocol": "leave-latest-positive-out (rating >= 4.0) with min_train_history = 5",
        },
    }
