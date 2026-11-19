"""Phase 9 retrieval metrics (independent from ranking metrics)."""

import math
import statistics

from ..data import Movie


def recall_at_m(pool_ids, relevant_ids):
    """Macro-ready per-user recall: relevant items retrieved / ALL relevant items.

    The denominator is never pool-restricted.
    """
    if not relevant_ids:
        return 0.0
    return len(set(pool_ids) & set(relevant_ids)) / len(relevant_ids)


def user_retrieval_coverage(recall_by_user):
    """Fraction of users with at least one relevant item in their pool (recall > 0)."""
    if not recall_by_user:
        return 0.0
    return sum(1.0 for value in recall_by_user.values() if value > 0.0) / len(recall_by_user)


def exclusion_summary(recall_by_user, total_relevant_by_user):
    """Exclusion stats computed from recall and per-user relevant totals."""
    excluded = [total * (1.0 - recall) for total, recall in zip(total_relevant_by_user.values(), recall_by_user.values())]
    retrieved = [total * recall for total, recall in zip(total_relevant_by_user.values(), recall_by_user.values())]
    return {
        "mean_excluded_per_user": statistics.mean(excluded) if excluded else 0.0,
        "median_excluded_per_user": statistics.median(excluded) if excluded else 0.0,
        "total_relevant_retrieved": int(round(sum(retrieved))),
        "total_relevant_missed": int(round(sum(excluded))),
        "total_relevant": int(round(sum(total_relevant_by_user.values()))),
        "users": len(recall_by_user),
    }


def candidate_precision(pool_ids, relevant_ids):
    """Proportion of generated candidates that are relevant (0 if pool empty)."""
    if not pool_ids:
        return 0.0
    return len(set(pool_ids) & set(relevant_ids)) / len(pool_ids)


def _genre_set(movie):
    return set(movie.genres) if movie else set()


def genre_diversity(pool_ids, movies):
    """Pairwise genre Jaccard dissimilarity and unique-genre count for a candidate pool."""
    ids = list(pool_ids)
    if len(ids) < 2:
        return {"pairwise_genre_dissimilarity": 0.0, "unique_genres": len(_genre_set(movies.get(ids[0]))) if ids else 0}
    total = 0.0
    pairs = 0
    for i in range(len(ids)):
        left = _genre_set(movies.get(ids[i]))
        for j in range(i + 1, len(ids)):
            right = _genre_set(movies.get(ids[j]))
            union = left | right
            dissim = 1.0 - (len(left & right) / len(union)) if union else 1.0
            total += dissim
            pairs += 1
    unique = set().union(*(_genre_set(movies.get(mid)) for mid in ids))
    return {
        "pairwise_genre_dissimilarity": total / pairs,
        "unique_genres": len(unique),
    }


def pool_overlap_sample(pools_by_user, sample_size=300, seed=2026):
    """Mean pairwise Jaccard overlap between users' pools over a deterministic user sample."""
    import random

    users = sorted(pools_by_user)
    if len(users) < 2:
        return {"mean_pool_overlap_jaccard": 1.0, "pairs": 0}
    rng = random.Random(seed)
    sampled = rng.sample(users, min(sample_size, len(users)))
    total = 0.0
    pairs = 0
    for i in range(len(sampled)):
        left = set(pools_by_user[sampled[i]])
        for j in range(i + 1, len(sampled)):
            right = set(pools_by_user[sampled[j]])
            union = left | right
            total += (len(left & right) / len(union)) if union else 1.0
            pairs += 1
    return {"mean_pool_overlap_jaccard": total / pairs if pairs else 1.0, "pairs": pairs}
