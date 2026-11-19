"""Ranking metrics for offline evaluation.

All metrics are computed per user on a ranked ID list and a set of relevant
IDs, then pooled as per-user means. Users with no relevant item are skipped by
the caller, not counted as zero.
"""

import math


def precision_at_k(recommended, relevant, k):
    return sum(movie_id in relevant for movie_id in recommended[:k]) / k if k else 0.0


def recall_at_k(recommended, relevant, k):
    if not relevant:
        return 0.0
    return sum(movie_id in relevant for movie_id in recommended[:k]) / len(relevant)


def average_precision_at_k(recommended, relevant, k):
    """Binary-relevance AP; denominator is min(relevant, k), the best reachable score."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, movie_id in enumerate(recommended[:k], 1):
        if movie_id in relevant:
            hits += 1
            total += hits / rank
    return total / min(len(relevant), k)


def ndcg_at_k(recommended, relevant, k):
    """Binary-gain NDCG; ideal list is the first min(relevant, k) positions hit."""
    if not relevant:
        return 0.0
    dcg = sum(1 / math.log2(rank + 1) for rank, movie_id in enumerate(recommended[:k], 1) if movie_id in relevant)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal


def hit_rate_at_k(recommended, relevant, k):
    """1.0 if at least one relevant item appears in top-k, else 0.0."""
    if not relevant or k <= 0:
        return 0.0
    return 1.0 if any(movie_id in relevant for movie_id in recommended[:k]) else 0.0


def _cosine(left, right):
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def intra_list_diversity(recommended, vectors, k):
    """Mean pairwise TF-IDF cosine dissimilarity (1 - cos) across top-k items."""
    items = recommended[:k] if k > 0 else []
    if len(items) < 2:
        return 0.0
    total = 0.0
    pairs = 0
    for i in range(len(items)):
        left = vectors.get(items[i], {})
        for j in range(i + 1, len(items)):
            total += 1.0 - _cosine(left, vectors.get(items[j], {}))
            pairs += 1
    return total / pairs if pairs else 0.0


def evaluate_user(recommended, relevant, k):
    return {
        "precision": precision_at_k(recommended, relevant, k),
        "recall": recall_at_k(recommended, relevant, k),
        "ndcg": ndcg_at_k(recommended, relevant, k),
        "map": average_precision_at_k(recommended, relevant, k),
    }


def evaluate_user_extended(recommended, relevant, k, vectors=None):
    """Full ranking plus hit-rate and intra-list diversity metrics for diagnostics."""
    metrics = evaluate_user(recommended, relevant, k)
    metrics["hr"] = hit_rate_at_k(recommended, relevant, k)
    metrics["ild"] = intra_list_diversity(recommended, vectors, k) if vectors is not None else 0.0
    return metrics


def summarize(per_user, coverage):
    """Mean of each per-user metric plus catalog coverage."""
    users = len(per_user)
    if not users:
        raise ValueError("no evaluated users")
    summary = {
        name: sum(values[name] for values in per_user.values()) / users
        for name in ("precision", "recall", "ndcg", "map")
    }
    summary["coverage"] = coverage
    summary["users"] = users
    return summary
