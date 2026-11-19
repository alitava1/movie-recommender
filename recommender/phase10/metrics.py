"""Phase 10 recommendation-level metrics: novelty decomposition, long-tail share, popularity concentration."""

import math
import statistics

from ..phase7.metrics import evaluate_user_phase7  # noqa: F401 (re-exported for convenience)


def novelty_stats(recommended_lists, popularity_counts, long_tail_threshold=None):
    """Per recommendation lists: mean self-information, long-tail share, mean train popularity.

    Novelty uses the Phase 7/9 definition: -log2(p) with p = positive-training-count share.
    Long-tail share: fraction of recommended items whose training positive count is below
    `long_tail_threshold` (defaults to the 80th percentile of the popularity distribution).
    """
    counts = sorted(popularity_counts.values())
    if long_tail_threshold is None:
        if counts:
            pos = 0.8 * (len(counts) - 1)
            lo, hi = int(math.floor(pos)), int(math.ceil(pos))
            long_tail_threshold = counts[lo] if lo == hi else counts[lo] * (hi - pos) + counts[hi] * (pos - lo)
        else:
            long_tail_threshold = 0.0
    total = sum(popularity_counts.values()) or 1
    novelty_values = []
    long_tail_shares = []
    mean_pop_values = []
    for recs in recommended_lists:
        if not recs:
            continue
        infos = []
        tail = 0
        pops = []
        for mid in recs:
            c = popularity_counts.get(mid, 0)
            p = c / total
            infos.append(-math.log2(p) if p > 0 else -math.log2(1.0 / (total + 1)))
            tail += 1 if c < long_tail_threshold else 0
            pops.append(c)
        novelty_values.append(sum(infos) / len(infos))
        long_tail_shares.append(tail / len(recs))
        mean_pop_values.append(sum(pops) / len(pops))
    return {
        "novelty": statistics.mean(novelty_values) if novelty_values else 0.0,
        "long_tail_share": statistics.mean(long_tail_shares) if long_tail_shares else 0.0,
        "mean_train_popularity_of_recs": statistics.mean(mean_pop_values) if mean_pop_values else 0.0,
        "long_tail_threshold": long_tail_threshold,
        "lists": len(novelty_values),
    }


def unique_genre_representation(recommended_lists, movies):
    """Average number of distinct genres represented in each top-10 list."""
    values = []
    for recs in recommended_lists:
        genres = set()
        for mid in recs:
            movie = movies.get(mid)
            if movie:
                genres.update(movie.genres)
        values.append(len(genres))
    return statistics.mean(values) if values else 0.0


def popularity_concentration(recommended_lists, popularity_counts):
    """Gini-style concentration: share of recommendations falling in the top popularity decile."""
    if not popularity_counts:
        return 0.0
    ordered = sorted(popularity_counts.values(), reverse=True)
    decile = max(1, int(len(ordered) * 0.1))
    head_ids_threshold = ordered[decile - 1]
    total_recs = 0
    head_recs = 0
    for recs in recommended_lists:
        for mid in recs:
            total_recs += 1
            if popularity_counts.get(mid, 0) >= head_ids_threshold:
                head_recs += 1
    return head_recs / total_recs if total_recs else 0.0
