"""Popularity baseline with Bayesian rating shrinkage."""


def popularity_scores(ratings, prior_count=10.0):
    """Count, mean and Bayesian-adjusted mean per movie.

    Bayesian mean = (count * mean + prior_count * global_mean) / (count + prior_count),
    so a single 5-star rating cannot outrank a well-established movie.
    """
    if prior_count < 0:
        raise ValueError("prior_count must be non-negative")
    counts, sums = {}, {}
    for _, movie_id, rating, _ in ratings:
        counts[movie_id] = counts.get(movie_id, 0) + 1
        sums[movie_id] = sums.get(movie_id, 0.0) + rating
    if not counts:
        return {}
    global_mean = sum(sums.values()) / sum(counts.values())
    return {
        movie_id: {
            "count": counts[movie_id],
            "mean": sums[movie_id] / counts[movie_id],
            "bayesian": (sums[movie_id] + prior_count * global_mean) / (counts[movie_id] + prior_count),
        }
        for movie_id in counts
    }


def ranked(scores):
    """Movie IDs by Bayesian mean, ties broken by movie ID for determinism."""
    return [movie_id for movie_id, _ in sorted(scores.items(), key=lambda item: (-item[1]["bayesian"], item[0]))]
