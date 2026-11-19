"""Weighted hybrid ranking with per-signal normalization and seen-item exclusion."""

import heapq

from . import content

DEFAULT_WEIGHTS = {"cf": 0.5, "content": 0.3, "popularity": 0.2}
KNOWN_SIGNALS = frozenset(DEFAULT_WEIGHTS)


def normalize(values):
    """Min-max scale to [0, 1]; constant or empty signals map to neutral 0.5."""
    if not values:
        return {}
    low, high = min(values.values()), max(values.values())
    if high == low:
        return {key: 0.5 for key in values}
    return {key: (value - low) / (high - low) for key, value in values.items()}


def blend(signals, weights=None):
    """Weighted mean over the signals available for each movie.

    A movie missing one signal is not penalized for it; its remaining weights
    are renormalized, so sparse cold-start movies stay comparable.
    """
    weights = weights or DEFAULT_WEIGHTS
    unknown = set(weights) - KNOWN_SIGNALS
    if unknown:
        raise ValueError(f"weights reference unknown signals: {sorted(unknown)}")
    scored = {}
    for name, values in signals.items():
        if name not in weights:
            continue
        for movie_id, value in normalize(values).items():
            total, weight = scored.get(movie_id, (0.0, 0.0))
            scored[movie_id] = (total + weights[name] * value, weight + weights[name])
    return {movie_id: total / weight for movie_id, (total, weight) in scored.items() if weight}


def user_signals(user_ratings, movies, popularity, cf=None, vectors=None, exclude_genres=()):
    """Candidate set plus raw per-signal scores, shared by every weighting.

    Evaluation reuses one signal computation per user and only re-blends, which
    keeps a four-model comparison cheap and makes weight differences the only
    variable between models.
    """
    seen = set(user_ratings)
    excluded = set(exclude_genres)
    candidates = {
        movie_id for movie_id, movie in movies.items()
        if movie_id not in seen and not (excluded & set(movie.genres))
    }
    signals = {
        "popularity": {movie_id: score["bayesian"] for movie_id, score in popularity.items() if movie_id in candidates},
    }
    if cf is not None and user_ratings:
        signals["cf"] = cf.predict(user_ratings, candidates)
    if vectors is not None and user_ratings:
        signals["content"] = content.score(content.user_profile(vectors, user_ratings), vectors, candidates)
    return candidates, signals


def rank(movies, signals, k=10, weights=None):
    """Blend precomputed signals into ranked rows with their score components."""
    combined = blend(signals, weights)
    best = heapq.nsmallest(k, combined.items(), key=lambda item: (-item[1], item[0]))
    return [{
        "movie_id": movie_id,
        "title": movies[movie_id].title,
        "genres": list(movies[movie_id].genres),
        "score": score,
        "components": {name: values.get(movie_id) for name, values in signals.items()},
    } for movie_id, score in best]


def recommend(user_ratings, movies, popularity, cf=None, vectors=None, k=10, weights=None, exclude_genres=()):
    """Rank unseen movies and return rows with the score components that produced them."""
    _, signals = user_signals(user_ratings, movies, popularity, cf, vectors, exclude_genres)
    return rank(movies, signals, k, weights)
