"""Phase 9 independent candidate generators with a common deterministic interface.

Every generator returns a list of (movie_id, score) sorted by (-score, movie_id),
excludes the user's seen items, and is capped at max_candidates. Generators receive
only training interactions and static metadata — never validation or test targets.
"""

from dataclasses import dataclass

from .. import content as content_mod
from ..data import group_by_user
from ..phase7.cf_adapter import build_cf

POSITIVE_THRESHOLD = 4.0


@dataclass(frozen=True)
class CandidateList:
    source: str
    items: tuple  # ((movie_id, score), ...) sorted by (-score, movie_id)

    def ids(self, max_candidates=None):
        rows = self.items if max_candidates is None else self.items[:max_candidates]
        return [movie_id for movie_id, _ in rows]


def _finalize(pairs, max_candidates):
    pairs.sort(key=lambda row: (-row[1], row[0]))
    return tuple(pairs[:max_candidates])


class CFCandidateGenerator:
    """Phase 7 exact co-rating CF; scores are predicted ratings clipped to [0.5, 5]."""

    source = "cf"

    def __init__(self, train_ratings, neighbors=100, shrinkage=10.0, prediction_shrinkage=300.0):
        self.model = build_cf(
            train_ratings,
            neighbors=neighbors,
            shrinkage=shrinkage,
            prediction_shrinkage=0.0,
        )
        self.prediction_shrinkage = prediction_shrinkage
        self.users = group_by_user(train_ratings)

    def generate(self, user_ratings, catalog, max_candidates):
        if not user_ratings:
            return CandidateList(self.source, ())
        candidates = catalog.keys() - user_ratings.keys()
        evidence = self.model.evidence(user_ratings, candidates)
        pairs = [
            (movie_id, min(5.0, max(0.5, mean + num / (den + self.prediction_shrinkage))))
            for movie_id, (mean, num, den) in evidence.items()
        ]
        return CandidateList(self.source, _finalize(pairs, max_candidates))


class PopularityCandidateGenerator:
    """Training-only popularity retrieval; signal = positive (rating >= 4.0) interaction count."""

    source = "popularity"

    def __init__(self, train_ratings):
        counts = {}
        for _, movie_id, rating, _ in train_ratings:
            if rating >= POSITIVE_THRESHOLD:
                counts[movie_id] = counts.get(movie_id, 0) + 1
        self.positive_counts = counts

    def generate(self, user_ratings, catalog, max_candidates):
        pairs = [
            (movie_id, float(count))
            for movie_id, count in self.positive_counts.items()
            if movie_id in catalog and movie_id not in user_ratings
        ]
        return CandidateList(self.source, _finalize(pairs, max_candidates))


class ContentCandidateGenerator:
    """TF-IDF user-profile cosine retrieval over static title+genre metadata."""

    source = "content"

    def __init__(self, movies, vectors):
        self.movies = movies
        self.vectors = vectors

    def generate(self, user_ratings, catalog, max_candidates):
        if not user_ratings:
            return CandidateList(self.source, ())
        profile = content_mod.user_profile(self.vectors, user_ratings)
        if not profile:
            return CandidateList(self.source, ())
        scores = content_mod.score(profile, self.vectors, catalog)
        # Items with empty TF-IDF vectors (no metadata) are not retrievable; documented.
        pairs = [(movie_id, score) for movie_id, score in scores.items() if score > 0.0]
        return CandidateList(self.source, _finalize(pairs, max_candidates))
