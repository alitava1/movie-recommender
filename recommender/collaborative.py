"""Centered item-item kNN collaborative filtering for explicit ratings.

Predictions are user-mean + weighted average of neighbor rating deviations.
Missing ratings are never treated as zeros or implicit positives.
"""

from math import sqrt

from .data import group_by_user

ARTIFACT_KIND = "item-item-cf"


class ItemItemCF:
    def __init__(self, neighbors, shrinkage, item_neighbors=None, prediction_shrinkage=100.0):
        if neighbors <= 0:
            raise ValueError("neighbors must be positive")
        if shrinkage < 0:
            raise ValueError("shrinkage must be non-negative")
        if prediction_shrinkage < 0:
            raise ValueError("prediction_shrinkage must be non-negative")
        self.neighbors = neighbors
        self.shrinkage = shrinkage
        self.prediction_shrinkage = prediction_shrinkage
        self.item_neighbors = item_neighbors or {}
        reverse = {}
        for movie_id, rows in self.item_neighbors.items():
            for rank, (neighbor_id, similarity) in enumerate(rows):
                reverse.setdefault(neighbor_id, []).append((movie_id, rank, similarity))
        self._reverse_neighbors = reverse

    @classmethod
    def build(cls, ratings, neighbors=50, shrinkage=10.0, prediction_shrinkage=100.0):
        by_user = group_by_user(ratings)
        deviations = {}
        for user_id, items in by_user.items():
            mean = sum(items.values()) / len(items)
            deviations[user_id] = {movie_id: rating - mean for movie_id, rating in items.items()}

        # Pairwise dot products and norms use co-rating users only; missing pairs are not evidence.
        pair = {}
        for user_deviation in deviations.values():
            entries = sorted(user_deviation.items())
            for left in range(len(entries)):
                left_id, left_value = entries[left]
                for right in range(left + 1, len(entries)):
                    right_id, right_value = entries[right]
                    entry = pair.setdefault((left_id, right_id), [0.0, 0.0, 0.0, 0])
                    entry[0] += left_value * right_value
                    entry[1] += left_value * left_value
                    entry[2] += right_value * right_value
                    entry[3] += 1

        neighbors_by_item = {}
        for (left_id, right_id), (dot, left_sq, right_sq, co_ratings) in pair.items():
            norm = sqrt(left_sq * right_sq)
            if norm == 0 or dot <= 0:
                continue
            similarity = dot / norm * co_ratings / (co_ratings + shrinkage)
            neighbors_by_item.setdefault(left_id, []).append((right_id, similarity))
            neighbors_by_item.setdefault(right_id, []).append((left_id, similarity))
        for movie_id, candidates in neighbors_by_item.items():
            candidates.sort(key=lambda item: (-item[1], item[0]))
            del candidates[neighbors:]
            neighbors_by_item[movie_id] = candidates
        return cls(neighbors, shrinkage, neighbors_by_item, prediction_shrinkage)

    def evidence(self, user_ratings, candidates=None):
        if not user_ratings:
            return {}
        mean = sum(user_ratings.values()) / len(user_ratings)
        deviations = {movie_id: rating - mean for movie_id, rating in user_ratings.items()}
        candidates = candidates if candidates is not None else self.item_neighbors.keys()
        evidence = {}
        for neighbor_id, deviation in deviations.items():
            for movie_id, rank, similarity in self._reverse_neighbors.get(neighbor_id, ()):
                if movie_id in candidates and movie_id not in user_ratings:
                    evidence.setdefault(movie_id, []).append((rank, similarity * deviation, abs(similarity)))
        result = {}
        for movie_id, contributions in evidence.items():
            contributions.sort(key=lambda row: row[0])
            numerator = sum(row[1] for row in contributions)
            denominator = sum(row[2] for row in contributions)
            if denominator:
                result[movie_id] = mean, numerator, denominator
        return result

    def predict(self, user_ratings, candidates=None):
        # Dividing by |similarity| alone lets thin evidence produce full-strength extremes.
        return {
            movie_id: min(5.0, max(0.5, mean + numerator / (denominator + self.prediction_shrinkage)))
            for movie_id, (mean, numerator, denominator) in self.evidence(user_ratings, candidates).items()
        }

    def to_payload(self):
        return {
            "kind": ARTIFACT_KIND,
            "neighbors": self.neighbors,
            "shrinkage": self.shrinkage,
            "prediction_shrinkage": self.prediction_shrinkage,
            "item_neighbors": {str(key): value for key, value in self.item_neighbors.items()},
        }

    @classmethod
    def from_payload(cls, payload):
        # Older artifacts predate prediction_shrinkage; they keep their original behavior.
        return cls(
            int(payload["neighbors"]),
            float(payload["shrinkage"]),
            {int(key): [(int(other), float(sim)) for other, sim in value] for key, value in payload["item_neighbors"].items()},
            float(payload.get("prediction_shrinkage", 0.0)),
        )
