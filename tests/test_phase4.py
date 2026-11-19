import sys
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import nested, popularity
from recommender.collaborative import ItemItemCF
from recommender.data import Movie


class Phase4Tests(unittest.TestCase):
    def setUp(self):
        self.movies = {
            movie_id: Movie(movie_id, f"Film {movie_id}", None, ("drama" if movie_id % 2 else "comedy",))
            for movie_id in range(1, 13)
        }
        self.ratings = [
            (user, movie, float(1 + (movie + user) % 5), 100 + movie)
            for user in range(1, 5)
            for movie in range(1, 9)
        ]
        self.test = [(user, 9, 5.0, 200) for user in range(1, 5)]
        self.settings = {
            "popularity": {"prior_count": 2.0},
            "cf": {"neighbors": 2, "shrinkage": 1.0, "prediction_shrinkage": 10.0},
            "hybrid_weights": {"cf": 0.5, "content": 0.3, "popularity": 0.2},
        }
        self.nested_settings = {
            "inner_validation_fraction": 0.2,
            "cf_grid": {"neighbors": [1, 2], "prediction_shrinkage": [1.0, 10.0]},
            "hybrid_weight_grid": [
                {"cf": 0.5, "content": 0.3, "popularity": 0.2},
                {"cf": 0.4, "content": 0.4, "popularity": 0.2},
            ],
        }

    def test_inner_split_is_chronological_disjoint_and_complete(self):
        rows = [(1, 2, 4.0, 20), (1, 1, 3.0, 10), (1, 3, 5.0, 20), (2, 4, 4.0, 5)]
        train, validation = nested.split_inner(rows, 0.5)
        self.assertEqual({(r[0], r[1]) for r in train} & {(r[0], r[1]) for r in validation}, set())
        self.assertEqual(set(train) | set(validation), set(rows))
        self.assertEqual([r[1] for r in validation if r[0] == 1], [3])  # timestamp tie uses movie ID
        self.assertEqual([r[1] for r in train if r[0] == 1], [1, 2])
        self.assertEqual([r[1] for r in train if r[0] == 2], [4])
        self.assertFalse(validation and any(r[0] == 2 for r in validation))

    def test_selection_fits_only_inner_training_rows(self):
        expected_train, expected_validation = nested.split_inner(self.ratings, 0.2)
        captured = []
        original = nested._model_parts

        def record_fit(ratings, *args, **kwargs):
            captured.append(list(ratings))
            return original(ratings, *args, **kwargs)

        with patch.object(nested, "_model_parts", side_effect=record_fit):
            nested.select_parameters(
                self.ratings, self.movies, self.settings, self.nested_settings, 10, 4.0
            )
        self.assertEqual(captured, [expected_train])
        self.assertFalse(set(captured[0]) & set(expected_validation))

    def test_outer_test_cannot_change_selected_parameters(self):
        first = nested.evaluate_fold(
            1, self.ratings, self.test, self.movies, self.settings, self.nested_settings, 10, 4.0
        )
        changed_test = [(user, 9, 1.0, 200) for user in range(1, 5)]
        second = nested.evaluate_fold(
            1, self.ratings, changed_test, self.movies, self.settings, self.nested_settings, 10, 4.0
        )
        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(first["metrics"]["hybrid_tuned"]["users"], 4)
        self.assertEqual(second["metrics"]["hybrid_tuned"]["users"], 0)

    def test_empty_or_insufficient_training_falls_back(self):
        selection = nested.select_parameters([], self.movies, self.settings, self.nested_settings, 10, 4.0)
        self.assertTrue(selection["fallback"])
        self.assertEqual(selection["evaluated_users"], 0)
        one = [(1, 1, 5.0, 10)]
        selection = nested.select_parameters(one, self.movies, self.settings, self.nested_settings, 10, 4.0)
        self.assertTrue(selection["fallback"])
        self.assertEqual(selection["inner_validation_interactions"], 0)

    def test_bootstrap_is_reproducible_and_constant_difference_exact(self):
        left = {f"u1:{user}": {"ndcg": 0.75} for user in range(4)}
        right = {f"u1:{user}": {"ndcg": 0.25} for user in range(4)}
        first = nested.paired_bootstrap(left, right, resamples=100, seed=7)
        second = nested.paired_bootstrap(left, right, resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["left_minus_right"], 0.5)
        self.assertEqual((first["lower"], first["upper"]), (0.5, 0.5))

    def test_aggregation_uses_user_weighting_and_fold_sample_std(self):
        def fold(value, users, coverage):
            metrics = {metric: value for metric in ("precision", "recall", "ndcg", "map")}
            metrics.update({"users": users, "coverage": coverage})
            per_user = {f"u:{i}": dict(metrics, users=1) for i in range(users)}
            return {"metrics": {"cf_default": metrics}, "per_user": {"cf_default": per_user}, "_coverage": {"cf_default": set(range(int(coverage * 100)))}}

        result = nested.aggregate_folds({"u1": fold(1.0, 1, 0.2), "u2": fold(0.0, 3, 0.4)}, 100, 20, 0.95, 1)
        summary = result["pooled"]["cf_default"]
        self.assertEqual(summary["ndcg"], 0.25)
        self.assertAlmostEqual(summary["fold_mean_std"]["ndcg"]["mean"], 0.5)
        self.assertAlmostEqual(summary["fold_mean_std"]["ndcg"]["std"], 2 ** -0.5)
        self.assertEqual(summary["coverage"], 0.4)

    def test_legacy_cf_artifact_keeps_original_prediction_behavior(self):
        legacy = {"neighbors": 2, "shrinkage": 1.0, "item_neighbors": {"2": [["1", 0.9]]}}
        restored = ItemItemCF.from_payload(legacy)
        self.assertEqual(restored.prediction_shrinkage, 0.0)
        self.assertEqual(restored.predict({1: 5.0, 3: 3.0})[2], 5.0)

    def test_indexed_cf_evidence_matches_neighbor_scan(self):
        model = ItemItemCF(3, 0.0, {
            10: [(1, 0.9), (2, 0.4)],
            11: [(2, 0.7), (1, 0.2)],
            12: [(3, 0.8)],
        }, 5.0)
        ratings = {1: 5.0, 2: 3.0, 4: 4.0}
        candidates = {10, 11, 12}
        mean = sum(ratings.values()) / len(ratings)
        expected = {}
        for movie_id in candidates:
            numerator = denominator = 0.0
            for neighbor_id, similarity in model.item_neighbors.get(movie_id, ()):
                if neighbor_id in ratings:
                    numerator += similarity * (ratings[neighbor_id] - mean)
                    denominator += abs(similarity)
            if denominator:
                expected[movie_id] = mean, numerator, denominator
        self.assertEqual(model.evidence(ratings, candidates), expected)
        self.assertEqual(model.predict(ratings, candidates), {
            movie_id: min(5.0, max(0.5, m + n / (d + 5.0)))
            for movie_id, (m, n, d) in expected.items()
        })

    def test_fast_selection_ranking_matches_engine_order(self):
        movies = {
            movie_id: Movie(movie_id, f"Film {movie_id}", None, ("drama",))
            for movie_id in range(1, 30)
        }
        signals = {
            "popularity": {movie_id: float(movie_id % 4) for movie_id in movies},
            "content": {movie_id: float((movie_id * 3) % 5) for movie_id in movies},
        }
        weights = {"cf": 0.5, "content": 0.3, "popularity": 0.2}
        for cf in ({movie_id: float(movie_id % 3) for movie_id in movies}, {}):
            combined = dict(signals)
            if cf:
                combined["cf"] = cf
            normalized = {name: nested.hybrid.normalize(values) for name, values in combined.items()}
            expected = [row["movie_id"] for row in nested.hybrid.rank(movies, combined, 10, weights)]
            if cf:
                self.assertEqual(nested._blend_ids(normalized, weights, 10), expected)
                self.assertEqual(nested._top_ids(normalized["cf"], 10), [
                    row["movie_id"] for row in nested.hybrid.rank(movies, {"cf": cf}, 10, {"cf": 1.0})
                ])


if __name__ == "__main__":
    unittest.main()
