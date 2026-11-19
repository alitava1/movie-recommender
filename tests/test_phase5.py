import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import content, diagnostics, evaluation, hybrid
from recommender.data import Movie


class Phase5Tests(unittest.TestCase):
    def setUp(self):
        self.movies = {
            1: Movie(1, "Space Alpha (1995)", 1995, ("sci-fi", "action")),
            2: Movie(2, "Space Beta (1996)", 1996, ("sci-fi", "action")),
            3: Movie(3, "Quiet Garden (1994)", 1994, ("drama",)),
            4: Movie(4, "Comedy Night (1995)", 1995, ("comedy",)),
            5: Movie(5, "Space Gamma (1995)", 1995, ("sci-fi",)),
            6: Movie(6, "Old Classic (1950)", 1950, ("drama",)),
            7: Movie(7, "Mystery Lake (1993)", 1993, ("mystery",)),
            8: Movie(8, "Action Hero (1995)", 1995, ("action",)),
            9: Movie(9, "Space Delta (1997)", 1997, ("sci-fi", "action")),
            10: Movie(10, "Romance Tale (1996)", 1996, ("romance",)),
        }
        self.ratings = [
            (user, movie, float(1 + (user + movie) % 5), 100 + movie)
            for user in range(1, 5)
            for movie in range(1, 8)
        ]
        self.test = [(user, 9, 5.0, 200) for user in range(1, 5)]
        self.settings = {
            "popularity": {"prior_count": 2.0},
            "cf": {"neighbors": 2, "shrinkage": 1.0, "prediction_shrinkage": 10.0},
            "hybrid_weights": {"cf": 0.5, "content": 0.3, "popularity": 0.2},
        }
        self.nested_settings = {
            "inner_validation_fraction": 0.2,
            "selection_k": 5,
            "cf_grid": {"neighbors": [1, 2], "prediction_shrinkage": [1.0, 10.0]},
            "hybrid_weight_grid": [
                {"cf": 0.5, "content": 0.3, "popularity": 0.2},
                {"cf": 0.6, "content": 0.2, "popularity": 0.2},
            ],
        }

    def test_hit_rate_and_intra_list_diversity(self):
        self.assertEqual(evaluation.hit_rate_at_k([1, 2, 3], {3, 9}, 3), 1.0)
        self.assertEqual(evaluation.hit_rate_at_k([1, 2, 3], {9}, 3), 0.0)
        self.assertEqual(evaluation.hit_rate_at_k([1, 2, 3], {3}, 2), 0.0)

        vectors = {
            1: {"a": 1.0},
            2: {"a": 1.0},
            3: {"b": 1.0},
        }
        self.assertAlmostEqual(evaluation.intra_list_diversity([1, 2], vectors, 2), 0.0)
        self.assertAlmostEqual(evaluation.intra_list_diversity([1, 3], vectors, 2), 1.0)
        self.assertAlmostEqual(evaluation.intra_list_diversity([1, 2, 3], vectors, 3), 2.0 / 3.0)
        self.assertEqual(evaluation.intra_list_diversity([1], vectors, 1), 0.0)

    def test_cf_first_gating_excludes_non_cf_items_unlike_full_blend(self):
        # Movie 3 lacks a CF score; full hybrid.blend renormalizes weights (content+pop)/0.5 = 1.0
        # and ranks movie 3 above CF-scored movies 1 and 2. CF-first gating keeps only CF candidates.
        signals = {
            "cf": {1: 4.5, 2: 4.2},
            "content": {1: 0.1, 2: 0.1, 3: 0.9},
            "popularity": {1: 2.0, 2: 2.0, 3: 4.5},
        }
        blended = hybrid.blend(signals, {"cf": 0.5, "content": 0.3, "popularity": 0.2})
        self.assertGreater(blended[3], blended[1])

        top_cf = diagnostics.cf_top_m_candidates(signals["cf"], top_m=2)
        self.assertEqual(top_cf, [1, 2])
        reranked = diagnostics.rerank_pool_scores(top_cf, signals["cf"], signals["content"], cf_weight=0.8)
        self.assertEqual(set(reranked), {1, 2})
        self.assertNotIn(3, reranked)

    def test_enriched_content_uses_genre_and_release_year(self):
        vectors = content.build_tfidf(self.movies)
        user_ratings = {1: 5.0, 2: 5.0, 3: 1.0}  # likes 1995-96 sci-fi/action, dislikes 1994 drama
        scores = diagnostics.enriched_content_scores(user_ratings, self.movies, vectors, [6, 9], threshold=4.0)
        self.assertGreater(scores[9], scores[6])

    def test_mmr_is_independent_and_increases_diversity(self):
        vectors = {
            1: {"sci": 1.0},
            2: {"sci": 1.0},
            3: {"drama": 1.0},
            4: {"comedy": 1.0},
        }
        rel = {1: 1.0, 2: 0.99, 3: 0.80, 4: 0.60}
        pure = diagnostics.mmr_rerank(rel, vectors, k=3, lam=1.0)
        self.assertEqual(pure, [1, 2, 3])
        diversified = diagnostics.mmr_rerank(rel, vectors, k=3, lam=0.5)
        self.assertEqual(diversified, [1, 3, 4])
        self.assertGreater(
            evaluation.intra_list_diversity(diversified, vectors, 3),
            evaluation.intra_list_diversity(pure, vectors, 3),
        )

    def test_outer_test_mutation_cannot_change_phase5_selection_or_audit(self):
        first = diagnostics.evaluate_phase5_fold(
            1, self.ratings, self.test, self.movies, self.settings, self.nested_settings,
            k=5, threshold=4.0, top_m=5, cf_first_weights=(0.9, 0.7), mmr_lambdas=(1.0, 0.7), primary_mmr_lambda=0.7,
        )
        mutated_test = [(user, 9, 1.0, 200) for user in range(1, 5)]
        second = diagnostics.evaluate_phase5_fold(
            1, self.ratings, mutated_test, self.movies, self.settings, self.nested_settings,
            k=5, threshold=4.0, top_m=5, cf_first_weights=(0.9, 0.7), mmr_lambdas=(1.0, 0.7), primary_mmr_lambda=0.7,
        )
        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(first["audit"], second["audit"])
        self.assertEqual(first["metrics"]["cf_tuned"]["users"], 4)
        self.assertEqual(second["metrics"]["cf_tuned"]["users"], 0)

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "hybrid_diagnostics.py"), "--output-dir", str(ROOT.parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
