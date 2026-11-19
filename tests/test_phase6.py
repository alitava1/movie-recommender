import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import content, diagnostics, nested, robustness
from recommender.data import Movie


class Phase6Tests(unittest.TestCase):
    def setUp(self):
        self.movies = {
            1: Movie(1, "Space Alpha (1995)", 1995, ("sci-fi",)),
            2: Movie(2, "Space Beta (1995)", 1995, ("sci-fi",)),
            3: Movie(3, "Drama Past (1950)", 1950, ("drama",)),
            4: Movie(4, "Comedy Modern (1995)", 1995, ("comedy",)),
            5: Movie(5, "Mystery Future (1995)", 1995, ("mystery",)),
            6: Movie(6, "Old Drama (1950)", 1950, ("drama",)),
            7: Movie(7, "Space Gamma (1950)", 1950, ("sci-fi",)),
            8: Movie(8, "Action Hero (1996)", 1996, ("action",)),
            9: Movie(9, "Space Delta (1995)", 1995, ("sci-fi",)),
        }
        self.ratings = [
            (user, movie, float(1 + (user + movie) % 5), 100 + movie)
            for user in range(1, 5)
            for movie in range(1, 7)
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
            "seed": 2026,
            "bootstrap_resamples": 50,
            "confidence": 0.95,
            "cf_grid": {"neighbors": [1, 2], "prediction_shrinkage": [1.0, 10.0]},
            "hybrid_weight_grid": [{"cf": 0.5, "content": 0.3, "popularity": 0.2}],
        }

    def test_no_outer_test_leakage_across_ablation_and_segments(self):
        first = robustness.evaluate_ablation_fold(
            1, self.ratings, self.test, self.movies, self.settings, self.nested_settings,
            k=5, threshold=4.0, top_m=5, cf_first_weights=(0.9, 0.7), collect_lists=True,
        )
        mutated_test = [(user, 9, 1.0, 200) for user in range(1, 5)]
        second = robustness.evaluate_ablation_fold(
            1, self.ratings, mutated_test, self.movies, self.settings, self.nested_settings,
            k=5, threshold=4.0, top_m=5, cf_first_weights=(0.9, 0.7), collect_lists=True,
        )
        self.assertEqual(first["selection"], second["selection"])
        self.assertEqual(first["segment_cutoffs"], second["segment_cutoffs"])
        self.assertEqual(first["metrics"]["cf_top100_complete_enrichment"]["users"], 4)
        self.assertEqual(second["metrics"]["cf_top100_complete_enrichment"]["users"], 0)

    def test_paired_user_cluster_bootstrap_resamples_users_together(self):
        # User 1 has 4 observations (+1.0 vs 0.5); User 2 has 1 observation (0.0 vs 0.5).
        # Cluster draws of 2 users can only produce pooled abs diffs: +0.5 (U1,U1), +0.3 (U1,U2), or -0.5 (U2,U2).
        left = {
            "u1:1": {"ndcg": 1.0}, "u2:1": {"ndcg": 1.0}, "u3:1": {"ndcg": 1.0}, "u4:1": {"ndcg": 1.0},
            "u1:2": {"ndcg": 0.0},
        }
        right = {
            "u1:1": {"ndcg": 0.5}, "u2:1": {"ndcg": 0.5}, "u3:1": {"ndcg": 0.5}, "u4:1": {"ndcg": 0.5},
            "u1:2": {"ndcg": 0.5},
        }
        res = robustness.paired_cluster_bootstrap(left, right, metric="ndcg", resamples=200, confidence=0.95, seed=11)
        self.assertEqual(res["clusters"], 2)
        self.assertEqual(res["observations"], 5)
        self.assertAlmostEqual(res["abs_diff"], 0.3)
        self.assertAlmostEqual(res["rel_diff"], 0.6)
        self.assertGreaterEqual(res["abs_lower"], -0.5)
        self.assertLessEqual(res["abs_upper"], 0.5)

        res_same = robustness.paired_cluster_bootstrap(left, right, metric="ndcg", resamples=200, confidence=0.95, seed=11)
        self.assertEqual(res, res_same)

    def test_component_ablation_isolation_and_complete_equivalence(self):
        vectors = content.build_tfidf(self.movies)
        # User likes 1995 sci-fi (movie 1) and dislikes 1950 drama (movie 3).
        user_ratings = {1: 5.0, 3: 1.0}
        # Candidate 4 is 1995 comedy (matches year only); candidate 7 is 1950 sci-fi (matches genre/TF-IDF, not year).
        cands = [4, 7]
        year_only = robustness.component_content_scores(user_ratings, self.movies, vectors, cands, ("year",), 4.0)
        self.assertGreater(year_only[4], year_only[7])

        genre_only = robustness.component_content_scores(user_ratings, self.movies, vectors, cands, ("genre",), 4.0)
        self.assertGreater(genre_only[7], genre_only[4])

        tfidf_only = robustness.component_content_scores(user_ratings, self.movies, vectors, cands, ("tfidf",), 4.0)
        self.assertGreater(tfidf_only[7], tfidf_only[4])

        complete = robustness.component_content_scores(
            user_ratings, self.movies, vectors, [4, 7, 8, 9], ("tfidf", "genre", "year"), 4.0
        )
        phase5_equiv = diagnostics.enriched_content_scores(user_ratings, self.movies, vectors, [4, 7, 8, 9], 4.0)
        for mid in (4, 7, 8, 9):
            self.assertAlmostEqual(complete[mid], phase5_equiv[mid], places=12)

    def test_candidate_recall_audit_and_excluded_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            ddir = Path(tmp)
            # 6 movies; user 1 and user 2 share movie 1..3 in inner train and rate 4,5,6 in validation.
            item_lines = [
                f"{m}|Film {m} (1995)||||1" + "|0" * 18
                for m in range(1, 7)
            ]
            (ddir / "u.item").write_text("\n".join(item_lines) + "\n", encoding="latin-1")
            base_rows = [
                "1\t1\t5\t10", "1\t2\t5\t20", "1\t3\t2\t30", "1\t4\t5\t40", "1\t5\t5\t50",
                "2\t1\t5\t10", "2\t2\t5\t20", "2\t4\t5\t30", "2\t5\t1\t40", "2\t6\t5\t50",
            ]
            (ddir / "u1.base").write_text("\n".join(base_rows) + "\n", encoding="ascii")
            movies = robustness.load_movies("ml-100k", ddir / "u.item")
            res = robustness.audit_candidate_recall_inner(
                ddir, "ml-100k", [1], movies, self.settings, self.nested_settings,
                pool_sizes=(1, 2), k=2, threshold=4.0, cf_first_weights=(0.9,),
            )
            s1 = res["summary"]["1"]
            s2 = res["summary"]["2"]
            self.assertEqual(s1["total_relevant_items"], s2["total_relevant_items"])
            self.assertEqual(s1["captured_relevant_items"] + s1["excluded_relevant_items"], s1["total_relevant_items"])
            self.assertGreaterEqual(s2["pool_recall"], s1["pool_recall"])

    def test_two_independent_runs_are_identical(self):
        def _single_pass():
            f1 = robustness.evaluate_ablation_fold(
                1, self.ratings, self.test, self.movies, self.settings, self.nested_settings,
                k=5, threshold=4.0, top_m=5, cf_first_weights=(0.9, 0.7), collect_lists=True,
            )
            folds = {"u1": f1}
            summary = robustness.aggregate_ablation_folds(folds, len(self.movies), resamples=20, seed=2026)
            segments = robustness.analyze_segments(folds, resamples=20, seed=2026)
            return {
                "folds": folds,
                "deterministic_payload": {"selection": f1["selection"], "metrics": f1["metrics"], "summary": summary, "segments": segments},
            }

        repro = robustness.compare_runs_reproducibility(_single_pass(), _single_pass())
        self.assertTrue(repro["identical_candidate_sets"])
        self.assertTrue(repro["identical_rankings"])
        self.assertTrue(repro["identical_metrics_and_bootstrap"])

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase6_robustness.py"), "--output-dir", str(ROOT.parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
