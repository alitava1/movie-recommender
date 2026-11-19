import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.data import Movie
from recommender.phase10.audit import audit_phase9_sources
from recommender.phase10.diversity import (
    genre_constraint_order,
    genre_redundancy_scores,
    mmr_scores,
    popularity_penalty_scores,
    rank_candidates,
)
from recommender.phase10.experiments import evaluate_configurations, selection_on_inner, segment_analysis
from recommender.phase10.pareto import dominance, pareto_front, pick_representatives
from recommender.phase10.scoring import combined_score, normalize_components, rrf_score


def make_movies():
    return {
        mid: Movie(mid, f"Film {mid} ({1970 + mid % 30})", 1970 + mid % 30, ("drama" if mid % 2 else "comedy",))
        for mid in range(1, 41)
    }


def make_ratings():
    rows = []
    for user in range(1, 9):
        for step, mid in enumerate(range(1, 41)):
            rating = 5.0 if (mid + user) % 3 == 0 else (4.0 if (mid + user) % 3 == 1 else 2.0)
            rows.append((user, mid, rating, 1000 + step))
    return rows


class Phase10Tests(unittest.TestCase):
    def setUp(self):
        self.movies = make_movies()
        self.ratings = make_ratings()
        self.settings = {
            "popularity": {"prior_count": 10.0},
            "cf": {"neighbors": 10, "shrinkage": 10.0, "prediction_shrinkage": 100.0},
            "cf_tuning": {"neighbors": 10, "prediction_shrinkage": 100.0},
        }

    def test_identical_pools_and_cf_scoring_completeness(self):
        from recommender.phase10.experiments import build_pools

        ur = {mid: 4.0 for mid in range(1, 26)}
        vecs = __import__("recommender.content", fromlist=["build_tfidf"]).build_tfidf(self.movies)
        pools, _ = build_pools(self.ratings, self.movies, vecs, ur, self.settings, pool_m=15)
        self.assertEqual(len(pools["cf"]), 15)
        self.assertEqual(len(pools["popularity"]), 15)
        self.assertEqual(len(pools["cf_popularity"]), 15)
        # All rankers receive the exact same IDs: verified via scoring completeness.
        components, norm, coverage = None, None, None
        from recommender.phase10.scoring import extract_components
        cf_gen_scores = {}
        from recommender.phase9.generators import CFCandidateGenerator

        gen = CFCandidateGenerator(self.ratings, neighbors=10, shrinkage=10.0, prediction_shrinkage=100.0)
        evidence = gen.model.evidence(ur, self.movies.keys() - ur.keys())
        cf_gen_scores = {mid: mean + num / den for mid, (mean, num, den) in evidence.items()}
        components, coverage = extract_components(
            ur, pools["popularity"], cf_gen_scores, dict(gen2 := __import__("recommender.phase9.generators", fromlist=["PopularityCandidateGenerator"]).PopularityCandidateGenerator(self.ratings).positive_counts), {}, self.movies,
        )
        self.assertEqual(coverage["candidates"], 15)
        self.assertEqual(coverage["popularity_scored"], 15)
        self.assertLessEqual(coverage["cf_scored"], 15)

    def test_cf_fallback_and_combined_missing_signal_handling(self):
        norm = {"cf": {1: 1.0, 2: 0.5}, "popularity": {1: 0.2, 2: 0.6, 3: 0.9}}
        # Fallback: candidate 3 has no CF score -> popularity used.
        _, scores, diag = rank_candidates([1, 2, 3], norm, "cf_pop_fallback")
        self.assertAlmostEqual(scores[3], 0.9)
        self.assertEqual(diag["fallback_candidates"], 1)
        # Combined with renormalize: candidate 3 scored from popularity alone (weight renorm).
        scores_c = combined_score([1, 2, 3], norm, {"cf": 0.5, "popularity": 0.5})
        self.assertAlmostEqual(scores_c[3], 0.9)
        self.assertAlmostEqual(scores_c[1], 0.5 * 1.0 + 0.5 * 0.2)
        # Weights must be positive-sum and non-negative.
        with self.assertRaises(ValueError):
            combined_score([1], norm, {"cf": 0.0, "popularity": 0.0})

    def test_normalization_is_per_pool_and_deterministic(self):
        norm = normalize_components({"popularity": {1: 100.0, 2: 10.0, 3: 10.0}})
        self.assertAlmostEqual(norm["popularity"][1], 1.0)
        self.assertAlmostEqual(norm["popularity"][2], 0.0)  # log1p-compressed min
        self.assertAlmostEqual(norm["popularity"][3], 0.0)
        # Constant component maps to neutral 0.5.
        const = normalize_components({"popularity": {1: 10.0, 2: 10.0}})
        self.assertAlmostEqual(const["popularity"][1], 0.5)
        self.assertEqual(norm, normalize_components({"popularity": {1: 100.0, 2: 10.0, 3: 10.0}}))

    def test_rrf_rank_fusion_prefers_items_in_both_rankings(self):
        norm = {"cf": {1: 1.0, 2: 0.9, 3: 0.1}, "popularity": {2: 1.0, 1: 0.9, 4: 0.1}}
        fused = rrf_score(norm, rrf_k=60)
        self.assertGreater(fused[2], fused[3])
        self.assertGreater(fused[2], fused[4])
        self.assertAlmostEqual(fused[1], 1 / 61 + 1 / 62)

    def test_popularity_penalty_monotone(self):
        rel = {1: 0.9, 2: 0.8}
        pop = {1: 1000, 2: 5}
        adj = popularity_penalty_scores(rel, pop, [1, 2], lam=0.5)
        # Popular item 1 penalized more; the adjusted gap shrinks.
        self.assertLess(adj[1] - adj[2], rel[1] - rel[2])
        zero = popularity_penalty_scores(rel, pop, [1, 2], lam=0.0)
        self.assertEqual(zero, rel)

    def test_mmr_endpoints_and_diversity_increase(self):
        rel = {1: 1.0, 2: 0.99, 3: 0.8, 4: 0.6}
        movies = {
            1: Movie(1, "A", None, ("sci-fi",)),
            2: Movie(2, "B", None, ("sci-fi",)),
            3: Movie(3, "C", None, ("drama",)),
            4: Movie(4, "D", None, ("comedy",)),
        }
        _, order_pure = mmr_scores(rel, [1, 2, 3, 4], movies, lam=1.0)
        self.assertEqual(order_pure, [1, 2, 3, 4])
        _, order_div = mmr_scores(rel, [1, 2, 3, 4], movies, lam=0.0)
        # lam=0 prioritizes diversity only: first pick is the most relevant (tie-break),
        # subsequent picks avoid genre repeats.
        self.assertNotEqual(order_div[:2], [1, 2])
        with self.assertRaises(ValueError):
            mmr_scores(rel, [1, 2, 3, 4], movies, lam=1.5)

    def test_genre_redundancy_and_constraint(self):
        rel = {1: 1.0, 2: 0.95, 3: 0.5}
        movies = {
            1: Movie(1, "A", None, ("sci-fi",)),
            2: Movie(2, "B", None, ("sci-fi",)),
            3: Movie(3, "C", None, ("drama",)),
        }
        adjusted = genre_redundancy_scores(rel, [1, 2, 3], movies, lam=1.0)
        self.assertEqual(list(adjusted.keys())[0], 1)
        self.assertEqual(list(adjusted.keys())[1], 3)  # drama avoids sci-fi redundancy

        order = genre_constraint_order(rel, movies, max_per_primary=1)
        self.assertIn(3, order[:2])  # second sci-fi item deferred behind drama

    def test_phase9_audit_set_intersections(self):
        from recommender.phase7.temporal import split_outer_temporal

        outer_train, outer_test, _ = split_outer_temporal(self.ratings)
        audit = audit_phase9_sources(
            outer_train, self.movies, self.settings,
            user_filter={1, 2}, pool_m=10, cf_neighbors=10, cf_prediction_shrinkage=100.0,
        )
        self.assertEqual(audit["users"], 2)
        # Union recall must be >= each single source recall (set inclusion).
        self.assertGreaterEqual(audit["recall_union"], audit["recall_cf"])
        self.assertGreaterEqual(audit["recall_union"], audit["recall_pop"])
        # Per-user counts must satisfy inclusion-exclusion on the aggregate.
        self.assertEqual(
            audit["relevant_union"],
            audit["relevant_cf"] + audit["relevant_pop"] - audit["relevant_intersection"],
        )

    def test_end_to_end_inner_outer_and_leak_free_selection(self):
        from recommender.phase7.temporal import split_outer_temporal

        outer_train, outer_test, _ = split_outer_temporal(self.ratings)
        vecs = __import__("recommender.content", fromlist=["build_tfidf"]).build_tfidf(self.movies)
        relevance_configs = [
            {"name": "popularity_only", "method": "popularity_only", "pool": "popularity"},
            {"name": "cf_pop_fallback", "method": "cf_pop_fallback", "pool": "popularity"},
            {"name": "combined_a50", "method": "combined", "params": {"alpha": 0.5}, "pool": "popularity"},
        ]
        inner = evaluate_configurations(
            outer_train, {u: {m} for u, m, r, t in outer_test}, self.movies, vecs,
            self.settings, user_filter={1, 2}, relevance_configs=relevance_configs,
        )
        selection = selection_on_inner(inner)
        self.assertIn(selection["best_accuracy"], {c["name"] for c in relevance_configs})
        # Repeated inner run is deterministic (no leakage, deterministic tie-breaks).
        inner2 = evaluate_configurations(
            outer_train, {u: {m} for u, m, r, t in outer_test}, self.movies, vecs,
            self.settings, user_filter={1, 2}, relevance_configs=relevance_configs,
        )
        self.assertEqual(
            {k: v["ndcg"] for k, v in inner["summary"].items()},
            {k: v["ndcg"] for k, v in inner2["summary"].items()},
        )
        segments = segment_analysis(outer_train, inner["per_user"])
        self.assertTrue(all(doc["users"] >= 0 for doc in segments.values()))

    def test_pareto_dominance_and_representatives(self):
        configs = {
            "a": {"ndcg": 0.03, "coverage": 0.10, "ild": 0.96, "novelty": 8.9},
            "b": {"ndcg": 0.025, "coverage": 0.50, "ild": 0.97, "novelty": 10.1},
            "c": {"ndcg": 0.020, "coverage": 0.40, "ild": 0.96, "novelty": 9.5},  # dominated by b
        }
        front, dominated, report = pareto_front(configs)
        self.assertEqual(set(front), {"a", "b"})
        self.assertIn("c", dominated)
        self.assertEqual(report["c"]["dominated_by"], "b")
        reps = pick_representatives(configs, front)
        self.assertEqual(reps["best_accuracy"], "a")
        self.assertEqual(reps["best_catalog_coverage"], "b")
        self.assertTrue(dominance(configs["b"], configs["c"]))
        self.assertFalse(dominance(configs["a"], configs["b"]))
        self.assertFalse(dominance(configs["a"], configs["c"]))

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase10_candidate_aware_ranking.py"),
             "--smoke", "--output-dir", str(ROOT.parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
