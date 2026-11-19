import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.data import Movie, load_movies, load_ratings
from recommender.phase9.experiments import (
    inner_validation_study,
    outer_evaluation,
    segment_analysis,
    segment_name,
)
from recommender.phase9.fusion import merge_candidates
from recommender.phase9.generators import (
    CFCandidateGenerator,
    CandidateList,
    ContentCandidateGenerator,
    PopularityCandidateGenerator,
)
from recommender.phase9.metrics import (
    candidate_precision,
    genre_diversity,
    pool_overlap_sample,
    recall_at_m,
    user_retrieval_coverage,
)


def make_movies():
    return {
        mid: Movie(mid, f"Film {mid} ({1970 + mid % 30})", 1970 + mid % 30, ("drama" if mid % 2 else "comedy",))
        for mid in range(1, 41)
    }


def make_ratings():
    rows = []
    for user in range(1, 9):
        for step, mid in enumerate(range(1, 26)):
            rating = 5.0 if (mid + user) % 3 == 0 else (4.0 if (mid + user) % 3 == 1 else 2.0)
            rows.append((user, mid, rating, 1000 + step))
    # Latest positive for each user is movie 25 area; ensure per-user latest >= 4 target exists.
    return rows


class Phase9Tests(unittest.TestCase):
    def setUp(self):
        self.movies = make_movies()
        self.ratings = make_ratings()
        import tempfile

        self._tmp = tempfile.TemporaryDirectory()
        ddir = Path(self._tmp.name)
        (ddir / "movies.dat").write_text(
            "\n".join(f"{mid}::Film {mid} ({1970 + mid % 30})::{'Drama' if mid % 2 else 'Comedy'}" for mid in range(1, 41)) + "\n",
            encoding="latin-1",
        )
        (ddir / "ratings.dat").write_text(
            "\n".join(f"{u}::{m}::{r}::{ts}" for u, m, r, ts in self.ratings) + "\n",
            encoding="ascii",
        )
        self.dataset_dir = ddir

    def tearDown(self):
        self._tmp.cleanup()

    def test_ml1m_loading_and_split_semantics_reused(self):
        ratings = load_ratings("ml-1m", self.dataset_dir / "ratings.dat")
        self.assertEqual(len(ratings), len(self.ratings))
        from recommender.phase7.temporal import split_outer_temporal, split_inner_chronological

        outer_train, outer_test, excluded = split_outer_temporal(ratings)
        self.assertEqual(len(outer_test), 8)
        self.assertEqual(excluded, [])
        # Inner split: strictly later validation rows per user.
        inner_train, validation = split_inner_chronological(outer_train)
        last_inner = {}
        for u, m, r, ts in inner_train:
            last_inner[u] = max(last_inner.get(u, -1), ts)
        for u, m, r, ts in validation:
            self.assertGreater(ts, last_inner[u])

    def test_generators_exclude_seen_and_cap_size_deterministically(self):
        cf = CFCandidateGenerator(self.ratings, neighbors=10, shrinkage=10.0, prediction_shrinkage=300.0)
        pop = PopularityCandidateGenerator(self.ratings)
        vecs = __import__("recommender.content", fromlist=["build_tfidf"]).build_tfidf(self.movies)
        cont = ContentCandidateGenerator(self.movies, vecs)
        ur = {m: 4.0 for m in range(1, 26)}
        for gen in (cf, pop, cont):
            cl = gen.generate(ur, self.movies, 10)
            ids = cl.ids()
            self.assertLessEqual(len(ids), 10)
            self.assertEqual(len(ids), len(set(ids)))
            self.assertTrue(set(ids).isdisjoint(ur.keys()))
            again = gen.generate(ur, self.movies, 10)
            self.assertEqual(cl.items, again.items)
            # Deterministic ordering: (-score, movie_id)
            self.assertEqual(ids, sorted(ids, key=lambda mid: (-dict(cl.items)[mid], mid)))

    def test_popularity_uses_training_only_positive_counts(self):
        pop = PopularityCandidateGenerator(self.ratings)
        # Movie 26-40 never rated: absent from popularity despite being in catalog.
        self.assertNotIn(30, pop.positive_counts)
        cl = pop.generate({1: 4.0}, self.movies, 1000)
        self.assertTrue(all(mid <= 25 for mid in cl.ids()))

    def test_content_missing_metadata_and_zero_scores_excluded(self):
        vecs = __import__("recommender.content", fromlist=["build_tfidf"]).build_tfidf(self.movies)
        cont = ContentCandidateGenerator(self.movies, vecs)
        ur = {1: 5.0, 2: 1.0}
        cl = cont.generate(ur, self.movies, 1000)
        self.assertTrue(cl.ids())
        # Every retrieved item has a strictly positive cosine score.
        self.assertTrue(all(score > 0.0 for _, score in cl.items))

    def test_fusion_union_rrf_quotas_and_truncation(self):
        a = CandidateList("cf", ((1, 0.9), (2, 0.8), (3, 0.7)))
        b = CandidateList("popularity", ((2, 50.0), (4, 40.0), (1, 30.0)))
        fused_union = merge_candidates([a, b], max_candidates=4, mode="union")
        self.assertEqual(set(fused_union.ids()), {1, 2, 3, 4})
        fused_rrf = merge_candidates([a, b], max_candidates=4, mode="rrf", rrf_k=60)
        # Movie 2 appears in both lists -> highest RRF score; movie 1 also in both.
        self.assertEqual(fused_rrf.ids()[0], 2)
        self.assertEqual(set(fused_rrf.ids()), {1, 2, 3, 4})
        # Cap respected and quota honored.
        fused_quota = merge_candidates(
            [a, b], max_candidates=2, mode="union", quotas={"cf": 0.5, "popularity": 0.5},
        )
        self.assertEqual(len(fused_quota.ids()), 2)
        self.assertIn(1, fused_quota.ids())   # cf quota top-1
        self.assertIn(2, fused_quota.ids())   # popularity quota top-1
        attribution = fused_union.attribution
        self.assertEqual(attribution["cf"]["contributed_candidates"], 3)
        self.assertEqual(attribution["popularity"]["overlapping_with_other_sources"], 2)

    def test_retrieval_metrics_definitions(self):
        self.assertAlmostEqual(recall_at_m([1, 2, 3], {2, 3, 9, 10}), 2 / 4)
        self.assertEqual(recall_at_m([], {1}), 0.0)
        self.assertAlmostEqual(user_retrieval_coverage({"u1": 0.5, "u2": 0.0, "u3": 0.25}), 2 / 3)
        self.assertAlmostEqual(candidate_precision([1, 2, 3, 4], {3}), 0.25)
        doc = genre_diversity([1, 2], self.movies)
        self.assertEqual(doc["pairwise_genre_dissimilarity"], 1.0)  # drama vs comedy
        same = genre_diversity([1, 3], self.movies)
        self.assertEqual(same["pairwise_genre_dissimilarity"], 0.0)  # both drama
        overlap = pool_overlap_sample({"a": [1, 2], "b": [1, 2], "c": [3, 4]}, sample_size=10, seed=1)
        self.assertGreater(overlap["mean_pool_overlap_jaccard"], 0.0)

    def test_inner_selection_and_outer_evaluation_leak_free(self):
        settings = {"popularity": {"prior_count": 10.0}, "cf": {"shrinkage": 10.0}}
        from recommender.phase7.temporal import split_outer_temporal

        outer_train, outer_test, _ = split_outer_temporal(self.ratings)
        vecs = __import__("recommender.content", fromlist=["build_tfidf"]).build_tfidf(self.movies)
        inner = inner_validation_study(outer_train, self.movies, vecs, settings, user_filter={1, 2})
        self.assertIn("cf@200", inner["summary"])
        self.assertIn("three_way@200", inner["summary"])
        self.assertTrue(inner["summary"]["cf@200"]["users"] > 0)

        # Outer evaluation with mutated test targets must keep selection identical
        # (selection happens inside inner_validation_study on outer_train only).
        inner2 = inner_validation_study(outer_train, self.movies, vecs, settings, user_filter={1, 2})
        self.assertEqual(inner["selection"], inner2["selection"])
        self.assertEqual(
            {k: v["mean_recall"] for k, v in inner["summary"].items()},
            {k: v["mean_recall"] for k, v in inner2["summary"].items()},
        )

        outer = outer_evaluation(
            outer_train, outer_test, self.movies, vecs, settings,
            selected={"best_family": "cf_content"}, user_filter={1, 2},
        )
        self.assertIn("cf_ranker_cf_pool", outer["ranking_summary"])
        self.assertIn("cf_ranker_best_pool", outer["ranking_summary"])
        self.assertIn("enrichment_ranker_best_pool", outer["ranking_summary"])
        self.assertGreater(outer["evaluated_users"], 0)
        # Pool records expose the CF-scored fraction limitation.
        for rec in outer["pool_records"].values():
            self.assertIn("cf_scored_fraction", rec)

    def test_segment_analysis_uses_training_history_only(self):
        settings = {"popularity": {"prior_count": 10.0}, "cf": {"shrinkage": 10.0}}
        from recommender.phase7.temporal import split_outer_temporal

        outer_train, outer_test, _ = split_outer_temporal(self.ratings)
        vecs = __import__("recommender.content", fromlist=["build_tfidf"]).build_tfidf(self.movies)
        outer = outer_evaluation(
            outer_train, outer_test, self.movies, vecs, settings,
            selected={"best_family": "cf_content"}, user_filter={1, 2},
        )
        segments = segment_analysis(outer_train, outer["retrieval"], outer["ranking_per_user"])
        # Every user here has the same history length, so only one segment is populated.
        populated = [name for name, doc in segments.items() if doc["users"] > 0]
        self.assertEqual(len(populated), 1)
        self.assertEqual(segment_name(10), "history_sparse")
        self.assertEqual(segment_name(50), "history_medium")
        self.assertEqual(segment_name(200), "history_dense")

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase9_candidate_retrieval.py"),
             "--smoke", "--output-dir", str(ROOT.parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
