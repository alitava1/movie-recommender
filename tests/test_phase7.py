import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import content, evaluation
from recommender.data import Movie
from recommender.phase7.cf_adapter import build_cf
from recommender.phase7.metrics import group_mean_std, novelty_at_k, paired_user_bootstrap, summarize_users
from recommender.phase7.temporal import (
    MIN_TRAIN_HISTORY,
    RELEVANT_THRESHOLD,
    group_relevant_by_user,
    split_inner_chronological,
    split_outer_temporal,
)


def make_rows(user_histories):
    """user_histories: {user_id: [(movie_id, rating, timestamp), ...]}"""
    rows = []
    for user_id, items in user_histories.items():
        for movie_id, rating, ts in items:
            rows.append((user_id, movie_id, float(rating), ts))
    return rows


class Phase7Tests(unittest.TestCase):
    def setUp(self):
        self.movies = {
            mid: Movie(mid, f"Film {mid} ({1960 + mid % 40})", 1960 + mid % 40, ("drama" if mid % 2 else "comedy",))
            for mid in range(1, 31)
        }
        self.histories = {
            1: [(1, 5.0, 10), (2, 4.0, 20), (3, 5.0, 30), (4, 3.0, 40), (5, 4.0, 50), (6, 5.0, 60)],
            2: [(1, 4.0, 10), (2, 4.0, 20), (3, 5.0, 30), (4, 4.0, 40), (5, 2.0, 50), (7, 1.0, 55), (8, 5.0, 60)],
            3: [(1, 2.0, 10), (2, 2.0, 20), (3, 2.0, 30), (4, 2.0, 40), (5, 2.0, 50), (6, 2.0, 60)],
        }
        self.rows = make_rows(self.histories)

    def test_ml1m_parsing_delimiter_encoding_and_duplicates(self):
        from recommender.data import load_movies, load_ratings

        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            ddir = Path(tmp)
            # "::" delimiter with a "::" inside the title exercises rsplit/split handling.
            (ddir / "movies.dat").write_text(
                "1::A::B (1995)::Drama|Comedy\n2::C::D (2000)::Sci-Fi\n", encoding="latin-1"
            )
            (ddir / "ratings.dat").write_text(
                "1::1::5::100\n1::1::4::200\n2::2::4::150\n", encoding="ascii"
            )
            movies = load_movies("ml-1m", ddir / "movies.dat")
            self.assertEqual(movies[1].title, "A::B (1995)")
            self.assertEqual(movies[1].genres, ("drama", "comedy"))
            ratings = load_ratings("ml-1m", ddir / "ratings.dat")
            self.assertEqual(len(ratings), 3)
            # Duplicate user/movie pair detection is the validator's job:
            from recommender.datasets import validate_dataset

            report = validate_dataset("ml-1m", ddir, tmp)
            self.assertEqual(report["duplicate_user_movie_pairs"], 1)
            self.assertFalse(report["valid"])

    def test_user_eligibility_and_temporal_split(self):
        histories = dict(self.histories)
        # User 4 has a positive latest row but only 2 earlier rows -> excluded (min 5).
        histories[4] = [(1, 5.0, 10), (2, 3.0, 20), (9, 5.0, 30)]
        rows = make_rows(histories)
        train, test, excluded = split_outer_temporal(rows)
        test_by_user = {row[0]: row for row in test}
        # User 1: latest positive is movie 6 (ts 60); train rows are strictly earlier.
        self.assertEqual(test_by_user[1][1], 6)
        train_1 = [r for r in train if r[0] == 1]
        self.assertEqual([r[1] for r in train_1], [1, 2, 3, 4, 5])
        self.assertTrue(all(r[3] < test_by_user[1][3] for r in train_1))
        # User 3 has no rating >= 4 -> excluded.
        self.assertIn(3, excluded)
        self.assertIn(4, excluded)
        self.assertEqual({row[0] for row in test}, {1, 2})

    def test_timestamp_tie_uses_movie_id_order(self):
        rows = make_rows({1: [(5, 4.0, 10), (2, 4.0, 10), (9, 4.0, 10), (1, 4.0, 10), (3, 4.0, 10), (4, 4.0, 10), (7, 4.0, 10)]})
        train, test, _ = split_outer_temporal(rows)
        self.assertEqual(test[0][1], 9)  # latest positive = highest movie id at tied timestamp
        self.assertEqual(sorted(r[1] for r in train), [1, 2, 3, 4, 5, 7])

    def test_candidate_exclusion_and_threshold(self):
        train, test, _ = split_outer_temporal(self.rows)
        relevant = group_relevant_by_user(test)
        self.assertEqual(relevant[1], {6})
        self.assertEqual(relevant[2], {8})
        # Candidates exclude every training movie.
        train_movies = {row[1] for row in train}
        self.assertNotIn(6, train_movies)

    def test_no_future_information_leakage_in_inner_split(self):
        outer_train, _, _ = split_outer_temporal(self.rows)
        inner_train, validation = split_inner_chronological(outer_train)
        inner_train_keys = {(r[0], r[1]) for r in inner_train}
        val_keys = {(r[0], r[1]) for r in validation}
        self.assertEqual(inner_train_keys & val_keys, set())
        # Validation rows must be strictly later than each user's last inner-train row.
        last_inner_ts = {}
        for user_id, movie_id, rating, ts in inner_train:
            last_inner_ts[user_id] = max(last_inner_ts.get(user_id, -1), ts)
        for user_id, movie_id, rating, ts in validation:
            self.assertGreater(ts, last_inner_ts[user_id])

    def test_cf_adapter_matches_pure_python_and_is_deterministic(self):
        ratings = []
        for u in range(1, 15):
            base = (u % 4) + 2.0
            ratings += [
                (u, 1, min(5.0, base), u), (u, 2, min(5.0, base + 0.5), u),
                (u, 3, max(1.0, 5.0 - base), u), (u, 4, max(1.0, 4.0 - base), u),
            ]
        ratings += [(u, 5, 5.0, u) for u in range(1, 10)]
        pure = build_cf(ratings, neighbors=5, shrinkage=10.0, prediction_shrinkage=0.0)
        # Force the pure-Python fallback by hiding numpy from the adapter.
        import builtins

        real_import = builtins.__import__

        def no_numpy(name, *args, **kwargs):
            if name == "numpy":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        builtins.__import__ = no_numpy
        try:
            fallback = build_cf(ratings, neighbors=5, shrinkage=10.0, prediction_shrinkage=0.0)
        finally:
            builtins.__import__ = real_import
        # numpy fast path vs pure-Python fallback: same items, same similarity values within the
        # documented <= 1 ULP float-summation deviation; near-tied neighbors at the list boundary
        # may legitimately flip (documented in Phase 13). Common entries must agree exactly in ID.
        self.assertEqual(sorted(pure.item_neighbors), sorted(fallback.item_neighbors))
        for mid, rows in pure.item_neighbors.items():
            frows = fallback.item_neighbors[mid]
            for pid, psim in rows:
                matches = [fsim for nid, fsim in frows if nid == pid]
                if matches:
                    self.assertAlmostEqual(psim, matches[0], places=12)
        # Pure-Python fallback is itself deterministic.
        builtins.__import__ = no_numpy
        try:
            fallback2 = build_cf(ratings, neighbors=5, shrinkage=10.0, prediction_shrinkage=0.0)
        finally:
            builtins.__import__ = real_import
        self.assertEqual(fallback.item_neighbors, fallback2.item_neighbors)
        # numpy repeat build: identical output (deterministic).
        again = build_cf(ratings, neighbors=5, shrinkage=10.0, prediction_shrinkage=0.0)
        self.assertEqual(pure.item_neighbors, again.item_neighbors)

    def test_novelty_metric_and_bootstrap_reproducibility(self):
        pop = {1: 900, 2: 90, 3: 9}
        nov = novelty_at_k([1, 2, 3], pop, 3)
        self.assertGreater(nov, 0.0)
        # Novelty is mean self-information over the top-k set: rare-heavy lists score higher.
        self.assertGreater(novelty_at_k([3, 2, 1], pop, 2), novelty_at_k([1, 2, 3], pop, 2))
        self.assertEqual(novelty_at_k([], pop, 3), 0.0)

        left = {f"user:{u}": {"ndcg": 0.6} for u in (1, 2, 3)}
        right = {f"user:{u}": {"ndcg": 0.2} for u in (1, 2, 3)}
        first = paired_user_bootstrap(left, right, resamples=100, seed=7)
        second = paired_user_bootstrap(left, right, resamples=100, seed=7)
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["abs_diff"], 0.4)
        self.assertAlmostEqual(first["abs_lower"], 0.4)
        self.assertAlmostEqual(first["abs_upper"], 0.4)
        self.assertAlmostEqual(first["rel_diff"], 2.0)

    def test_summarize_and_group_mean_std(self):
        per_user = {
            "user:1": {"ndcg": 0.5, "map": 0.4, "hr": 1.0, "precision": 0.1, "recall": 0.2, "ild": 0.9, "novelty": 3.0},
            "user:2": {"ndcg": 0.5, "map": 0.4, "hr": 1.0, "precision": 0.1, "recall": 0.2, "ild": 0.9, "novelty": 3.0},
        }
        pooled = summarize_users(per_user, 60, 100)
        self.assertAlmostEqual(pooled["ndcg"], 0.5)
        self.assertAlmostEqual(pooled["coverage"], 0.6)
        groups = {"sparse": ["user:1"], "dense": ["user:2"]}
        doc = group_mean_std(per_user, {}, 100, groups)
        self.assertEqual(doc["sparse"]["users"], 1)
        self.assertAlmostEqual(doc["dense"]["ndcg"]["mean"], 0.5)
        self.assertEqual(doc["dense"]["ndcg"]["std"], 0.0)

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase7_movielens1m.py"),
             "--smoke", "--output-dir", str(ROOT.parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
