import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts, content, hybrid, popularity
from recommender.collaborative import ARTIFACT_KIND, ItemItemCF
from recommender.data import DatasetValidationError, Movie, load_dataset, load_ratings


class Phase2Tests(unittest.TestCase):
    def test_loaders_parse_both_dataset_formats(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            ml100 = root / "ml-100k"
            ml100.mkdir()
            (ml100 / "u.data").write_text("1\t2\t4.5\t10\n", encoding="ascii")
            (ml100 / "u.item").write_text("2|Film (2001)||||1" + "|0" * 18 + "\n", encoding="latin-1")
            ratings, movies = load_dataset("ml-100k", ml100)
            self.assertEqual(ratings, [(1, 2, 4.5, 10)])
            self.assertEqual(movies[2], Movie(2, "Film (2001)", 2001, ("unknown",)))

            ml1 = root / "ml-1m"
            ml1.mkdir()
            (ml1 / "ratings.dat").write_text("1::10::4::100\n", encoding="ascii")
            (ml1 / "movies.dat").write_text("10::A :: Title (2000)::Drama|Comedy\n", encoding="latin-1")
            ratings, movies = load_dataset("ml-1m", ml1)
            self.assertEqual(ratings[0], (1, 10, 4.0, 100))
            self.assertEqual(movies[10].title, "A :: Title (2000)")
            self.assertEqual(movies[10].genres, ("drama", "comedy"))

    def test_popularity_bayesian_mean(self):
        scores = popularity.popularity_scores([(1, 7, 5, 0), (2, 7, 3, 0), (1, 8, 5, 0)], prior_count=2)
        self.assertAlmostEqual(scores[7]["mean"], 4.0)
        self.assertAlmostEqual(scores[7]["bayesian"], (8 + 2 * (13 / 3)) / 4)
        self.assertEqual(popularity.ranked(scores)[0], 8)

    def test_cf_similarity_uses_shared_users_only_and_excludes_seen(self):
        ratings = [
            (1, 10, 5, 0), (1, 20, 4, 0), (1, 30, 1, 0),
            (2, 10, 5, 0), (2, 20, 4, 0), (2, 30, 1, 0),
            (3, 10, 5, 0), (3, 40, 1, 0),
        ]
        model = ItemItemCF.build(ratings, neighbors=10, shrinkage=0, prediction_shrinkage=0)
        self.assertAlmostEqual(dict(model.item_neighbors[10])[20], 1.0)
        self.assertNotIn(10, model.predict({10: 5.0, 30: 1.0}))

    def test_thin_cf_evidence_stays_near_user_mean(self):
        neighbor = {1: [(2, 0.9)], 2: [(1, 0.9)]}
        ratings = {1: 5.0, 3: 3.0}  # mean 4.0, so item 1 deviates by +1.0
        self.assertAlmostEqual(ItemItemCF(10, 0.0, neighbor, 0).predict(ratings)[2], 4.0 + 0.9 * 1.0 / 0.9)
        self.assertAlmostEqual(ItemItemCF(10, 0.0, neighbor, 9.0).predict(ratings)[2], 4.0 + 0.9 * 1.0 / (0.9 + 9.0))

    def test_tfidf_profile_and_scores(self):
        movies = {
            1: Movie(1, "Space Story (1990)", 1990, ("sci-fi",)),
            2: Movie(2, "Space Journey (1991)", 1991, ("sci-fi",)),
            3: Movie(3, "Quiet Garden (1992)", 1992, ("drama",)),
        }
        vectors = content.build_tfidf(movies)
        profile = content.user_profile(vectors, {1: 5.0, 3: 1.0})
        scores = content.score(profile, vectors)
        self.assertGreater(scores[2], scores[3])
        self.assertAlmostEqual(math.sqrt(sum(value * value for value in vectors[1].values())), 1.0)

    def test_hybrid_seen_exclusion_cold_start_and_weight_renormalization(self):
        movies = {
            1: Movie(1, "Seen", None, ("drama",)),
            2: Movie(2, "Popular", None, ("comedy",)),
            3: Movie(3, "Other", None, ("drama",)),
        }
        popularity_scores = {2: {"bayesian": 4.0}, 3: {"bayesian": 2.0}}
        self.assertEqual(hybrid.blend({"cf": {2: 1.0}, "popularity": {3: 4.0}}, {"cf": 0.5, "popularity": 0.5}), {2: 0.5, 3: 0.5})
        rows = hybrid.recommend({}, movies, popularity_scores)
        self.assertEqual(rows[0]["movie_id"], 2)
        self.assertEqual({row["movie_id"] for row in rows}, {2, 3})
        rows = hybrid.recommend({1: 4.0}, movies, popularity_scores, exclude_genres=("drama",))
        self.assertEqual([row["movie_id"] for row in rows], [2])

    def test_artifact_round_trip_and_kind_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.json"
            model = ItemItemCF(5, 2.0, {1: [(2, 0.5)]})
            artifacts.save(path, model.to_payload(), ARTIFACT_KIND)
            restored = ItemItemCF.from_payload(artifacts.load(path, ARTIFACT_KIND))
            self.assertEqual(restored.item_neighbors, model.item_neighbors)
            with self.assertRaisesRegex(ValueError, "artifact kind"):
                artifacts.load(path, "wrong-kind")
            document = json.loads(path.read_text(encoding="utf-8"))
            document["version"] = -1
            path.write_text(json.dumps(document), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "version"):
                artifacts.load(path, ARTIFACT_KIND)

    def test_bad_rating_reports_line_number(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "ratings.dat"
            path.write_text("1::2::6::100\n", encoding="ascii")
            with self.assertRaisesRegex(DatasetValidationError, "line 1"):
                load_ratings("ml-1m", path)


if __name__ == "__main__":
    unittest.main()
