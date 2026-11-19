import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import evaluation, hybrid
from recommender.data import Movie


class Phase3Tests(unittest.TestCase):
    def test_metrics_hand_computed(self):
        recommended = [1, 2, 3, 4, 5]
        relevant = {2, 4, 9}
        self.assertAlmostEqual(evaluation.precision_at_k(recommended, relevant, 5), 2 / 5)
        self.assertAlmostEqual(evaluation.recall_at_k(recommended, relevant, 5), 2 / 3)
        self.assertAlmostEqual(evaluation.average_precision_at_k(recommended, relevant, 5), (1 / 2 + 2 / 4) / 3)
        expected_dcg = 1 / math.log2(3) + 1 / math.log2(5)
        ideal = 1 + 1 / math.log2(3) + 1 / math.log2(4)
        self.assertAlmostEqual(evaluation.ndcg_at_k(recommended, relevant, 5), expected_dcg / ideal)

    def test_metric_edges(self):
        self.assertEqual(evaluation.precision_at_k([], {1}, 10), 0.0)
        self.assertEqual(evaluation.recall_at_k([1], set(), 10), 0.0)
        self.assertEqual(evaluation.ndcg_at_k([1], set(), 10), 0.0)
        self.assertAlmostEqual(evaluation.average_precision_at_k([1], {1}, 10), 1.0)
        self.assertAlmostEqual(evaluation.precision_at_k([1], {1}, 0), 0.0)

    def test_rank_respects_single_signal_weights_and_ties(self):
        movies = {1: Movie(1, "A", None, ()), 2: Movie(2, "B", None, ()), 3: Movie(3, "C", None, ())}
        signals = {"popularity": {1: 5.0, 2: 4.0, 3: 4.0}, "cf": {1: 0.0, 2: 5.0, 3: 4.0}}
        rows = hybrid.rank(movies, signals, 2, {"popularity": 1.0})
        self.assertEqual([row["movie_id"] for row in rows], [1, 2])
        rows = hybrid.rank(movies, signals, 2, {"cf": 1.0})
        self.assertEqual([row["movie_id"] for row in rows], [2, 3])

    def test_user_signals_excludes_seen_and_missing_signals(self):
        movies = {1: Movie(1, "Seen", None, ()), 2: Movie(2, "Other", None, ())}
        popularity_scores = {1: {"bayesian": 5.0}, 2: {"bayesian": 4.0}}
        candidates, signals = hybrid.user_signals({1: 5.0}, movies, popularity_scores)
        self.assertEqual(candidates, {2})
        self.assertEqual(set(signals), {"popularity"})
        self.assertEqual(signals["popularity"], {2: 4.0})

    def test_summarize_pools_users(self):
        per_user = {
            1: {"precision": 1.0, "recall": 0.5, "ndcg": 1.0, "map": 1.0},
            2: {"precision": 0.0, "recall": 0.0, "ndcg": 0.0, "map": 0.0},
        }
        summary = evaluation.summarize(per_user, 0.25)
        self.assertEqual(summary["users"], 2)
        self.assertAlmostEqual(summary["precision"], 0.5)
        self.assertEqual(summary["coverage"], 0.25)
        with self.assertRaises(ValueError):
            evaluation.summarize({}, 0.0)


if __name__ == "__main__":
    unittest.main()
