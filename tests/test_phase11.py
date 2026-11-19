import math
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import content
from recommender.data import Movie
from recommender.personality import TRAITS
from recommender.personality.preference_model import RidgePersonalityModel
from recommender.personality.profile import ObservedProfile, generate_population_profiles, randomize_profiles, shuffle_profiles
from recommender.personality.questionnaire import build_schema
from recommender.personality.simulator import build_movie_attributes, generate_synthetic_dataset
from recommender.phase11.experiments import run_track_a
from recommender.phase11.personality import assert_observed_profile, build_control_profiles, mask_dimensions
from recommender.phase11.ranking import rank_with_personality
from recommender.phase11.statistics import holm_bonferroni, paired_bootstrap_with_p


def make_movies():
    genres = ("drama", "comedy", "sci-fi", "thriller", "adventure")
    return {
        mid: Movie(mid, f"Film {mid} ({1970 + mid % 30})", 1970 + mid % 30, (genres[mid % len(genres)],))
        for mid in range(1, 51)
    }


class Phase11Tests(unittest.TestCase):
    def setUp(self):
        self.movies = make_movies()
        self.vectors = content.build_tfidf(self.movies)
        self.schema = build_schema()
        self.settings = {
            "popularity": {"prior_count": 10.0},
            "cf": {"neighbors": 10, "shrinkage": 10.0, "prediction_shrinkage": 100.0},
            "cf_tuning": {"neighbors": 10, "prediction_shrinkage": 100.0},
        }

    def test_data_integrity_and_missing_dimensions(self):
        latent, responses, profiles = generate_population_profiles([1, 2, 3], self.schema, seed=42)
        self.assertEqual(set(TRAITS), {"openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism"})
        self.assertEqual(set(profiles.keys()), {1, 2, 3})
        for uid, prof in profiles.items():
            self.assertIsInstance(prof, ObservedProfile)
            self.assertEqual(prof.source, "synthetic_questionnaire")
            for trait in TRAITS:
                self.assertTrue(0.0 <= prof.traits[trait] <= 1.0)
            for val in responses[uid].values():
                self.assertIn(val, (1, 2, 3, 4, 5, None))
        masked = mask_dimensions(profiles, n_missing_dims=2, seed=9)
        for uid, prof in masked.items():
            self.assertEqual(prof.source, "masked_2_dims")
            self.assertAlmostEqual(prof.completeness, 0.6)
            self.assertGreaterEqual(sum(1 for t in TRAITS if prof.traits[t] == 0.5), 2)

    def test_leakage_guards_and_controls(self):
        # Passing a raw latent trait dict instead of ObservedProfile must fail loudly.
        with self.assertRaises(TypeError):
            assert_observed_profile({t: 0.9 for t in TRAITS})
        with self.assertRaises(TypeError):
            rank_with_personality(
                [1, 2, 3], {1: 0.9, 2: 0.8, 3: 0.7},
                {t: 0.9 for t in TRAITS}, build_movie_attributes(self.movies, seed=1),
                RidgePersonalityModel(), personality_weight=0.2, enabled=True,
            )

        controls = build_control_profiles([1, 2, 3, 4], self.schema, seed=77)
        # Shuffled profiles are deranged (no user keeps own traits) and reproducible.
        for uid in (1, 2, 3, 4):
            self.assertNotEqual(controls["shuffled"][uid].traits, controls["correct"][uid].traits)
            self.assertEqual(controls["shuffled"][uid].source, "shuffled_control")
            self.assertEqual(controls["random"][uid].source, "random_control")
        self.assertEqual(randomize_profiles([1, 2, 3], seed=5), randomize_profiles([1, 2, 3], seed=5))

    def test_disabled_and_zero_weight_reproduce_baseline_bit_for_bit(self):
        attrs = build_movie_attributes(self.movies, seed=11)
        _, _, profiles = generate_population_profiles([1, 2], self.schema, seed=11)
        train = [(1, m, 5.0 if m % 2 else 2.0, m) for m in range(1, 21)]
        model = RidgePersonalityModel(alpha=5.0).fit(train, profiles, attrs)
        pool = list(range(21, 35))
        base = {mid: 1.0 - (mid - 21) * 0.03 for mid in pool}

        r_base = [mid for mid, _ in sorted(base.items(), key=lambda r: (-r[1], r[0]))][:10]
        r_dis, s_dis, diag_dis = rank_with_personality(
            pool, base, profiles[1], attrs, model, personality_weight=0.25, enabled=False, k=10,
        )
        r_zero, s_zero, diag_zero = rank_with_personality(
            pool, base, profiles[1], attrs, model, personality_weight=0.0, enabled=True, k=10,
        )
        self.assertFalse(diag_dis["personality_active"])
        self.assertFalse(diag_zero["personality_active"])
        self.assertEqual(r_dis, r_base)
        self.assertEqual(r_zero, r_base)
        self.assertEqual(s_dis, base)
        self.assertEqual(s_zero, base)

    def test_bounded_scores_nan_guard_and_determinism(self):
        attrs = build_movie_attributes(self.movies, seed=12)
        _, _, profiles = generate_population_profiles([1], self.schema, seed=12)
        model = RidgePersonalityModel(alpha=5.0).fit([(1, 1, 5.0, 1), (1, 2, 1.0, 2)], profiles, attrs)
        pool = [10, 11, 12, 13]
        base = {mid: 0.5 for mid in pool}

        r1, s1, _ = rank_with_personality(pool, base, profiles[1], attrs, model, personality_weight=0.2, enabled=True, k=4)
        r2, s2, _ = rank_with_personality(pool, base, profiles[1], attrs, model, personality_weight=0.2, enabled=True, k=4)
        self.assertEqual(r1, r2)
        self.assertEqual(s1, s2)
        self.assertTrue(all(0.0 <= v <= 1.0 and not math.isnan(v) and not math.isinf(v) for v in s1.values()))

        # Missing personality profile falls back safely to baseline without NaNs.
        r_miss, s_miss, _ = rank_with_personality(pool, base, None, attrs, model, personality_weight=0.2, enabled=True, k=4)
        self.assertEqual(s_miss, base)
        self.assertEqual(r_miss, [10, 11, 12, 13])

        # NaN score from corrupted model raises rather than silently passing NaN.
        class BadModel:
            def score_candidates(self, *_args):
                return {10: float("nan")}

        with self.assertRaises(ValueError):
            rank_with_personality(pool, base, profiles[1], attrs, BadModel(), personality_weight=0.2, enabled=True)

    def test_statistics_paired_bootstrap_and_holm_bonferroni(self):
        left = {"s1:1": {"ndcg": 0.4}, "s2:1": {"ndcg": 0.6}, "s1:2": {"ndcg": 0.8}, "s2:2": {"ndcg": 0.8}}
        right = {"s1:1": {"ndcg": 0.2}, "s2:1": {"ndcg": 0.2}, "s1:2": {"ndcg": 0.4}, "s2:2": {"ndcg": 0.4}}
        comp = paired_bootstrap_with_p(left, right, metric="ndcg", resamples=200, seed=2026)
        self.assertAlmostEqual(comp["abs_diff"], 0.35)
        self.assertEqual(comp["users"], 2)  # clustered across s1/s2
        self.assertFalse(comp["includes_zero"])
        self.assertLess(comp["p_value"], 0.05)

        adjusted = holm_bonferroni([
            {"name": "p1", "family": "primary", "p_value": 0.01},
            {"name": "p2", "family": "primary", "p_value": 0.04},
            {"name": "e1", "family": "exploratory", "p_value": 0.001},
        ])
        self.assertAlmostEqual(adjusted[0]["p_holm"], 0.02)
        self.assertAlmostEqual(adjusted[1]["p_holm"], 0.04)
        self.assertTrue(adjusted[0]["significant_holm_05"])
        self.assertIsNone(adjusted[2]["p_holm"])

    def test_synthetic_track_a_null_and_strong_smoke(self):
        res = run_track_a(
            self.movies, self.vectors, self.settings,
            scenarios=("A_null", "D_strong"), seeds=(101,), n_users=24, interactions_per_user=15,
        )
        self.assertIn("A_null", res["scenarios"])
        self.assertIn("D_strong", res["scenarios"])
        # Disabled must equal baseline in Track A too.
        for sc in ("A_null", "D_strong"):
            ws = res["scenarios"][sc]["warm_summary"]
            self.assertAlmostEqual(ws["three_w3_baseline"]["ndcg"], ws["personality_disabled"]["ndcg"], places=12)

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase11_personality_ranking.py"),
             "--smoke", "--output-dir", str(ROOT.parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
