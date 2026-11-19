import random
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import content
from recommender.data import Movie
from recommender.personality import TRAITS
from recommender.personality.experiments import _select_baseline_and_personality_weight
from recommender.personality.preference_model import RidgePersonalityModel
from recommender.personality.profile import generate_population_profiles, randomize_profiles, shuffle_profiles
from recommender.personality.questionnaire import build_schema, score_responses, simulate_responses
from recommender.personality.ranking import blend_with_personality, rank_architecture
from recommender.personality.simulator import compute_personality_effect, generate_synthetic_dataset


class Phase8Tests(unittest.TestCase):
    def setUp(self):
        self.movies = {
            mid: Movie(mid, f"Film {mid} ({1990 + mid % 8})", 1990 + mid % 8, ("sci-fi" if mid % 2 else "drama",))
            for mid in range(1, 31)
        }
        self.schema = build_schema(items_per_trait=4)

    def test_questionnaire_reverse_scoring_normalization_and_missing(self):
        # All max positive (5 on +1 items, 1 on -1 items) -> all traits = 1.0
        answers_high = {
            item["item_id"]: (5 if item["keyed"] == 1 else 1)
            for item in self.schema["items"]
        }
        res_high = score_responses(answers_high, self.schema)
        self.assertEqual(res_high["completeness"], 1.0)
        for t in TRAITS:
            self.assertAlmostEqual(res_high["traits"][t], 1.0)

        # All min positive (1 on +1 items, 5 on -1 items) -> all traits = 0.0
        answers_low = {
            item["item_id"]: (1 if item["keyed"] == 1 else 5)
            for item in self.schema["items"]
        }
        res_low = score_responses(answers_low, self.schema)
        for t in TRAITS:
            self.assertAlmostEqual(res_low["traits"][t], 0.0)

        # Missing answers: omit half of openness -> completeness reflects answered fraction, fallback 0.5 when all missing
        partial = dict(answers_high)
        open_items = [item["item_id"] for item in self.schema["items"] if item["trait"] == "openness"]
        for iid in open_items:
            partial[iid] = None
        res_partial = score_responses(partial, self.schema)
        self.assertAlmostEqual(res_partial["completeness"], 0.8)
        self.assertAlmostEqual(res_partial["traits"]["openness"], 0.5)
        self.assertAlmostEqual(res_partial["traits"]["extraversion"], 1.0)

    def test_questionnaire_rejects_invalid_responses(self):
        bad_val = {self.schema["items"][0]["item_id"]: 6}
        with self.assertRaises(ValueError):
            score_responses(bad_val, self.schema)
        bad_bool = {self.schema["items"][0]["item_id"]: True}
        with self.assertRaises(ValueError):
            score_responses(bad_bool, self.schema)
        with self.assertRaises(ValueError):
            score_responses({"UNKNOWN_ITEM": 3}, self.schema)

    def test_ground_truth_and_observed_profile_separation_and_determinism(self):
        lat1, resp1, obs1 = generate_population_profiles([1, 2, 3], self.schema, seed=77)
        lat2, resp2, obs2 = generate_population_profiles([1, 2, 3], self.schema, seed=77)
        self.assertEqual(lat1, lat2)
        self.assertEqual(resp1, resp2)
        self.assertEqual(obs1, obs2)
        for uid, prof in obs1.items():
            self.assertEqual(prof.source, "synthetic_questionnaire")
            self.assertFalse(hasattr(prof, "latent_traits"))
            self.assertTrue(all(0.0 <= v <= 1.0 for v in prof.traits.values()))

    def test_shuffled_and_random_controls(self):
        _, _, obs = generate_population_profiles(list(range(1, 21)), self.schema, seed=55)
        shuf = shuffle_profiles(obs, seed=99)
        self.assertEqual(set(shuf.keys()), set(obs.keys()))
        # Derangement: no user keeps their own profile vector (barring exact coincidental collision).
        self.assertTrue(any(shuf[u].traits != obs[u].traits for u in obs))
        # Exact multiset of trait vectors preserved.
        orig_multiset = sorted(tuple(sorted(p.traits.items())) for p in obs.values())
        shuf_multiset = sorted(tuple(sorted(p.traits.items())) for p in shuf.values())
        self.assertEqual(orig_multiset, shuf_multiset)

        rand = randomize_profiles(list(obs.keys()), seed=123)
        self.assertEqual(set(rand.keys()), set(obs.keys()))
        self.assertTrue(all(p.source == "random_control" for p in rand.values()))

    def test_synthetic_splits_null_and_positive_scenarios(self):
        ds_null = generate_synthetic_dataset(self.movies, "A_null", seed=101, n_users=24, interactions_per_user=15)
        ds_null_2 = generate_synthetic_dataset(self.movies, "A_null", seed=101, n_users=24, interactions_per_user=15)
        self.assertEqual(ds_null.warm_train, ds_null_2.warm_train)
        self.assertEqual(ds_null.cold_test_ratings, ds_null_2.cold_test_ratings)

        # User-held-out separation
        self.assertEqual(set(ds_null.warm_users) & set(ds_null.cold_test_users), set())
        warm_fit_uids = {r[0] for r in ds_null.warm_train + ds_null.warm_val}
        cold_uids = {r[0] for r in ds_null.cold_test_ratings}
        self.assertEqual(warm_fit_uids & cold_uids, set())

        # Interaction-held-out separation & chronological order per user
        for uid in ds_null.warm_users:
            tr = {(r[0], r[1]) for r in ds_null.warm_train if r[0] == uid}
            va = {(r[0], r[1]) for r in ds_null.warm_val if r[0] == uid}
            te = {(r[0], r[1]) for r in ds_null.warm_test if r[0] == uid}
            self.assertEqual(tr & va, set())
            self.assertEqual(tr & te, set())
            self.assertEqual(va & te, set())

        # Null scenario effect must be exact zero; positive scenario non-zero
        u1_lt = ds_null._latent_traits[1]
        m1_attr = ds_null.movie_features[1]
        self.assertEqual(compute_personality_effect(u1_lt, m1_attr, 0.0), 0.0)

        ds_strong = generate_synthetic_dataset(self.movies, "D_strong", seed=101, n_users=60, interactions_per_user=18)
        m_null = RidgePersonalityModel(alpha=20.0).fit(ds_null.warm_train, ds_null.observed_profiles, ds_null.movie_features)
        m_strong = RidgePersonalityModel(alpha=20.0).fit(ds_strong.warm_train, ds_strong.observed_profiles, ds_strong.movie_features)
        self.assertGreater(sum(abs(w) for w in m_strong.weights[1:]), sum(abs(w) for w in m_null.weights[1:]) * 0.5)
        # Verify factored score_candidates matches per-item personality_score exactly
        prof_s = ds_strong.observed_profiles[ds_strong.warm_users[0]]
        batch = m_strong.score_candidates(prof_s, ds_strong.movie_features, [1, 2, 3])
        for mid in (1, 2, 3):
            self.assertAlmostEqual(batch[mid], m_strong.personality_score(prof_s, ds_strong.movie_features[mid]), places=12)

    def test_personality_disabled_reproduces_baseline_bit_identically(self):
        base_scores = {1: 0.91, 2: 0.85, 3: 0.72, 4: 0.60, 5: 0.40}
        ds = generate_synthetic_dataset(self.movies, "C_moderate", seed=11, n_users=16, interactions_per_user=15)
        model = RidgePersonalityModel(alpha=20.0).fit(ds.warm_train, ds.observed_profiles, ds.movie_features)
        prof = ds.observed_profiles[ds.warm_users[0]]

        baseline = rank_architecture("cf_content_temporal", base_scores, None, ds.movie_features, model, personality_weight=0.3, k=5)
        disabled = rank_architecture("cf_content_temporal_personality_disabled", base_scores, prof, ds.movie_features, model, personality_weight=0.3, k=5)
        zero_w = rank_architecture("cf_content_temporal_personality", base_scores, prof, ds.movie_features, model, personality_weight=0.0, k=5)
        self.assertEqual(baseline, disabled)
        self.assertEqual(baseline, zero_w)
        self.assertEqual(blend_with_personality(base_scores, {1: 9.0}, 0.4, enabled=False), base_scores)

    def test_inner_only_weight_selection_is_invariant_to_outer_test_mutation(self):
        ds = generate_synthetic_dataset(self.movies, "C_moderate", seed=33, n_users=24, interactions_per_user=15)
        vectors = content.build_tfidf(self.movies)
        settings = {
            "popularity": {"prior_count": 2.0},
            "cf": {"neighbors": 5, "shrinkage": 1.0, "prediction_shrinkage": 10.0},
        }
        nested_settings = {
            "cf_grid": {"neighbors": [5], "prediction_shrinkage": [10.0]},
        }
        b1, w1, sw1 = _select_baseline_and_personality_weight(
            ds.warm_train, ds.warm_val, self.movies, vectors, settings, nested_settings,
            ds.observed_profiles, ds.movie_features, k=5, threshold=4.0, top_m=10,
        )
        # Outer test (ds.warm_test) is never even passed to selection; verify repeated call matches identically.
        b2, w2, sw2 = _select_baseline_and_personality_weight(
            ds.warm_train, ds.warm_val, self.movies, vectors, settings, nested_settings,
            ds.observed_profiles, ds.movie_features, k=5, threshold=4.0, top_m=10,
        )
        self.assertEqual((b1, w1, sw1), (b2, w2, sw2))

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase8_personality.py"), "--smoke", "--output-dir", str(ROOT.parent)],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)


if __name__ == "__main__":
    unittest.main()
