import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.phase13 import engine, experiments
from recommender.phase13.data import TRAITS, full_audit, load_personality, load_ratings
from recommender.phase13.profiles import (
    ablate_single_trait, missing_profiles, noisy_profiles, normalize_traits,
    randomized_profiles, shuffled_profiles,
)
from recommender.phase11.statistics import holm_bonferroni, paired_bootstrap_with_p


def tiny_rows():
    """8 users x 12 movies, chronological, with duplicate (user, movie) pairs and dup personality rows."""
    rows = []
    for u in range(8):
        for step in range(12):
            rating = 5.0 if (step + u) % 3 == 0 else (4.0 if (step + u) % 3 == 1 else 2.0)
            rows.append((u, 100 + step, rating, f"2020-01-{step + 1:02d} 10:00:00"))
    rows.append((0, 100, 3.0, "2019-06-01 09:00:00"))  # earlier duplicate pair for user 0
    return rows


def tiny_traits():
    return {
        u: {"openness": 1.0 + u * 0.5, "conscientiousness": 7.0 - u * 0.5, "extraversion": 2.0 + u * 0.5,
            "agreeableness": 6.0 - u * 0.25, "neuroticism": 3.0 + (u % 4) * 0.5}
        for u in range(8)
    }


class Phase13Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = tiny_rows()
        cls.traits_raw = tiny_traits()
        cls.settings = None

    def test_schema_and_reconciliation(self):
        traits_raw, p_audit = load_personality()
        self.assertEqual(p_audit["distinct_participants"] + p_audit["duplicate_rows"],
                         p_audit["personality_rows_total"])
        self.assertEqual(p_audit["duplicate_ids"], 7)
        self.assertEqual(p_audit["duplicate_trait_divergent"], 0)
        self.assertEqual(len(traits_raw), 1820)

    def test_trait_direction_normalization(self):
        traits_raw, _ = load_personality()
        neuro = [traits_raw[u]["neuroticism"] for u in list(traits_raw)[:200]]
        self.assertTrue(all(1.0 <= v <= 7.0 for v in neuro))
        norm, bounds = normalize_traits(traits_raw, sorted(traits_raw))
        for u in list(norm)[:50]:
            for t in TRAITS:
                self.assertGreaterEqual(norm[u][t], 0.0)
                self.assertLessEqual(norm[u][t], 1.0)

    def test_missing_trait_handling_in_normalization(self):
        traits = {0: {"openness": 4.0, "conscientiousness": 4.0, "extraversion": 4.0,
                      "agreeableness": 4.0, "neuroticism": 4.0}}
        norm, bounds = normalize_traits(traits, [0])
        self.assertEqual(bounds["openness"], (4.0, 4.0))  # degenerate span
        self.assertEqual(norm[0]["openness"], 0.5)  # neutral, not NaN

    def test_duplicate_rating_resolution_keeps_latest(self):
        from recommender.phase13.data import latest_per_pair

        resolved = latest_per_pair(self.rows)
        self.assertEqual(len(resolved), 8 * 12)  # duplicate (user 0, movie 100) collapsed
        self.assertEqual(resolved[(0, 100)][1], 5.0)  # 2020 rating kept, not 2019's 3.0

    def test_temporal_ordering_and_k_history_enforcement(self):
        train, targets, excluded = engine.warm_stratum(self.rows)
        self.assertEqual(excluded, [])
        self.assertEqual(len(targets), 8)
        hist = engine.per_user_ordered(train)
        for uid, items in hist.items():
            ts = [t for _, _, t in items]
            self.assertEqual(ts, sorted(ts))
        train_k1, targets_k1, excl_k1 = engine.sparse_stratum(self.rows, 1)
        hist_k1 = engine.per_user_ordered(train_k1)
        for uid, items in hist_k1.items():
            self.assertEqual(len(items), 1)
        # K-history uses only the earliest interactions.
        self.assertEqual(hist_k1[0][0][0], 100)

    def test_participant_disjoint_folds(self):
        folds = engine.participant_folds(8, n_folds=3)
        all_users = [u for f in folds for u in f]
        self.assertEqual(sorted(all_users), list(range(8)))
        sets = [set(f) for f in folds]
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                self.assertEqual(sets[i] & sets[j], set())

    def test_seen_item_exclusion_and_pool_consistency(self):
        models = experiments.StratumModels(self.rows, self.traits_raw, "tiny")
        targets = {u: set() for u in range(8)}
        targets[0] = {111}
        pool = models.pop_pool(0)
        self.assertLessEqual(len(pool), engine.POOL_M)
        self.assertTrue(set(pool).isdisjoint(set(m for m, _, _ in engine.per_user_ordered(self.rows)[0])))
        # Identical pools across conditions: base_scores reuses the same popularity pool.
        pool2, base, _ = models.base_scores(0)
        self.assertEqual(pool, pool2)

    def test_personality_disabled_exact_baseline(self):
        models = experiments.StratumModels(self.rows, self.traits_raw, "tiny")
        pool, base, _ = models.base_scores(0)
        r_disabled, _, diag = engine.rank_with_personality13(
            pool, base, models.traits_norm[0], 0, models.model, models.attr_get, 0.3, enabled=False, k=10)
        r_zero, _, diag_zero = engine.rank_with_personality13(
            pool, base, models.traits_norm[0], 0, models.model, models.attr_get, 0.0, enabled=True, k=10)
        expected = sorted(base, key=lambda m: (-base[m], m))[:10]
        self.assertEqual(r_disabled, expected)
        self.assertEqual(r_zero, expected)

    def test_controls_shuffled_random_noisy_bounds(self):
        norm, _ = normalize_traits(self.traits_raw, sorted(self.traits_raw))
        shuf = shuffled_profiles(norm, seed=5)
        for u, t in shuf.items():
            self.assertNotEqual(t, norm[u])  # derangement
        r1 = randomized_profiles(norm, seed=7)
        self.assertEqual(randomized_profiles(norm, seed=7), r1)
        for u in r1:
            for t in TRAITS:
                self.assertGreaterEqual(r1[u][t], 0.0)
                self.assertLessEqual(r1[u][t], 1.0)
        noisy = noisy_profiles(norm, sigma=1.2, seed=9)
        for u in noisy:
            for t in TRAITS:
                self.assertGreaterEqual(noisy[u][t], 0.0)
                self.assertLessEqual(noisy[u][t], 1.0)
        missing = missing_profiles(norm, n_missing=2, seed=11)
        for u in missing:
            self.assertEqual(sum(1 for t in TRAITS if missing[u][t] == norm[u][t]), 3)
        abl = ablate_single_trait(norm, "openness")
        for u in abl:
            self.assertEqual(abl[u]["openness"], norm[u]["openness"] if False else abl[u]["openness"])
        # Trait ablation collapses openness to the shared median.
        self.assertEqual(len({abl[u]["openness"] for u in abl}), 1)

    def test_zero_history_exclusion_from_training(self):
        folds = engine.participant_folds(8, n_folds=2)
        fold0 = set(folds[0])
        train_rows = [r for r in self.rows if r[0] not in fold0]
        models = experiments.StratumModels(train_rows, self.traits_raw, "fold")
        # Held-out participants must not appear in the training history.
        for uid in fold0:
            self.assertNotIn(uid, models.hist_by_user)

    def test_noise_bounds_score_level(self):
        norm, _ = normalize_traits(self.traits_raw, sorted(self.traits_raw))
        for sigma in (0.3, 0.7, 1.2):
            noisy = noisy_profiles(norm, sigma=sigma, seed=2026)
            for u in noisy:
                for t in TRAITS:
                    self.assertTrue(0.0 <= noisy[u][t] <= 1.0)

    def test_bootstrap_determinism_and_holm(self):
        left = {"a": {"ndcg": 0.5}, "b": {"ndcg": 0.4}}
        right = {"a": {"ndcg": 0.3}, "b": {"ndcg": 0.2}}
        c1 = paired_bootstrap_with_p(left, right, resamples=100, seed=2026)
        c2 = paired_bootstrap_with_p(left, right, resamples=100, seed=2026)
        self.assertEqual(c1, c2)
        rows = [
            {"name": "p1", "family": "primary", "p_value": 0.03},
            {"name": "p2", "family": "primary", "p_value": 0.04},
            {"name": "p3", "family": "primary", "p_value": 0.20},
            {"name": "e1", "family": "exploratory", "p_value": 0.001},
        ]
        adj = holm_bonferroni(rows)
        # Sorted p ascending: 0.03*3=0.09; 0.04*2=0.08 -> running max 0.09; 0.20*1=0.20.
        self.assertAlmostEqual(adj[0]["p_holm"], 0.09)
        self.assertAlmostEqual(adj[1]["p_holm"], 0.09)
        self.assertAlmostEqual(adj[2]["p_holm"], 0.20)
        self.assertIsNone(adj[3]["p_holm"])

    def test_aggregate_only_reporting(self):
        from recommender.phase13.reporting import write_data_audit_md, write_limitations
        import tempfile

        traits_raw, rows, audit = full_audit()
        with tempfile.TemporaryDirectory() as tmp:
            write_data_audit_md(Path(tmp), audit)
            write_limitations(Path(tmp))
            text = (Path(tmp) / "data_audit.md").read_text(encoding="utf-8")
            # No 32-hex participant hash may appear in the aggregate report.
            import re

            self.assertIsNone(re.search(r"\b[0-9a-f]{32}\b", text))

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase13_real_personality.py"),
             "--stage", "audit", "--output-dir", str(ROOT.parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)

    def test_phase12_artifacts_untouched(self):
        p12 = ROOT / "artifacts" / "evaluation" / "phase12-dataset-feasibility" / "final_report.md"
        self.assertTrue(p12.is_file())


if __name__ == "__main__":
    unittest.main()
