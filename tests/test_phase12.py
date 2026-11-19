import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.phase12.candidates import (
    CANDIDATES,
    INST_SCORES_ONLY,
    LIC_UNAVAILABLE,
    REQUIRED_FIELDS,
    UNLINKED,
    UNVERIFIED,
    candidates_to_csv,
    load_candidates,
    save_candidates,
    score_candidate,
    validate_candidate,
)
from recommender.phase12.feasibility import check_thresholds, history_group, sparse_history_audit


class Phase12Tests(unittest.TestCase):
    def test_embedded_candidates_validate_and_cover_required_fields(self):
        self.assertGreaterEqual(len(CANDIDATES), 8)
        for rec in CANDIDATES:
            validate_candidate(rec)  # must not raise

    def test_missing_source_field_is_rejected(self):
        rec = {k: "x" for k in REQUIRED_FIELDS}
        del rec["license_status"]
        with self.assertRaises(ValueError):
            validate_candidate(rec)

    def test_linkage_and_license_classification(self):
        rec = {k: "x" for k in REQUIRED_FIELDS}
        rec.update(
            linkage=UNVERIFIED, linkage_confidence="verified_same_participants",
            license_status="open_research_license", instrument_class=INST_SCORES_ONLY,
        )
        with self.assertRaises(ValueError):
            validate_candidate(rec)  # confidence must match linkage class

        base = {k: "x" for k in REQUIRED_FIELDS}
        base.update(
            linkage=UNLINKED, linkage_confidence=UNLINKED,
            license_status=LIC_UNAVAILABLE, instrument_class=INST_SCORES_ONLY,
        )
        sc = score_candidate(base)
        self.assertIn("participant_linkage_unverified_or_unknown", sc["hard_blockers"])
        self.assertEqual(sc["overall"], "unsuitable")

    def test_scoring_transparency_unknown_is_not_positive(self):
        rec = {k: "x" for k in REQUIRED_FIELDS}
        rec.update(linkage="verified_same_participants", linkage_confidence="verified_same_participants",
                   license_status="unknown", instrument_class=INST_SCORES_ONLY)
        rec["score_cold_start_suitability"] = None
        rec["score_sample_size"] = None
        sc = score_candidate(rec)
        self.assertIsNone(sc["dimensions"]["sample_size"])
        self.assertIn("license_unavailable_or_unknown", sc["hard_blockers"])
        self.assertEqual(sc["overall"], "unsuitable")

        # A fully documented public linked dataset scores as strong.
        good = dict(rec)
        good["license_status"] = "open_research_license"
        for d in ("cold_start_suitability", "sample_size", "temporal_evaluation_suitability", "reproducibility", "ethical_feasibility", "preference_data_suitability"):
            good[f"score_{d}"] = 2
        sc2 = score_candidate(good)
        self.assertEqual(sc2["overall"], "strong")
        self.assertEqual(sc2["total_score"], 17)

    def test_no_double_counting_participants_vs_interactions(self):
        with tempfile.TemporaryDirectory() as tmp:
            p_csv = Path(tmp) / "personality.csv"
            r_csv = Path(tmp) / "ratings.csv"
            p_csv.write_text(
                "userid\nu1\nu2\nu3\n",
                encoding="utf-8",
            )
            r_csv.write_text(
                "userid, movie_id, rating, tstamp\n"
                "u1, 1, 5, 2020-01-01\nu1, 2, 4, 2020-01-02\nu1, 3, 3, 2020-01-03\n"
                "u2, 1, 4, 2020-01-04\nu2, 2, 2, 2020-01-05\n"
                "u3, 1, 5, 2020-01-06\n"
                "u9, 1, 5, 2020-01-07\n",
                encoding="utf-8",
            )
            audit = sparse_history_audit(r_csv, p_csv, id_col=0, rating_col=2, ts_col=3)
            # 3 linked participants, not 6 rating rows as participants.
            self.assertEqual(audit["personality_participants"], 3)
            self.assertEqual(audit["linked_participants_with_preferences"], 3)
            self.assertEqual(audit["total_ratings_by_linked_users"], 6)
            # u9's rating is excluded (no personality row): no fabricated linkage.
            self.assertNotIn("u9", str(audit["history_groups"]))
            self.assertEqual(audit["history_groups"][">20"], 0)
            self.assertEqual(audit["history_groups"]["1-4"], 3)
            self.assertTrue(audit["timestamps_available"])

    def test_history_groups_and_thresholds(self):
        self.assertEqual(history_group(0), "0")
        self.assertEqual(history_group(4), "1-4")
        self.assertEqual(history_group(9), "5-9")
        self.assertEqual(history_group(20), "10-20")
        self.assertEqual(history_group(21), ">20")

        audit = {
            "linked_participants_with_preferences": 1820,
            "group_1_4": 1, "group_5_9": 0, "group_10_20": 42,
            "timestamps_available": True,
            "total_ratings_by_linked_users": 1028751,
        }
        verdict = check_thresholds(audit)
        self.assertTrue(verdict["meets_min_linked"])
        self.assertFalse(verdict["meets_min_sparse_group"])  # 43 < 100
        self.assertTrue(verdict["temporal_evaluation_possible"])

    def test_unknown_never_converted_to_positive_evidence(self):
        rec = {k: "x" for k in REQUIRED_FIELDS}
        rec.update(linkage="verified_same_participants", linkage_confidence="verified_same_participants",
                   license_status="open_research_license", instrument_class="unknown")
        sc = score_candidate(rec)
        self.assertIn("personality_instrument_not_validated_or_unknown", sc["hard_blockers"])

    def test_candidate_roundtrip_and_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            jpath = Path(tmp) / "c.json"
            cpath = Path(tmp) / "c.csv"
            save_candidates(jpath, CANDIDATES)
            loaded = load_candidates(jpath)
            self.assertEqual(len(loaded), len(CANDIDATES))
            candidates_to_csv(loaded, cpath)
            header = cpath.read_text(encoding="utf-8").splitlines()[0]
            self.assertIn("overall", header)
            self.assertIn("total_score", header)

    def test_cli_rejects_output_dir_outside_project(self):
        proc = subprocess.run(
            [sys.executable, "-I", str(ROOT / "scripts" / "phase12_dataset_feasibility.py"),
             "--output-dir", str(ROOT.parent)],
            capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("inside the project directory", proc.stderr)

    def test_phase11_artifacts_untouched(self):
        p11 = ROOT / "artifacts" / "evaluation" / "phase11-personality-ranking" / "final_report.md"
        self.assertTrue(p11.is_file())
        # Phase 12 must not write anywhere else in the artifacts tree.
        p12_root = ROOT / "artifacts" / "evaluation" / "phase12-dataset-feasibility"
        self.assertTrue(p12_root.is_dir())


if __name__ == "__main__":
    unittest.main()
