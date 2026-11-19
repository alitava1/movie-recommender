"""Phase 12 CLI: dataset discovery, validity audit, cold-start feasibility, and GO/NO-GO decision."""

import argparse
import hashlib
import json
import platform
import sys
import time
import tracemalloc
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts
from recommender.phase12.candidates import CANDIDATES, RUBRIC_DIMENSIONS, candidates_to_csv, load_candidates, save_candidates, score_candidate
from recommender.phase12.feasibility import check_thresholds, sparse_history_audit

OUT_DIR = ROOT / "artifacts" / "evaluation" / "phase12-dataset-feasibility"
DOWNLOADS = OUT_DIR / "downloads"
P18_DIR = DOWNLOADS / "personality-isf2018"

ACCESS_DATE = "2026-11-18"


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_md(path, text):
    Path(path).write_text(text, encoding="utf-8")


def build_candidate_records():
    """Validate embedded candidates and attach rubric scores; verified counts stay verbatim."""
    records = [dict(c) for c in CANDIDATES]
    for rec in records:
        rec["scoring"] = score_candidate(rec)
    return records


def p18_structural_audit():
    """Structural audit of the downloaded Personality 2018 dataset (authorized public download)."""
    if not (P18_DIR / "ratings.csv").is_file():
        return None
    audit = sparse_history_audit(
        P18_DIR / "ratings.csv",
        P18_DIR / "personality-data.csv",
        id_col=0, rating_col=2, ts_col=3,
    )
    audit["zip_sha256"] = sha256_file(DOWNLOADS / "personality-isf2018.zip")
    return audit


def main():
    parser = argparse.ArgumentParser(description="Phase 12 dataset feasibility research.")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if sys.platform != "win32":
        tracemalloc.start()

    records = build_candidate_records()
    save_candidates(out_dir / "dataset_candidates.json", records)
    candidates_to_csv(records, out_dir / "dataset_candidates.csv")

    p18 = p18_structural_audit()
    if p18:
        artifacts.save(out_dir / "cold_start_feasibility.json", p18, "evaluation")
        thresholds = check_thresholds(p18)
    else:
        thresholds = None

    runtime = {"wall_seconds": time.perf_counter() - t0, "python": sys.version, "platform": platform.platform()}
    artifacts.save(out_dir / "runtime_report.json", runtime, "evaluation")

    test_proc_args = [sys.executable, "-I", "-m", "unittest", "discover", "-s", "tests", "-v"]
    import subprocess

    test_proc = subprocess.run(test_proc_args, cwd=ROOT, capture_output=True, text=True, check=False)
    (out_dir / "test_report.txt").write_text(test_proc.stdout + "\n" + test_proc.stderr, encoding="utf-8")
    print(f"Phase 12 records validated: {len(records)}; P18 audit: {'yes' if p18 else 'missing download'}")
    print(f"Tests exit code: {test_proc.returncode}")


if __name__ == "__main__":
    main()
