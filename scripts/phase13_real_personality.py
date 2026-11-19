"""Phase 13 CLI: real personality-aware recommendation on GroupLens Personality 2018."""

import argparse
import csv
import ctypes
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import tracemalloc

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts
from recommender.phase13 import engine, experiments
from recommender.phase13.data import TRAITS, full_audit
from recommender.phase13.experiments import run_ablations, run_baseline, run_sparse, run_statistics, run_warm_personality, run_zero_history

OUT_DIR = ROOT / "artifacts" / "evaluation" / "phase13-real-personality"


def _peak_memory_bytes():
    if sys.platform == "win32":
        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
            ]
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        psapi.GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PROCESS_MEMORY_COUNTERS), ctypes.c_ulong]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.GetCurrentProcess()
        if psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return counters.PeakWorkingSetSize, "PeakWorkingSetSize"
    return tracemalloc.get_traced_memory()[1], "tracemalloc_peak_python_bytes"


def _write_csv(path, fields, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Phase 13 real personality experiment.")
    parser.add_argument("--stage", choices=("audit", "baseline", "sparse", "zero", "stats", "repro", "full"), default="full")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=120)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        tracemalloc.start()
    t0 = time.perf_counter()

    print("Stage A: dataset audit (aggregates only)...", flush=True)
    traits_raw, rows, audit = full_audit()
    artifacts.save(out_dir / "data_audit.json", audit, "evaluation")
    artifacts.save(out_dir / "dataset_checksums.json", audit["checksums"], "evaluation")
    artifacts.save(out_dir / "participant_linkage.json", audit["linkage"], "evaluation")
    artifacts.save(out_dir / "trait_distribution.json", audit["trait_ranges_raw_1_7"], "evaluation")
    print(f"  participants={audit['experiment_population']} ratings={audit['ratings']['rows_after_latest_per_pair']} "
          f"dups_removed={audit['ratings']['duplicate_user_movie_pairs_removed']} "
          f"personality_dup_rows={audit['personality']['duplicate_rows']}", flush=True)

    smoke_users = None
    if args.smoke:
        smoke_users = set(range(min(args.smoke_users, audit["experiment_population"])))
        rows = [r for r in rows if r[0] in smoke_users]

    if args.stage in ("baseline", "full"):
        print("Track B: warm-stratum non-personality baselines...", flush=True)
        baseline = run_baseline(traits_raw, rows, smoke_users)
        artifacts.save(out_dir / "baseline_results.json", {
            "train_rows": baseline["train_rows"], "targets": baseline["targets_count"],
            "excluded": baseline["excluded"], "catalog_size": baseline["catalog_size"],
            "candidate_recall": baseline["candidate_recall"], "summary": baseline["summary"],
        }, "evaluation")
        print(f"  done: pop={baseline['summary']['popularity_only']['ndcg']:.4f} "
              f"cf={baseline['summary']['cf_only']['ndcg']:.4f} "
              f"hybrid={baseline['summary']['personality_disabled']['ndcg']:.4f}", flush=True)
    if args.stage in ("stats", "full"):
        print("Track C: warm personality integration (P1-P3)...", flush=True)
        warm = run_warm_personality(baseline, smoke_users)
        print(f"  w_p={warm['inner_selected_weight']}; selected={warm['summary']['personality_selected']['ndcg']:.4f} "
              f"disabled={warm['summary']['personality_disabled']['ndcg']:.4f}", flush=True)
    if args.stage in ("sparse", "full"):
        print("Track D: simulated sparse histories K in {1,5,10,20}...", flush=True)
        sparse = run_sparse(traits_raw, rows, smoke_users)
        for k, res in sparse.items():
            print(f"  {k}: targets={res['targets']} w_p={res['inner_selected_weight']} "
                  f"correct={res['summary']['correct_selected']['ndcg']:.4f} "
                  f"baseline={res['summary']['baseline_disabled']['ndcg']:.4f}", flush=True)
    if args.stage in ("zero", "full"):
        print("Track E: zero-history leave-one-user-out...", flush=True)
        zero = run_zero_history(traits_raw, rows, smoke_users)
        print(f"  blend_w_p={zero['blend_weight']}; pers_only={zero['summary']['personality_only']['ndcg']:.4f} "
              f"pop_only={zero['summary']['popularity_only']['ndcg']:.4f}", flush=True)
    if args.stage in ("stats", "full"):
        print("Statistics: preregistered P1-P5 + Holm-Bonferroni...", flush=True)
        comparisons = experiments.run_statistics(baseline, warm, sparse, zero, resamples=200 if args.smoke else args.resamples)
        artifacts.save(out_dir / "bootstrap_results.json", comparisons, "evaluation")
        fields = ["comparison", "family", "metric", "abs_diff", "rel_diff", "ci_lower", "ci_upper",
                  "includes_zero", "p_value", "p_holm", "significant_holm_05", "users", "observations"]
        _write_csv(out_dir / "statistical_comparisons.csv", fields, comparisons)

    if args.stage == "audit":
        print(f"Stage audit done in {time.perf_counter() - t0:.1f} s")
        return

    if args.stage == "repro":
        print("Reproducibility: two independent passes on 400-participant subset...", flush=True)
        sample = set(range(min(400, audit["experiment_population"])))
        runs = []
        for pass_idx in (1, 2):
            b = run_baseline(traits_raw, rows, sample)
            w = run_warm_personality(b, sample)
            runs.append(json.dumps({
                "baseline_summary": b["summary"],
                "warm_summary": {k: v for k, v in w["summary"].items()},
                "w_p": w["inner_selected_weight"],
            }, sort_keys=True))
            print(f"  pass {pass_idx} done", flush=True)
        repro = {
            "match": runs[0] == runs[1],
            "tolerance": "1e-12 (exact equality of serialized aggregates)",
            "comparison_population": "first 400 participant indices; identical seeds and configs",
            "canonical_bytes": len(runs[0]),
        }
        artifacts.save(out_dir / "reproducibility_report.json", repro, "evaluation")
        print(f"  reproducibility match: {repro['match']}")
        return

    if args.stage != "full":
        print(f"Stage {args.stage} done in {time.perf_counter() - t0:.1f} s")
        return

    print("Ablations, segments, diversity, runtime, and reports...", flush=True)
    ablation_rows = run_ablations(baseline["models"], baseline["targets"], traits_raw)
    _write_csv(out_dir / "personality_ablation.csv", ["ablation", "level", "ndcg", "hr", "coverage"], ablation_rows)

    seg = experiments.segment_by_history(baseline["models"], baseline["per_user"]["personality_disabled"], baseline["targets"])
    seg_rows = []
    for name, doc in seg.items():
        seg_rows.append({"segment": name, "users": doc["users"], "cutpoints": str(doc["cutpoints"]),
                         **{m: round(doc.get(m, 0.0), 6) for m in ("ndcg", "map", "hr", "ild", "novelty")}})
    _write_csv(out_dir / "segment_results.csv", ["segment", "users", "cutpoints", "ndcg", "map", "hr", "ild", "novelty"], seg_rows)

    div_rows = []
    for track, summary in (("warm_baseline", baseline["summary"]), ("warm_personality", warm["summary"]),
                           ("zero_history", zero["summary"])):
        for cname, m in summary.items():
            div_rows.append({"track": track, "condition": cname, "coverage": round(m["coverage"], 6),
                             "ild": round(m["ild"], 6), "novelty": round(m["novelty"], 6),
                             "long_tail_share": round(m["long_tail_share"], 6),
                             "popularity_concentration": round(m["popularity_concentration"], 6)})
    for k, res in sparse.items():
        for cname, m in res["summary"].items():
            div_rows.append({"track": f"sparse_{k}", "condition": cname, "coverage": round(m["coverage"], 6),
                             "ild": round(m["ild"], 6), "novelty": round(m["novelty"], 6),
                             "long_tail_share": round(m["long_tail_share"], 6),
                             "popularity_concentration": round(m["popularity_concentration"], 6)})
    _write_csv(out_dir / "diversity_results.csv", ["track", "condition", "coverage", "ild", "novelty", "long_tail_share", "popularity_concentration"], div_rows)

    sparse_rows = []
    for k, res in sparse.items():
        for cname, m in res["summary"].items():
            sparse_rows.append({"stratum": k, "condition": cname, "users": m["users"],
                                "ndcg": round(m["ndcg"], 6), "map": round(m["map"], 6), "hr": round(m["hr"], 6),
                                "recall10": round(m["recall"], 6), "recall50": round(m["recall50"], 6)})
    _write_csv(out_dir / "sparse_history_results.csv", ["stratum", "condition", "users", "ndcg", "map", "hr", "recall10", "recall50"], sparse_rows)
    zero_rows = [{"condition": cname, "users": m["users"], "ndcg": round(m["ndcg"], 6), "map": round(m["map"], 6),
                  "hr": round(m["hr"], 6), "recall10": round(m["recall"], 6), "recall50": round(m["recall50"], 6)}
                 for cname, m in zero["summary"].items()]
    _write_csv(out_dir / "zero_history_results.csv", ["condition", "users", "ndcg", "map", "hr", "recall10", "recall50"], zero_rows)

    total_wall = time.perf_counter() - t0
    peak, mem_method = _peak_memory_bytes()
    artifacts.save(out_dir / "runtime_memory_report.json", {
        "total_wall_seconds": total_wall, "peak_memory_bytes": peak, "memory_measurement": mem_method,
        "python": sys.version, "platform": platform.platform(), "smoke_mode": args.smoke,
        "numpy": __import__("numpy").__version__,
    }, "evaluation")
    artifacts.save(out_dir / "experiment_config.json", {
        "phase": 13, "seed": 2026, "pool_m": engine.POOL_M, "relevance_threshold": engine.RELEVANCE,
        "weight_grid": list(engine.WEIGHT_GRID), "ridge_alpha": 5.0, "k_values": list(engine.K_VALUES),
        "folds_zero_history": 10, "min_train_history": engine.MIN_TRAIN_HISTORY,
        "bootstrap_resamples": 200 if args.smoke else args.resamples,
        "primary_comparisons": ["P1_personality_vs_baseline_warm", "P2_correct_vs_shuffled_warm",
                                "P3_correct_vs_random_warm", "P4_personality_only_vs_popularity_zero",
                                "P5_personality_vs_sparse_baseline_K5"],
        "disclaimer": "Real published questionnaire-derived scores; simulated sparse/zero-history strata; aggregate-only artifacts.",
    }, "evaluation")

    test_proc = subprocess.run([sys.executable, "-I", "-m", "unittest", "discover", "-s", "tests", "-v"],
                               cwd=ROOT, capture_output=True, text=True, check=False)
    (out_dir / "test_report.txt").write_text(test_proc.stdout + "\n" + test_proc.stderr, encoding="utf-8")

    from recommender.phase13.reporting import write_final_report, write_limitations, write_data_audit_md
    write_data_audit_md(out_dir, audit)
    write_limitations(out_dir)
    write_final_report(out_dir, audit, baseline, warm, sparse, zero, comparisons, ablation_rows, seg_rows, total_wall, peak, mem_method)

    print(f"Total wall: {total_wall:.1f} s; peak memory: {peak:,} bytes ({mem_method})")
    print(f"Tests exit code: {test_proc.returncode}")
    print(f"Deliverables saved under {out_dir}")


if __name__ == "__main__":
    main()
