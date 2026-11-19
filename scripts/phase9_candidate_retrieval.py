"""Phase 9 CLI: candidate generation and retrieval research on MovieLens 1M."""

import argparse
import ctypes
import json
import os
from pathlib import Path
import platform
import sys
import time
import tracemalloc

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(var, "1")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts, content
from recommender.data import load_movies, load_ratings
from recommender.phase7.temporal import MIN_TRAIN_HISTORY, RELEVANT_THRESHOLD, split_outer_temporal
from recommender.phase9.experiments import (
    EVAL_M,
    POOL_SIZES,
    bootstrap_comparisons,
    inner_validation_study,
    outer_evaluation,
    segment_analysis,
)


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


def _write_summary(out_dir, config, inner, outer, comparisons, segments, runtime, repro, catalog_size):
    lines = [
        "# Phase 9: Candidate Generation and Retrieval Research",
        "",
        "## 1. Research questions and methodology",
        "",
        "- **Questions:** (Q1) Does larger M improve retrieval recall? (Q2) Can popularity retrieval recover relevant items CF misses? (Q3) Can content retrieval? (Q4) Do hybrid generators improve recall? (Q5) Does a larger pool improve final NDCG@10? (Q6) Does retrieval improvement cost diversity/novelty/runtime? (Q7) Which segments benefit? (Q8) Does the best configuration generalize to outer test?",
        "- **Protocol:** Reuses Phase 7's temporal semantics unchanged — chronological leave-latest-positive-out outer split (latest rating >= 4.0 target, >= 5 train rows, 6,037 eligible users) and per-user chronological 80/20 inner validation. All generators, popularity statistics, profiles, fusion weights, and selections use training data only.",
        "- **Methods:** A = exact Phase 7 CF retrieval; B = training-only popularity (positive-count signal); C = TF-IDF profile cosine; D = CF+popularity (union, 50/50 quota); E = CF+content (union, 50/50); F = three-way (50/25/25). Pool sizes M in {100, 200, 500, 1000}; downstream ranking uses M=200 pools with both Phase 7 reference rankers on identical pools.",
        "",
        "## 2. Inner-validation retrieval results (selection stage)",
        "",
        "| Method | Recall@100 | Recall@200 | Recall@500 | Recall@1000 | Coverage@200 | Precision@200 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    s = inner["summary"]
    for method in ("cf", "popularity", "content", "cf_popularity", "cf_content", "three_way"):
        row = [f"{method}@{m}" in s and s[f"{method}@{m}"]["mean_recall"] or 0.0 for m in POOL_SIZES]
        cov = s.get(f"{method}@{EVAL_M}", {}).get("user_coverage", 0.0)
        prec = s.get(f"{method}@{EVAL_M}", {}).get("mean_candidate_precision", 0.0)
        lines.append(f"| {method} | " + " | ".join(f"{v:.4f}" for v in row) + f" | {cov:.4f} | {prec:.4f} |")

    lines.extend([
        "",
        f"- **Selection (inner only):** best single source at M=200: `{inner['selection']['best_single_at_eval_m']}`; "
        f"family pools for downstream ranking at M={EVAL_M}: {', '.join(f'{k} (quota {v})' for k, v in inner['selection'].items() if k != 'best_single_at_eval_m')}.",
        f"- Inner pass wall time: {inner['wall_seconds']:.1f} s over {inner['inner_eligible_users']} eligible users.",
        "",
        "## 3. Outer-test retrieval results (single evaluation, no peeking)",
        "",
        "| Method | Recall@200 | Coverage | Precision | Mean excluded/user |",
        "|---|---:|---:|---:|---:|",
    ])
    for name, stats in outer["retrieval"].items():
        ex = stats["exclusion"]
        lines.append(
            f"| {name} | {stats['mean_recall']:.4f} | {stats['user_coverage']:.4f} | {stats['mean_candidate_precision']:.4f} | {ex['mean_excluded_per_user']:.2f} |"
        )

    lines.extend([
        "",
        "## 4. Downstream ranking on identical candidate pools (outer test)",
        "",
        "Rankers score only CF-evidence items inside each pool; pool members retrieved purely by popularity/content lack CF predictions and cannot be ranked by the existing Phase 7 rankers (documented limitation, quantified by `cf_scored_fraction`).",
        "",
        "| Ranker | NDCG@10 | MAP@10 | HR@10 | P@10 | R@10 | Coverage | ILD@10 | Novelty | Users |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, pooled in outer["ranking_summary"].items():
        lines.append(
            f"| {name} | {pooled['ndcg']:.4f} | {pooled['map']:.4f} | {pooled['hr']:.4f} | {pooled['precision']:.4f} | "
            f"{pooled['recall']:.4f} | {pooled['coverage']:.4f} | {pooled['ild']:.4f} | {pooled['novelty']:.4f} | {pooled['users']} |"
        )

    cf_frac = [rec["cf_scored_fraction"] for rec in outer["pool_records"].values()]
    lines.append(
        f"- Mean CF-scored fraction of the ranked pool: {sum(cf_frac) / len(cf_frac):.4f} "
        f"(min {min(cf_frac):.4f}, max {max(cf_frac):.4f})."
    )

    lines.extend([
        "",
        "## 5. Paired user-level bootstrap (2,000 resamples, seed 2026, 95% CI)",
        "",
        "| Comparison | Mean diff | Rel diff | 95% CI | Includes zero |",
        "|---|---:|---:|---|---|",
    ])
    for label, comp in comparisons.items():
        includes = comp["abs_lower"] <= 0.0 <= comp["abs_upper"]
        rel = f"{100 * comp['rel_diff']:+.2f}%" if comp["rel_diff"] is not None else "—"
        lines.append(
            f"| {label} | {comp['abs_diff']:+.5f} | {rel} | [{comp['abs_lower']:+.5f}, {comp['abs_upper']:+.5f}] | {includes} |"
        )

    lines.extend([
        "",
        "## 6. User-segment analysis (training-only history cutoffs 30/80)",
        "",
        "| Segment | Users | Best retrieval recall@200 | CF ranker NDCG@10 (CF pool) | Enrichment ranker NDCG@10 (best pool) |",
        "|---|---:|---:|---:|---:|",
    ])
    for seg, doc in segments.items():
        best = max(
            (doc.get(name, {}).get("mean_recall") or 0.0 for name in ("cf@200", "popularity@200", "content@200", "cf_popularity@200", "cf_content@200", "three_way@200")),
            default=0.0,
        )
        cf_nd = doc.get("cf_ranker_cf_pool", {}).get("ndcg")
        en_nd = doc.get("enrichment_ranker_best_pool", {}).get("ndcg")
        lines.append(
            f"| {seg} | {doc['users']} | {best:.4f} | {cf_nd if cf_nd is not None else float('nan'):.4f} | {en_nd if en_nd is not None else float('nan'):.4f} |"
        )

    lines.extend([
        "",
        "## 7. Runtime, memory, reproducibility, and limitations",
        "",
        f"- Primary pass wall time: {runtime['primary_wall_seconds']:.1f} s; reproducibility passes: "
        f"{repro.get('pass1_wall_seconds', 0):.1f} s / {repro.get('pass2_wall_seconds', 0):.1f} s.",
        f"- Peak memory: {runtime['peak_memory_bytes']:,} bytes ({runtime['memory_measurement']}); BLAS threads capped at 1.",
        f"- Deterministic outputs match: `{repro['match']}`.",
        "- Limitations: single chronological split; existing rankers cannot score non-CF pool members (fusion pools are only partially exploitable downstream); fusion modes/quotas are the documented starting configurations selected on inner validation; exploratory comparisons without multiplicity correction.",
        "",
    ])
    (out_dir / "phase9_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 9 candidate generation and retrieval research.")
    parser.add_argument("--smoke", action="store_true", help="Run a limited-user smoke configuration.")
    parser.add_argument("--smoke-users", type=int, default=200)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repro-only", action="store_true", help="Repeat the experiment on a deterministic subsample and compare.")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "evaluation" / "phase9-candidate-retrieval",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")

    dataset_dir = ROOT / "data" / "raw" / "ml-1m"
    if not (dataset_dir / "ratings.dat").is_file():
        print("MovieLens 1M missing under data/raw/ml-1m; refusing to download without authorization.", file=sys.stderr)
        sys.exit(2)

    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    settings["cf_tuning"] = {"neighbors": 100, "prediction_shrinkage": 300.0}
    movies = load_movies("ml-1m", dataset_dir / "movies.dat")
    vectors = content.build_tfidf(movies)

    if sys.platform != "win32":
        tracemalloc.start()
    t0 = time.perf_counter()

    print("Loading ratings and building Phase 7 temporal split...", flush=True)
    ratings = load_ratings("ml-1m", dataset_dir / "ratings.dat")
    outer_train, outer_test_rows, excluded = split_outer_temporal(ratings)
    smoke_users = None
    if args.smoke:
        smoke_users = {row[0] for row in outer_test_rows if row[0] <= args.smoke_users}
        ratings_train = [r for r in outer_train if r[0] in smoke_users]
        ratings_test = [r for r in outer_test_rows if r[0] in smoke_users]
    else:
        ratings_train, ratings_test = outer_train, outer_test_rows

    print("Stage 5: inner-validation retrieval study...", flush=True)
    inner = inner_validation_study(
        ratings_train, movies, vectors, settings,
        user_filter=smoke_users,
    )
    best_family = max(
        (name for name in ("cf", "popularity", "content", "cf_popularity", "cf_content", "three_way")),
        key=lambda name: inner["summary"].get(f"{name}@{EVAL_M}", {}).get("mean_recall", 0.0),
    )
    print(f"  inner done in {inner['wall_seconds']:.1f} s; best family at M={EVAL_M}: {best_family}", flush=True)

    print("Stage 6-7: outer evaluation, ranking, segments, bootstrap...", flush=True)
    outer = outer_evaluation(
        ratings_train, ratings_test, movies, vectors, settings,
        selected={"best_family": best_family},
        user_filter=smoke_users,
    )
    segments = segment_analysis(ratings_train, outer["retrieval_per_user"], outer["ranking_per_user"])
    comparisons = bootstrap_comparisons(
        outer["retrieval_per_user"], outer["ranking_per_user"],
        resamples=args.resamples, confidence=0.95, seed=args.seed,
    )
    primary_wall = time.perf_counter() - t0
    peak_bytes, mem_method = _peak_memory_bytes()

    repro_doc = {"match": None}
    if not args.smoke:
        print("Stage 8: reproducibility pass (deterministic 400-user subsample, full-data models)...", flush=True)
        sample = sorted({row[0] for row in outer_test_rows})[:400]
        sample_set = set(sample)
        repro_runs = []
        for pass_idx in (1, 2):
            started = time.perf_counter()
            outer_r = outer_evaluation(
                outer_train, outer_test_rows, movies, vectors, settings,
                selected={"best_family": best_family},
                user_filter=sample_set,
            )
            repro_runs.append({"outer": outer_r, "wall": time.perf_counter() - started})
            print(f"  repro pass {pass_idx}: {repro_runs[-1]['wall']:.1f} s", flush=True)

        def _canon(run):
            return json.dumps({
                "retrieval": run["outer"]["retrieval"],
                "ranking": run["outer"]["ranking_summary"],
                "lists": run["outer"]["ranked_lists"],
            }, sort_keys=True)

        canon1, canon2 = _canon(repro_runs[0]), _canon(repro_runs[1])
        repro_doc = {
            "match": canon1 == canon2,
            "canonical_bytes": len(canon1),
            "comparison_population": f"first {len(sample)} eligible users; models refit on the complete outer_train in both passes",
            "pass1_wall_seconds": repro_runs[0]["wall"],
            "pass2_wall_seconds": repro_runs[1]["wall"],
            "timing_excluded": "wall-clock seconds excluded from canonical comparison",
            "python": sys.version,
            "platform": platform.platform(),
        }

    runtime = {
        "primary_wall_seconds": primary_wall,
        "peak_memory_bytes": peak_bytes,
        "memory_measurement": mem_method,
        "smoke_mode": args.smoke,
        "inner_wall_seconds": inner["wall_seconds"],
        "outer_wall_seconds": outer["wall_seconds"],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_audit = {
        "dataset": "ml-1m",
        "ratings": len(ratings),
        "eligible_outer_users": len(outer_test_rows),
        "excluded_users": len(excluded),
        "outer_train_rows": len(outer_train),
        "protocol": f"chronological leave-latest-positive-out (threshold {RELEVANT_THRESHOLD}, min_train {MIN_TRAIN_HISTORY})",
    }
    config_doc = {
        "phase": 9,
        "smoke_mode": args.smoke,
        "pool_sizes": list(POOL_SIZES),
        "eval_m": EVAL_M,
        "fusion_modes": {"cf_popularity": "union 50/50", "cf_content": "union 50/50", "three_way": "union 50/25/25"},
        "downstream_rankers": {
            "cf_ranker": {"neighbors": 100, "prediction_shrinkage": 300.0, "pools": ["cf@200", "best_family@200"]},
            "enrichment_ranker": {"neighbors": 100, "prediction_shrinkage": 300.0, "cf_weight": 0.7, "pools": ["best_family@200"]},
        },
        "selection_metric": "inner-validation mean Recall@200",
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95, "unit": "user"},
        "resources": runtime,
    }

    artifacts.save(out_dir / "experiment_config.json", config_doc, "evaluation")
    artifacts.save(out_dir / "dataset_audit.json", dataset_audit, "evaluation")
    artifacts.save(out_dir / "candidate_recall_results.json", {
        "inner": inner["summary"], "outer": {
            name: {k2: v2 for k2, v2 in stats.items() if k2 not in ("recall_by_user", "totals", "pools")}
            for name, stats in outer["retrieval"].items()
        },
    }, "evaluation")
    artifacts.save(out_dir / "candidate_coverage_results.json", {
        name: {"user_coverage": stats["user_coverage"], "users": stats["users"]}
        for name, stats in outer["retrieval"].items()
    }, "evaluation")
    artifacts.save(out_dir / "candidate_source_contributions.json", inner["source_diagnostics"], "evaluation")
    artifacts.save(out_dir / "ranking_results.json", {
        "summary": outer["ranking_summary"],
        "pool_records_sample": dict(list(outer["pool_records"].items())[:50]),
    }, "evaluation")
    artifacts.save(out_dir / "segment_results.json", segments, "evaluation")
    artifacts.save(out_dir / "bootstrap_results.json", comparisons, "evaluation")
    artifacts.save(out_dir / "runtime_memory_report.json", runtime, "evaluation")
    artifacts.save(out_dir / "reproducibility_report.json", repro_doc, "evaluation")
    _write_summary(out_dir, config_doc, inner, outer, comparisons, segments, runtime, repro_doc, len(movies))

    print(f"Total wall: {time.perf_counter() - t0:.1f} s; peak memory: {peak_bytes:,} bytes ({mem_method})")
    print(f"Reproducibility match: {repro_doc.get('match')}")
    print(f"Deliverables saved under {out_dir}")


if __name__ == "__main__":
    main()
