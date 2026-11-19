"""Nested train-only hyperparameter selection on MovieLens 100K u1-u5."""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts
from recommender.data import load_movies, load_ratings
from recommender.nested import aggregate_folds, evaluate_fold


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


def _report(payload, output):
    lines = [
        "# MovieLens 100K nested evaluation",
        "",
        "## Protocol",
        "",
        "- Outer evaluation uses official `u1`–`u5`: fit on each `uN.base`, evaluate once on untouched `uN.test`.",
        "- Inner validation splits each user's outer-training interactions by `(timestamp, movie_id)`; earliest 80% train, latest 20% validate. One-interaction users remain inner-train only.",
        "- Popularity, CF, and user profiles fit on inner-train during selection; TF-IDF uses fixed movie metadata only. Outer refit uses all outer-train interactions.",
        "- Relevant means held-out rating >= 4.0. Users need at least one train rating and one relevant held-out rating. Candidate set is all catalog movies not seen in training.",
        "- Primary selection metric: binary NDCG@10. Fixed baselines: popularity, content, default CF (50 neighbors, prediction shrinkage 100), and default hybrid.",
        "- CF search: 9 settings; hybrid jointly searches same CF settings × 3 predefined weight triples. Similarity shrinkage and popularity prior remain fixed.",
        "- MMR, latent factors, and regularization are not applicable; those models are not implemented.",
        "",
        "## Split audit",
        "",
        "Each official fold has 80,000 base and 20,000 test rows. Base/test interaction-pair overlap is zero; test sets are pairwise disjoint. Users occur in both partitions, so evaluation measures warm-start ranking. MovieLens `u1`–`u5` are block-based, not chronological outer folds.",
        "",
        "## Fold selection and metrics",
        "",
        "| Fold | Inner train | Inner validation | Selection users | CF choice (NDCG@10) | Hybrid choice (NDCG@10) |",
        "|---|---:|---:|---:|---|---|",
    ]
    for fold, result in payload["folds"].items():
        selection = result["selection"]
        cf = selection["cf"]
        hybrid = selection["hybrid"]
        lines.append(
            f"| {fold} | {selection['inner_train_interactions']} | {selection['inner_validation_interactions']} | {selection['evaluated_users']} | "
            f"`{json.dumps(cf['parameters'], sort_keys=True)}` ({cf['ndcg']}) | "
            f"`{json.dumps(hybrid['parameters'], sort_keys=True)}` ({hybrid['ndcg']}) |"
        )
    lines.extend([
        "",
        "| Model | Pooled P@10 | R@10 | NDCG@10 | MAP@10 | Coverage | User-folds | NDCG fold mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, values in payload["summary"]["pooled"].items():
        ndcg = values["fold_mean_std"]["ndcg"]
        lines.append(
            f"| {name} | {values['precision']:.4f} | {values['recall']:.4f} | {values['ndcg']:.4f} | {values['map']:.4f} | "
            f"{values['coverage']:.4f} | {values['users']} | {ndcg['mean']:.4f} ± {ndcg['std']:.4f} |"
        )
    lines.extend(["", "## Paired bootstrap comparisons", "", "Intervals use shared user-fold resamples; they are descriptive, not independent-user confirmatory intervals. No multiplicity correction applied.", ""])
    for name, result in payload["summary"]["comparisons"].items():
        if result is None:
            lines.append(f"- {name}: unavailable (no paired users).")
        else:
            lines.append(
                f"- {name}: mean difference {result['left_minus_right']:.5f}; "
                f"{result['confidence']:.0%} percentile CI [{result['lower']:.5f}, {result['upper']:.5f}], "
                f"n={result['paired_user_folds']}, resamples={result['resamples']}."
            )
    lines.extend(["", "## Resources", "", f"- Total wall time: {payload['resources']['wall_seconds']:.1f} s", f"- Peak memory: {payload['resources']['peak_memory_bytes']} bytes ({payload['resources']['memory_measurement']})", "- Per-fold phase timings and interaction counts are in JSON artifact.", "", "## Limitations", "", "Fold-to-fold standard deviation describes split variation, not significance. User-fold rows repeat users across folds; bootstrap intervals do not model all dependence induced by repeated users or overlapping training sets. These intervals are exploratory; no multiplicity adjustment was applied. Inner validation is chronological per user, but outer official folds are block-based. No claim of statistically significant superiority follows from a larger mean alone.", ""])
    output.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Nested model selection and evaluation on MovieLens 100K.")
    parser.add_argument("--dataset", default="ml-100k", choices=["ml-100k"])
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "ml-100k-nested.json")
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or any(fold not in range(1, 6) for fold in args.folds):
        parser.error("--folds must contain unique values from 1-5")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT):
        parser.error("--output must be inside the project directory")
    experiment = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    nested_settings = experiment["nested"]
    k = args.k if args.k is not None else experiment["evaluation"]["k"]
    selection_k = nested_settings["selection_k"]
    if k <= 0:
        parser.error("--k must be positive")
    threshold = experiment["evaluation"]["relevant_threshold"]
    dataset_dir = ROOT / "data" / "raw" / args.dataset
    movies = load_movies(args.dataset, dataset_dir / "u.item")

    if sys.platform != "win32":
        tracemalloc.start()
    wall_started = time.perf_counter()
    folds = {}
    for fold in args.folds:
        train = load_ratings(args.dataset, dataset_dir / f"u{fold}.base")
        test = load_ratings(args.dataset, dataset_dir / f"u{fold}.test")
        result = evaluate_fold(
            fold, train, test, movies, settings, nested_settings,
            k, threshold, selection_k=selection_k,
        )
        folds[f"u{fold}"] = result
        print(f"u{fold}: selection {result['resources']['selection_seconds']:.1f}s, refit {result['resources']['refit_seconds']:.1f}s, outer score {result['resources']['outer_scoring_seconds']:.1f}s, users={result['metrics']['hybrid_tuned']['users']}")
    wall_seconds = time.perf_counter() - wall_started
    peak_bytes, memory_measurement = _peak_memory_bytes()
    summary = aggregate_folds(
        folds, len(movies), nested_settings["bootstrap_resamples"], nested_settings["confidence"], nested_settings["seed"]
    )
    public_folds = {
        fold: {key: value for key, value in result.items() if not key.startswith("_")} | {
            "resources": result["resources"],
            "per_user_ndcg": {
                name: {user: values["ndcg"] for user, values in rows.items()}
                for name, rows in result["per_user"].items()
            },
        }
        for fold, result in folds.items()
    }
    payload = {
        "dataset": args.dataset,
        "protocol": "nested-per-user-chronological-inner-validation",
        "k": k,
        "relevant_threshold": threshold,
        "folds": public_folds,
        "summary": summary,
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_memory_bytes": peak_bytes,
            "memory_measurement": memory_measurement,
        },
        "selection": {
            "metric": "ndcg",
            "inner_validation_fraction": nested_settings["inner_validation_fraction"],
            "cf_grid": nested_settings["cf_grid"],
            "hybrid_weight_grid": nested_settings["hybrid_weight_grid"],
            "seed": nested_settings["seed"],
        },
        "limitations": [
            "Official outer folds are block-based, not chronological.",
            "Users repeat across folds; bootstrap intervals do not model all dependence.",
            "No multiplicity correction; confidence intervals are exploratory.",
            "No personality, MMR, latent-factor, or GPU components are evaluated.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    artifacts.save(output, payload, "evaluation")
    report_path = output.with_suffix(".md")
    _report(payload, report_path)
    print(f"Total wall time: {wall_seconds:.1f}s; peak memory: {peak_bytes} bytes ({memory_measurement})")
    print(f"Results: {output}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()
