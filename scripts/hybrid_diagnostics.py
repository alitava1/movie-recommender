"""Phase 5 hybrid recommender diagnostics and CF-first nested evaluation on MovieLens 100K."""

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
from recommender.diagnostics import (
    DEFAULT_CF_FIRST_WEIGHTS,
    DEFAULT_MMR_LAMBDAS,
    PRIMARY_MMR_LAMBDA,
    aggregate_audit,
    aggregate_phase5_folds,
    evaluate_phase5_fold,
)


CORE_MODELS = (
    "cf_tuned",
    "hybrid_tuned",
    "cf_top100_content_enrichment",
    "cf_top100_content_reranking",
    "cf_top100_content_reranking_mmr",
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


def _determine_best_validated(summary):
    pooled = summary["pooled"]
    comparisons = summary["comparisons"]
    candidates = [
        name for name in ("cf_top100_content_enrichment", "cf_top100_content_reranking", "cf_top100_content_reranking_mmr")
        if pooled.get(name, {}).get("ndcg") is not None
    ]
    best_cf_first = max(candidates, key=lambda name: pooled[name]["ndcg"]) if candidates else None
    cf_ndcg = pooled["cf_tuned"]["ndcg"]
    reproducible_beat = False
    if best_cf_first is not None:
        comp = comparisons.get(f"{best_cf_first}_vs_cf_tuned_ndcg")
        if comp is not None and comp["lower"] > 0.0:
            reproducible_beat = True
    return {
        "best_overall_reference": best_cf_first if reproducible_beat else "cf_tuned",
        "replace_cf_tuned": reproducible_beat,
        "best_cf_first_variant": best_cf_first,
        "best_cf_first_ndcg": pooled[best_cf_first]["ndcg"] if best_cf_first else None,
        "cf_tuned_ndcg": cf_ndcg,
        "criterion": "Replace cf_tuned only if a validated CF-first configuration beats cf_tuned with 95% paired bootstrap CI strictly above zero.",
    }


def _fmt_mean_sd(row, metric):
    stat = row["fold_mean_std"][metric]
    return f"{stat['mean']:.4f} ± {stat['std']:.4f}"


def _write_report(payload, output_path):
    audit = payload["audit"]
    pooled = payload["summary"]["pooled"]
    comparisons = payload["summary"]["comparisons"]
    best = payload["best_validated"]
    intrusion = audit["pooled_intrusion"]
    ablation = audit["mean_inner_ndcg_ablation"]
    corrs = audit["mean_correlations"]
    pools = audit["mean_candidate_pools"]
    dists = audit["mean_score_distributions"]

    lines = [
        "# Phase 5: Hybrid Recommender Diagnostics and CF-First Evaluation",
        "",
        "## Executive diagnostic summary",
        "",
        f"- **Reference baseline:** `cf_tuned` (pooled NDCG@10 `{pooled['cf_tuned']['ndcg']:.4f}`, MAP@10 `{pooled['cf_tuned']['map']:.4f}`, Coverage `{pooled['cf_tuned']['coverage']:.4f}`).",
        f"- **Best validated configuration:** `{best['best_overall_reference']}` (`replace_cf_tuned={best['replace_cf_tuned']}`).",
        f"- **Why existing `hybrid_tuned` underperforms `cf_tuned`:**",
        f"  1. **Missing-signal weight renormalization:** Across inner validation splits, `cf` scores `{pools['cf_scored']['mean']:.1f}` of `{pools['all_unseen']['mean']:.1f}` unseen candidates, leaving `{pools['missing_cf']['mean']:.1f}` unlinked items without a CF score. Because `hybrid.blend()` divides by the sum of *present* signal weights (`content + popularity`), items without CF evidence receive a full-scale `(0.3 * content + 0.2 * popularity) / 0.5` score and enter top-10.",
        f"  2. **Irrelevant out-of-pool intrusion:** Across the 5 inner folds, `{intrusion['no_cf_slots']}` / `{intrusion['total_hybrid_top10_slots']}` (`{100 * intrusion['no_cf_slots'] / max(1, intrusion['total_hybrid_top10_slots']):.1f}%`) hybrid top-10 recommendations had **no CF score**, yielding `{intrusion['no_cf_hits']}` hits (`precision={intrusion['no_cf_precision']:.4f}`), while displacing `{intrusion['displaced_cf_slots']}` CF top-10 items (`precision={intrusion['displaced_cf_precision']:.4f}`).",
        f"  3. **Removing renormalization immediately recovers accuracy on inner validation:** Fixing missing CF to `0.0` raises inner NDCG@10 from `{ablation['hybrid_default_renorm']['mean']:.4f}` to `{ablation['hybrid_zero_missing_cf']['mean']:.4f}`, and gating to CF-scored items reaches `{ablation['hybrid_cf_scored_only']['mean']:.4f}` (vs `cf_default` `{ablation['cf_default']['mean']:.4f}`).",
        f"  4. **Distribution & alignment mismatch:** Raw CF predictions span `{dists['cf']['raw_spread_mean']['mean']:.4f}` (std `{dists['cf']['raw_std_mean']['mean']:.4f}`), whereas content spans `{dists['content']['raw_spread_mean']['mean']:.4f}` with near-zero CF correlation (Pearson `{corrs['pearson_cf_content']['mean']:.4f}`). Min-max normalization stretches both into `[0, 1]`, injecting TF-IDF noise into top ranks, while popularity (`Pearson={corrs['pearson_cf_popularity']['mean']:.4f}`) pulls recommendations toward popular items and lowers coverage.",
        "",
        "## Score distributions, correlations, and candidate overlap (inner validation)",
        "",
        "| Signal | Raw min | Raw max | Raw spread | Raw std | Norm mean | Norm std | Norm (relevant) | Norm (non-relevant) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for sig in ("cf", "content", "popularity"):
        d = dists[sig]
        lines.append(
            f"| {sig} | {d['raw_min_mean']['mean']:.4f} | {d['raw_max_mean']['mean']:.4f} | {d['raw_spread_mean']['mean']:.4f} | "
            f"{d['raw_std_mean']['mean']:.4f} | {d['norm_mean']['mean']:.4f} | {d['norm_std_mean']['mean']:.4f} | "
            f"{d['norm_relevant_mean']['mean']:.4f} | {d['norm_nonrelevant_mean']['mean']:.4f} |"
        )
    lines.extend([
        "",
        "| Diagnostic metric (mean ± fold SD across u1–u5 inner val) | Value |",
        "|---|---:|",
        f"| Unseen candidates per user | {pools['all_unseen']['mean']:.1f} ± {pools['all_unseen']['std']:.1f} |",
        f"| CF-scored candidates per user | {pools['cf_scored']['mean']:.1f} ± {pools['cf_scored']['std']:.1f} |",
        f"| Missing-CF candidates per user | {pools['missing_cf']['mean']:.1f} ± {pools['missing_cf']['std']:.1f} |",
        f"| CF top-100 recall ceiling | {pools['cf_top100_recall']['mean']:.4f} ± {pools['cf_top100_recall']['std']:.4f} |",
        f"| All-CF recall ceiling | {pools['cf_all_recall']['mean']:.4f} ± {pools['cf_all_recall']['std']:.4f} |",
        f"| Jaccard@10 (CF vs Content) | {pools['jaccard_top10_cf_content']['mean']:.4f} ± {pools['jaccard_top10_cf_content']['std']:.4f} |",
        f"| Jaccard@100 (CF vs Content) | {pools['jaccard_top100_cf_content']['mean']:.4f} ± {pools['jaccard_top100_cf_content']['std']:.4f} |",
        f"| Jaccard@10 (CF vs Popularity) | {pools['jaccard_top10_cf_popularity']['mean']:.4f} ± {pools['jaccard_top10_cf_popularity']['std']:.4f} |",
        f"| Jaccard@100 (CF vs Popularity) | {pools['jaccard_top100_cf_popularity']['mean']:.4f} ± {pools['jaccard_top100_cf_popularity']['std']:.4f} |",
        f"| Pearson(CF, Content) | {corrs['pearson_cf_content']['mean']:.4f} ± {corrs['pearson_cf_content']['std']:.4f} |",
        f"| Spearman(CF, Content) | {corrs['spearman_cf_content']['mean']:.4f} ± {corrs['spearman_cf_content']['std']:.4f} |",
        f"| Pearson(CF, Popularity) | {corrs['pearson_cf_popularity']['mean']:.4f} ± {corrs['pearson_cf_popularity']['std']:.4f} |",
        f"| Spearman(CF, Popularity) | {corrs['spearman_cf_popularity']['mean']:.4f} ± {corrs['spearman_cf_popularity']['std']:.4f} |",
        "",
        "## Outer test evaluation: core configurations (pooled across u1–u5)",
        "",
        "| Model | NDCG@10 | MAP@10 | HR@10 | P@10 | R@10 | Coverage | ILD@10 | NDCG fold mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name in CORE_MODELS:
        row = pooled[name]
        lines.append(
            f"| {name} | {row['ndcg']:.4f} | {row['map']:.4f} | {row['hr']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['coverage']:.4f} | {row['ild']:.4f} | {_fmt_mean_sd(row, 'ndcg')} |"
        )

    lines.extend([
        "",
        "## Fold-level mean ± SD across u1–u5",
        "",
        "| Model | NDCG@10 (mean ± SD) | MAP@10 (mean ± SD) | HR@10 (mean ± SD) | Coverage (mean ± SD) | ILD@10 (mean ± SD) |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for name in CORE_MODELS:
        row = pooled[name]
        lines.append(
            f"| {name} | {_fmt_mean_sd(row, 'ndcg')} | {_fmt_mean_sd(row, 'map')} | "
            f"{_fmt_mean_sd(row, 'hr')} | {_fmt_mean_sd(row, 'coverage')} | {_fmt_mean_sd(row, 'ild')} |"
        )

    lines.extend([
        "",
        "## Per-fold outer test metrics and inner-selected parameters",
        "",
        "| Fold | Model | Selected parameters | NDCG@10 | MAP@10 | HR@10 | P@10 | R@10 | Coverage | ILD@10 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold_name, fold_data in payload["folds"].items():
        sel = fold_data["selection"]
        for name in CORE_MODELS:
            m = fold_data["metrics"][name]
            if name == "cf_top100_content_reranking_mmr":
                params = f"{json.dumps(sel['cf_top100_content_reranking']['parameters'], sort_keys=True)} + lam={PRIMARY_MMR_LAMBDA}"
            else:
                params = json.dumps(sel[name]["parameters"], sort_keys=True)
            lines.append(
                f"| {fold_name} | {name} | `{params}` | {m['ndcg']:.4f} | {m['map']:.4f} | {m['hr']:.4f} | "
                f"{m['precision']:.4f} | {m['recall']:.4f} | {m['coverage']:.4f} | {m['ild']:.4f} |"
            )

    lines.extend([
        "",
        "## Independent MMR trade-off sweep (relevance fixed on inner validation)",
        "",
        "| Configuration | Lambda | NDCG@10 | MAP@10 | HR@10 | Coverage | ILD@10 | NDCG diff vs cf_tuned (95% CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for base, prefix in (
        ("CF top-100 + content reranking", "cf_top100_content_reranking_mmr_lam_"),
        ("CF tuned top-100", "cf_tuned_top100_mmr_lam_"),
    ):
        for lam in DEFAULT_MMR_LAMBDAS:
            name = f"{prefix}{lam:.2f}"
            row = pooled[name]
            comp = comparisons.get(f"{name}_vs_cf_tuned_ndcg")
            ci_str = f"{comp['left_minus_right']:+.5f} [{comp['lower']:+.5f}, {comp['upper']:+.5f}]" if comp else "—"
            lines.append(
                f"| {base} | {lam:.2f} | {row['ndcg']:.4f} | {row['map']:.4f} | {row['hr']:.4f} | "
                f"{row['coverage']:.4f} | {row['ild']:.4f} | {ci_str} |"
            )

    lines.extend(["", "## Paired bootstrap comparisons (3,717 user-folds, 2,000 resamples)", ""])
    for key in (
        "hybrid_tuned_vs_cf_tuned_ndcg",
        "cf_top100_content_enrichment_vs_cf_tuned_ndcg",
        "cf_top100_content_reranking_vs_cf_tuned_ndcg",
        "cf_top100_content_reranking_mmr_vs_cf_tuned_ndcg",
        "cf_top100_content_reranking_vs_hybrid_tuned_ndcg",
        "cf_top100_content_enrichment_vs_hybrid_tuned_ndcg",
        "cf_top100_content_reranking_mmr_vs_cf_tuned_ild",
    ):
        comp = comparisons.get(key)
        if comp:
            lines.append(
                f"- `{key}`: mean diff `{comp['left_minus_right']:+.5f}`, 95% CI `[{comp['lower']:+.5f}, {comp['upper']:+.5f}]`."
            )

    lines.extend([
        "",
        "## Resources and limitations",
        "",
        f"- Total wall time: {payload['resources']['wall_seconds']:.1f} s",
        f"- Peak memory: {payload['resources']['peak_memory_bytes']} bytes ({payload['resources']['memory_measurement']})",
        "- Parameters were selected strictly inside per-user chronological inner validation splits of `uN.base`; outer `uN.test` was evaluated once per fold.",
        "- Official `u1`–`u5` folds are block-based and share users across folds; bootstrap intervals remain exploratory without multiplicity adjustment.",
        "- Ranking quality (NDCG@10, MAP@10, HR@10) is evaluated separately from catalog coverage and intra-list diversity (ILD@10).",
        "",
    ])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 5 hybrid diagnostics and CF-first evaluation.")
    parser.add_argument("--dataset", default="ml-100k", choices=["ml-100k"])
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--top-m", type=int, default=100)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "evaluation" / "hybrid-diagnostics",
    )
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds) or any(fold not in range(1, 6) for fold in args.folds):
        parser.error("--folds must contain unique values from 1-5")
    if args.top_m <= 0:
        parser.error("--top-m must be positive")
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")

    experiment = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    nested_settings = experiment["nested"]
    k = args.k if args.k is not None else experiment["evaluation"]["k"]
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
        result = evaluate_phase5_fold(
            fold, train, test, movies, settings, nested_settings,
            k=k, threshold=threshold, top_m=args.top_m,
            cf_first_weights=DEFAULT_CF_FIRST_WEIGHTS,
            mmr_lambdas=DEFAULT_MMR_LAMBDAS,
            primary_mmr_lambda=PRIMARY_MMR_LAMBDA,
        )
        folds[f"u{fold}"] = result
        res = result["resources"]
        print(
            f"u{fold}: audit {res['audit_seconds']:.1f}s, sel {res['selection_seconds']:.1f}s, "
            f"refit {res['refit_seconds']:.1f}s, score {res['outer_scoring_seconds']:.1f}s | "
            f"cf={result['metrics']['cf_tuned']['ndcg']:.4f}, hyb={result['metrics']['hybrid_tuned']['ndcg']:.4f}, "
            f"rerank={result['metrics']['cf_top100_content_reranking']['ndcg']:.4f}, "
            f"enrich={result['metrics']['cf_top100_content_enrichment']['ndcg']:.4f}",
            flush=True,
        )

    wall_seconds = time.perf_counter() - wall_started
    peak_bytes, memory_measurement = _peak_memory_bytes()
    summary = aggregate_phase5_folds(
        folds, len(movies), nested_settings["bootstrap_resamples"], nested_settings["confidence"], nested_settings["seed"]
    )
    audit_summary = aggregate_audit(folds)
    best_validated = _determine_best_validated(summary)

    public_folds = {
        fold: {
            "selection": {
                key: {k2: v2 for k2, v2 in val.items() if k2 != "candidates"}
                if isinstance(val, dict) and "parameters" in val else val
                for key, val in res["selection"].items()
            },
            "metrics": res["metrics"],
            "resources": res["resources"],
        }
        for fold, res in folds.items()
    }

    payload = {
        "dataset": args.dataset,
        "protocol": "phase5-hybrid-diagnostics-and-cf-first-nested-evaluation",
        "k": k,
        "top_m": args.top_m,
        "relevant_threshold": threshold,
        "core_models": list(CORE_MODELS),
        "cf_first_weight_grid": list(DEFAULT_CF_FIRST_WEIGHTS),
        "mmr_lambdas": list(DEFAULT_MMR_LAMBDAS),
        "primary_mmr_lambda": PRIMARY_MMR_LAMBDA,
        "best_validated": best_validated,
        "audit": audit_summary,
        "folds": public_folds,
        "summary": summary,
        "resources": {
            "wall_seconds": wall_seconds,
            "peak_memory_bytes": peak_bytes,
            "memory_measurement": memory_measurement,
        },
        "limitations": [
            "Official outer folds are block-based, not chronological.",
            "Users repeat across folds; bootstrap intervals do not model all cross-fold dependence.",
            "No multiplicity correction; confidence intervals are exploratory.",
            "No personality, transformer, GPU, or MovieLens 1M components are used.",
        ],
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts.save(out_dir / "diagnostics.json", payload, "evaluation")
    (out_dir / "per_fold_results.json").write_text(
        json.dumps({"folds": public_folds, "fold_mean_std": {m: summary["pooled"][m]["fold_mean_std"] for m in summary["pooled"]}}, indent=2),
        encoding="utf-8",
    )
    (out_dir / "score_analysis.json").write_text(
        json.dumps(audit_summary, indent=2),
        encoding="utf-8",
    )
    _write_report(payload, out_dir / "report.md")
    print(f"Total wall time: {wall_seconds:.1f}s; peak memory: {peak_bytes} bytes ({memory_measurement})")
    print(f"Best validated reference: {best_validated['best_overall_reference']} (replace_cf_tuned={best_validated['replace_cf_tuned']})")
    print(f"Deliverables saved under {out_dir}")


if __name__ == "__main__":
    main()
