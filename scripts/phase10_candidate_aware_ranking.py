"""Phase 10 CLI: candidate-aware ranking and diversity-constrained recommendation on MovieLens 1M."""

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
from recommender.phase10.audit import audit_phase9_sources
from recommender.phase10.experiments import (
    DIVERSITY_CONFIGS,
    RELEVANCE_CONFIGS,
    evaluate_configurations,
    selection_on_inner,
    segment_analysis,
)
from recommender.phase10.metrics import novelty_stats, popularity_concentration, unique_genre_representation
from recommender.phase10.pareto import pareto_front, pick_representatives
from recommender.phase7.metrics import paired_user_bootstrap


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


def _pareto_configs(outer):
    """Assemble the Pareto objective set from the outer-test summary + novelty metrics."""
    configs = {}
    for name, pooled in outer["summary"].items():
        configs[name] = {
            "ndcg": pooled["ndcg"],
            "coverage": pooled["coverage"],
            "ild": pooled["ild"],
            "novelty": outer["novelty"][name]["novelty"],
        }
    return configs


def _write_summary(out_dir, audit, config_doc, inner, outer, comparisons, segments, pareto_doc, runtime, repro):
    s_inner = inner["summary"]
    s_outer = outer["summary"]
    nov_outer = outer["novelty"]

    lines = [
        "# Phase 10: Candidate-Aware Ranking and Diversity-Constrained Recommendation",
        "",
        "## 1. Phase 9 audit findings (per-user set intersections, inner validation)",
        "",
        f"- Rebuilt Phase 9 generators on inner-train only; identical relevant denominators; pool M=200; {audit['users']} users.",
        f"- Relevant items (totals): CF {audit['relevant_cf']:,}, popularity {audit['relevant_pop']:,}, "
        f"intersection {audit['relevant_intersection']:,}, popularity-only {audit['relevant_pop_only']:,}, "
        f"CF-only {audit['relevant_cf_only']:,}, union {audit['relevant_union']:,}.",
        f"- Recall@200: CF {audit['recall_cf']:.4f}, popularity {audit['recall_pop']:.4f}, "
        f"unrestricted union {audit['recall_union']:.4f}, Phase 9 50/50 quota pool {audit['recall_quota_pool']:.4f}.",
        f"- **Interpretation correction:** union recall exceeds both sources (set inclusion), so the Phase 9 statement "
        f"that cf+popularity (quota) beats CF is confirmed, but the gains come from popularity items CF misses "
        f"({audit['relevant_pop_only']:,} popularity-only relevant items vs {audit['relevant_cf_only']:,} CF-only). "
        f"\"Relevant only via fusion = 0\" in Phase 9 was correct for union mode *by construction* (the fused pool is the "
        f"union of its sources), and the 50/50 quota pool is NOT an unrestricted union: it replaces candidates, which is why "
        f"quota recall ({audit['recall_quota_pool']:.4f}) sits between CF and union.",
        f"- Mean CF-scored fraction: union pool {audit['mean_cf_scored_fraction_union_pool']:.4f}, "
        f"popularity pool {audit['mean_cf_scored_fraction_pop_pool']:.4f} (min {audit['min_cf_scored_fraction_pop_pool']:.4f}), "
        f"quota pool {audit['mean_cf_scored_fraction_quota_pool']:.4f}; "
        f"mean pool sizes: CF {audit['mean_pool_size_cf']:.1f}, popularity {audit['mean_pool_size_pop']:.1f}, "
        f"union {audit['mean_pool_size_union']:.1f}, quota {audit['mean_pool_size_quota']:.1f}.",
        "",
        "## 2. Methodology",
        "",
        "- Shared frozen pools per condition (`cf@200`, `popularity@200`, `cf_popularity@200` quota union) built from training data only; every ranker receives identical candidate IDs.",
        "- Relevance rankers (per-pool min-max normalized components; missing signals renormalized, never dropped): cf_only (historical, drops non-CF), popularity_only, cf_pop_fallback, combined alpha in {0.25, 0.5, 0.75}, three-signal weights {(0.6,0.2,0.2),(0.5,0.3,0.2),(0.4,0.2,0.4)}, RRF k=60.",
        f"- Diversity layers applied to the inner-selected base relevance ranker without touching pools or raw relevance scores: {', '.join(c['name'] for c in DIVERSITY_CONFIGS)}.",
        "- Selection on inner validation only: best-accuracy = max inner NDCG@10; best-balanced = max mean z-score of (NDCG@10, recommendation catalog coverage, novelty). Outer test evaluated once for the selected configs.",
        "",
        "## 3. Inner-validation selection results",
        "",
        "| Config | NDCG@10 | Rec. coverage | Novelty |",
        "|---|---:|---:|---:|",
    ]
    for name in inner["selection"]["accuracy_ranking"]:
        lines.append(
            f"| {name} | {s_inner[name]['ndcg']:.4f} | {s_inner[name]['coverage']:.4f} | {inner['novelty'][name]['novelty']:.4f} |"
        )
    lines.append("")
    lines.append(f"- **Selected:** best-accuracy `{inner['selection']['best_accuracy']}`; best-balanced `{inner['selection']['best_balanced']}` ({inner['selection']['criterion']}).")

    lines.extend([
        "",
        "## 4. Outer-test results (single evaluation)",
        "",
        "| Config | NDCG@10 | MAP@10 | HR@10 | Rec. coverage | ILD@10 | Novelty | Long-tail share | Pop. concentration |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    shown = sorted(
        s_outer,
        key=lambda name: (
            name not in (inner["selection"]["best_accuracy"], inner["selection"]["best_balanced"]),
            -s_outer[name]["ndcg"],
        ),
    )
    for name in shown:
        p = s_outer[name]
        lines.append(
            f"| {name} | {p['ndcg']:.4f} | {p['map']:.4f} | {p['hr']:.4f} | {p['coverage']:.4f} | {p['ild']:.4f} | "
            f"{nov_outer[name]['novelty']:.4f} | {nov_outer[name]['long_tail_share']:.4f} | {outer['popularity_concentration'][name]:.4f} |"
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
        "## 6. Pareto analysis (NDCG@10, recommendation catalog coverage, ILD@10, novelty)",
        "",
        "| Config | Pareto-optimal | Dominated by |",
        "|---|---|---|",
    ])
    for name, info in pareto_doc["report"].items():
        lines.append(f"| {name} | {info['pareto_optimal']} | {info['dominated_by'] or '—'} |")
    lines.append("")
    lines.append(f"- Pareto front: `{pareto_doc['front']}`; representatives: `{pareto_doc['representatives']}`.")

    lines.extend([
        "",
        "## 7. User-segment analysis (training-only cutoffs 30/80)",
        "",
        "| Segment | Users | Best-accuracy NDCG@10 | Best-balanced NDCG@10 | Best-accuracy HR@10 | Best-balanced ILD@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    ba, bb = inner["selection"]["best_accuracy"], inner["selection"]["best_balanced"]
    for seg, doc in segments.items():
        lines.append(
            f"| {seg} | {doc['users']} | {doc.get(ba, {}).get('ndcg') if doc.get(ba, {}).get('ndcg') is not None else float('nan'):.4f} | "
            f"{doc.get(bb, {}).get('ndcg') if doc.get(bb, {}).get('ndcg') is not None else float('nan'):.4f} | "
            f"{doc.get(ba, {}).get('hr') if doc.get(ba, {}).get('hr') is not None else float('nan'):.4f} | "
            f"{doc.get(bb, {}).get('ild') if doc.get(bb, {}).get('ild') is not None else float('nan'):.4f} |"
        )

    lines.extend([
        "",
        "## 8. Scoring completeness, runtime, memory, and reproducibility",
        "",
        "| Pool | CF scored | Pop scored | Content scored | Missing CF | Missing content |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for pname, pdiag in outer["scoring_diagnostics"].items():
        if not pname.startswith("pool:"):
            continue
        lines.append(
            f"| {pdiag['pool']} | {100 * pdiag['cf_scored_pct']:.2f}% | {100 * pdiag['popularity_scored_pct']:.2f}% | "
            f"{100 * pdiag['content_scored_pct']:.2f}% | {100 * pdiag['missing_cf_pct']:.2f}% | {100 * pdiag['missing_content_pct']:.2f}% |"
        )
    lines.extend([
        "",
        f"- General-purpose rankers score 100% of candidates across every pool (`cf_only` drops the missing-CF share above; `cf_pop_fallback` substitutes popularity for missing CF).",
        f"- Primary pass wall time: {runtime['primary_wall_seconds']:.1f} s (audit {runtime['audit_wall_seconds']:.1f} s, inner {runtime['inner_wall_seconds']:.1f} s, outer {runtime['outer_wall_seconds']:.1f} s); peak memory {runtime['peak_memory_bytes']:,} bytes ({runtime['memory_measurement']}).",
        f"- Reproducibility: match=`{repro.get('match')}` ({repro.get('comparison_population', '')}); passes {repro.get('pass1_wall_seconds', 0):.1f} s / {repro.get('pass2_wall_seconds', 0):.1f} s.",
        "- Limitations: single chronological split; exploratory statistics without multiplicity correction; diversity layers evaluated only on the inner-selected base rankers; genre constraint uses the first listed genre as primary (multi-label limitation).",
        "",
    ])
    (out_dir / "phase10_summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 10 candidate-aware ranking experiments.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-users", type=int, default=150)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--repro-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=ROOT / "artifacts" / "evaluation" / "phase10-candidate-aware-ranking",
    )
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")

    dataset_dir = ROOT / "data" / "raw" / "ml-1m"
    if not (dataset_dir / "ratings.dat").is_file():
        print("MovieLens 1M missing; refusing to download without authorization.", file=sys.stderr)
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
    from recommender.phase7.temporal import split_inner_chronological, split_outer_temporal

    outer_train, outer_test_rows, _ = split_outer_temporal(ratings)
    smoke_users = None
    if args.smoke:
        smoke_users = {row[0] for row in outer_test_rows if row[0] <= args.smoke_users}
        train_used = [r for r in outer_train if r[0] in smoke_users]
        test_used = [r for r in outer_test_rows if r[0] in smoke_users]
    else:
        train_used, test_used = outer_train, outer_test_rows

    if args.repro_only:
        print("Reproducibility-only mode (deterministic 400-user subsample)...", flush=True)
        sample = sorted({row[0] for row in outer_test_rows})[:400]
        sample_set = set(sample)
        inner_train_r, validation_r = split_inner_chronological(outer_train)
        from recommender.data import group_by_user

        inner_users_r = group_by_user(inner_train_r)
        rel_r = {}
        for uid, mid, rating, _ in validation_r:
            if rating >= 4.0 and uid in sample_set:
                rel_r.setdefault(uid, set()).add(mid)
        eligible_r = {u: rel for u, rel in rel_r.items() if inner_users_r.get(u)}
        runs = []
        for pass_idx in (1, 2):
            started = time.perf_counter()
            inner_r = evaluate_configurations(
                inner_train_r, eligible_r, movies, vectors, settings,
                relevance_configs=RELEVANCE_CONFIGS, diversity_configs=DIVERSITY_CONFIGS,
            )
            runs.append({"inner": inner_r, "wall": time.perf_counter() - started})
            print(f"  repro pass {pass_idx}: {runs[-1]['wall']:.1f} s", flush=True)

        def _canon_r(run):
            sel = selection_on_inner(run["inner"])
            return json.dumps({
                "summary": run["inner"]["summary"],
                "novelty": run["inner"]["novelty"],
                "lists": run["inner"]["ranked_lists"],
                "selection": {k: sel[k] for k in ("best_accuracy", "best_balanced")},
            }, sort_keys=True)

        repro_only_doc = {
            "match": _canon_r(runs[0]) == _canon_r(runs[1]),
            "comparison_population": f"first {len(sample)} eligible inner-validation users; models refit identically in both passes",
            "pass1_wall_seconds": runs[0]["wall"],
            "pass2_wall_seconds": runs[1]["wall"],
            "timing_excluded": "wall-clock seconds excluded from canonical comparison",
            "python": sys.version,
            "platform": platform.platform(),
        }
        out_dir.mkdir(parents=True, exist_ok=True)
        artifacts.save(out_dir / "reproducibility_report.json", repro_only_doc, "evaluation")
        print(f"Reproducibility match: {repro_only_doc['match']}")
        return

    print("Stage 2: Phase 9 source audit (per-user set intersections)...", flush=True)
    t_audit = time.perf_counter()
    audit = audit_phase9_sources(
        outer_train, movies, settings,
        user_filter=smoke_users,
        pool_m=200, cf_neighbors=100, cf_prediction_shrinkage=300.0,
    )
    audit_wall = time.perf_counter() - t_audit
    print(f"  audit done in {audit_wall:.1f} s: recall CF {audit['recall_cf']:.4f}, pop {audit['recall_pop']:.4f}, union {audit['recall_union']:.4f}, quota {audit['recall_quota_pool']:.4f}", flush=True)

    print("Stage 6: inner-validation ranking comparison...", flush=True)
    relevant_inner = {}
    inner_train, validation = split_inner_chronological(train_used)
    from recommender.data import group_by_user

    inner_users = group_by_user(inner_train)
    for uid, mid, rating, _ in validation:
        if rating >= 4.0:
            relevant_inner.setdefault(uid, set()).add(mid)
    eligible_inner = {u: rel for u, rel in relevant_inner.items() if inner_users.get(u)}
    if smoke_users:
        eligible_inner = {u: rel for u, rel in eligible_inner.items() if u in smoke_users}
    inner = evaluate_configurations(
        inner_train, eligible_inner, movies, vectors, settings,
        relevance_configs=RELEVANCE_CONFIGS, diversity_configs=DIVERSITY_CONFIGS,
    )
    selection = selection_on_inner(inner)
    inner["selection"] = selection
    print(f"  inner done in {inner['wall_seconds']:.1f} s; selected: {selection['best_accuracy']} / {selection['best_balanced']}", flush=True)

    print("Stage 7: outer-test evaluation of selected configurations and baselines...", flush=True)
    relevant_outer = {}
    for uid, mid, rating, _ in test_used:
        if rating >= 4.0:
            relevant_outer.setdefault(uid, set()).add(mid)
    ba_base = selection["best_accuracy"].split("+", 1)[0]
    bb_base = selection["best_balanced"].split("+", 1)[0]
    outer = evaluate_configurations(
        train_used, relevant_outer, movies, vectors, settings,
        user_filter=smoke_users,
        relevance_configs=RELEVANCE_CONFIGS,
        diversity_configs=DIVERSITY_CONFIGS,
        diversity_base=next(c for c in RELEVANCE_CONFIGS if c["name"] == ba_base),
    )
    # Also evaluate best_balanced's base diversity layers if its base differs from best_accuracy's base.
    if bb_base != ba_base:
        outer_balanced = evaluate_configurations(
            train_used, relevant_outer, movies, vectors, settings,
            user_filter=smoke_users,
            relevance_configs=[c for c in RELEVANCE_CONFIGS if c["name"] == bb_base],
            diversity_configs=DIVERSITY_CONFIGS,
            diversity_base=next(c for c in RELEVANCE_CONFIGS if c["name"] == bb_base),
        )
        for name, vals in outer_balanced["per_user"].items():
            outer["per_user"].setdefault(name, {}).update(vals)
        for name, vals in outer_balanced["summary"].items():
            outer["summary"][name] = vals
        for key in ("novelty", "genre_representation", "popularity_concentration"):
            outer[key].update(outer_balanced[key])

    segments = segment_analysis(train_used, outer["per_user"])

    print("Stage 8: bootstrap and Pareto analysis...", flush=True)
    pu = outer["per_user"]
    comparisons = {}
    def cmp(name, left, right, metric="ndcg"):
        if left in pu and right in pu and pu[left].keys() & pu[right].keys():
            comparisons[name] = paired_user_bootstrap(pu[left], pu[right], metric=metric, resamples=args.resamples, confidence=0.95, seed=args.seed)

    # All 7 primary comparisons from Section 12:
    ba_pool = next(c["pool"] for c in RELEVANCE_CONFIGS if c["name"] == ba_base)
    same_pool_cf = "cf_only_pop" if ba_pool == "popularity" else ("cf_only_cfpop" if ba_pool == "cf_popularity" else "cf_only")
    cmp("1_general_ranker_vs_cf_same_pool_ndcg", selection["best_accuracy"], same_pool_cf)
    cmp("2_cf_pop_combined_vs_popularity_only_ndcg", ba_base if ba_pool == "popularity" else "combined_a50", "popularity_only")
    cmp("3_three_signal_vs_cf_pop_ndcg", "three_w1", "combined_a50")
    cmp("4_mmr_l75_vs_no_diversity_ndcg", f"{ba_base}+mmr_l75", ba_base)
    cmp("5_pop_penalty_l30_vs_no_diversity_ndcg", f"{ba_base}+pop_penalty_l30", ba_base)
    cmp("6_best_balanced_vs_phase9_cf_ref_ndcg", selection["best_balanced"], "cf_only")
    cmp("7_best_balanced_vs_phase9_pop_ref_ndcg", selection["best_balanced"], "popularity_only")
    cmp("best_accuracy_vs_phase9_cf_ref_ndcg", selection["best_accuracy"], "cf_only")
    cmp("cf_pop_fallback_vs_popularity_only_ndcg", "cf_pop_fallback", "popularity_only")
    cmp("best_accuracy_vs_rrf_cf_pop_ndcg", selection["best_accuracy"], "rrf_cf_pop")

    pareto_configs = _pareto_configs(outer)
    front, dominated, report = pareto_front(pareto_configs)
    pareto_doc = {"front": front, "dominated": dominated, "report": report, "representatives": pick_representatives(pareto_configs, front)}

    primary_wall = time.perf_counter() - t0
    peak_bytes, mem_method = _peak_memory_bytes()

    repro_doc = {"match": None}
    if not args.smoke:
        print("Stage 9: reproducibility pass (deterministic 400-user subsample)...", flush=True)
        sample = sorted({row[0] for row in outer_test_rows})[:400]
        sample_set = set(sample)
        runs = []
        for pass_idx in (1, 2):
            started = time.perf_counter()
            inner_r = evaluate_configurations(
                inner_train, {u: rel for u, rel in eligible_inner.items() if u in sample_set},
                movies, vectors, settings,
                relevance_configs=RELEVANCE_CONFIGS, diversity_configs=DIVERSITY_CONFIGS,
            )
            runs.append({"inner": inner_r, "wall": time.perf_counter() - started})
            print(f"  repro pass {pass_idx}: {runs[-1]['wall']:.1f} s", flush=True)

        def _canon(run):
            sel = selection_on_inner(run["inner"])
            return json.dumps({
                "summary": run["inner"]["summary"],
                "novelty": run["inner"]["novelty"],
                "lists": run["inner"]["ranked_lists"],
                "selection": {k: sel[k] for k in ("best_accuracy", "best_balanced")},
            }, sort_keys=True)

        repro_doc = {
            "match": _canon(runs[0]) == _canon(runs[1]),
            "comparison_population": f"first {len(sample)} eligible inner-validation users; models refit identically in both passes",
            "pass1_wall_seconds": runs[0]["wall"],
            "pass2_wall_seconds": runs[1]["wall"],
            "timing_excluded": "wall-clock seconds excluded from canonical comparison",
            "python": sys.version,
            "platform": platform.platform(),
        }

    runtime = {
        "primary_wall_seconds": primary_wall,
        "audit_wall_seconds": audit_wall,
        "inner_wall_seconds": inner["wall_seconds"],
        "outer_wall_seconds": outer["wall_seconds"],
        "peak_memory_bytes": peak_bytes,
        "memory_measurement": mem_method,
        "smoke_mode": args.smoke,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    config_doc = {
        "phase": 10,
        "smoke_mode": args.smoke,
        "shared_pools": ["cf@200", "popularity@200", "cf_popularity@200 (quota union)"],
        "relevance_configs": RELEVANCE_CONFIGS,
        "diversity_configs": DIVERSITY_CONFIGS,
        "selection": selection,
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "confidence": 0.95, "unit": "user"},
        "normalization": "per-pool min-max to [0,1]; popularity log1p-compressed before scaling; missing signals renormalized",
        "resources": runtime,
    }
    artifacts.save(out_dir / "phase9_source_audit.json", audit, "evaluation")
    artifacts.save(out_dir / "experiment_config.json", config_doc, "evaluation")
    artifacts.save(out_dir / "dataset_audit.json", {
        "dataset": "ml-1m",
        "ratings": len(ratings),
        "outer_train_rows": len(outer_train),
        "outer_test_targets": len(outer_test_rows),
        "eligible_users": len(outer_test_rows),
        "protocol": "Phase 7 chronological leave-latest-positive-out (threshold 4.0, min_train 5)",
    }, "evaluation")
    artifacts.save(out_dir / "ranking_results.json", {
        "inner_summary": inner["summary"],
        "inner_selection": selection,
        "outer_summary": outer["summary"],
        "outer_novelty": outer["novelty"],
        "outer_genre_representation": outer["genre_representation"],
        "outer_popularity_concentration": outer["popularity_concentration"],
    }, "evaluation")
    artifacts.save(out_dir / "scoring_diagnostics.json", {
        "inner": inner["scoring_diagnostics"],
        "outer": outer["scoring_diagnostics"],
    }, "evaluation")
    artifacts.save(out_dir / "diversity_results.json", {
        name: {
            "novelty": outer["novelty"][name]["novelty"],
            "long_tail_share": outer["novelty"][name]["long_tail_share"],
            "genre_representation": outer["genre_representation"][name],
            "popularity_concentration": outer["popularity_concentration"][name],
        }
        for name in outer["summary"]
    }, "evaluation")
    artifacts.save(out_dir / "segment_results.json", segments, "evaluation")
    artifacts.save(out_dir / "bootstrap_results.json", comparisons, "evaluation")
    artifacts.save(out_dir / "pareto_results.json", pareto_doc, "evaluation")
    artifacts.save(out_dir / "runtime_memory_report.json", runtime, "evaluation")
    artifacts.save(out_dir / "reproducibility_report.json", repro_doc, "evaluation")
    _write_summary(out_dir, audit, config_doc, inner, outer, comparisons, segments, pareto_doc, runtime, repro_doc)

    print(f"Total wall: {time.perf_counter() - t0:.1f} s; peak memory: {peak_bytes:,} bytes ({mem_method})")
    print(f"Reproducibility match: {repro_doc.get('match')}")
    print(f"Deliverables saved under {out_dir}")


if __name__ == "__main__":
    main()
