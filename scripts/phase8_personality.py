"""Phase 8 CLI: Synthetic Big Five personality modeling and controlled experiments."""

import argparse
import ctypes
import json
from pathlib import Path
import sys
import time
import tracemalloc

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender.data import load_movies
from recommender.personality.evaluation import render_scenario_svg, render_weight_sweep_svg
from recommender.personality.experiments import (
    estimate_resources,
    run_environment_a,
    run_environment_b,
)
from recommender.personality.questionnaire import build_schema
from recommender.personality.ranking import ARCHITECTURES, PERSONALITY_WEIGHT_GRID
from recommender.personality.simulator import SCENARIOS


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


def _write_report(
    out_dir,
    config_doc,
    env_a,
    env_b,
    repro_doc,
    wall_seconds,
    peak_bytes,
    mem_method,
):
    lines = [
        "# Phase 8: Synthetic Big Five Personality Modeling and Controlled Experiments",
        "",
        "> **Scientific boundary:** All Big Five questionnaire responses, user traits, and Environment B movie preferences in Phase 8 are **synthetic simulations**. These results test whether our modeling pipeline recovers controlled simulated effects and rejects null/shuffled controls; they are **NOT** evidence that real-world Big Five traits predict real human movie preferences. Real-participant validation remains future work.",
        "",
        "## 1. Research question & synthetic methodology",
        "",
        "- **Question:** Can self-reported Big Five questionnaire estimates provide incremental ranking value beyond the Phase 6 non-personality baseline (`cf_top100_content_enrichment`: CF + TF-IDF + genre + release-year enrichment)?",
        "- **Questionnaire engine:** 5 traits (`openness`, `conscientiousness`, `extraversion`, `agreeableness`, `neuroticism`) normalized to `[0.0, 1.0]` from simulated ordinal `1–5` Likert responses with positively and reverse-keyed placeholder items (`recommender/personality/questionnaire.py`). Recommenders receive ONLY noisy questionnaire-derived estimates (`ObservedProfile`), never latent ground truth.",
        "- **Preference model:** Pure-Python regularized Ridge regression (`RidgePersonalityModel`, 42 bilinear trait×attribute + main-effect features) trained strictly on training interactions and observed profiles.",
        "- **Architectures (`A`–`F`):** `cf_content_temporal` (unchanged Phase 6 baseline), `personality_only`, `cf_content_temporal_personality` (inner-validation weight sweep `w_p ∈ {0.0, 0.05, 0.10, 0.20, 0.30, 0.40}`), `cf_content_temporal_random_personality`, `cf_content_temporal_shuffled_personality`, and `cf_content_temporal_personality_disabled`.",
        "",
        "## 2. Environment A — MovieLens 100K null-control experiment",
        "",
        "Independent synthetic Big Five profiles assigned to real MovieLens 100K users (`u1`–`u5` or selected folds). Because profiles are independent of MovieLens ratings, personality should not yield consistent gains.",
        "",
        "| Architecture | NDCG@10 | MAP@10 | HR@10 | P@10 | R@10 | Coverage | ILD@10 | Diff vs Baseline (95% cluster CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    a_pooled = env_a["pooled"]
    a_comp = env_a["comparisons"]
    for arch in ARCHITECTURES:
        row = a_pooled[arch]
        if arch == "cf_content_temporal":
            ci = "—"
        else:
            c = a_comp[f"{arch}_vs_baseline_ndcg"]
            ci = f"{c['abs_diff']:+.5f} [{c['abs_lower']:+.5f}, {c['abs_upper']:+.5f}]"
        lines.append(
            f"| {arch} | {row['ndcg']:.4f} | {row['map']:.4f} | {row['hr']:.4f} | "
            f"{row['precision']:.4f} | {row['recall']:.4f} | {row['coverage']:.4f} | {row['ild']:.4f} | {ci} |"
        )

    all_disabled_eq_a = all(f["disabled_equals_baseline"] for f in env_a["per_fold"].values())
    lines.extend([
        "",
        f"- **Personality-disabled bit-exactness (Env A):** `{all_disabled_eq_a}`.",
        f"- **Inner-selected `personality_weight` per fold:** `{ {k: v['selected_personality_weight'] for k, v in env_a['per_fold'].items()} }`.",
        "",
        "## 3. Environment B — Controlled synthetic benchmark (Scenarios A–E)",
        "",
        "### Warm interaction-held-out evaluation (pooled across seeds)",
        "",
        "| Scenario | Mean `w_p` | Baseline NDCG | +Personality NDCG | Shuffled NDCG | Random NDCG | Personality-Only NDCG | +Pers vs Base (95% cluster CI) |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ])
    for sc, sdata in env_b["scenario_results"].items():
        wm = sdata["warm_interaction_held_out"]["models"]
        wc = sdata["warm_interaction_held_out"]["comparisons"]["cf_content_temporal_personality_vs_baseline_ndcg"]
        lines.append(
            f"| {sc} | {sdata['mean_selected_personality_weight']:.3f} | "
            f"{wm['cf_content_temporal']['ndcg']:.4f} | {wm['cf_content_temporal_personality']['ndcg']:.4f} | "
            f"{wm['cf_content_temporal_shuffled_personality']['ndcg']:.4f} | {wm['cf_content_temporal_random_personality']['ndcg']:.4f} | "
            f"{wm['personality_only']['ndcg']:.4f} | {wc['abs_diff']:+.5f} [{wc['abs_lower']:+.5f}, {wc['abs_upper']:+.5f}] |"
        )

    lines.extend([
        "",
        "### Cold user-held-out evaluation (disjoint users absent from training, pooled across seeds)",
        "",
        "| Scenario | Baseline (Pop.) NDCG | +Personality NDCG | Shuffled NDCG | Random NDCG | Personality-Only NDCG | +Pers vs Base (95% cluster CI) |",
        "|---|---:|---:|---:|---:|---:|---|",
    ])
    for sc, sdata in env_b["scenario_results"].items():
        cm = sdata["cold_user_held_out"]["models"]
        cc = sdata["cold_user_held_out"]["comparisons"]["cf_content_temporal_personality_vs_baseline_ndcg"]
        lines.append(
            f"| {sc} | {cm['cf_content_temporal']['ndcg']:.4f} | {cm['cf_content_temporal_personality']['ndcg']:.4f} | "
            f"{cm['cf_content_temporal_shuffled_personality']['ndcg']:.4f} | {cm['cf_content_temporal_random_personality']['ndcg']:.4f} | "
            f"{cm['personality_only']['ndcg']:.4f} | {cc['abs_diff']:+.5f} [{cc['abs_lower']:+.5f}, {cc['abs_upper']:+.5f}] |"
        )

    lines.extend([
        "",
        "## 4. Personality diagnostics & questionnaire ablation (Scenario C, synthetic)",
        "",
        f"- **Environment A trait recovery:** mean Pearson `r = {env_a['trait_recovery']['mean_pearson_r']:.4f}`, MAE `{env_a['trait_recovery']['overall_mae']:.4f}`, RMSE `{env_a['trait_recovery']['overall_rmse']:.4f}`.",
        f"- **Environment B Scenario C trait recovery:** mean Pearson `r = {env_b['scenario_results']['C_moderate']['mean_trait_pearson_r']:.4f}`, MAE `{env_b['scenario_results']['C_moderate']['mean_trait_mae']:.4f}`.",
        "",
        "| Noise `sigma` | Trait `r` | Trait MAE | Cold `personality_only` NDCG@10 |",
        "|---:|---:|---:|---:|",
    ])
    for row in env_b["questionnaire_ablation"]["noise_sensitivity"]:
        lines.append(f"| {row['response_noise']:.2f} | {row['trait_pearson_r']:.4f} | {row['trait_mae']:.4f} | {row['cold_personality_only_ndcg']:.4f} |")

    lines.extend([
        "",
        "| Missing rate | Completeness | Trait `r` | Trait MAE | Cold `personality_only` NDCG@10 |",
        "|---:|---:|---:|---:|---:|",
    ])
    for row in env_b["questionnaire_ablation"]["missingness_sensitivity"]:
        lines.append(f"| {row['missing_rate']:.2f} | {row['mean_completeness']:.4f} | {row['trait_pearson_r']:.4f} | {row['trait_mae']:.4f} | {row['cold_personality_only_ndcg']:.4f} |")

    lines.extend([
        "",
        "## 5. Charts (SYNTHETIC SIMULATION)",
        "",
        "- [Scenario NDCG comparison (SVG)](charts/scenario_ndcg_comparison.svg)",
        "- [Inner-validation weight sweep (SVG)](charts/weight_sweep.svg)",
        "",
        "## 6. Reproducibility, resources, limitations & next steps",
        "",
        f"- **Deterministic reproducibility:** Env A match=`{repro_doc['env_a_canonical_match']}`, Env B match=`{repro_doc['env_b_canonical_match']}`.",
        f"- **Runtime & peak memory:** Total wall time `{wall_seconds:.1f} s`; peak memory `{peak_bytes}` bytes (`{mem_method}`).",
        "- **Limitations:** Synthetic Big Five items, synthetic item attributes, and controlled preference equations do not validate psychological theory on real humans. Scenario E demonstrates that confounded traits can make `personality_only` appear predictive even when causal personality weight is lower or noisy, though shuffled/random controls remain essential safeguards.",
        "- **Next steps toward real-participant validation:** Replace `SYNTHETIC_ITEM_*` placeholders with a properly licensed validated Big Five instrument (e.g., BFI-2 / IPIP with permission), obtain IRB/ethics approval for consenting human participants, and evaluate prospectively against the preserved Phase 6 non-personality baseline.",
        "",
    ])
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 8 synthetic personality recommendation experiments.")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke configuration (1 fold, 1 seed, 100 users).")
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    parser.add_argument("--n-users", type=int, default=360)
    parser.add_argument("--interactions-per-user", type=int, default=50)
    parser.add_argument("--items-per-trait", type=int, default=4)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "artifacts" / "evaluation" / "phase8-personality",
    )
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")
    if len(set(args.folds)) != len(args.folds) or any(f not in range(1, 6) for f in args.folds):
        parser.error("--folds must contain unique values from 1-5")

    folds_list = [1] if args.smoke else args.folds
    seeds_list = [101] if args.smoke else args.seeds
    n_users = 100 if args.smoke else args.n_users
    inter_per_user = 35 if args.smoke else args.interactions_per_user
    resamples = 200 if args.smoke else 2000

    est = estimate_resources(folds_list, n_users, inter_per_user, len(SCENARIOS), len(seeds_list))
    print(
        f"Resource estimate ({'SMOKE' if args.smoke else 'FULL'}): "
        f"~{est['estimated_wall_seconds']}s wall, ~{est['estimated_peak_memory_mb']} MB peak memory "
        f"({est['env_a_folds']} Env-A folds, {est['env_b_runs']} Env-B runs).",
        flush=True,
    )
    if est["estimated_wall_seconds"] > 600:
        raise RuntimeError(f"Estimated runtime {est['estimated_wall_seconds']}s exceeds safe local limit (600s).")

    experiment_cfg = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    nested_settings = dict(experiment_cfg["nested"])
    nested_settings["bootstrap_resamples"] = resamples
    dataset_dir = ROOT / "data" / "raw" / "ml-100k"
    movies = load_movies("ml-100k", dataset_dir / "u.item")
    schema = build_schema(items_per_trait=args.items_per_trait)

    if sys.platform != "win32":
        tracemalloc.start()
    t0 = time.perf_counter()

    print("Running Environment A (MovieLens 100K null control)...", flush=True)
    env_a = run_environment_a(
        dataset_dir, movies, settings, nested_settings,
        folds_list=tuple(folds_list), schema=schema, resamples=resamples,
    )
    print(f"Environment A done in {env_a['wall_seconds']:.1f}s.", flush=True)

    print("Running Environment B (Synthetic benchmark Scenarios A-E)...", flush=True)
    env_b = run_environment_b(
        movies, seeds=tuple(seeds_list), n_users=n_users, interactions_per_user=inter_per_user,
        schema=schema, resamples=resamples, confidence=nested_settings["confidence"],
        checkpoint_dir=out_dir / "checkpoints",
    )
    print(f"Environment B done in {env_b['wall_seconds']:.1f}s.", flush=True)

    # Reproducibility check: repeat Env A (first fold) and Env B (Scenario C, first seed) independently.
    print("Verifying two-pass reproducibility...", flush=True)
    repro_a = run_environment_a(
        dataset_dir, movies, settings, nested_settings,
        folds_list=(folds_list[0],), schema=schema, resamples=resamples,
    )
    repro_b = run_environment_b(
        movies, seeds=(seeds_list[0],), scenarios=("C_moderate",),
        n_users=n_users, interactions_per_user=inter_per_user,
        schema=schema, resamples=resamples, confidence=nested_settings["confidence"],
    )
    env_a_canon_1 = json.dumps(env_a["per_fold"][f"u{folds_list[0]}"], sort_keys=True)
    env_a_canon_2 = json.dumps(repro_a["per_fold"][f"u{folds_list[0]}"], sort_keys=True)
    env_b_canon_1 = json.dumps(env_b["per_seed_results"][f"C_moderate:seed{seeds_list[0]}"], sort_keys=True)
    env_b_canon_2 = json.dumps(repro_b["per_seed_results"][f"C_moderate:seed{seeds_list[0]}"], sort_keys=True)
    repro_doc = {
        "env_a_checked_fold": f"u{folds_list[0]}",
        "env_a_canonical_match": env_a_canon_1 == env_a_canon_2,
        "env_b_checked_case": f"C_moderate:seed{seeds_list[0]}",
        "env_b_canonical_match": env_b_canon_1 == env_b_canon_2,
        "all_disabled_equal_baseline_env_a": all(f["disabled_equals_baseline"] for f in env_a["per_fold"].values()),
        "all_disabled_equal_baseline_env_b": all(r["warm_disabled_equals_baseline"] for r in env_b["per_seed_results"].values()),
    }

    wall_seconds = time.perf_counter() - t0
    peak_bytes, mem_method = _peak_memory_bytes()

    config_doc = {
        "phase": 8,
        "smoke_mode": args.smoke,
        "env_a_folds": folds_list,
        "env_b_seeds": seeds_list,
        "n_users": n_users,
        "interactions_per_user": inter_per_user,
        "items_per_trait": args.items_per_trait,
        "personality_weight_grid": list(PERSONALITY_WEIGHT_GRID),
        "scenarios": SCENARIOS,
        "architectures": list(ARCHITECTURES),
        "resources": {
            "estimated": est,
            "actual_wall_seconds": wall_seconds,
            "peak_memory_bytes": peak_bytes,
            "memory_measurement": mem_method,
        },
        "disclaimer": "All personality responses and Scenario B preferences are synthetic simulations.",
    }
    control_doc = {
        "environment_a_null_control": {
            "pooled": env_a["pooled"],
            "comparisons": env_a["comparisons"],
            "history_size_breakdown": env_a["history_size_breakdown"],
            "per_fold": env_a["per_fold"],
        },
        "environment_b_controls_by_scenario": {
            sc: {
                "warm_comparisons": sdata["warm_interaction_held_out"]["comparisons"],
                "cold_comparisons": sdata["cold_user_held_out"]["comparisons"],
            }
            for sc, sdata in env_b["scenario_results"].items()
        },
    }
    personality_metrics_doc = {
        "environment_a_trait_recovery": env_a["trait_recovery"],
        "environment_b_trait_recovery_by_scenario": {
            sc: {
                "mean_pearson_r": sdata["mean_trait_pearson_r"],
                "mean_mae": sdata["mean_trait_mae"],
                "mean_selected_personality_weight": sdata["mean_selected_personality_weight"],
            }
            for sc, sdata in env_b["scenario_results"].items()
        },
        "environment_a_history_size_breakdown": env_a["history_size_breakdown"],
        "environment_b_list_change_rates": {
            key: {
                "warm": val["warm_list_changes"],
                "cold": val["cold_list_changes"],
            }
            for key, val in env_b["per_seed_results"].items()
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    charts_dir = out_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    render_scenario_svg(env_b["scenario_results"], charts_dir / "scenario_ndcg_comparison.svg")
    c_mod_key = f"C_moderate:seed{seeds_list[0]}"
    render_weight_sweep_svg(env_b["per_seed_results"][c_mod_key]["weight_sweep"], charts_dir / "weight_sweep.svg")

    (out_dir / "configuration.json").write_text(json.dumps(config_doc, indent=2), encoding="utf-8")
    (out_dir / "questionnaire_schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")
    (out_dir / "synthetic_dataset_summary.json").write_text(json.dumps(env_b["dataset_summary"], indent=2), encoding="utf-8")
    (out_dir / "scenario_results.json").write_text(json.dumps(env_b["scenario_results"], indent=2), encoding="utf-8")
    (out_dir / "personality_metrics.json").write_text(json.dumps(personality_metrics_doc, indent=2), encoding="utf-8")
    (out_dir / "ablation_results.json").write_text(json.dumps(env_b["questionnaire_ablation"], indent=2), encoding="utf-8")
    (out_dir / "control_results.json").write_text(json.dumps(control_doc, indent=2), encoding="utf-8")
    (out_dir / "per_seed_results.json").write_text(json.dumps(env_b["per_seed_results"], indent=2), encoding="utf-8")
    (out_dir / "reproducibility_report.json").write_text(json.dumps(repro_doc, indent=2), encoding="utf-8")
    _write_report(out_dir, config_doc, env_a, env_b, repro_doc, wall_seconds, peak_bytes, mem_method)

    print(f"Total wall time: {wall_seconds:.1f}s; peak memory: {peak_bytes} bytes ({mem_method})")
    print(f"Reproducibility: env_a={repro_doc['env_a_canonical_match']}, env_b={repro_doc['env_b_canonical_match']}")
    print(f"Deliverables saved under {out_dir}")


if __name__ == "__main__":
    main()
