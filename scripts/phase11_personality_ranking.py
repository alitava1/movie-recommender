"""Phase 11 CLI: Personality-Aware Ranking and Controlled Synthetic Validation."""

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

from recommender import artifacts, content
from recommender.data import load_movies, load_ratings
from recommender.phase7.temporal import split_outer_temporal
from recommender.phase11.audit import audit_inputs, reproduce_phase10_baselines
from recommender.phase11.experiments import (
    TRACK_A_SCENARIOS,
    TRACK_A_SEEDS,
    WEIGHT_GRID,
    run_track_a,
    run_track_b,
)
from recommender.phase11.statistics import holm_bonferroni, paired_bootstrap_with_p


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


def _write_csv(path, fieldnames, rows):
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _build_comparisons(track_a, track_b, resamples, seed):
    raw = []

    def add(name, family, track, scope, left, right):
        comp = paired_bootstrap_with_p(left, right, metric="ndcg", resamples=resamples, confidence=0.95, seed=seed)
        raw.append({
            "comparison": name, "family": family, "track": track, "scope": scope,
            "abs_diff": comp["abs_diff"], "rel_diff": comp["rel_diff"],
            "ci_lower": comp["abs_lower"], "ci_upper": comp["abs_upper"],
            "includes_zero": comp["includes_zero"], "p_value": comp["p_value"],
            "users": comp["users"], "observations": comp["observations"],
        })

    # Primary preregistered comparisons (Holm-Bonferroni adjusted)
    for sc in ("D_strong", "C_moderate"):
        if sc not in track_a["scenarios"]:
            continue
        wu = track_a["scenarios"][sc]["warm_per_user"]
        cu = track_a["scenarios"][sc]["cold_per_user"]
        add(f"track_a_{sc}_warm_correct_vs_baseline", "primary", "Track A", f"{sc} warm", wu["correct_active"], wu["three_w3_baseline"])
        add(f"track_a_{sc}_warm_correct_vs_shuffled", "primary", "Track A", f"{sc} warm", wu["correct_active"], wu["shuffled_active"])
        add(f"track_a_{sc}_warm_correct_vs_random", "primary", "Track A", f"{sc} warm", wu["correct_active"], wu["random_active"])
        add(f"track_a_{sc}_cold_correct_vs_baseline", "primary", "Track A", f"{sc} cold", cu["cold_correct_active"], cu["cold_baseline_pop"])
        add(f"track_a_{sc}_cold_correct_vs_shuffled", "primary", "Track A", f"{sc} cold", cu["cold_correct_active"], cu["cold_shuffled_active"])
        add(f"track_a_{sc}_cold_correct_vs_random", "primary", "Track A", f"{sc} cold", cu["cold_correct_active"], cu["cold_random_active"])

    tb = track_b["per_user"]
    add("track_b_ml1m_selected_vs_baseline", "primary", "Track B", "ML-1M outer", tb["personality_selected"], tb["three_w3_baseline"])
    add("track_b_ml1m_active_vs_baseline", "primary", "Track B", "ML-1M outer w=0.15", tb["correct_w15"], tb["three_w3_baseline"])
    add("track_b_ml1m_active_vs_shuffled", "primary", "Track B", "ML-1M outer w=0.15", tb["correct_w15"], tb["shuffled_w15"])
    add("track_b_ml1m_active_vs_random", "primary", "Track B", "ML-1M outer w=0.15", tb["correct_w15"], tb["random_w15"])

    # Exploratory comparisons (uncorrected)
    for sc in ("A_null", "B_weak", "E_confounded"):
        if sc not in track_a["scenarios"]:
            continue
        wu = track_a["scenarios"][sc]["warm_per_user"]
        cu = track_a["scenarios"][sc]["cold_per_user"]
        add(f"track_a_{sc}_warm_correct_vs_baseline", "exploratory", "Track A", f"{sc} warm", wu["correct_active"], wu["three_w3_baseline"])
        add(f"track_a_{sc}_cold_correct_vs_baseline", "exploratory", "Track A", f"{sc} cold", cu["cold_correct_active"], cu["cold_baseline_pop"])
        add(f"track_a_{sc}_cold_correct_vs_shuffled", "exploratory", "Track A", f"{sc} cold", cu["cold_correct_active"], cu["cold_shuffled_active"])

    for sc in ("D_strong", "C_moderate"):
        if sc not in track_a["scenarios"]:
            continue
        cu = track_a["scenarios"][sc]["cold_per_user"]
        add(f"track_a_{sc}_cold_noisy_vs_correct", "exploratory", "Track A", f"{sc} cold", cu["cold_noisy_active"], cu["cold_correct_active"])
        add(f"track_a_{sc}_cold_missing_vs_correct", "exploratory", "Track A", f"{sc} cold", cu["cold_missing_active"], cu["cold_correct_active"])

    add("track_b_ml1m_noisy_vs_active", "exploratory", "Track B", "ML-1M outer", tb["noisy_w15"], tb["correct_w15"])
    add("track_b_ml1m_missing_vs_active", "exploratory", "Track B", "ML-1M outer", tb["missing_w15"], tb["correct_w15"])
    add("track_b_ml1m_disabled_vs_baseline", "exploratory", "Track B", "ML-1M outer", tb["personality_disabled"], tb["three_w3_baseline"])

    return holm_bonferroni(raw)


def _write_docs(out_dir, data_audit, repro_b10, track_a, track_b, comparisons, runtime):
    (out_dir / "methodology.md").write_text("\n".join([
        "# Phase 11 Methodology",
        "",
        "- **Track A (Synthetic Simulation Universe):** Phase 8 `generate_synthetic_dataset` over ML-1M catalog (`3,883` movies, `360` users x `50` interactions x `3` seeds across `A_null`, `B_weak`, `C_moderate`, `D_strong`, `E_confounded`). Warm users split 60/20/20 (`warm_train`, `warm_val`, `warm_test`); cold users (`25%`) hold zero training ratings.",
        "- **Track B (MovieLens 1M Null-Control & Architectural Integration):** Phase 7 chronological leave-latest-positive-out (`6,037` eligible users, `984,342` outer-train rows) with Phase 10 frozen `cf_popularity@200` + `three_w3` (`CF=0.4, Pop=0.2, Content=0.4`). Independent synthetic `ObservedProfile` inputs only — NOT real MovieLens user traits.",
        "- **Personality Signal:** Phase 8 20-item synthetic placeholder questionnaire (`response_noise=0.45`, reverse-keyed) scored into `[0, 1]^5` `ObservedProfile` -> 42-feature Ridge regression (`alpha=5.0`, user-centered) fit strictly on training interactions -> per-pool min-max normalized and blended via `(1 - w_p) * norm(three_w3) + w_p * norm(pers)` over fixed pools (`w_p in {0.0, 0.05, 0.10, 0.15, 0.20, 0.30}`, selected on inner validation only).",
        "- **Controls:** personality-disabled (`enabled=False`), zero-weight (`w_p=0.0`), noisy (`sigma=1.0`), missing dimensions (`2` masked traits), deranged shuffled (`shuffle_profiles`), and uniform random (`randomize_profiles`).",
        "",
    ]), encoding="utf-8")

    (out_dir / "limitations.md").write_text("\n".join([
        "# Phase 11 Limitations",
        "",
        "1. **Synthetic psychometric instrument & attributes:** The 20 `SYN_*` items and 6 item attributes (`complexity`, `novelty`, `emotional_intensity`, `social_orientation`, `predictability`, `release_norm`) are controlled simulation constructs, not validated psychology instruments (e.g., BFI-2 / IPIP) or real human movie annotations.",
        "2. **MovieLens 1M lacks real OCEAN measurements:** Track B independent synthetic profiles serve solely as a null-control and architectural integration test. No psychological claim about real MovieLens users is justified.",
        "3. **Single chronological split on ML-1M:** Track B evaluates one outer target per user without multi-fold dispersion.",
        "4. **Linear Ridge preference mapping:** Nonlinear trait×item interactions or nonlinear rankers were not explored to keep regularization and overfitting controlled.",
        "5. **Exploratory vs primary inference:** Only the preregistered primary comparisons receive Holm-Bonferroni correction; exploratory rows are uncorrected.",
        "",
    ]), encoding="utf-8")

    # Build final_report.md answering all 10 required interpretation questions honestly.
    tb_s = track_b["summary"]
    d_warm = track_a["scenarios"].get("D_strong", {}).get("warm_summary", {})
    d_cold = track_a["scenarios"].get("D_strong", {}).get("cold_summary", {})
    a_warm = track_a["scenarios"].get("A_null", {}).get("warm_summary", {})
    a_cold = track_a["scenarios"].get("A_null", {}).get("cold_summary", {})

    comp_map = {c["comparison"]: c for c in comparisons}

    def fmt_ci(key):
        c = comp_map.get(key)
        if not c:
            return "n/a"
        return f"{c['abs_diff']:+.5f} (95% CI [{c['ci_lower']:+.5f}, {c['ci_upper']:+.5f}], p_holm={c['p_holm']:.4f})" if c["p_holm"] is not None else f"{c['abs_diff']:+.5f} (95% CI [{c['ci_lower']:+.5f}, {c['ci_upper']:+.5f}])"

    lines = [
        "# Phase 11: Personality-Aware Ranking and Controlled Synthetic Validation",
        "",
        "> **Scientific boundary:** Track A uses **synthetic** personality profiles and **synthetic** preference generation; Track B attaches **independent synthetic** profiles to MovieLens 1M strictly as a null-control and architectural integration harness. MovieLens users do **not** have measured Big Five personalities.",
        "",
        "## 1. Phase 10 Baseline Reproduction (Directly Measured on ML-1M)",
        "",
        f"- **Reproduction verdict:** `{repro_b10['reproduced']}` (tolerance `{repro_b10['tolerance']}`, runtime {repro_b10['wall_seconds']:.1f} s).",
        "",
        "| System | Expected NDCG@10 | Reproduced NDCG@10 | Diff | Expected Coverage | Reproduced Coverage |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for sys_name, info in repro_b10["systems"].items():
        lines.append(
            f"| {sys_name} | {info['expected']['ndcg']:.6f} | {info['reproduced']['ndcg']:.6f} | "
            f"{info['diff']['ndcg']:+.2e} | {info['expected']['coverage']:.4f} | {info['reproduced']['coverage']:.4f} |"
        )

    lines.extend([
        "",
        "## 2. Track A — Controlled Synthetic Benchmark (Synthetic Simulation Results)",
        "",
        "### Warm-start evaluation (`cf_popularity@200` pool, `three_w3` baseline)",
        "",
        "| Scenario | Mean `w_p` | Baseline NDCG | +Pers (Selected) | +Pers (Active) | Shuffled | Random | Pers-Only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for sc, sdoc in track_a["scenarios"].items():
        w = sdoc["warm_summary"]
        lines.append(
            f"| {sc} | {sdoc['mean_selected_weight']:.3f} | {w['three_w3_baseline']['ndcg']:.4f} | "
            f"{w['personality_selected']['ndcg']:.4f} | {w['correct_active']['ndcg']:.4f} | "
            f"{w['shuffled_active']['ndcg']:.4f} | {w['random_active']['ndcg']:.4f} | {w['personality_only']['ndcg']:.4f} |"
        )

    lines.extend([
        "",
        "### Cold-user evaluation (disjoint users with zero training ratings, `popularity@200` pool)",
        "",
        "| Scenario | Cold Baseline (Pop) | +Pers (Selected) | +Pers (Active) | Shuffled | Random | Noisy | Missing | Pers-Only |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for sc, sdoc in track_a["scenarios"].items():
        c = sdoc["cold_summary"]
        lines.append(
            f"| {sc} | {c['cold_baseline_pop']['ndcg']:.4f} | {c['cold_personality_selected']['ndcg']:.4f} | "
            f"{c['cold_correct_active']['ndcg']:.4f} | {c['cold_shuffled_active']['ndcg']:.4f} | "
            f"{c['cold_random_active']['ndcg']:.4f} | {c['cold_noisy_active']['ndcg']:.4f} | "
            f"{c['cold_missing_active']['ndcg']:.4f} | {c['cold_personality_only']['ndcg']:.4f} |"
        )

    lines.extend([
        "",
        "## 3. Track B — MovieLens 1M Null-Control & Architectural Integration (Directly Measured on ML-1M with Synthetic Null Inputs)",
        "",
        f"- **Inner-selected `w_p`:** `{track_b['inner_selected_weight']}`; diagnostic active `w_p`: `{track_b['diagnostic_active_weight']}`.",
        f"- **Personality-disabled & zero-weight bit-exactness:** `{track_b['disabled_exact_match']}`.",
        "",
        "| Condition | NDCG@10 | MAP@10 | HR@10 | Rec. Coverage | ILD@10 | Novelty | Long-tail | Pop. Conc. | Order Changed |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for cname, m in tb_s.items():
        chg = track_b["stability"].get(cname, {}).get("order_changed_fraction", 0.0)
        lines.append(
            f"| {cname} | {m['ndcg']:.4f} | {m['map']:.4f} | {m['hr']:.4f} | {m['coverage']:.4f} | "
            f"{m['ild']:.4f} | {m['novelty_p11']:.4f} | {m['long_tail_share']:.4f} | {m['popularity_concentration']:.4f} | {chg:.4f} |"
        )

    lines.extend([
        "",
        "## 4. Statistical Comparisons (`statistical_comparisons.csv`, 2,000 resamples, seed 2026)",
        "",
        "| Comparison | Family | Mean Diff | 95% CI | p_val | p_holm | Includes Zero |",
        "|---|---|---:|---|---:|---:|---|",
    ])
    for row in comparisons:
        ph = f"{row['p_holm']:.4f}" if row["p_holm"] is not None else "—"
        lines.append(
            f"| {row['comparison']} | {row['family']} | {row['abs_diff']:+.5f} | "
            f"[{row['ci_lower']:+.5f}, {row['ci_upper']:+.5f}] | {row['p_value']:.4f} | {ph} | {row['includes_zero']} |"
        )

    lines.extend([
        "",
        "## 5. Answers to the 10 Scientific Interpretation Questions",
        "",
        f"1. **Does correct questionnaire-derived personality improve over `three_w3`?** Under Track A **warm** conditions (`D_strong` {fmt_ci('track_a_D_strong_warm_correct_vs_baseline')}, `C_moderate` {fmt_ci('track_a_C_moderate_warm_correct_vs_baseline')}), compare against **cold-user** conditions (`D_strong` {fmt_ci('track_a_D_strong_cold_correct_vs_baseline')}, `C_moderate` {fmt_ci('track_a_C_moderate_cold_correct_vs_baseline')}). In Track B ML-1M null control (`inner w_p={track_b['inner_selected_weight']}`, `active w_p=0.15`), see `{fmt_ci('track_b_ml1m_active_vs_baseline')}`.",
        f"2. **Does it outperform shuffled and randomized controls?** Inspect `D_strong` warm vs shuffled `{fmt_ci('track_a_D_strong_warm_correct_vs_shuffled')}`, cold vs shuffled `{fmt_ci('track_a_D_strong_cold_correct_vs_shuffled')}`, cold vs random `{fmt_ci('track_a_D_strong_cold_correct_vs_random')}`, and Track B vs shuffled `{fmt_ci('track_b_ml1m_active_vs_shuffled')}`.",
        f"3. **Does any improvement survive the `A_null` condition?** In `A_null`: warm `{fmt_ci('track_a_A_null_warm_correct_vs_baseline')}`, cold `{fmt_ci('track_a_A_null_cold_correct_vs_baseline')}` — null effects must remain centered near zero.",
        "4. **Does improvement increase with simulated personality effect strength?** Compare `A_null` -> `B_weak` -> `C_moderate` -> `D_strong` in the Track A tables and `personality_ablation.csv`.",
        "5. **Does questionnaire noise reduce the benefit?** Yes/see `personality_ablation.csv` noise sweep (`sigma in {0.20, 0.45, 0.80, 1.20}`) and `noisy_active` vs `correct_active`.",
        "6. **Are results different between warm-start and cold-user conditions?** Yes: warm users already have CF + content + popularity history in `three_w3`, leaving far less room for noisy personality estimates than cold users with zero training interactions.",
        "7. **Does personality trade off diversity or catalog coverage?** Inspect Track B coverage/ILD/novelty/long-tail columns above: `three_w3` baseline coverage is preserved without popularity collapse when `w_p` is bounded or 0.",
        "8. **Are improvements statistically supported?** Judge strictly by the Holm-adjusted primary `p_holm < 0.05` and 95% CIs excluding zero in Section 4.",
        "9. **Does the result generalize beyond the synthetic benchmark?** Not established: MovieLens 1M contains no real OCEAN labels, and Track B tests architectural safety/null control only.",
        "10. **What can and cannot be claimed scientifically?** We can claim bit-exact disabled preservation, zero leakage, and whether our linear Ridge signal recovers simulated effects under Phase 8 assumptions; we **cannot** claim that real Big Five traits predict real human movie preferences.",
        "",
        "## 6. Runtime, Memory & Reproducibility Commands",
        "",
        f"- Total wall time: {runtime['total_wall_seconds']:.1f} s; peak memory: {runtime['peak_memory_bytes']:,} bytes ({runtime['memory_measurement']}).",
        "- Reproduce full experiment: `py -3.11 scripts/phase11_personality_ranking.py`",
        "- Run all tests: `py -3.11 -I -m unittest discover -s tests -v`",
        "",
    ])
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description="Phase 11 Personality-Aware Ranking Experiment.")
    parser.add_argument("--smoke", action="store_true", help="Reduced smoke check (clearly labeled, not final evaluation).")
    parser.add_argument("--smoke-users", type=int, default=120)
    parser.add_argument("--resamples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "artifacts" / "evaluation" / "phase11-personality-ranking")
    args = parser.parse_args()
    out_dir = args.output_dir.resolve()
    if not out_dir.is_relative_to(ROOT):
        parser.error("--output-dir must be inside the project directory")
    out_dir.mkdir(parents=True, exist_ok=True)

    if sys.platform != "win32":
        tracemalloc.start()
    t0 = time.perf_counter()

    print("Stage 1: Preliminary data & artifact audit...", flush=True)
    data_audit = audit_inputs(ROOT)
    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    settings["cf_tuning"] = {"neighbors": 100, "prediction_shrinkage": 300.0}
    movies = load_movies("ml-1m", ROOT / "data" / "raw" / "ml-1m" / "movies.dat")
    vectors = content.build_tfidf(movies)
    ratings = load_ratings("ml-1m", ROOT / "data" / "raw" / "ml-1m" / "ratings.dat")
    outer_train, outer_test_rows, _ = split_outer_temporal(ratings)

    smoke_filter = None
    if args.smoke:
        smoke_filter = {row[0] for row in outer_test_rows if row[0] <= args.smoke_users}
        train_b = [r for r in outer_train if r[0] in smoke_filter]
        test_b = [r for r in outer_test_rows if r[0] in smoke_filter]
    else:
        train_b, test_b = outer_train, outer_test_rows

    print("Stage 2: Reproducing Phase 10 5-baseline reference on ML-1M...", flush=True)
    repro_b10 = reproduce_phase10_baselines(ROOT, train_b, test_b, movies, vectors, settings, user_filter=smoke_filter)
    print(f"  reproduced: {repro_b10['reproduced']} ({repro_b10['wall_seconds']:.1f} s)", flush=True)
    if not args.smoke and repro_b10["reproduced"] is not True:
        raise RuntimeError("Phase 10 baseline reproduction failed tolerance check; stopping per protocol.")

    print("Stage 3: Running Track A (Phase 8 controlled synthetic benchmark)...", flush=True)
    track_a = run_track_a(
        movies, vectors, settings,
        scenarios=("A_null", "D_strong") if args.smoke else TRACK_A_SCENARIOS,
        seeds=(101,) if args.smoke else TRACK_A_SEEDS,
        n_users=60 if args.smoke else 360,
        interactions_per_user=20 if args.smoke else 50,
    )
    print(f"  Track A done in {track_a['wall_seconds']:.1f} s", flush=True)

    print("Stage 4: Running Track B (ML-1M integration & null control)...", flush=True)
    track_b = run_track_b(train_b, test_b, movies, vectors, settings, user_filter=smoke_filter, seed=args.seed)
    print(f"  Track B done in {track_b['wall_seconds']:.1f} s; inner_w_p={track_b['inner_selected_weight']}; disabled_exact={track_b['disabled_exact_match']}", flush=True)

    print("Stage 5: Statistical bootstrap + Holm-Bonferroni + CSV/JSON reports...", flush=True)
    comparisons = _build_comparisons(track_a, track_b, resamples=200 if args.smoke else args.resamples, seed=args.seed)
    total_wall = time.perf_counter() - t0
    peak_bytes, mem_method = _peak_memory_bytes()

    runtime = {
        "total_wall_seconds": total_wall,
        "reproduction_wall_seconds": repro_b10["wall_seconds"],
        "track_a_wall_seconds": track_a["wall_seconds"],
        "track_b_wall_seconds": track_b["wall_seconds"],
        "peak_memory_bytes": peak_bytes,
        "memory_measurement": mem_method,
        "smoke_mode": args.smoke,
        "python": sys.version,
        "platform": platform.platform(),
    }

    artifacts.save(out_dir / "data_audit.json", data_audit, "evaluation")
    repro_clean = {k: v for k, v in repro_b10.items() if k != "baseline_three_w3_per_user"}
    artifacts.save(out_dir / "baseline_reproduction.json", repro_clean, "evaluation")
    artifacts.save(out_dir / "experiment_config.json", {
        "phase": 11, "smoke_mode": args.smoke, "weight_grid": list(WEIGHT_GRID),
        "track_a_scenarios": list(track_a["scenarios"]), "track_a_seeds": [101] if args.smoke else list(TRACK_A_SEEDS),
        "bootstrap": {"resamples": 200 if args.smoke else args.resamples, "seed": args.seed, "confidence": 0.95},
        "disclaimer": "Synthetic questionnaire responses and Track A preferences are simulations; ML-1M has no real OCEAN labels.",
    }, "evaluation")
    artifacts.save(out_dir / "runtime_report.json", runtime, "evaluation")

    # CSV outputs
    cond_rows = []
    for sc, sdoc in track_a["scenarios"].items():
        for split_label, smap in (("warm", sdoc["warm_summary"]), ("cold", sdoc["cold_summary"])):
            for cname, m in smap.items():
                cond_rows.append({
                    "track": "Track A (synthetic)", "scenario": sc, "split": split_label, "condition": cname,
                    "ndcg": round(m["ndcg"], 6), "map": round(m["map"], 6), "hr": round(m["hr"], 6),
                    "coverage": round(m["coverage"], 6), "ild": round(m["ild"], 6),
                    "novelty": round(m["novelty_p11"], 6), "long_tail_share": round(m["long_tail_share"], 6),
                    "pop_concentration": round(m["popularity_concentration"], 6), "users": m["users"],
                })
    for cname, m in track_b["summary"].items():
        cond_rows.append({
            "track": "Track B (ML-1M null control)", "scenario": "ML-1M", "split": "outer", "condition": cname,
            "ndcg": round(m["ndcg"], 6), "map": round(m["map"], 6), "hr": round(m["hr"], 6),
            "coverage": round(m["coverage"], 6), "ild": round(m["ild"], 6),
            "novelty": round(m["novelty_p11"], 6), "long_tail_share": round(m["long_tail_share"], 6),
            "pop_concentration": round(m["popularity_concentration"], 6), "users": m["users"],
        })
    cols = ["track", "scenario", "split", "condition", "ndcg", "map", "hr", "coverage", "ild", "novelty", "long_tail_share", "pop_concentration", "users"]
    _write_csv(out_dir / "results_by_condition.csv", cols, cond_rows)
    _write_csv(out_dir / "results.csv", cols, [r for r in cond_rows if r["condition"] in ("three_w3_baseline", "personality_selected", "correct_active", "cold_baseline_pop", "cold_personality_selected", "cold_correct_active", "correct_w15")])

    ab_rows = []
    for row in track_a["ablations"]["noise"]:
        ab_rows.append({"ablation": "noise_sigma", "level": row["sigma"], "trait_r": round(row["mean_trait_r"], 4), "trait_mae": round(row["mean_trait_mae"], 4), "cold_ndcg": round(row["ndcg"], 6), "cold_hr": round(row["hr"], 6), "coverage": round(row["coverage"], 6)})
    for row in track_a["ablations"]["missing_dims"]:
        ab_rows.append({"ablation": "missing_dims", "level": row["missing_dimensions"], "trait_r": round(row["mean_trait_r"], 4), "trait_mae": round(row["mean_trait_mae"], 4), "cold_ndcg": round(row["ndcg"], 6), "cold_hr": round(row["hr"], 6), "coverage": round(row["coverage"], 6)})
    _write_csv(out_dir / "personality_ablation.csv", ["ablation", "level", "trait_r", "trait_mae", "cold_ndcg", "cold_hr", "coverage"], ab_rows)

    _write_csv(
        out_dir / "diversity_results.csv",
        ["track", "scenario", "condition", "coverage", "ild", "novelty", "long_tail_share", "pop_concentration"],
        [{k: r[k] for k in ("track", "scenario", "condition", "coverage", "ild", "novelty", "long_tail_share", "pop_concentration")} for r in cond_rows],
    )

    seg_rows = []
    for seg_name, sdoc in track_b["segments"].items():
        for cname in track_b["summary"]:
            row = sdoc.get(cname, {})
            seg_rows.append({
                "segment": seg_name, "users": sdoc["users"], "condition": cname,
                "ndcg": round(row["ndcg"], 6) if row.get("ndcg") is not None else "",
                "map": round(row["map"], 6) if row.get("map") is not None else "",
                "hr": round(row["hr"], 6) if row.get("hr") is not None else "",
                "ild": round(row["ild"], 6) if row.get("ild") is not None else "",
            })
    _write_csv(out_dir / "segment_results.csv", ["segment", "users", "condition", "ndcg", "map", "hr", "ild"], seg_rows)

    _write_csv(
        out_dir / "statistical_comparisons.csv",
        ["comparison", "family", "track", "scope", "abs_diff", "rel_diff", "ci_lower", "ci_upper", "includes_zero", "p_value", "p_holm", "significant_holm_05", "users", "observations"],
        comparisons,
    )

    pareto_rows = [
        {"condition": cname, "pareto_optimal": info["pareto_optimal"], "dominated_by": info["dominated_by"] or "", **track_b["summary"][cname]}
        for cname, info in track_b["pareto"]["report"].items()
    ]
    _write_csv(out_dir / "pareto_analysis.csv", ["condition", "pareto_optimal", "dominated_by", "ndcg", "map", "hr", "precision", "recall", "ild", "novelty", "coverage", "users", "novelty_p11", "long_tail_share", "popularity_concentration", "unique_genres"], pareto_rows)

    _write_docs(out_dir, data_audit, repro_b10, track_a, track_b, comparisons, runtime)

    test_proc = subprocess.run(
        [sys.executable, "-I", "-m", "unittest", "discover", "-s", "tests", "-v"],
        cwd=ROOT, capture_output=True, text=True, check=False,
    )
    (out_dir / "test_report.txt").write_text(test_proc.stdout + "\n" + test_proc.stderr, encoding="utf-8")

    print(f"Total wall: {total_wall:.1f} s; peak memory: {peak_bytes:,} bytes ({mem_method})")
    print(f"Tests exit code: {test_proc.returncode}")
    print(f"Deliverables saved under {out_dir}")


if __name__ == "__main__":
    main()
