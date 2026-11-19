"""Phase 13 report writers (aggregate-only; no participant rows or hashed ids)."""


def write_data_audit_md(out_dir, audit):
    text = f"""# Phase 13 Data Audit (aggregate information only)

## Reconciliation of the 1,834 / 1,820 discrepancy

- `personality-data.csv` contains **{audit['personality']['personality_rows_total']} non-empty rows** but only **{audit['personality']['distinct_participants']} distinct participant ids**.
- Exactly **{audit['personality']['duplicate_ids']} ids appear twice** ({audit['personality']['duplicate_rows']} duplicate rows): 1,820 + 14 = 1,834. The shipped README's "1,834 users" counts **rows**, not participants.
- Duplicate rows carry **{audit['personality']['duplicate_trait_divergent']} divergent trait-value sets** across the duplicated ids; the preregistered rule keeps the **first occurrence**. No record was silently discarded: both counts are reported here.
- Experiment population: **{audit['experiment_population']} participants** (all distinct ids).

## Ratings integrity

- {audit['ratings']['rating_rows_total']:,} raw rows -> **{audit['ratings']['rows_after_latest_per_pair']:,}** after keeping the most recent rating per (user, movie) pair (**{audit['ratings']['duplicate_user_movie_pairs_removed']:,}** duplicate pairs removed; rule preregistered).
- {audit['ratings']['distinct_raters']} distinct raters; {audit['ratings']['distinct_movies']:,} distinct movies; rating values exactly {audit['ratings']['rating_values']}.
- Timestamps {audit['ratings']['timestamp_min']} to {audit['ratings']['timestamp_max']}.

## Participant linkage

- Personality users with ratings: {audit['linkage']['personality_users_with_ratings']}; raters without personality: {audit['linkage']['raters_without_personality']}; personality users without ratings: {audit['linkage']['personality_users_without_ratings']}.
- Verified same-participant linkage: **{audit['linkage']['verified_same_participants']}**.

## Trait schema and direction (raw 1-7 scale)

| Trait | Min | Max | Mean | Distinct values | Direction |
|---|---:|---:|---:|---:|---|
""" + "\n".join(
        f"| {t} | {v['min']} | {v['max']} | {v['mean']} | {v['distinct_values']} | "
        + ("reverse of emotional_stability: neuroticism = 8 - emotional_stability" if t == "neuroticism" else "identity")
        for t, v in audit["trait_ranges_raw_1_7"].items()
    ) + """

No missing trait values were found. Raw questionnaire item responses are not distributed (score-level data only).
"""
    (out_dir / "data_audit.md").write_text(text, encoding="utf-8")


def write_limitations(out_dir):
    text = """# Phase 13 Limitations

1. **Score-level personality only:** raw item responses are not distributed, so reliability cannot be recomputed and the noise conditions are score-level perturbations (documented sensitivity analyses), not measured questionnaire noise.
2. **Simulated sparse histories:** the 1-20 interaction strata (K1/K5/K10/K20) are produced by deterministic history truncation of established participants; only ~43 participants are naturally sparse. Simulated sparsity is not natural sparsity and results are labelled as such.
3. **Simulated zero-history:** Track E is a leave-one-user-out construction over established participants; the dataset contains no naturally zero-history users. This is NOT evidence about genuinely new users.
4. **No movie metadata:** GroupLens Personality 2018 ships no genres/years, so the Phase 10 content signal cannot be reproduced; the hybrid baseline is CF + popularity, and ILD uses training co-rating Jaccard instead of genre vectors. `three_w3` remains a historical reference that could not be replicated exactly.
5. **Single chronological target (warm stratum):** one held-out positive per participant; no multi-fold dispersion on the warm stratum.
6. **Exploratory multiplicity:** only the five preregistered comparisons receive Holm-Bonferroni correction; all other rows are uncorrected and exploratory.
7. **Popularity-pool ceiling:** fixed popularity@200 pools cap reachable accuracy; candidate-recall diagnostics quantify this rather than hiding it.
8. **No causality claim:** observed associations under a controlled protocol do not establish that personality traits cause preference differences.
9. **Trait direction:** neuroticism is derived as 8 - emotional_stability per the published schema; any miscoding would invert that dimension's effect (guarded by unit tests).
10. **Ethics determination:** project-owner authorization recorded; no institutional determination exists in the project materials.
"""
    (out_dir / "limitations.md").write_text(text, encoding="utf-8")


def write_final_report(out_dir, audit, baseline, warm, sparse, zero, comparisons, ablation_rows, seg_rows, total_wall, peak, mem_method):
    bs, ws = baseline["summary"], warm["summary"]
    prim = [c for c in comparisons if c["family"] == "primary"]

    def fmt(name):
        c = next((x for x in comparisons if x["comparison"] == name), None)
        if not c:
            return "n/a"
        ph = f", p_holm={c['p_holm']:.4f}" if c["p_holm"] is not None else ""
        return (f"{c['abs_diff']:+.5f} (95% CI [{c['ci_lower']:+.5f}, {c['ci_upper']:+.5f}], "
                f"p={c['p_value']:.4f}{ph}, n={c['users']})")

    lines = [
        "# Phase 13 Final Report: Real Personality-Aware Recommendation on GroupLens Personality 2018",
        "",
        "> **Scientific boundary:** all personality inputs are **real published questionnaire-derived scores** (IPIP-based, score-level). The sparse-history (K1/K5/K10/K20) and zero-history strata are **simulated** by deterministic history truncation and leave-one-user-out constructions over established participants; the dataset contains no naturally sparse or zero-history users. Aggregate-only reporting; no participant records appear in any artifact.",
        "",
        "## 1. Direct empirical observations",
        "",
        f"- Population: {audit['experiment_population']} linked participants; {audit['ratings']['rows_after_latest_per_pair']:,} ratings after the preregistered latest-per-pair rule ({audit['ratings']['duplicate_user_movie_pairs_removed']:,} duplicate pairs removed); catalog {audit['ratings']['distinct_movies']:,} movies.",
        f"- Warm stratum: {baseline['targets_count']} evaluated targets ({baseline['excluded']} participants excluded by the >= 5-history rule); catalog size in-stratum {baseline['catalog_size']:,}.",
        f"- Candidate recall of the fixed popularity@200 pool (warm): recall@100 = {baseline['candidate_recall']['recall_at_100']:.4f}, recall@200 = {baseline['candidate_recall']['recall_at_200']:.4f}; user retrieval coverage@200 = {baseline['candidate_recall']['user_retrieval_coverage_at_200']:.4f}.",
        "",
        "### Warm stratum (Track B/C)",
        "",
        "| System | NDCG@10 | MAP@10 | HR@10 | Recall@10 | Coverage | ILD@10 | Novelty |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name in ("popularity_only", "cf_only", "personality_disabled", "personality_selected", "correct_w15", "shuffled_w15", "random_w15", "noisy_w15", "missing_w15"):
        if name in ws:
            m = ws[name]
        elif name in bs:
            m = bs[name]
        else:
            continue
        lines.append(f"| {name} | {m['ndcg']:.4f} | {m['map']:.4f} | {m['hr']:.4f} | {m['recall']:.4f} | {m['coverage']:.4f} | {m['ild']:.4f} | {m['novelty']:.4f} |")
    lines.append("")
    lines.append(f"- Inner-selected personality weight (warm): `w_p = {warm['inner_selected_weight']}`.")

    lines.extend(["", "### Simulated sparse histories (Track D)", "",
                  "| Stratum | w_p | Baseline NDCG@10 | Correct (selected) | Correct w=0.15 | Shuffled | Random | Noisy | Targets |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for k, res in sparse.items():
        s = res["summary"]
        lines.append(
            f"| {k} | {res['inner_selected_weight']} | {s['baseline_disabled']['ndcg']:.4f} | "
            f"{s['correct_selected']['ndcg']:.4f} | {s['correct_w15']['ndcg']:.4f} | {s['shuffled_w15']['ndcg']:.4f} | "
            f"{s['random_w15']['ndcg']:.4f} | {s['noisy_w15']['ndcg']:.4f} | {res['targets']} |")

    lines.extend(["", "### Zero-history leave-one-user-out (Track E)", "",
                  "| Condition | NDCG@10 | MAP@10 | HR@10 | Recall@10 | Coverage | Novelty |",
                  "|---|---:|---:|---:|---:|---:|---:|"])
    for name in ("personality_only", "popularity_only", "personality_popularity", "random_profile"):
        m = zero["summary"][name]
        lines.append(f"| {name} | {m['ndcg']:.4f} | {m['map']:.4f} | {m['hr']:.4f} | {m['recall']:.4f} | {m['coverage']:.4f} | {m['novelty']:.4f} |")
    lines.append(f"\n- Personality+popularity blend weight (zero-history): `w_p = {zero['blend_weight']}` (selected on an inner 80% slice of training participants, identical across folds).")

    lines.extend([
        "",
        "## 2. Statistical conclusions (2,000-resample paired participant bootstrap, seed 2026, Holm-Bonferroni over the 5 preregistered comparisons)",
        "",
        "| Comparison | Mean diff | 95% CI | Includes zero |",
        "|---|---:|---|---|",
    ])
    for c in prim:
        lines.append(f"| {c['comparison']} | {c['abs_diff']:+.5f} | [{c['ci_lower']:+.5f}, {c['ci_upper']:+.5f}] | {c['includes_zero']} (p_holm={c['p_holm']:.4f}) |")

    lines.extend([
        "",
        "## 3. Answers to the required research questions",
        "",
        f"1. **Does real Big Five personality information improve movie recommendations?** P1 (warm, personality-selected vs disabled): {fmt('P1_personality_vs_baseline_warm')}.",
        f"2. **Does the contribution depend on behavioral history?** Compare P1 (warm) with P5 (K=5 sparse): {fmt('P5_personality_vs_sparse_baseline_K5')}; exploratory K1/K10/K20 rows in statistical_comparisons.csv.",
        f"3. **Does personality-only recommendation provide meaningful signal?** P4 (zero-history, personality-only vs popularity-only): {fmt('P4_personality_only_vs_popularity_zero')}.",
        f"4. **Does correct participant-profile matching beat shuffled/randomized controls?** P2: {fmt('P2_correct_vs_shuffled_warm')}; P3: {fmt('P3_correct_vs_random_warm')}.",
        "5. **Which personality dimensions contribute most?** See personality_ablation.csv (per-trait ablation at w=0.15).",
        "6. **Does personality improve accuracy without damaging diversity?** Compare coverage/ILD/novelty/long-tail between personality and disabled conditions in diversity_results.csv.",
        "7. **Are improvements statistically significant after multiple-comparison correction?** Judge strictly by p_holm < 0.05 with CIs excluding zero in Section 2.",
        "8. **Comparison with Phases 10/11:** Phase 10's `three_w3` (0.025093 NDCG@10 on ML-1M) is a different dataset and could not be replicated exactly (no content metadata here). Phase 11's synthetic cold-start gains are or are not replicated by the real-data analogues in P4/P5; the warm-data expectation from Phase 11 (no significant warm gain) is tested by P1/P2/P3.",
        "9. **Strongest methodological limitations:** see limitations.md (score-level personality only; simulated sparse/zero strata; no content metadata; single warm target; exploratory multiplicity).",
        "10. **Next research:** depends on the observed pattern of P1-P5; stated in Section 4.",
        "",
        "## 4. Outcome assignment (preregistered mapping)",
        "",
        _assign_outcome(prim),
        "",
        "## 5. Interpretation rules honoured",
        "",
        "- A confidence interval containing zero is NOT interpreted as evidence of equivalence.",
        "- Significance requires Holm-adjusted p < 0.05 with the CI excluding zero in the direction of the claim.",
        "- Personality systems are compared against the strongest non-personality baseline and the shuffled/random controls, not merely against popularity-only.",
        "- Simulated sparse and zero-history strata are explicitly distinguished from naturally occurring cold-start.",
        "- No causal or generalization claims are made.",
        "",
        "## 6. Reproducibility and resources",
        "",
        f"- Total wall time: {total_wall:.1f} s; peak memory: {peak:,} bytes ({mem_method}).",
        "- Reproduce: `py -3.11 scripts/phase13_real_personality.py --stage full` (or `--stage repro` for the deterministic two-pass check; `--smoke` for the reduced labeled run).",
        "- Tests: `py -3.11 -I -m unittest discover -s tests` (all Phase 1-13 tests must pass; see test_report.txt).",
        "",
    ])
    (out_dir / "final_report.md").write_text("\n".join(lines), encoding="utf-8")


def _assign_outcome(prim):
    def sig(name, positive=True):
        c = next((x for x in prim if x["comparison"] == name), None)
        if not c:
            return False
        return (not c["includes_zero"]) and c["significant_holm_05"] and (
            c["abs_diff"] > 0 if positive else c["abs_diff"] < 0)

    p1 = sig("P1_personality_vs_baseline_warm")
    p2 = sig("P2_correct_vs_shuffled_warm")
    p3 = sig("P3_correct_vs_random_warm")
    p4 = sig("P4_personality_only_vs_popularity_zero")
    p5 = sig("P5_personality_vs_sparse_baseline_K5")
    harms = sig("P1_personality_vs_baseline_warm", positive=False) or sig("P5_personality_vs_sparse_baseline_K5", positive=False)
    if p1 and p2 and p3:
        return "**Outcome A - strong empirical personality benefit** (P1-P3 all Holm-significant in favour of correct profiles)."
    if p5 and not p1:
        return "**Outcome B - sparse-history benefit only** (P5 significant; P1 not)."
    if p4:
        return "**Outcome C - personality-only benefit at zero history** (P4 significant), plus whatever warm/sparse outcomes hold."
    if harms:
        return "**Outcome E - personality integration harms recommendations** (a primary comparison is significantly negative); cause documented in the ablation tables."
    return "**Outcome D - no statistically established benefit** (no preregistered comparison is Holm-significant in favour of personality). This negative result is preserved honestly."
