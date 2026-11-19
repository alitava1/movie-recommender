"""Phase 11 statistical analysis: user-clustered paired bootstrap + Holm-Bonferroni correction."""

import random

from .. import robustness


def paired_bootstrap_with_p(left_per_user, right_per_user, metric="ndcg", resamples=2000, confidence=0.95, seed=2026):
    """User-clustered paired bootstrap plus two-sided empirical p-value from the resample null."""
    comp = robustness.paired_cluster_bootstrap(
        left_per_user, right_per_user,
        metric=metric, resamples=resamples, confidence=confidence, seed=seed,
    )
    # Recompute bootstrap resample diffs to derive a two-sided percentile p-value.
    common = sorted(left_per_user.keys() & right_per_user.keys())
    clusters = {}
    for key in common:
        uid = key.split(":", 1)[1] if ":" in key else key
        clusters.setdefault(uid, []).append(left_per_user[key][metric] - right_per_user[key][metric])
    cluster_diffs = [sum(vals) / len(vals) for vals in clusters.values()]
    n = len(cluster_diffs)
    rng = random.Random(seed)
    le_zero = 0
    ge_zero = 0
    for _ in range(resamples):
        sample_mean = sum(cluster_diffs[rng.randrange(n)] for _ in range(n)) / n if n else 0.0
        if sample_mean <= 0.0:
            le_zero += 1
        if sample_mean >= 0.0:
            ge_zero += 1
    p_val = min(1.0, 2.0 * (min(le_zero, ge_zero) + 1) / (resamples + 1)) if n else 1.0
    return {**comp, "users": comp["clusters"], "p_value": p_val, "includes_zero": comp["abs_lower"] <= 0.0 <= comp["abs_upper"]}


def holm_bonferroni(comparisons):
    """Apply Holm-Bonferroni adjustment across rows labeled `family == 'primary'`.

    Returns updated list of dicts with `p_holm` and `significant_holm_05`.
    Exploratory rows receive `p_holm = None` and `significant_holm_05 = None`.
    """
    primary_idx = [i for i, row in enumerate(comparisons) if row.get("family") == "primary"]
    m = len(primary_idx)
    ordered = sorted(primary_idx, key=lambda i: comparisons[i]["p_value"])
    running_max = 0.0
    adjusted = {}
    for rank, idx in enumerate(ordered):
        p_adj = min(1.0, comparisons[idx]["p_value"] * (m - rank))
        running_max = max(running_max, p_adj)
        adjusted[idx] = running_max

    out = []
    for i, row in enumerate(comparisons):
        if i in adjusted:
            out.append({**row, "p_holm": adjusted[i], "significant_holm_05": adjusted[i] < 0.05})
        else:
            out.append({**row, "p_holm": None, "significant_holm_05": None})
    return out
