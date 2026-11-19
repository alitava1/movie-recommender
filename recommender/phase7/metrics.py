"""Phase 7 metrics, novelty, and paired user bootstrap (cluster-safe)."""

import math
import random
import statistics

from .. import evaluation
from ..diagnostics import _quantile

DIAGNOSTIC_METRICS = ("ndcg", "map", "hr", "precision", "recall", "ild", "novelty")


def novelty_at_k(recommended, popularity_counts, k):
    """Mean self-information (-log2 p) of recommended items; p from training counts only."""
    if k <= 0 or not popularity_counts:
        return 0.0
    total = sum(popularity_counts.values())
    if total <= 0:
        return 0.0
    values = []
    for movie_id in recommended[:k]:
        p = popularity_counts.get(movie_id, 0) / total
        if p <= 0:
            values.append(-math.log2(1.0 / (total + 1)))
        else:
            values.append(-math.log2(p))
    return sum(values) / len(values) if values else 0.0


def evaluate_user_phase7(recommended, relevant, k, vectors=None, popularity_counts=None):
    """All 7 Phase 7 metrics for one user (reuses shared evaluation math)."""
    metrics = evaluation.evaluate_user_extended(recommended, relevant, k, vectors)
    metrics["novelty"] = novelty_at_k(recommended, popularity_counts, k) if popularity_counts is not None else 0.0
    return metrics


def _cluster_id(obs_key):
    return obs_key.split(":", 1)[1] if ":" in obs_key else obs_key


def paired_user_bootstrap(
    left,
    right,
    metric="ndcg",
    resamples=2000,
    confidence=0.95,
    seed=2026,
):
    """Paired user-level bootstrap resampling whole users (all their observations together)."""
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be between 0 and 1")
    common = sorted(left.keys() & right.keys())
    if not common:
        raise ValueError("paired user bootstrap needs at least one matching user")

    clusters = {}
    for key in common:
        clusters.setdefault(_cluster_id(key), []).append(key)

    cluster_stats = []
    total_l = total_r = 0.0
    total_obs = 0
    for cid in sorted(clusters):
        keys = clusters[cid]
        s_l = sum(left[k][metric] for k in keys)
        s_r = sum(right[k][metric] for k in keys)
        cluster_stats.append((len(keys), s_l, s_r))
        total_l += s_l
        total_r += s_r
        total_obs += len(keys)

    l_mean = total_l / total_obs
    r_mean = total_r / total_obs
    abs_diff = l_mean - r_mean
    rel_diff = abs_diff / r_mean if r_mean != 0 else None

    rng = random.Random(seed)
    n = len(cluster_stats)
    abs_samples = []
    rel_samples = []
    for _ in range(resamples):
        b_cnt = b_l = b_r = 0
        for _ in range(n):
            cnt, sl, sr = rng.choice(cluster_stats)
            b_cnt += cnt
            b_l += sl
            b_r += sr
        lm = b_l / b_cnt
        rm = b_r / b_cnt
        d = lm - rm
        abs_samples.append(d)
        if rm != 0:
            rel_samples.append(d / rm)
    abs_samples.sort()
    rel_samples.sort()
    tail = (1.0 - confidence) / 2.0
    return {
        "metric": metric,
        "left_mean": l_mean,
        "right_mean": r_mean,
        "abs_diff": abs_diff,
        "abs_lower": _quantile(abs_samples, tail),
        "abs_upper": _quantile(abs_samples, 1.0 - tail),
        "rel_diff": rel_diff,
        "rel_lower": _quantile(rel_samples, tail) if len(rel_samples) == resamples else None,
        "rel_upper": _quantile(rel_samples, 1.0 - tail) if len(rel_samples) == resamples else None,
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "users": n,
        "observations": total_obs,
    }


def summarize_users(per_user, coverage_count, catalog_size):
    """Pooled per-user means plus catalog coverage; distinct from group means."""
    metrics = DIAGNOSTIC_METRICS
    n = len(per_user)
    if not n:
        return {**{m: 0.0 for m in metrics}, "coverage": 0.0, "users": 0}
    out = {m: sum(row[m] for row in per_user.values()) / n for m in metrics}
    out["coverage"] = coverage_count / catalog_size if catalog_size else 0.0
    out["users"] = n
    return out


def group_mean_std(per_user, coverage_by_group, catalog_size, groups):
    """Mean +/- sample SD of per-user metrics within each training-defined group."""
    result = {}
    for group_name, keys in groups.items():
        sub = {k: per_user[k] for k in keys if k in per_user}
        if not sub:
            result[group_name] = {m: None for m in DIAGNOSTIC_METRICS} | {"users": 0, "coverage": None}
            continue
        vals = {m: [row[m] for row in sub.values()] for m in DIAGNOSTIC_METRICS}
        result[group_name] = {
            m: {
                "mean": statistics.mean(vals[m]),
                "std": statistics.stdev(vals[m]) if len(vals[m]) > 1 else 0.0,
            }
            for m in DIAGNOSTIC_METRICS
        }
        result[group_name]["users"] = len(sub)
        cov = coverage_by_group.get(group_name) if coverage_by_group else None
        result[group_name]["coverage"] = len(cov) / catalog_size if cov else 0.0
    return result
