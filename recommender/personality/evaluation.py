"""Phase 8 personality diagnostics, metrics aggregation, cluster bootstrap, and SVG charts."""

import math
import statistics

from .. import evaluation, robustness


def trait_recovery_diagnostics(latent_traits, observed_profiles):
    """Correlation, MAE, RMSE, and mean completeness between latent traits and questionnaire estimates."""
    traits = list(next(iter(latent_traits.values())).keys()) if latent_traits else []
    user_ids = sorted(latent_traits.keys() & observed_profiles.keys())
    per_trait = {}
    all_err = []
    for trait in traits:
        xs = [latent_traits[u][trait] for u in user_ids]
        ys = [observed_profiles[u].traits[trait] for u in user_ids]
        errs = [y - x for x, y in zip(xs, ys)]
        all_err.extend(errs)
        mx = sum(xs) / len(xs)
        my = sum(ys) / len(ys)
        num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
        den = math.sqrt(sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys))
        per_trait[trait] = {
            "pearson_r": num / den if den > 0 else 0.0,
            "mae": sum(abs(e) for e in errs) / len(errs),
            "rmse": math.sqrt(sum(e * e for e in errs) / len(errs)),
        }
    completeness = [observed_profiles[u].completeness for u in user_ids]
    return {
        "users": len(user_ids),
        "mean_completeness": statistics.mean(completeness) if completeness else 0.0,
        "overall_mae": sum(abs(e) for e in all_err) / len(all_err) if all_err else 0.0,
        "overall_rmse": math.sqrt(sum(e * e for e in all_err) / len(all_err)) if all_err else 0.0,
        "mean_pearson_r": statistics.mean(v["pearson_r"] for v in per_trait.values()) if per_trait else 0.0,
        "per_trait": per_trait,
    }


def list_change_rate(baseline_lists, other_lists):
    """Fraction of users whose top-k list changed, and mean Jaccard overlap."""
    keys = sorted(baseline_lists.keys() & other_lists.keys())
    if not keys:
        return {"users": 0, "order_changed_fraction": 0.0, "set_changed_fraction": 0.0, "mean_jaccard": 1.0}
    order_diff = 0
    set_diff = 0
    jaccards = []
    for k in keys:
        a = baseline_lists[k]
        b = other_lists[k]
        if a != b:
            order_diff += 1
        sa, sb = set(a), set(b)
        if sa != sb:
            set_diff += 1
        jaccards.append(len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0)
    n = len(keys)
    return {
        "users": n,
        "order_changed_fraction": order_diff / n,
        "set_changed_fraction": set_diff / n,
        "mean_jaccard": sum(jaccards) / n,
    }


def summarize_per_user(per_user, coverage_set, catalog_size):
    """Compute all 7 standard ranking + diversity metrics over per-user evaluations."""
    metrics = ("ndcg", "map", "hr", "precision", "recall", "ild")
    n = len(per_user)
    if not n:
        return {**{m: 0.0 for m in metrics}, "coverage": 0.0, "users": 0}
    out = {m: sum(row[m] for row in per_user.values()) / n for m in metrics}
    out["coverage"] = len(coverage_set) / catalog_size if catalog_size else 0.0
    out["users"] = n
    return out


def paired_user_bootstrap(left_per_user, right_per_user, metric="ndcg", resamples=2000, confidence=0.95, seed=2026):
    """Cluster-aware paired bootstrap (groups u{fold}:{uid} or {seed}:{uid} by user ID)."""
    return robustness.paired_cluster_bootstrap(
        left_per_user,
        right_per_user,
        metric=metric,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


def render_scenario_svg(scenario_summary, output_path):
    """Render a lightweight pure-Python SVG bar chart clearly labeled as SYNTHETIC SIMULATION."""
    scenarios = list(scenario_summary.keys())
    width, height = 760, 380
    pad_l, pad_r, pad_t, pad_b = 70, 30, 65, 80
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    models = [
        ("cf_content_temporal", "Baseline (No Pers.)", "#4a6fa5"),
        ("cf_content_temporal_personality", "Baseline + Pers.", "#2a9d8f"),
        ("cf_content_temporal_shuffled_personality", "Shuffled Control", "#e76f51"),
        ("personality_only", "Personality Only", "#9c6644"),
    ]
    max_val = max(
        scenario_summary[sc]["warm_interaction_held_out"]["models"][m]["ndcg"]
        for sc in scenarios for m, _, _ in models
    )
    y_max = max(0.20, math.ceil(max_val * 20.0) / 20.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="380" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold" fill="#111">SYNTHETIC SIMULATION: Scenario NDCG@10 Comparison (Environment B)</text>',
        '<text x="380" y="42" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Controlled synthetic preferences only - NOT real human personality data</text>',
    ]

    # Grid lines
    for i in range(5):
        val = y_max * i / 4.0
        y = pad_t + plot_h - (val / y_max) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e5e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{val:.2f}</text>')

    group_w = plot_w / max(1, len(scenarios))
    bar_w = group_w * 0.17
    for g_idx, sc in enumerate(scenarios):
        gx = pad_l + g_idx * group_w + group_w * 0.14
        for m_idx, (m_key, _, color) in enumerate(models):
            val = scenario_summary[sc]["warm_interaction_held_out"]["models"][m_key]["ndcg"]
            bh = (val / y_max) * plot_h
            bx = gx + m_idx * (bar_w + 3)
            by = pad_t + plot_h - bh
            parts.append(f'<rect x="{bx:.1f}" y="{by:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" fill="{color}"/>')
            parts.append(f'<text x="{bx + bar_w / 2:.1f}" y="{by - 4:.1f}" text-anchor="middle" font-family="sans-serif" font-size="9" fill="#222">{val:.3f}</text>')
        parts.append(f'<text x="{pad_l + (g_idx + 0.5) * group_w:.1f}" y="{height - pad_b + 20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#111">{sc}</text>')

    # Legend
    lx = pad_l
    ly = height - 24
    for m_idx, (_, label, color) in enumerate(models):
        x = lx + m_idx * 170
        parts.append(f'<rect x="{x}" y="{ly - 10}" width="12" height="12" fill="{color}"/>')
        parts.append(f'<text x="{x + 17}" y="{ly}" font-family="sans-serif" font-size="11" fill="#222">{label}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def render_weight_sweep_svg(sweep_rows, output_path):
    """Render pure-Python SVG chart for inner-validation personality weight sweep (SYNTHETIC)."""
    width, height = 700, 340
    pad_l, pad_r, pad_t, pad_b = 70, 30, 60, 65
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    weights = [row["weight"] for row in sweep_rows]
    all_vals = [row["ndcg"] for row in sweep_rows]
    y_min = max(0.0, math.floor((min(all_vals) - 0.01) * 50.0) / 50.0)
    y_max = math.ceil((max(all_vals) + 0.01) * 50.0) / 50.0
    span = max(1e-4, y_max - y_min)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="350" y="24" text-anchor="middle" font-family="sans-serif" font-size="15" font-weight="bold" fill="#111">SYNTHETIC SIMULATION: Inner-Validation Weight Sweep (Scenario C)</text>',
        '<text x="350" y="42" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#555">Synthetic controlled data only - NOT real human personality research</text>',
    ]
    for i in range(5):
        val = y_min + span * i / 4.0
        y = pad_t + plot_h - ((val - y_min) / span) * plot_h
        parts.append(f'<line x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        parts.append(f'<text x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="11" fill="#444">{val:.3f}</text>')

    pts = []
    for i, row in enumerate(sweep_rows):
        x = pad_l + (i / max(1, len(sweep_rows) - 1)) * plot_w
        y = pad_t + plot_h - ((row["ndcg"] - y_min) / span) * plot_h
        pts.append((x, y, row))
        parts.append(f'<text x="{x:.1f}" y="{height - pad_b + 20}" text-anchor="middle" font-family="sans-serif" font-size="11" fill="#111">w={row["weight"]:.2f}</text>')
    poly = " ".join(f"{x:.1f},{y:.1f}" for x, y, _ in pts)
    parts.append(f'<polyline fill="none" stroke="#2a9d8f" stroke-width="2.5" points="{poly}"/>')
    for x, y, row in pts:
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#2a9d8f"/>')
        parts.append(f'<text x="{x:.1f}" y="{y - 9:.1f}" text-anchor="middle" font-family="sans-serif" font-size="10" fill="#111">{row["ndcg"]:.4f}</text>')

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")
