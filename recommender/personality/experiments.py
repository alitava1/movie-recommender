"""Phase 8 experimental runners for Environment A (null control) and Environment B (synthetic benchmark)."""

import json
import math
from pathlib import Path
import statistics
import time

from .. import content, diagnostics, evaluation, hybrid, nested, popularity, robustness
from ..collaborative import ItemItemCF
from ..data import group_by_user, load_ratings
from .evaluation import (
    list_change_rate,
    paired_user_bootstrap,
    summarize_per_user,
    trait_recovery_diagnostics,
)
from .preference_model import RidgePersonalityModel
from .profile import generate_population_profiles, randomize_profiles, shuffle_profiles
from .questionnaire import build_schema
from .ranking import (
    ARCHITECTURES,
    PERSONALITY_WEIGHT_GRID,
    baseline_cf_content_temporal_scores,
    blend_with_personality,
    rank_architecture,
)
from .simulator import SCENARIOS, build_movie_attributes, generate_synthetic_dataset


def estimate_resources(env_a_folds, n_users, interactions_per_user, n_scenarios, n_seeds):
    """Conservative CPU runtime and memory estimate before launching full experiments."""
    env_a_s = len(env_a_folds) * 48.0
    env_b_runs = n_scenarios * n_seeds
    scale = (n_users / 360.0) * (interactions_per_user / 50.0)
    env_b_s = env_b_runs * max(0.3, 4.5 * scale) + 8.0
    repro_s = (48.0 if env_a_folds else 0.0) + max(0.3, 4.5 * scale)
    est_total_s = env_a_s + env_b_s + repro_s
    est_mem_mb = 420.0 if env_a_folds else 160.0
    return {
        "estimated_wall_seconds": round(est_total_s, 1),
        "estimated_peak_memory_mb": round(est_mem_mb, 1),
        "env_a_folds": len(env_a_folds),
        "env_b_runs": env_b_runs,
    }


def _select_baseline_and_personality_weight(
    inner_train,
    validation,
    movies,
    vectors,
    settings,
    nested_settings,
    observed_profiles,
    movie_features,
    k=10,
    threshold=4.0,
    top_m=100,
    ridge_alpha=5.0,
    weight_grid=PERSONALITY_WEIGHT_GRID,
):
    """Select Phase 6 baseline params and personality_weight strictly on inner validation."""
    cf_grid = nested_settings["cf_grid"]
    neighbors_grid = list(dict.fromkeys(int(v) for v in cf_grid["neighbors"]))
    shrinkage_grid = list(dict.fromkeys(float(v) for v in cf_grid["prediction_shrinkage"]))
    cf_first_weights = diagnostics.DEFAULT_CF_FIRST_WEIGHTS
    max_neighbors = max(neighbors_grid)

    cf_model = ItemItemCF.build(
        inner_train, neighbors=max_neighbors, shrinkage=settings["cf"]["shrinkage"], prediction_shrinkage=0.0
    )
    pers_model = RidgePersonalityModel(alpha=ridge_alpha).fit(inner_train, observed_profiles, movie_features)

    inner_users = group_by_user(inner_train)
    relevant_by_user = {}
    for uid, mid, rating, _ in validation:
        if rating >= threshold:
            relevant_by_user.setdefault(uid, set()).add(mid)
    eligible = {u: rel for u, rel in relevant_by_user.items() if inner_users.get(u)}

    if not eligible:
        default_base = {
            "neighbors": settings["cf"]["neighbors"],
            "prediction_shrinkage": settings["cf"].get("prediction_shrinkage", 100.0),
            "cf_weight": cf_first_weights[0],
        }
        return default_base, 0.0, [{"weight": w, "ndcg": None} for w in weight_grid]

    cf_views = {n: nested._neighbor_view(cf_model, n) for n in neighbors_grid}
    cf_configs = [(n, s) for n in neighbors_grid for s in shrinkage_grid]
    cf_first_configs = [(n, s, w_cf) for n, s in cf_configs for w_cf in cf_first_weights]
    base_scores_sum = [0.0] * len(cf_first_configs)

    sorted_users = sorted(eligible)
    user_base_cache = {}
    for uid in sorted_users:
        ur = inner_users[uid]
        cands = movies.keys() - ur.keys()
        profile = content.user_profile(vectors, ur)
        tfidf_cache = {}
        pool_cache = {}
        user_base_cache[uid] = {}

        for n_idx, n in enumerate(neighbors_grid):
            ev = cf_views[n].evidence(ur, cands)
            for s_idx, s in enumerate(shrinkage_grid):
                cfg_idx = n_idx * len(shrinkage_grid) + s_idx
                preds = nested._cf_predictions(ev, s)
                top_ids = diagnostics.cf_top_m_candidates(preds, top_m)
                cf_n = hybrid.normalize({mid: preds[mid] for mid in top_ids})
                pkey = tuple(top_ids)
                if pkey not in pool_cache:
                    missing = [mid for mid in top_ids if mid not in tfidf_cache]
                    if missing:
                        if profile:
                            tfidf_cache.update(content.score(profile, vectors, missing))
                        else:
                            for mid in missing:
                                tfidf_cache[mid] = 0.0
                    comp_maps = robustness._component_maps(
                        ur, movies, vectors, top_ids, threshold=threshold,
                        profile=profile, tfidf_raw=tfidf_cache,
                    )
                    pool_cache[pkey] = hybrid.normalize(
                        robustness.combine_enrichment_components(comp_maps, top_ids, ("tfidf", "genre", "year"))
                    )
                sec_n = pool_cache[pkey]
                for w_idx, w_cf in enumerate(cf_first_weights):
                    idx = cfg_idx * len(cf_first_weights) + w_idx
                    w_sec = 1.0 - w_cf
                    reranked = {mid: w_cf * cf_n[mid] + w_sec * sec_n[mid] for mid in top_ids}
                    user_base_cache[uid][(n, s, w_cf)] = reranked
                    base_scores_sum[idx] += evaluation.ndcg_at_k(nested._top_ids(reranked, k), eligible[uid], k)

    best_idx = max(range(len(cf_first_configs)), key=lambda i: base_scores_sum[i])
    bn, bs, bw = cf_first_configs[best_idx]
    best_base = {"neighbors": bn, "prediction_shrinkage": bs, "cf_weight": bw}

    # Sweep personality_weight strictly on inner validation using the fixed best baseline scores.
    sweep_rows = []
    for w_p in weight_grid:
        ndcg_sum = 0.0
        for uid in sorted_users:
            base_map = user_base_cache[uid][(bn, bs, bw)]
            prof = observed_profiles.get(uid)
            if w_p == 0.0 or prof is None:
                recs = nested._top_ids(base_map, k)
            else:
                pers_map = pers_model.score_candidates(prof, movie_features, base_map.keys())
                blended = blend_with_personality(base_map, pers_map, personality_weight=w_p, enabled=True)
                recs = nested._top_ids(blended, k)
            ndcg_sum += evaluation.ndcg_at_k(recs, eligible[uid], k)
        sweep_rows.append({"weight": float(w_p), "ndcg": ndcg_sum / len(sorted_users)})

    best_w_p = max(sweep_rows, key=lambda r: r["ndcg"])["weight"]
    return best_base, best_w_p, sweep_rows


def run_environment_a(
    dataset_dir,
    movies,
    settings,
    nested_settings,
    folds_list=(1, 2, 3, 4, 5),
    schema=None,
    profile_seed=20260,
    ridge_alpha=5.0,
    k=10,
    threshold=4.0,
    top_m=100,
    resamples=2000,
    seed=2026,
):
    """Environment A: MovieLens 100K null-control experiment with independent synthetic profiles."""
    schema = schema or build_schema()
    vectors = content.build_tfidf(movies)
    movie_features = build_movie_attributes(movies, seed=profile_seed + 7)

    # Collect all MovieLens user IDs across folds and generate questionnaire profiles independently of ratings.
    all_user_ids = set()
    fold_data = {}
    for fold in folds_list:
        tr = load_ratings("ml-100k", dataset_dir / f"u{fold}.base")
        te = load_ratings("ml-100k", dataset_dir / f"u{fold}.test")
        fold_data[fold] = (tr, te)
        all_user_ids.update(r[0] for r in tr)
        all_user_ids.update(r[0] for r in te)

    latent_traits, _, observed_profiles = generate_population_profiles(
        sorted(all_user_ids), schema, seed=profile_seed, response_noise=0.45, missing_rate=0.0
    )
    shuffled_profiles = shuffle_profiles(observed_profiles, seed=profile_seed + 101)
    random_profiles = randomize_profiles(sorted(all_user_ids), seed=profile_seed + 202)
    trait_diag = trait_recovery_diagnostics(latent_traits, observed_profiles)

    per_user = {arch: {} for arch in ARCHITECTURES}
    coverage = {arch: set() for arch in ARCHITECTURES}
    user_size_seg = {}
    fold_results = {}
    t_start = time.perf_counter()

    for fold in folds_list:
        outer_train, outer_test = fold_data[fold]
        inner_train, validation = nested.split_inner(outer_train, nested_settings["inner_validation_fraction"])
        best_base, best_w_p, sweep = _select_baseline_and_personality_weight(
            inner_train, validation, movies, vectors, settings, nested_settings,
            observed_profiles, movie_features, k=nested_settings.get("selection_k", k),
            threshold=threshold, top_m=top_m, ridge_alpha=ridge_alpha,
        )

        cf_model = ItemItemCF.build(
            outer_train, neighbors=best_base["neighbors"], shrinkage=settings["cf"]["shrinkage"], prediction_shrinkage=0.0
        )
        pers_model = RidgePersonalityModel(alpha=ridge_alpha).fit(outer_train, observed_profiles, movie_features)
        train_users = group_by_user(outer_train)

        relevant_by_user = {}
        for uid, mid, rating, _ in outer_test:
            if rating >= threshold:
                relevant_by_user.setdefault(uid, set()).add(mid)

        fold_per_user = {arch: {} for arch in ARCHITECTURES}
        fold_cov = {arch: set() for arch in ARCHITECTURES}
        fold_lists = {arch: {} for arch in ARCHITECTURES}
        fold_sizes = {}

        for uid in sorted(relevant_by_user):
            ur = train_users.get(uid, {})
            if not ur:
                continue
            rel = relevant_by_user[uid]
            all_cands = movies.keys() - ur.keys()
            base_scores = baseline_cf_content_temporal_scores(
                ur, movies, vectors, cf_model, best_base["prediction_shrinkage"], best_base["cf_weight"], top_m, threshold
            )
            prof_map = {
                "cf_content_temporal": None,
                "personality_only": observed_profiles.get(uid),
                "cf_content_temporal_personality": observed_profiles.get(uid),
                "cf_content_temporal_random_personality": random_profiles.get(uid),
                "cf_content_temporal_shuffled_personality": shuffled_profiles.get(uid),
                "cf_content_temporal_personality_disabled": observed_profiles.get(uid),
            }
            uid_key = f"u{fold}:{uid}"
            n_hist = len(ur)
            fold_sizes[uid_key] = "sparse" if n_hist <= 30 else "medium" if n_hist <= 80 else "dense"
            for arch in ARCHITECTURES:
                recs = rank_architecture(
                    arch, base_scores, prof_map[arch], movie_features, pers_model,
                    personality_weight=best_w_p, all_candidates=all_cands, k=k,
                )
                metrics = evaluation.evaluate_user_extended(recs, rel, k, vectors)
                fold_per_user[arch][uid_key] = metrics
                per_user[arch][uid_key] = metrics
                fold_cov[arch].update(recs)
                coverage[arch].update(recs)
                fold_lists[arch][uid_key] = recs

        fold_metrics = {
            arch: summarize_per_user(fold_per_user[arch], fold_cov[arch], len(movies))
            for arch in ARCHITECTURES
        }
        user_size_seg.update(fold_sizes)
        fold_results[f"u{fold}"] = {
            "selected_baseline": best_base,
            "selected_personality_weight": best_w_p,
            "weight_sweep": sweep,
            "disabled_equals_baseline": fold_lists["cf_content_temporal_personality_disabled"] == fold_lists["cf_content_temporal"],
            "list_changes_vs_baseline": list_change_rate(fold_lists["cf_content_temporal"], fold_lists["cf_content_temporal_personality"]),
            "metrics": fold_metrics,
        }

    pooled = {arch: summarize_per_user(per_user[arch], coverage[arch], len(movies)) for arch in ARCHITECTURES}
    comparisons = {}
    for arch in ARCHITECTURES:
        if arch == "cf_content_temporal":
            continue
        comparisons[f"{arch}_vs_baseline_ndcg"] = paired_user_bootstrap(
            per_user[arch], per_user["cf_content_temporal"], metric="ndcg",
            resamples=resamples, confidence=nested_settings["confidence"], seed=seed,
        )

    history_breakdown = {}
    for seg in ("sparse", "medium", "dense"):
        keys = [k2 for k2, v2 in user_size_seg.items() if v2 == seg]
        if keys:
            history_breakdown[seg] = {
                "user_folds": len(keys),
                "baseline_ndcg": sum(per_user["cf_content_temporal"][k2]["ndcg"] for k2 in keys) / len(keys),
                "personality_ndcg": sum(per_user["cf_content_temporal_personality"][k2]["ndcg"] for k2 in keys) / len(keys),
            }

    return {
        "environment": "A_movielens_null_control",
        "warning": "Synthetic independent profiles assigned to real MovieLens 100K users as a negative control.",
        "wall_seconds": time.perf_counter() - t_start,
        "trait_recovery": trait_diag,
        "pooled": pooled,
        "comparisons": comparisons,
        "history_size_breakdown": history_breakdown,
        "per_fold": fold_results,
    }


def _evaluate_synthetic_seed(
    movies,
    vectors,
    scenario_name,
    seed,
    n_users,
    interactions_per_user,
    schema,
    ridge_alpha=5.0,
    k=10,
    threshold=4.0,
    top_m=100,
    weight_grid=PERSONALITY_WEIGHT_GRID,
):
    """Run one synthetic scenario seed for both interaction-held-out and user-held-out splits."""
    ds = generate_synthetic_dataset(
        movies, scenario_name, seed,
        n_users=n_users, interactions_per_user=interactions_per_user, schema=schema,
    )
    trait_diag = trait_recovery_diagnostics(ds._latent_traits, ds.observed_profiles)
    shuffled_profiles = shuffle_profiles(ds.observed_profiles, seed=seed + 501)
    random_profiles = randomize_profiles(sorted(ds.observed_profiles), seed=seed + 601)

    # 1. Inner-validation selection on warm_train -> warm_val (interaction-held-out)
    cf_inner = ItemItemCF.build(ds.warm_train, neighbors=50, shrinkage=10.0, prediction_shrinkage=0.0)
    pers_inner = RidgePersonalityModel(alpha=ridge_alpha).fit(ds.warm_train, ds.observed_profiles, ds.movie_features)
    train_users = group_by_user(ds.warm_train)
    val_rel = {}
    for uid, mid, r, _ in ds.warm_val:
        if r >= threshold:
            val_rel.setdefault(uid, set()).add(mid)
    eligible_val = {u: rel for u, rel in val_rel.items() if train_users.get(u)}

    # Select cf_weight in {0.7, 0.8, 0.9} & prediction_shrinkage in {10, 100} on warm_val.
    cf_cfgs = [(10.0, 0.7), (100.0, 0.7), (100.0, 0.8), (100.0, 0.9)]
    val_base_cache = {}
    cfg_sums = [0.0] * len(cf_cfgs)
    for uid in sorted(eligible_val):
        ur = train_users[uid]
        cands = movies.keys() - ur.keys()
        ev = cf_inner.evidence(ur, cands)
        val_base_cache[uid] = {}
        for i, (ps, w_cf) in enumerate(cf_cfgs):
            preds = nested._cf_predictions(ev, ps)
            top_ids = diagnostics.cf_top_m_candidates(preds, top_m)
            enriched = diagnostics.enriched_content_scores(ur, movies, vectors, top_ids, threshold)
            bscores = diagnostics.rerank_pool_scores(top_ids, preds, enriched, w_cf)
            val_base_cache[uid][(ps, w_cf)] = bscores
            cfg_sums[i] += evaluation.ndcg_at_k(nested._top_ids(bscores, k), eligible_val[uid], k)
    best_ps, best_cf_w = cf_cfgs[max(range(len(cf_cfgs)), key=lambda i: cfg_sums[i])]

    # Sweep personality_weight strictly on warm_val
    weight_sweep = []
    for w_p in weight_grid:
        ndcg_s = 0.0
        for uid in sorted(eligible_val):
            bscores = val_base_cache[uid][(best_ps, best_cf_w)]
            pers_map = pers_inner.score_candidates(ds.observed_profiles[uid], ds.movie_features, bscores.keys())
            blended = blend_with_personality(bscores, pers_map, personality_weight=w_p, enabled=True)
            ndcg_s += evaluation.ndcg_at_k(nested._top_ids(blended, k), eligible_val[uid], k)
        weight_sweep.append({"weight": float(w_p), "ndcg": ndcg_s / len(eligible_val) if eligible_val else 0.0})
    best_w_p = max(weight_sweep, key=lambda r: r["ndcg"])["weight"]

    # 2. Refit on warm_train + warm_val (zero test users, zero test interactions)
    fit_ratings = ds.warm_train + ds.warm_val
    cf_full = ItemItemCF.build(fit_ratings, neighbors=50, shrinkage=10.0, prediction_shrinkage=0.0)
    pop_full = popularity.popularity_scores(fit_ratings, prior_count=10.0)
    pers_full = RidgePersonalityModel(alpha=ridge_alpha).fit(fit_ratings, ds.observed_profiles, ds.movie_features)
    fit_users = group_by_user(fit_ratings)

    # 3A. Interaction-held-out evaluation (warm_test)
    warm_rel = {}
    for uid, mid, r, _ in ds.warm_test:
        if r >= threshold:
            warm_rel.setdefault(uid, set()).add(mid)
    warm_per_user = {arch: {} for arch in ARCHITECTURES}
    warm_cov = {arch: set() for arch in ARCHITECTURES}
    warm_lists = {arch: {} for arch in ARCHITECTURES}

    for uid in sorted(warm_rel):
        ur = fit_users.get(uid, {})
        if not ur:
            continue
        rel = warm_rel[uid]
        all_cands = movies.keys() - ur.keys()
        base_scores = baseline_cf_content_temporal_scores(
            ur, movies, vectors, cf_full, best_ps, best_cf_w, top_m, threshold
        )
        prof_map = {
            "cf_content_temporal": None,
            "personality_only": ds.observed_profiles[uid],
            "cf_content_temporal_personality": ds.observed_profiles[uid],
            "cf_content_temporal_random_personality": random_profiles[uid],
            "cf_content_temporal_shuffled_personality": shuffled_profiles[uid],
            "cf_content_temporal_personality_disabled": ds.observed_profiles[uid],
        }
        obs_key = f"s{seed}:{uid}"
        for arch in ARCHITECTURES:
            recs = rank_architecture(
                arch, base_scores, prof_map[arch], ds.movie_features, pers_full,
                personality_weight=best_w_p, all_candidates=all_cands, k=k,
            )
            warm_per_user[arch][obs_key] = evaluation.evaluate_user_extended(recs, rel, k, vectors)
            warm_cov[arch].update(recs)
            warm_lists[arch][obs_key] = recs

    # 3B. User-held-out cold-start evaluation (cold_test_users never present in fit_ratings)
    cold_rel = {}
    for uid, mid, r, _ in ds.cold_test_ratings:
        if r >= threshold:
            cold_rel.setdefault(uid, set()).add(mid)
    cold_per_user = {arch: {} for arch in ARCHITECTURES}
    cold_cov = {arch: set() for arch in ARCHITECTURES}
    cold_lists = {arch: {} for arch in ARCHITECTURES}

    for uid in sorted(cold_rel):
        rel = cold_rel[uid]
        all_cands = movies.keys()
        # Cold users have zero interactions in training: baseline falls back to popularity.
        base_scores = baseline_cf_content_temporal_scores(
            {}, movies, vectors, cf_full, best_ps, best_cf_w, top_m, threshold, popularity_scores=pop_full
        )
        prof_map = {
            "cf_content_temporal": None,
            "personality_only": ds.observed_profiles[uid],
            "cf_content_temporal_personality": ds.observed_profiles[uid],
            "cf_content_temporal_random_personality": random_profiles[uid],
            "cf_content_temporal_shuffled_personality": shuffled_profiles[uid],
            "cf_content_temporal_personality_disabled": ds.observed_profiles[uid],
        }
        obs_key = f"s{seed}:{uid}"
        for arch in ARCHITECTURES:
            recs = rank_architecture(
                arch, base_scores, prof_map[arch], ds.movie_features, pers_full,
                personality_weight=best_w_p, all_candidates=all_cands, k=k,
            )
            cold_per_user[arch][obs_key] = evaluation.evaluate_user_extended(recs, rel, k, vectors)
            cold_cov[arch].update(recs)
            cold_lists[arch][obs_key] = recs

    return {
        "scenario": scenario_name,
        "seed": seed,
        "dataset_stats": {
            "total_users": n_users,
            "warm_users": len(ds.warm_users),
            "cold_test_users": len(ds.cold_test_users),
            "warm_train_interactions": len(ds.warm_train),
            "warm_val_interactions": len(ds.warm_val),
            "warm_test_interactions": len(ds.warm_test),
            "cold_test_interactions": len(ds.cold_test_ratings),
        },
        "trait_recovery": trait_diag,
        "selected_baseline": {"prediction_shrinkage": best_ps, "cf_weight": best_cf_w},
        "selected_personality_weight": best_w_p,
        "weight_sweep": weight_sweep,
        "warm_disabled_equals_baseline": warm_lists["cf_content_temporal_personality_disabled"] == warm_lists["cf_content_temporal"],
        "warm_list_changes": list_change_rate(warm_lists["cf_content_temporal"], warm_lists["cf_content_temporal_personality"]),
        "cold_list_changes": list_change_rate(cold_lists["cf_content_temporal"], cold_lists["cf_content_temporal_personality"]),
        "warm_metrics": {arch: summarize_per_user(warm_per_user[arch], warm_cov[arch], len(movies)) for arch in ARCHITECTURES},
        "cold_metrics": {arch: summarize_per_user(cold_per_user[arch], cold_cov[arch], len(movies)) for arch in ARCHITECTURES},
        "_warm_per_user": warm_per_user,
        "_cold_per_user": cold_per_user,
        "_warm_cov": warm_cov,
        "_cold_cov": cold_cov,
    }


def run_environment_b(
    movies,
    seeds=(101, 202, 303),
    scenarios=tuple(SCENARIOS),
    n_users=360,
    interactions_per_user=50,
    schema=None,
    resamples=2000,
    confidence=0.95,
    checkpoint_dir=None,
):
    """Environment B: Multi-seed synthetic controlled benchmark across Scenarios A-E."""
    schema = schema or build_schema()
    vectors = content.build_tfidf(movies)
    t_start = time.perf_counter()

    per_seed = {}
    scenario_summary = {}

    for sc in scenarios:
        seed_runs = []
        for s in seeds:
            ckpt_path = Path(checkpoint_dir) / f"{sc}_seed{s}.json" if checkpoint_dir else None
            res = _evaluate_synthetic_seed(
                movies, vectors, sc, s, n_users=n_users, interactions_per_user=interactions_per_user, schema=schema
            )
            if ckpt_path is not None:
                ckpt_path.parent.mkdir(parents=True, exist_ok=True)
                public_ckpt = {k: v for k, v in res.items() if not k.startswith("_")}
                ckpt_path.write_text(json.dumps(public_ckpt, indent=2), encoding="utf-8")
            seed_runs.append(res)
            per_seed[f"{sc}:seed{s}"] = {k: v for k, v in res.items() if not k.startswith("_")}

        # Aggregate warm & cold across seeds using user-cluster bootstrap
        def _agg_split(split_key, cov_key):
            pooled_pu = {
                arch: {k: v for sr in seed_runs for k, v in sr[split_key][arch].items()}
                for arch in ARCHITECTURES
            }
            pooled_cov = {
                arch: set().union(*(sr[cov_key][arch] for sr in seed_runs))
                for arch in ARCHITECTURES
            }
            models_summary = {
                arch: summarize_per_user(pooled_pu[arch], pooled_cov[arch], len(movies))
                for arch in ARCHITECTURES
            }
            comparisons = {}
            for arch in ARCHITECTURES:
                if arch == "cf_content_temporal":
                    continue
                if pooled_pu[arch].keys() & pooled_pu["cf_content_temporal"].keys():
                    comparisons[f"{arch}_vs_baseline_ndcg"] = paired_user_bootstrap(
                        pooled_pu[arch], pooled_pu["cf_content_temporal"], metric="ndcg",
                        resamples=resamples, confidence=confidence, seed=2026,
                    )
            return {"models": models_summary, "comparisons": comparisons}

        scenario_summary[sc] = {
            "config": SCENARIOS[sc],
            "seeds": list(seeds),
            "mean_selected_personality_weight": statistics.mean(sr["selected_personality_weight"] for sr in seed_runs),
            "mean_trait_pearson_r": statistics.mean(sr["trait_recovery"]["mean_pearson_r"] for sr in seed_runs),
            "mean_trait_mae": statistics.mean(sr["trait_recovery"]["overall_mae"] for sr in seed_runs),
            "warm_interaction_held_out": _agg_split("_warm_per_user", "_warm_cov"),
            "cold_user_held_out": _agg_split("_cold_per_user", "_cold_cov"),
        }

    # Sensitivity ablation on Scenario C_moderate (first seed) for noise & missingness
    ref_seed = seeds[0]
    noise_ablation = []
    for noise in (0.20, 0.45, 0.80, 1.20):
        ds_n = generate_synthetic_dataset(
            movies, "C_moderate", ref_seed, n_users=min(240, n_users), interactions_per_user=interactions_per_user,
            schema=schema, response_noise=noise, missing_rate=0.0,
        )
        td = trait_recovery_diagnostics(ds_n._latent_traits, ds_n.observed_profiles)
        pers_m = RidgePersonalityModel(alpha=5.0).fit(ds_n.warm_train, ds_n.observed_profiles, ds_n.movie_features)
        # Measure personality_only NDCG on cold_test_users as direct signal stability metric
        cold_rel = {}
        for uid, mid, r, _ in ds_n.cold_test_ratings:
            if r >= 4.0:
                cold_rel.setdefault(uid, set()).add(mid)
        ndcgs = [
            evaluation.ndcg_at_k(
                rank_architecture("personality_only", {}, ds_n.observed_profiles[u], ds_n.movie_features, pers_m, all_candidates=movies.keys(), k=10),
                rel, 10,
            )
            for u, rel in cold_rel.items()
        ]
        noise_ablation.append({
            "response_noise": noise,
            "trait_pearson_r": td["mean_pearson_r"],
            "trait_mae": td["overall_mae"],
            "cold_personality_only_ndcg": statistics.mean(ndcgs) if ndcgs else 0.0,
        })

    missing_ablation = []
    for miss in (0.0, 0.25, 0.50, 0.75):
        ds_m = generate_synthetic_dataset(
            movies, "C_moderate", ref_seed, n_users=min(240, n_users), interactions_per_user=interactions_per_user,
            schema=schema, response_noise=0.45, missing_rate=miss,
        )
        td = trait_recovery_diagnostics(ds_m._latent_traits, ds_m.observed_profiles)
        pers_m = RidgePersonalityModel(alpha=5.0).fit(ds_m.warm_train, ds_m.observed_profiles, ds_m.movie_features)
        cold_rel = {}
        for uid, mid, r, _ in ds_m.cold_test_ratings:
            if r >= 4.0:
                cold_rel.setdefault(uid, set()).add(mid)
        ndcgs = [
            evaluation.ndcg_at_k(
                rank_architecture("personality_only", {}, ds_m.observed_profiles[u], ds_m.movie_features, pers_m, all_candidates=movies.keys(), k=10),
                rel, 10,
            )
            for u, rel in cold_rel.items()
        ]
        missing_ablation.append({
            "missing_rate": miss,
            "mean_completeness": td["mean_completeness"],
            "trait_pearson_r": td["mean_pearson_r"],
            "trait_mae": td["overall_mae"],
            "cold_personality_only_ndcg": statistics.mean(ndcgs) if ndcgs else 0.0,
        })

    return {
        "environment": "B_synthetic_controlled_benchmark",
        "disclaimer": "All users, questionnaire responses, attributes, and ratings are synthetic simulations.",
        "wall_seconds": time.perf_counter() - t_start,
        "dataset_summary": per_seed[f"{scenarios[0]}:seed{seeds[0]}"]["dataset_stats"],
        "scenario_results": scenario_summary,
        "per_seed_results": per_seed,
        "questionnaire_ablation": {
            "noise_sensitivity": noise_ablation,
            "missingness_sensitivity": missing_ablation,
        },
    }
