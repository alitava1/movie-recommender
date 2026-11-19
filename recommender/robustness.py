"""Phase 6 robustness, component ablation, user-cluster bootstrap, segments, candidate ceiling, and reproducibility."""

import json
import math
import random
import statistics
import time

from . import content, diagnostics, evaluation, hybrid, nested
from .data import group_by_user, load_movies, load_ratings

COMPONENT_WEIGHTS = {"tfidf": 0.50, "genre": 0.35, "year": 0.15}

ABLATION_SPECS = (
    ("cf_tuned", None),
    ("cf_top100_tfidf_only", ("tfidf",)),
    ("cf_top100_genre_only", ("genre",)),
    ("cf_top100_year_only", ("year",)),
    ("cf_top100_tfidf_genre", ("tfidf", "genre")),
    ("cf_top100_tfidf_year", ("tfidf", "year")),
    ("cf_top100_genre_year", ("genre", "year")),
    ("cf_top100_complete_enrichment", ("tfidf", "genre", "year")),
)

ABLATION_MODEL_NAMES = tuple(name for name, _ in ABLATION_SPECS)


def _component_maps(user_ratings, movies, vectors, candidates, threshold=4.0, profile=None, tfidf_raw=None):
    """Compute per-component normalized enrichment maps once per candidate pool."""
    if not user_ratings or not candidates:
        return {"tfidf": {}, "genre": {}, "year": {}}
    if tfidf_raw is None:
        if profile is None:
            profile = content.user_profile(vectors, user_ratings)
        tfidf_raw = content.score(profile, vectors, candidates) if profile else {m: 0.0 for m in candidates}
    else:
        tfidf_raw = {m: tfidf_raw.get(m, 0.0) for m in candidates}
    genre_prefs = diagnostics.build_genre_preferences(user_ratings, movies)
    pref_year = diagnostics.preferred_release_year(user_ratings, movies, threshold)

    genre_raw = {}
    year_raw = {}
    for movie_id in candidates:
        movie = movies.get(movie_id)
        if not movie:
            continue
        if movie.genres and genre_prefs:
            genre_raw[movie_id] = sum(genre_prefs.get(g, 0.0) for g in movie.genres) / len(movie.genres)
        else:
            genre_raw[movie_id] = 0.0
        if pref_year is not None and movie.year is not None:
            year_raw[movie_id] = math.exp(-abs(movie.year - pref_year) / 15.0)
        else:
            year_raw[movie_id] = 0.5

    return {
        "tfidf": hybrid.normalize(tfidf_raw),
        "genre": hybrid.normalize(genre_raw),
        "year": hybrid.normalize(year_raw),
    }


def combine_enrichment_components(component_maps, candidates, components):
    """Combine a subset of normalized enrichment components with fixed renormalized weights."""
    active = tuple(components)
    if not active:
        raise ValueError("at least one enrichment component must be specified")
    unknown = set(active) - set(COMPONENT_WEIGHTS)
    if unknown:
        raise ValueError(f"unknown enrichment components: {sorted(unknown)}")
    total_w = sum(COMPONENT_WEIGHTS[c] for c in active)
    return {
        movie_id: sum(COMPONENT_WEIGHTS[c] * component_maps[c].get(movie_id, 0.5) for c in active) / total_w
        for movie_id in candidates
    }


def component_content_scores(user_ratings, movies, vectors, candidates, components, threshold=4.0):
    """Compute subset-ablated content enrichment scores over candidates."""
    if not user_ratings or not candidates:
        return {}
    maps = _component_maps(user_ratings, movies, vectors, candidates, threshold=threshold)
    return combine_enrichment_components(maps, candidates, components)


def select_ablation_parameters(
    outer_train,
    movies,
    settings,
    nested_settings,
    k=10,
    threshold=4.0,
    top_m=100,
    cf_first_weights=diagnostics.DEFAULT_CF_FIRST_WEIGHTS,
):
    """Select parameters for cf_tuned and ablation variants A-H strictly on inner validation."""
    inner_train, validation = nested.split_inner(outer_train, nested_settings["inner_validation_fraction"])
    cf_grid = nested_settings["cf_grid"]
    neighbors_grid = list(dict.fromkeys(int(v) for v in cf_grid["neighbors"]))
    shrinkage_grid = list(dict.fromkeys(float(v) for v in cf_grid["prediction_shrinkage"]))
    cf_first_weights = tuple(float(w) for w in cf_first_weights)
    if not neighbors_grid or min(neighbors_grid) <= 0 or not shrinkage_grid or min(shrinkage_grid) < 0:
        raise ValueError("CF tuning grid must contain positive neighbors and non-negative prediction shrinkage")
    if not cf_first_weights:
        raise ValueError("cf_first_weights must not be empty")

    max_neighbors = max(neighbors_grid)
    _, vectors, cf_model = nested._model_parts(inner_train, movies, settings, max_neighbors)
    inner_users = group_by_user(inner_train)
    relevant_by_user = {}
    for user_id, movie_id, rating, _ in validation:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)
    eligible = {u: rel for u, rel in relevant_by_user.items() if inner_users.get(u)}

    cf_configs = [(n, s) for n in neighbors_grid for s in shrinkage_grid]
    cf_first_configs = [(n, s, w_cf) for n, s in cf_configs for w_cf in cf_first_weights]
    enrichment_names = [name for name, comps in ABLATION_SPECS if comps is not None]

    if not eligible:
        default_cf = {
            "neighbors": settings["cf"]["neighbors"],
            "prediction_shrinkage": settings["cf"].get("prediction_shrinkage", 100.0),
        }
        default_w_cf = cf_first_weights[0]
        result = {
            "fallback": True,
            "evaluated_users": 0,
            "inner_train_interactions": len(inner_train),
            "inner_validation_interactions": len(validation),
            "cf_tuned": {"parameters": default_cf, "ndcg": None, "candidates": []},
        }
        for name in enrichment_names:
            result[name] = {"parameters": {**default_cf, "cf_weight": default_w_cf}, "ndcg": None, "candidates": []}
        return result

    cf_scores = [0.0] * len(cf_configs)
    variant_scores = {name: [0.0] * len(cf_first_configs) for name in enrichment_names}
    cf_views = {n: nested._neighbor_view(cf_model, n) for n in neighbors_grid}

    for user_id in sorted(eligible):
        relevant = eligible[user_id]
        user_ratings = inner_users[user_id]
        candidates = movies.keys() - user_ratings.keys()
        profile = content.user_profile(vectors, user_ratings)
        tfidf_cache = {}
        pool_cache = {}

        for neighbor_index, neighbors in enumerate(neighbors_grid):
            evidence = cf_views[neighbors].evidence(user_ratings, candidates)
            for shrinkage_index, prediction_shrinkage in enumerate(shrinkage_grid):
                config_index = neighbor_index * len(shrinkage_grid) + shrinkage_index
                predictions = nested._cf_predictions(evidence, prediction_shrinkage)
                cf_scores[config_index] += evaluation.ndcg_at_k(nested._top_ids(predictions, k), relevant, k)

                top_m_ids = diagnostics.cf_top_m_candidates(predictions, top_m)
                cf_pool_n = hybrid.normalize({mid: predictions[mid] for mid in top_m_ids})
                pool_key = tuple(top_m_ids)
                if pool_key not in pool_cache:
                    missing = [mid for mid in top_m_ids if mid not in tfidf_cache]
                    if missing:
                        if profile:
                            tfidf_cache.update(content.score(profile, vectors, missing))
                        else:
                            for mid in missing:
                                tfidf_cache[mid] = 0.0
                    comp_maps = _component_maps(
                        user_ratings, movies, vectors, top_m_ids, threshold=threshold,
                        profile=profile, tfidf_raw=tfidf_cache,
                    )
                    pool_cache[pool_key] = {
                        name: hybrid.normalize(combine_enrichment_components(comp_maps, top_m_ids, comps))
                        for name, comps in ABLATION_SPECS if comps is not None
                    }
                variant_pools_n = pool_cache[pool_key]

                for w_index, w_cf in enumerate(cf_first_weights):
                    idx = config_index * len(cf_first_weights) + w_index
                    w_sec = 1.0 - w_cf
                    for name in enrichment_names:
                        sec_n = variant_pools_n[name]
                        reranked = {mid: w_cf * cf_pool_n[mid] + w_sec * sec_n[mid] for mid in top_m_ids}
                        variant_scores[name][idx] += evaluation.ndcg_at_k(nested._top_ids(reranked, k), relevant, k)

    count = len(eligible)
    cf_cands = [
        {"parameters": {"neighbors": n, "prediction_shrinkage": s}, "ndcg": sc / count}
        for (n, s), sc in zip(cf_configs, cf_scores)
    ]
    best_cf = max(cf_cands, key=lambda row: row["ndcg"])
    result = {
        "fallback": False,
        "evaluated_users": count,
        "inner_train_interactions": len(inner_train),
        "inner_validation_interactions": len(validation),
        "cf_tuned": {**best_cf, "candidates": cf_cands},
    }
    for name in enrichment_names:
        cands = [
            {"parameters": {"neighbors": n, "prediction_shrinkage": s, "cf_weight": w}, "ndcg": sc / count}
            for (n, s, w), sc in zip(cf_first_configs, variant_scores[name])
        ]
        best = max(cands, key=lambda row: row["ndcg"])
        result[name] = {**best, "candidates": cands}
    return result


def compute_user_train_segments(train_users, movies):
    """Define user history-size and genre-diversity segments strictly from outer_train."""
    entropies = {}
    sizes = {}
    for user_id, ratings in train_users.items():
        sizes[user_id] = len(ratings)
        genre_counts = {}
        for movie_id in ratings:
            movie = movies.get(movie_id)
            if not movie:
                continue
            for genre in movie.genres:
                genre_counts[genre] = genre_counts.get(genre, 0) + 1
        total = sum(genre_counts.values())
        if total > 0:
            ent = -sum((c / total) * math.log2(c / total) for c in genre_counts.values())
        else:
            ent = 0.0
        entropies[user_id] = ent

    sorted_ent = sorted(entropies.values())
    if sorted_ent:
        q33 = diagnostics._quantile(sorted_ent, 1.0 / 3.0)
        q67 = diagnostics._quantile(sorted_ent, 2.0 / 3.0)
    else:
        q33, q67 = 0.0, 0.0

    user_segments = {}
    for user_id in train_users:
        n = sizes[user_id]
        if n <= 30:
            size_seg = "sparse"
        elif n <= 80:
            size_seg = "medium"
        else:
            size_seg = "dense"

        ent = entropies[user_id]
        if ent <= q33:
            div_seg = "narrow"
        elif ent <= q67:
            div_seg = "moderate"
        else:
            div_seg = "diverse"

        user_segments[user_id] = {
            "train_count": n,
            "genre_entropy": ent,
            "history_size": size_seg,
            "genre_diversity": div_seg,
        }
    return user_segments, {"entropy_q33": q33, "entropy_q67": q67, "size_cutoffs": [30, 80]}


def evaluate_ablation_fold(
    fold,
    outer_train,
    outer_test,
    movies,
    settings,
    nested_settings,
    k=10,
    threshold=4.0,
    top_m=100,
    cf_first_weights=diagnostics.DEFAULT_CF_FIRST_WEIGHTS,
    collect_lists=False,
):
    """Run train-only selection, outer refit, and single outer-test scoring for ablation A-H."""
    sel_started = time.perf_counter()
    selection = select_ablation_parameters(
        outer_train, movies, settings, nested_settings,
        k=nested_settings.get("selection_k", k),
        threshold=threshold,
        top_m=top_m,
        cf_first_weights=cf_first_weights,
    )
    selection_seconds = time.perf_counter() - sel_started

    max_neighbors = max(selection[name]["parameters"]["neighbors"] for name in ABLATION_MODEL_NAMES)
    refit_started = time.perf_counter()
    _, vectors, cf_model = nested._model_parts(outer_train, movies, settings, max_neighbors)
    refit_seconds = time.perf_counter() - refit_started

    train_users = group_by_user(outer_train)
    user_segments, segment_cutoffs = compute_user_train_segments(train_users, movies)

    relevant_by_user = {}
    for user_id, movie_id, rating, _ in outer_test:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)

    per_user = {name: {} for name in ABLATION_MODEL_NAMES}
    coverage = {name: set() for name in ABLATION_MODEL_NAMES}
    fold_segments = {}
    ranked_lists = {name: {} for name in ABLATION_MODEL_NAMES} if collect_lists else None
    candidate_pools = {} if collect_lists else None

    needed_cf = {
        (selection[name]["parameters"]["neighbors"], selection[name]["parameters"]["prediction_shrinkage"])
        for name in ABLATION_MODEL_NAMES
    }
    cf_views = {n: nested._neighbor_view(cf_model, n) for n, _ in needed_cf}

    score_started = time.perf_counter()
    for user_id in sorted(relevant_by_user):
        relevant = relevant_by_user[user_id]
        user_ratings = train_users.get(user_id, {})
        if not user_ratings:
            continue
        candidates = movies.keys() - user_ratings.keys()
        profile = content.user_profile(vectors, user_ratings)
        tfidf_cache = {}

        ev_cache = {}
        preds_cache = {}
        for n, s in needed_cf:
            if n not in ev_cache:
                ev_cache[n] = cf_views[n].evidence(user_ratings, candidates)
            preds_cache[(n, s)] = nested._cf_predictions(ev_cache[n], s)

        pool_cache = {}
        uid_key = f"u{fold}:{user_id}"
        fold_segments[uid_key] = user_segments[user_id]

        for name, comps in ABLATION_SPECS:
            params = selection[name]["parameters"]
            cf_preds = preds_cache[(params["neighbors"], params["prediction_shrinkage"])]
            if comps is None:
                recs = nested._top_ids(cf_preds, k)
                if collect_lists:
                    candidate_pools[uid_key] = diagnostics.cf_top_m_candidates(cf_preds, top_m)
            else:
                top_m_ids = diagnostics.cf_top_m_candidates(cf_preds, top_m)
                pool_key = tuple(top_m_ids)
                if pool_key not in pool_cache:
                    missing = [mid for mid in top_m_ids if mid not in tfidf_cache]
                    if missing:
                        if profile:
                            tfidf_cache.update(content.score(profile, vectors, missing))
                        else:
                            for mid in missing:
                                tfidf_cache[mid] = 0.0
                    pool_cache[pool_key] = _component_maps(
                        user_ratings, movies, vectors, top_m_ids, threshold=threshold,
                        profile=profile, tfidf_raw=tfidf_cache,
                    )
                comp_score = combine_enrichment_components(pool_cache[pool_key], top_m_ids, comps)
                reranked = diagnostics.rerank_pool_scores(top_m_ids, cf_preds, comp_score, params["cf_weight"])
                recs = nested._top_ids(reranked, k)
                if collect_lists and name == "cf_top100_complete_enrichment":
                    candidate_pools[f"{uid_key}:complete_enrichment"] = top_m_ids

            per_user[name][uid_key] = evaluation.evaluate_user_extended(recs, relevant, k, vectors)
            coverage[name].update(recs)
            if collect_lists:
                ranked_lists[name][uid_key] = recs

    score_seconds = time.perf_counter() - score_started
    metrics = {
        name: diagnostics._summarize_extended(values, len(coverage[name]), len(movies))
        for name, values in per_user.items()
    }
    return {
        "selection": selection,
        "metrics": metrics,
        "per_user": per_user,
        "segments": fold_segments,
        "segment_cutoffs": segment_cutoffs,
        "_coverage": coverage,
        "ranked_lists": ranked_lists,
        "candidate_pools": candidate_pools,
        "resources": {
            "selection_seconds": selection_seconds,
            "refit_seconds": refit_seconds,
            "outer_scoring_seconds": score_seconds,
        },
    }


def _extract_user_id(obs_key):
    if ":" in obs_key:
        return obs_key.split(":", 1)[1]
    return obs_key


def paired_cluster_bootstrap(
    left,
    right,
    metric="ndcg",
    resamples=2000,
    confidence=0.95,
    seed=2026,
    cluster_fn=_extract_user_id,
):
    """Paired user-cluster bootstrap resampling all fold observations of each user together."""
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be between 0 and 1")
    common_keys = sorted(left.keys() & right.keys())
    if not common_keys:
        raise ValueError("paired cluster bootstrap needs at least one matching observation")

    clusters = {}
    for key in common_keys:
        cid = cluster_fn(key)
        clusters.setdefault(cid, []).append(key)

    cluster_stats = []
    total_obs = 0
    total_left = 0.0
    total_right = 0.0
    for cid in sorted(clusters, key=lambda x: (str(x))):
        keys = clusters[cid]
        s_left = sum(left[k][metric] for k in keys)
        s_right = sum(right[k][metric] for k in keys)
        cnt = len(keys)
        cluster_stats.append((cnt, s_left, s_right))
        total_obs += cnt
        total_left += s_left
        total_right += s_right

    obs_left_mean = total_left / total_obs
    obs_right_mean = total_right / total_obs
    abs_estimate = obs_left_mean - obs_right_mean
    rel_estimate = abs_estimate / obs_right_mean if obs_right_mean != 0 else None

    rng = random.Random(seed)
    n_clusters = len(cluster_stats)
    abs_samples = []
    rel_samples = []
    for _ in range(resamples):
        b_cnt = 0
        b_left = 0.0
        b_right = 0.0
        for _ in range(n_clusters):
            cnt, s_l, s_r = rng.choice(cluster_stats)
            b_cnt += cnt
            b_left += s_l
            b_right += s_r
        l_mean = b_left / b_cnt
        r_mean = b_right / b_cnt
        diff = l_mean - r_mean
        abs_samples.append(diff)
        if r_mean != 0:
            rel_samples.append(diff / r_mean)

    abs_samples.sort()
    tail = (1.0 - confidence) / 2.0
    abs_lower = nested._percentile(abs_samples, tail)
    abs_upper = nested._percentile(abs_samples, 1.0 - tail)

    if len(rel_samples) == resamples:
        rel_samples.sort()
        rel_lower = nested._percentile(rel_samples, tail)
        rel_upper = nested._percentile(rel_samples, 1.0 - tail)
    else:
        rel_lower = None
        rel_upper = None

    return {
        "metric": metric,
        "left_mean": obs_left_mean,
        "right_mean": obs_right_mean,
        "abs_diff": abs_estimate,
        "abs_lower": abs_lower,
        "abs_upper": abs_upper,
        "rel_diff": rel_estimate,
        "rel_lower": rel_lower,
        "rel_upper": rel_upper,
        "confidence": confidence,
        "resamples": resamples,
        "seed": seed,
        "clusters": n_clusters,
        "observations": total_obs,
    }


def aggregate_ablation_folds(folds, catalog_size, resamples=2000, confidence=0.95, seed=2026):
    """Aggregate per-fold metrics, fold mean ± SD, and cluster/unclustered bootstrap CIs."""
    pooled = {}
    for name in ABLATION_MODEL_NAMES:
        fold_rows = [res["metrics"][name] for res in folds.values() if res["metrics"][name]["users"]]
        total_users = sum(row["users"] for row in fold_rows)
        pooled[name] = {
            metric: sum(row[metric] * row["users"] for row in fold_rows) / total_users if total_users else None
            for metric in diagnostics.DIAGNOSTIC_METRICS
        }
        all_cov = set().union(*(res["_coverage"][name] for res in folds.values()))
        pooled[name].update({"coverage": len(all_cov) / catalog_size, "users": total_users})
        pooled[name]["fold_mean_std"] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in fold_rows) if fold_rows else None,
                "std": statistics.stdev(row[metric] for row in fold_rows) if len(fold_rows) > 1 else 0.0 if fold_rows else None,
            }
            for metric in (*diagnostics.DIAGNOSTIC_METRICS, "coverage")
        }

    per_user = {
        name: {key: val for res in folds.values() for key, val in res["per_user"][name].items()}
        for name in ABLATION_MODEL_NAMES
    }

    cluster_targets = (
        "cf_top100_complete_enrichment",
        "cf_top100_tfidf_only",
        "cf_top100_genre_only",
        "cf_top100_year_only",
        "cf_top100_tfidf_genre",
        "cf_top100_tfidf_year",
        "cf_top100_genre_year",
    )
    cluster_bootstrap = {}
    unclustered_comparison = {}
    for candidate in cluster_targets:
        cluster_bootstrap[candidate] = {}
        for metric in diagnostics.DIAGNOSTIC_METRICS:
            cluster_bootstrap[candidate][metric] = paired_cluster_bootstrap(
                per_user[candidate], per_user["cf_tuned"], metric=metric,
                resamples=resamples, confidence=confidence, seed=seed,
            )
        unclustered_comparison[candidate] = nested.paired_bootstrap(
            per_user[candidate], per_user["cf_tuned"], metric="ndcg",
            resamples=resamples, confidence=confidence, seed=seed,
        )

    return {
        "pooled": pooled,
        "cluster_bootstrap": cluster_bootstrap,
        "unclustered_ndcg_bootstrap": unclustered_comparison,
    }


def analyze_segments(folds, resamples=2000, confidence=0.95, seed=2026):
    """Aggregate segment performance defined strictly from outer_train histories."""
    segment_meta = {key: val for res in folds.values() for key, val in res["segments"].items()}
    per_user = {
        name: {key: val for res in folds.values() for key, val in res["per_user"][name].items()}
        for name in ("cf_tuned", "cf_top100_complete_enrichment", "cf_top100_tfidf_only", "cf_top100_genre_only", "cf_top100_year_only")
    }

    axes = {
        "history_size": ("sparse", "medium", "dense"),
        "genre_diversity": ("narrow", "moderate", "diverse"),
    }
    results = {
        "definitions": {
            "history_size": "outer_train rating count only: sparse <= 30, medium 31-80, dense > 80",
            "genre_diversity": "outer_train Shannon entropy tertiles computed per fold on all outer_train users only",
            "fold_entropy_cutoffs": {fold: res["segment_cutoffs"] for fold, res in folds.items()},
        },
        "segments": {},
    }

    for axis, labels in axes.items():
        axis_out = {}
        for label in labels:
            keys = sorted(k for k, meta in segment_meta.items() if meta[axis] == label)
            unique_users = len({_extract_user_id(k) for k in keys})
            seg_models = {}
            for name, rows in per_user.items():
                sub = {k: rows[k] for k in keys if k in rows}
                if sub:
                    seg_models[name] = {
                        metric: sum(r[metric] for r in sub.values()) / len(sub)
                        for metric in diagnostics.DIAGNOSTIC_METRICS
                    }
                else:
                    seg_models[name] = {metric: None for metric in diagnostics.DIAGNOSTIC_METRICS}
            comp_ndcg = paired_cluster_bootstrap(
                {k: per_user["cf_top100_complete_enrichment"][k] for k in keys},
                {k: per_user["cf_tuned"][k] for k in keys},
                metric="ndcg", resamples=resamples, confidence=confidence, seed=seed,
            ) if keys else None
            comp_hr = paired_cluster_bootstrap(
                {k: per_user["cf_top100_complete_enrichment"][k] for k in keys},
                {k: per_user["cf_tuned"][k] for k in keys},
                metric="hr", resamples=resamples, confidence=confidence, seed=seed,
            ) if keys else None
            axis_out[label] = {
                "user_folds": len(keys),
                "unique_users": unique_users,
                "models": seg_models,
                "complete_vs_cf_tuned_ndcg": comp_ndcg,
                "complete_vs_cf_tuned_hr": comp_hr,
            }
        results["segments"][axis] = axis_out
    return results


def audit_candidate_recall_inner(
    dataset_dir,
    dataset,
    folds_list,
    movies,
    settings,
    nested_settings,
    pool_sizes=(50, 100, 200),
    k=10,
    threshold=4.0,
    cf_first_weights=diagnostics.DEFAULT_CF_FIRST_WEIGHTS,
):
    """Audit candidate pool recall, excluded relevant items, conditional accuracy, and cost on inner validation ONLY."""
    pool_sizes = tuple(sorted(set(int(m) for m in pool_sizes)))
    cf_grid = nested_settings["cf_grid"]
    neighbors_grid = list(dict.fromkeys(int(v) for v in cf_grid["neighbors"]))
    shrinkage_grid = list(dict.fromkeys(float(v) for v in cf_grid["prediction_shrinkage"]))
    max_neighbors = max(neighbors_grid)

    per_fold = {}
    agg_totals = {
        m: {
            "users": 0,
            "total_relevant": 0,
            "captured_relevant": 0,
            "excluded_relevant": 0,
            "recall_sum": 0.0,
            "cond_users": 0,
            "cf_ndcg_all": 0.0,
            "enrich_ndcg_all": 0.0,
            "cf_map_all": 0.0,
            "enrich_map_all": 0.0,
            "cf_hr_all": 0.0,
            "enrich_hr_all": 0.0,
            "cf_ndcg_cond": 0.0,
            "enrich_ndcg_cond": 0.0,
            "cf_hr_cond": 0.0,
            "enrich_hr_cond": 0.0,
            "score_seconds": 0.0,
        }
        for m in pool_sizes
    }

    for fold in folds_list:
        outer_train = load_ratings(dataset, dataset_dir / f"u{fold}.base")
        inner_train, validation = nested.split_inner(outer_train, nested_settings["inner_validation_fraction"])
        _, vectors, cf_model = nested._model_parts(inner_train, movies, settings, max_neighbors)
        inner_users = group_by_user(inner_train)
        relevant_by_user = {}
        for user_id, movie_id, rating, _ in validation:
            if rating >= threshold:
                relevant_by_user.setdefault(user_id, set()).add(movie_id)
        eligible = {u: rel for u, rel in relevant_by_user.items() if inner_users.get(u)}

        # Precompute evidence & predictions for default CF (or grid-selected? Audit using inner-selected per M!).
        # To keep candidate-generation selection strictly inner-only, tune (n, s, w_cf) for each pool size M on inner val!
        cf_configs = [(n, s) for n in neighbors_grid for s in shrinkage_grid]
        cf_first_configs = [(n, s, w) for n, s in cf_configs for w in cf_first_weights]
        cf_views = {n: nested._neighbor_view(cf_model, n) for n in neighbors_grid}

        user_predictions = {}
        for user_id in sorted(eligible):
            user_ratings = inner_users[user_id]
            candidates = movies.keys() - user_ratings.keys()
            preds_by_cfg = {}
            for n in neighbors_grid:
                ev = cf_views[n].evidence(user_ratings, candidates)
                for s in shrinkage_grid:
                    preds_by_cfg[(n, s)] = nested._cf_predictions(ev, s)
            user_predictions[user_id] = preds_by_cfg

        # Pick best inner CF config once (independent of M), and best inner complete enrichment per M.
        sorted_users = sorted(eligible)
        best_cf_cfg = max(
            cf_configs,
            key=lambda cfg: sum(
                evaluation.ndcg_at_k(nested._top_ids(user_predictions[u][cfg], k), eligible[u], k)
                for u in sorted_users
            ),
        )

        fold_res = {}
        for m in pool_sizes:
            t0 = time.perf_counter()
            best_m_cfg = None
            best_m_score = -1.0
            pool_by_u_cfg = {}
            comp_cache = {}
            for u in sorted_users:
                user_ratings = inner_users[u]
                profile = content.user_profile(vectors, user_ratings)
                tfidf_cache = {}
                for cfg in cf_configs:
                    preds = user_predictions[u][cfg]
                    pool_ids = diagnostics.cf_top_m_candidates(preds, m)
                    pool_by_u_cfg[(u, cfg)] = pool_ids
                    pkey = (u, tuple(pool_ids))
                    if pkey not in comp_cache:
                        missing = [mid for mid in pool_ids if mid not in tfidf_cache]
                        if missing:
                            if profile:
                                tfidf_cache.update(content.score(profile, vectors, missing))
                            else:
                                for mid in missing:
                                    tfidf_cache[mid] = 0.0
                        maps = _component_maps(
                            user_ratings, movies, vectors, pool_ids, threshold=threshold,
                            profile=profile, tfidf_raw=tfidf_cache,
                        )
                        comp_cache[pkey] = hybrid.normalize(
                            combine_enrichment_components(maps, pool_ids, ("tfidf", "genre", "year"))
                        )
            cf_norm_cache = {
                (u, cfg): hybrid.normalize({mid: user_predictions[u][cfg][mid] for mid in pool_by_u_cfg[(u, cfg)]})
                for u in sorted_users for cfg in cf_configs
            }
            for n, s, w_cf in cf_first_configs:
                score_sum = 0.0
                w_sec = 1.0 - w_cf
                for u in sorted_users:
                    pool_ids = pool_by_u_cfg[(u, (n, s))]
                    cf_n = cf_norm_cache[(u, (n, s))]
                    sec_n = comp_cache[(u, tuple(pool_ids))]
                    reranked = {mid: w_cf * cf_n[mid] + w_sec * sec_n[mid] for mid in pool_ids}
                    score_sum += evaluation.ndcg_at_k(nested._top_ids(reranked, k), eligible[u], k)
                if score_sum > best_m_score:
                    best_m_score = score_sum
                    best_m_cfg = (n, s, w_cf)
            elapsed_m = time.perf_counter() - t0

            # Measure pool recall, excluded items, and conditional vs unconditional accuracy at M.
            m_stats = {
                "users": len(eligible),
                "total_relevant": 0,
                "captured_relevant": 0,
                "excluded_relevant": 0,
                "recall_sum": 0.0,
                "cond_users": 0,
                "cf_ndcg_all": 0.0,
                "enrich_ndcg_all": 0.0,
                "cf_map_all": 0.0,
                "enrich_map_all": 0.0,
                "cf_hr_all": 0.0,
                "enrich_hr_all": 0.0,
                "cf_ndcg_cond": 0.0,
                "enrich_ndcg_cond": 0.0,
                "cf_hr_cond": 0.0,
                "enrich_hr_cond": 0.0,
                "score_seconds": elapsed_m,
            }
            bn, bs, bw = best_m_cfg
            for u in sorted_users:
                rel = eligible[u]
                cf_preds = user_predictions[u][best_cf_cfg]
                enr_preds = user_predictions[u][(bn, bs)]
                pool_ids = pool_by_u_cfg[(u, (bn, bs))]
                pool_set = set(pool_ids)
                hits_in_pool = len(pool_set & rel)
                n_rel = len(rel)
                m_stats["total_relevant"] += n_rel
                m_stats["captured_relevant"] += hits_in_pool
                m_stats["excluded_relevant"] += (n_rel - hits_in_pool)
                m_stats["recall_sum"] += hits_in_pool / n_rel

                cf_recs = nested._top_ids(cf_preds, k)
                enrich_s = comp_cache[(u, tuple(pool_ids))]
                reranked = diagnostics.rerank_pool_scores(pool_ids, enr_preds, enrich_s, bw)
                enr_recs = nested._top_ids(reranked, k)

                cf_m = evaluation.evaluate_user_extended(cf_recs, rel, k)
                en_m = evaluation.evaluate_user_extended(enr_recs, rel, k)
                m_stats["cf_ndcg_all"] += cf_m["ndcg"]
                m_stats["enrich_ndcg_all"] += en_m["ndcg"]
                m_stats["cf_map_all"] += cf_m["map"]
                m_stats["enrich_map_all"] += en_m["map"]
                m_stats["cf_hr_all"] += cf_m["hr"]
                m_stats["enrich_hr_all"] += en_m["hr"]
                if hits_in_pool > 0:
                    m_stats["cond_users"] += 1
                    m_stats["cf_ndcg_cond"] += cf_m["ndcg"]
                    m_stats["enrich_ndcg_cond"] += en_m["ndcg"]
                    m_stats["cf_hr_cond"] += cf_m["hr"]
                    m_stats["enrich_hr_cond"] += en_m["hr"]

            for key, val in m_stats.items():
                agg_totals[m][key] += val
            n_u = m_stats["users"]
            c_u = m_stats["cond_users"]
            fold_res[str(m)] = {
                "selected_cf": {"neighbors": best_cf_cfg[0], "prediction_shrinkage": best_cf_cfg[1]},
                "selected_enrichment": {"neighbors": bn, "prediction_shrinkage": bs, "cf_weight": bw},
                "pool_recall": m_stats["recall_sum"] / n_u if n_u else 0.0,
                "total_relevant": m_stats["total_relevant"],
                "captured_relevant": m_stats["captured_relevant"],
                "excluded_relevant": m_stats["excluded_relevant"],
                "conditional_users": c_u,
                "unconditional_cf_ndcg": m_stats["cf_ndcg_all"] / n_u if n_u else 0.0,
                "unconditional_enrich_ndcg": m_stats["enrich_ndcg_all"] / n_u if n_u else 0.0,
                "conditional_cf_ndcg": m_stats["cf_ndcg_cond"] / c_u if c_u else 0.0,
                "conditional_enrich_ndcg": m_stats["enrich_ndcg_cond"] / c_u if c_u else 0.0,
                "selection_seconds": elapsed_m,
            }
        per_fold[f"u{fold}"] = fold_res

    summary = {}
    for m in pool_sizes:
        t = agg_totals[m]
        n_u = t["users"]
        c_u = t["cond_users"]
        summary[str(m)] = {
            "pool_size": m,
            "user_folds": n_u,
            "conditional_user_folds": c_u,
            "conditional_user_fraction": c_u / n_u if n_u else 0.0,
            "pool_recall": t["recall_sum"] / n_u if n_u else 0.0,
            "total_relevant_items": t["total_relevant"],
            "captured_relevant_items": t["captured_relevant"],
            "excluded_relevant_items": t["excluded_relevant"],
            "mean_excluded_per_user": t["excluded_relevant"] / n_u if n_u else 0.0,
            "unconditional": {
                "cf_tuned_ndcg": t["cf_ndcg_all"] / n_u if n_u else 0.0,
                "complete_enrichment_ndcg": t["enrich_ndcg_all"] / n_u if n_u else 0.0,
                "cf_tuned_map": t["cf_map_all"] / n_u if n_u else 0.0,
                "complete_enrichment_map": t["enrich_map_all"] / n_u if n_u else 0.0,
                "cf_tuned_hr": t["cf_hr_all"] / n_u if n_u else 0.0,
                "complete_enrichment_hr": t["enrich_hr_all"] / n_u if n_u else 0.0,
            },
            "conditional_on_relevant_in_pool": {
                "cf_tuned_ndcg": t["cf_ndcg_cond"] / c_u if c_u else 0.0,
                "complete_enrichment_ndcg": t["enrich_ndcg_cond"] / c_u if c_u else 0.0,
                "cf_tuned_hr": t["cf_hr_cond"] / c_u if c_u else 0.0,
                "complete_enrichment_hr": t["enrich_hr_cond"] / c_u if c_u else 0.0,
            },
            "inner_selection_seconds": t["score_seconds"],
            "ms_per_user_fold": (1000.0 * t["score_seconds"] / n_u) if n_u else 0.0,
        }
    return {"protocol": "inner-validation-only", "pool_sizes": list(pool_sizes), "summary": summary, "per_fold": per_fold}


def compare_runs_reproducibility(run1, run2):
    """Verify identical candidate sets, top-10 rankings, metrics, and canonical JSON payloads."""
    candidate_mismatches = 0
    ranking_mismatches = 0
    total_candidate_checks = 0
    total_ranking_checks = 0

    for fold_name in run1["folds"]:
        pools1 = run1["folds"][fold_name]["candidate_pools"] or {}
        pools2 = run2["folds"][fold_name]["candidate_pools"] or {}
        if pools1.keys() != pools2.keys():
            candidate_mismatches += 1
        for k in pools1.keys() & pools2.keys():
            total_candidate_checks += 1
            if pools1[k] != pools2[k]:
                candidate_mismatches += 1

        lists1 = run1["folds"][fold_name]["ranked_lists"] or {}
        lists2 = run2["folds"][fold_name]["ranked_lists"] or {}
        for model in ABLATION_MODEL_NAMES:
            m1 = lists1.get(model, {})
            m2 = lists2.get(model, {})
            if m1.keys() != m2.keys():
                ranking_mismatches += 1
            for k in m1.keys() & m2.keys():
                total_ranking_checks += 1
                if m1[k] != m2[k]:
                    ranking_mismatches += 1

    canon1 = json.dumps(run1["deterministic_payload"], sort_keys=True)
    canon2 = json.dumps(run2["deterministic_payload"], sort_keys=True)
    return {
        "identical_candidate_sets": candidate_mismatches == 0,
        "candidate_checks": total_candidate_checks,
        "candidate_mismatches": candidate_mismatches,
        "identical_rankings": ranking_mismatches == 0,
        "ranking_checks": total_ranking_checks,
        "ranking_mismatches": ranking_mismatches,
        "identical_metrics_and_bootstrap": canon1 == canon2,
        "canonical_payload_bytes": len(canon1.encode("utf-8")),
    }
