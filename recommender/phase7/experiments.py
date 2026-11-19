"""Phase 7 experiments: frozen Phase 6 transfer, independent ML-1M nested tuning, and diagnostics."""

import time

from .. import content, diagnostics, evaluation, hybrid, nested, popularity
from ..data import group_by_user
from .cf_adapter import build_cf
from .metrics import evaluate_user_phase7, summarize_users
from .temporal import split_inner_chronological

# Frozen Phase 6 configurations (modal per-fold selections from phase6-robustness).
FROZEN_CF_TUNED = {"neighbors": 50, "shrinkage": 10.0, "prediction_shrinkage": 100.0}
FROZEN_ENRICHMENT = {
    "neighbors": 100,
    "shrinkage": 10.0,
    "prediction_shrinkage": 100.0,
    "cf_weight": 0.7,
    "top_m": 100,
}

# Independent ML-1M nested tuning grid (Phase 6 grid, small and non-exhaustive).
TUNING_GRID = {
    "neighbors": (20, 50, 100),
    "prediction_shrinkage": (10.0, 100.0, 300.0),
    "cf_weight": (0.95, 0.90, 0.80, 0.70),
}

SEGMENT_SIZE_CUTOFFS = (30, 80)
SEGMENT_POP_CUTOFFS = (100, 1000)
SEGMENT_YEAR_CUTOFFS = (1979, 1995)

ENRICHMENT_COMPONENTS = ("tfidf", "genre", "year")


def _enrichment_scores(user_ratings, movies, vectors, pool_ids, threshold=4.0):
    return diagnostics.enriched_content_scores(user_ratings, movies, vectors, pool_ids, threshold)


def _pool_scores(cf_model, user_ratings, candidates, neighbors, prediction_shrinkage, top_m, movies, vectors, cf_weight, threshold=4.0):
    """Phase 6 cf_top100_complete_enrichment pipeline; returns (scores, pool_ids)."""
    view = _view_cache(cf_model, neighbors)
    evidence = view.evidence(user_ratings, candidates)
    preds = {
        movie_id: min(5.0, max(0.5, mean + num / (den + prediction_shrinkage)))
        for movie_id, (mean, num, den) in evidence.items()
    }
    pool_ids = diagnostics.cf_top_m_candidates(preds, top_m)
    if not pool_ids:
        return {}, []
    enriched = _enrichment_scores(user_ratings, movies, vectors, pool_ids, threshold)
    return diagnostics.rerank_pool_scores(pool_ids, preds, enriched, cf_weight), pool_ids


def _view_cache(cf_model, neighbors):
    from ..collaborative import ItemItemCF

    view = cf_model
    if neighbors < cf_model.neighbors:
        view = ItemItemCF(
            neighbors,
            cf_model.shrinkage,
            {m: rows[:neighbors] for m, rows in cf_model.item_neighbors.items()},
            cf_model.prediction_shrinkage,
        )
    return view


def _cf_scores(cf_model, user_ratings, candidates, neighbors, prediction_shrinkage):
    """Plain cf_tuned top-k scores (full candidate space, no enrichment)."""
    view = _view_cache(cf_model, neighbors)
    evidence = view.evidence(user_ratings, candidates)
    return {
        movie_id: min(5.0, max(0.5, mean + num / (den + prediction_shrinkage)))
        for movie_id, (mean, num, den) in evidence.items()
    }


def run_experiment(
    outer_train,
    outer_test_rows,
    movies,
    vectors,
    settings,
    k=10,
    threshold=4.0,
    top_m=100,
    frozen=True,
    grid=TUNING_GRID,
    collect_lists=False,
    cf_model=None,
    popularity_counts=None,
    user_filter=None,
):
    """Run one Phase 7 experiment (frozen transfer or nested tuned) and return per-user metrics.

    Frozen mode scores with fixed Phase 6 modal parameters. Tuned mode performs inner-only
    selection over grid, refits the winner on the full outer_train, and scores outer_test once.
    """
    train_users = group_by_user(outer_train)
    if cf_model is None:
        max_neighbors = max(FROZEN_CF_TUNED["neighbors"], FROZEN_ENRICHMENT["neighbors"]) if frozen else max(grid["neighbors"])
        cf_model = build_cf(
            outer_train,
            neighbors=max_neighbors,
            shrinkage=settings["cf"]["shrinkage"],
            prediction_shrinkage=0.0,
        )
    if popularity_counts is None:
        popularity_counts = {}
        for _, movie_id, _, _ in outer_train:
            popularity_counts[movie_id] = popularity_counts.get(movie_id, 0) + 1

    selection = {"mode": "frozen_phase6_transfer" if frozen else "independent_ml1m_tuning"}
    sweep_rows = []
    if frozen:
        cf_cfg = dict(FROZEN_CF_TUNED)
        enr_cfg = dict(FROZEN_ENRICHMENT)
    else:
        inner_train, validation = split_inner_chronological(outer_train)
        inner_users = group_by_user(inner_train)
        relevant_val = {}
        for user_id, movie_id, rating, _ in validation:
            if rating >= threshold:
                relevant_val.setdefault(user_id, set()).add(movie_id)
        eligible_val = {u: rel for u, rel in relevant_val.items() if inner_users.get(u)}
        if user_filter is not None:
            eligible_val = {u: rel for u, rel in eligible_val.items() if u in user_filter}
        max_n = max(grid["neighbors"])
        cf_inner = build_cf(inner_train, neighbors=max_n, shrinkage=settings["cf"]["shrinkage"], prediction_shrinkage=0.0)

        cf_scores = {}
        enr_scores = {}
        started = time.perf_counter()
        for user_id, relevant in eligible_val.items():
            ur = inner_users[user_id]
            cands = movies.keys() - ur.keys()
            for n in grid["neighbors"]:
                view = _view_cache(cf_inner, n)
                evidence = view.evidence(ur, cands)
                for s in grid["prediction_shrinkage"]:
                    preds = {
                        movie_id: min(5.0, max(0.5, mean + num / (den + s)))
                        for movie_id, (mean, num, den) in evidence.items()
                    }
                    cf_scores[(n, s)] = cf_scores.get((n, s), 0.0) + _ndcg_of(
                        _top_ids(preds, k), relevant, k
                    )
                    pool_ids = diagnostics.cf_top_m_candidates(preds, top_m)
                    if not pool_ids:
                        continue
                    enriched = _enrichment_scores(ur, movies, vectors, pool_ids, threshold)
                    cf_n = hybrid.normalize({mid: preds[mid] for mid in pool_ids})
                    sec_n = hybrid.normalize(enriched)
                    for w in grid["cf_weight"]:
                        key = (n, s, w)
                        w_sec = 1.0 - w
                        reranked = {mid: w * cf_n[mid] + w_sec * sec_n[mid] for mid in pool_ids}
                        enr_scores[key] = enr_scores.get(key, 0.0) + _ndcg_of(
                            _top_ids(reranked, k), relevant, k
                        )
        selection_seconds = time.perf_counter() - started
        count = max(1, len(eligible_val))
        cf_means = {key: sc / count for key, sc in cf_scores.items()}
        enr_means = {key: sc / count for key, sc in enr_scores.items()}
        best_cf_cfg = max(cf_means, key=cf_means.get)
        best_enr_cfg = max(enr_means, key=enr_means.get)
        cf_cfg = {"neighbors": best_cf_cfg[0], "shrinkage": settings["cf"]["shrinkage"], "prediction_shrinkage": best_cf_cfg[1]}
        enr_cfg = {
            "neighbors": best_enr_cfg[0],
            "shrinkage": settings["cf"]["shrinkage"],
            "prediction_shrinkage": best_enr_cfg[1],
            "cf_weight": best_enr_cfg[2],
            "top_m": top_m,
        }
        sweep_rows = (
            [{"model": "cf", **{"neighbors": n, "prediction_shrinkage": s}, "ndcg": cf_means[(n, s)]}
             for n in grid["neighbors"] for s in grid["prediction_shrinkage"]]
            + [{"model": "enrichment", "neighbors": n, "prediction_shrinkage": s, "cf_weight": w, "ndcg": enr_means[(n, s, w)]}
               for n in grid["neighbors"] for s in grid["prediction_shrinkage"] for w in grid["cf_weight"]]
        )
        selection.update({
            "inner_val_users": len(eligible_val),
            "inner_selection_seconds": selection_seconds,
            "cf_selected": cf_cfg,
            "enrichment_selected": enr_cfg,
            "sweep": sweep_rows,
        })

    relevant_test = {}
    for user_id, movie_id, rating, _ in outer_test_rows:
        if rating >= threshold:
            relevant_test.setdefault(user_id, set()).add(movie_id)

    per_user = {"cf_tuned": {}, "cf_top100_content_enrichment": {}}
    coverage = {"cf_tuned": set(), "cf_top100_content_enrichment": set()}
    ranked_lists = {name: {} for name in per_user}
    pool_lists = {}

    score_started = time.perf_counter()
    for user_id in sorted(relevant_test):
        if user_filter is not None and user_id not in user_filter:
            continue
        ur = train_users.get(user_id, {})
        if not ur:
            continue
        relevant = relevant_test[user_id]
        cands = movies.keys() - ur.keys()
        key = f"user:{user_id}"

        cf_scores = _cf_scores(cf_model, ur, cands, cf_cfg["neighbors"], cf_cfg["prediction_shrinkage"])
        cf_recs = _top_ids(cf_scores, k)
        enr_scores, enr_pool = _pool_scores(
            cf_model, ur, cands,
            enr_cfg["neighbors"], enr_cfg["prediction_shrinkage"], enr_cfg["top_m"],
            movies, vectors, enr_cfg["cf_weight"], threshold,
        )
        enr_recs = _top_ids(enr_scores, k)

        for name, recs in (("cf_tuned", cf_recs), ("cf_top100_content_enrichment", enr_recs)):
            per_user[name][key] = evaluate_user_phase7(recs, relevant, k, vectors, popularity_counts)
            coverage[name].update(recs)
            ranked_lists[name][key] = recs
        pool_lists[key] = enr_pool
    score_seconds = time.perf_counter() - score_started

    catalog_size = len(movies)
    pooled = {name: summarize_users(vals, len(coverage[name]), catalog_size) for name, vals in per_user.items()}
    return {
        "selection": selection,
        "pooled": pooled,
        "per_user": per_user,
        "ranked_lists": ranked_lists,
        "pool_lists": pool_lists,
        "_coverage": coverage,
        "catalog_size": catalog_size,
        "outer_scoring_seconds": score_seconds,
        "popularity_counts": popularity_counts,
    }


def _top_ids(values, k):
    return nested._top_ids(values, k)


def _ndcg_of(recs, relevant, k):
    return evaluation.ndcg_at_k(recs, relevant, k)
