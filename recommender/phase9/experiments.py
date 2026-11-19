"""Phase 9 experiment orchestration: inner selection, outer evaluation, ranking, segments, bootstrap."""

import time

from .. import content as content_mod
from .. import diagnostics, evaluation
from ..data import group_by_user
from ..phase7.metrics import evaluate_user_phase7, paired_user_bootstrap, summarize_users
from .fusion import merge_candidates
from .generators import CFCandidateGenerator, ContentCandidateGenerator, PopularityCandidateGenerator
from .metrics import (
    candidate_precision,
    exclusion_summary,
    genre_diversity,
    pool_overlap_sample,
    recall_at_m,
    user_retrieval_coverage,
)

POOL_SIZES = (100, 200, 500, 1000)
EVAL_M = 200          # pool size used for downstream ranking and family selection
SEGMENT_CUTOFFS = (30, 80)

FAMILY_CONFIGS = {
    "cf_popularity": {"mode": "union", "quotas": {"cf": 0.5, "popularity": 0.5}},
    "cf_content": {"mode": "union", "quotas": {"cf": 0.5, "content": 0.5}},
    "three_way": {"mode": "union", "quotas": {"cf": 0.5, "popularity": 0.25, "content": 0.25}},
}

SEGMENT_NAMES = ("history_sparse", "history_medium", "history_dense")


def segment_name(history_size):
    if history_size <= SEGMENT_CUTOFFS[0]:
        return "history_sparse"
    if history_size <= SEGMENT_CUTOFFS[1]:
        return "history_medium"
    return "history_dense"


def _recall_stats(pool_ids_by_user, relevant_by_user):
    recall = {}
    totals = {}
    precision_sum = 0.0
    for key, pool in pool_ids_by_user.items():
        relevant = relevant_by_user.get(key)
        if relevant is None:
            continue
        recall[key] = recall_at_m(pool, relevant)
        totals[key] = len(relevant)
        precision_sum += candidate_precision(pool, relevant)
    return {
        "recall_by_user": recall,
        "total_relevant_by_user": totals,
        "mean_recall": (sum(recall.values()) / len(recall)) if recall else 0.0,
        "user_coverage": user_retrieval_coverage(recall),
        "mean_candidate_precision": precision_sum / len(pool_ids_by_user) if pool_ids_by_user else 0.0,
        "exclusion": exclusion_summary(recall, totals),
        "users": len(recall),
    }


def _source_relevant_counts(lists, relevant):
    """Relevant items retrieved by each source before fusion (set-level)."""
    counts = {}
    for cl in lists:
        ids = set(cl.ids())
        counts[cl.source] = len(ids & relevant)
    return counts


def inner_validation_study(
    outer_train,
    movies,
    vectors,
    settings,
    inner_fraction=0.2,
    threshold=4.0,
    eval_m=EVAL_M,
    pool_sizes=POOL_SIZES,
    user_filter=None,
):
    """Stage 5: compare all retrieval methods on inner validation; select per-family configs."""
    from ..phase7.temporal import split_inner_chronological

    inner_train, validation = split_inner_chronological(outer_train, inner_fraction)
    inner_users = group_by_user(inner_train)
    relevant = {}
    for user_id, movie_id, rating, _ in validation:
        if rating >= threshold:
            relevant.setdefault(user_id, set()).add(movie_id)
    eligible = {u: rel for u, rel in relevant.items() if inner_users.get(u)}
    if user_filter is not None:
        eligible = {u: rel for u, rel in eligible.items() if u in user_filter}

    cf_gen = CFCandidateGenerator(
        inner_train,
        neighbors=settings.get("cf_tuning", {}).get("neighbors", 100),
        shrinkage=settings["cf"]["shrinkage"],
        prediction_shrinkage=settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0),
    )
    pop_gen = PopularityCandidateGenerator(inner_train)
    content_gen = ContentCandidateGenerator(movies, vectors)
    catalog = movies

    results = {}
    source_diag = {}
    t0 = time.perf_counter()
    for user_id in sorted(eligible):
        rel = eligible[user_id]
        ur = inner_users[user_id]
        cf_list = cf_gen.generate(ur, catalog, max(pool_sizes))
        pop_list = pop_gen.generate(ur, catalog, max(pool_sizes))
        cont_list = content_gen.generate(ur, catalog, max(pool_sizes))

        for method, pool in (
            ("cf", cf_list),
            ("popularity", pop_list),
            ("content", cont_list),
        ):
            for m in pool_sizes:
                key = f"{method}@{m}"
                ids = pool.ids(m)
                entry = results.setdefault(key, {"recall": {}, "totals": {}, "precision_sum": 0.0})
                entry["recall"][user_id] = recall_at_m(ids, rel)
                entry["totals"][user_id] = len(rel)
                entry["precision_sum"] += candidate_precision(ids, rel)

        for family, cfg in FAMILY_CONFIGS.items():
            sources = [cf_list, pop_list] if family == "cf_popularity" else (
                [cf_list, cont_list] if family == "cf_content" else [cf_list, pop_list, cont_list]
            )
            fused = merge_candidates(sources, max(pool_sizes), mode=cfg["mode"], quotas=cfg["quotas"])
            source_diag.setdefault(family, {}).setdefault("totals", {"contributed": {}, "in_final": {}, "unique": {}, "overlap": {}})
            diag = source_diag[family]["totals"]
            for src, info in fused.attribution.items():
                diag["contributed"][src] = diag["contributed"].get(src, 0) + info["contributed_candidates"]
                diag["in_final"][src] = diag["in_final"].get(src, 0) + info["in_final_pool"]
                diag["unique"][src] = diag["unique"].get(src, 0) + info["unique_to_source_before_cap"]
                diag["overlap"][src] = diag["overlap"].get(src, 0) + info["overlapping_with_other_sources"]
            sr = _source_relevant_counts(sources, rel)
            final_ids = set(fused.ids())
            source_diag[family].setdefault("relevant_by_source", {})
            for src, cnt in sr.items():
                source_diag[family]["relevant_by_source"].setdefault(src, 0)
                source_diag[family]["relevant_by_source"][src] += cnt
            source_diag[family]["relevant_only_via_fusion"] = source_diag[family].get("relevant_only_via_fusion", 0) + len(
                (final_ids - set(cf_list.ids()) - set(pop_list.ids()) - set(cont_list.ids())) & rel
            )
            for m in pool_sizes:
                key = f"{family}@{m}"
                ids = fused.ids()[:m]
                entry = results.setdefault(key, {"recall": {}, "totals": {}, "precision_sum": 0.0})
                entry["recall"][user_id] = recall_at_m(ids, rel)
                entry["totals"][user_id] = len(rel)
                entry["precision_sum"] += candidate_precision(ids, rel)

    wall = time.perf_counter() - t0

    summary = {}
    for key, entry in results.items():
        n = len(entry["recall"])
        recall_by_user = entry["recall"]
        totals = entry["totals"]
        summary[key] = {
            "mean_recall": sum(recall_by_user.values()) / n if n else 0.0,
            "user_coverage": user_retrieval_coverage(recall_by_user),
            "mean_candidate_precision": entry["precision_sum"] / n if n else 0.0,
            "exclusion": exclusion_summary(recall_by_user, totals),
            "users": n,
        }
    selection = {family: f"{family}@{eval_m}" for family in FAMILY_CONFIGS}
    best_single = max(
        (key for key in summary if key.startswith(("cf@", "popularity@", "content@")) and key.endswith(f"@{eval_m}")),
        key=lambda key: summary[key]["mean_recall"],
        default="cf@200",
    )
    selection["best_single_at_eval_m"] = best_single
    return {
        "summary": summary,
        "selection": selection,
        "source_diagnostics": source_diag,
        "wall_seconds": wall,
        "inner_eligible_users": len(eligible),
    }


def outer_evaluation(
    outer_train,
    outer_test_rows,
    movies,
    vectors,
    settings,
    selected,
    k=10,
    threshold=4.0,
    eval_m=EVAL_M,
    user_filter=None,
):
    """Stage 6: refit generators on outer_train; evaluate retrieval + downstream ranking once."""
    train_users = group_by_user(outer_train)
    relevant = {}
    for user_id, movie_id, rating, _ in outer_test_rows:
        if rating >= threshold:
            relevant.setdefault(user_id, set()).add(movie_id)

    cf_tuned_gen = CFCandidateGenerator(
        outer_train,
        neighbors=settings.get("cf_tuning", {}).get("neighbors", 100),
        shrinkage=settings["cf"]["shrinkage"],
        prediction_shrinkage=settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0),
    )
    pop_gen = PopularityCandidateGenerator(outer_train)
    content_gen = ContentCandidateGenerator(movies, vectors)

    popularity_counts = {}
    for _, movie_id, rating, _ in outer_train:
        if rating >= threshold:
            popularity_counts[movie_id] = popularity_counts.get(movie_id, 0) + 1

    retrieval_stats = {}
    per_user_ranking = {}
    coverage_sets = {}
    ranked_lists = {}
    pool_records = {}
    t0 = time.perf_counter()
    users = sorted(relevant)
    if user_filter is not None:
        users = [u for u in users if u in user_filter]

    for user_id in users:
        rel = relevant[user_id]
        ur = train_users.get(user_id, {})
        if not ur:
            continue
        key = f"user:{user_id}"
        cf_list = cf_tuned_gen.generate(ur, movies, eval_m)
        pop_list = pop_gen.generate(ur, movies, eval_m)
        cont_list = content_gen.generate(ur, movies, eval_m)

        pools = {
            "cf@200": cf_list.ids(),
            "popularity@200": pop_list.ids(),
            "content@200": cont_list.ids(),
            "cf_popularity@200": merge_candidates([cf_list, pop_list], eval_m, mode="union", quotas=FAMILY_CONFIGS["cf_popularity"]["quotas"]).ids(),
            "cf_content@200": merge_candidates([cf_list, cont_list], eval_m, mode="union", quotas=FAMILY_CONFIGS["cf_content"]["quotas"]).ids(),
            "three_way@200": merge_candidates([cf_list, pop_list, cont_list], eval_m, mode="union", quotas=FAMILY_CONFIGS["three_way"]["quotas"]).ids(),
        }
        best_family_key = selected.get("best_family", "cf_content") + "@200"
        if best_family_key not in pools:
            best_family_key = "cf_content@200"

        for name, ids in pools.items():
            st = retrieval_stats.setdefault(name, {"recall": {}, "totals": {}, "precision_sum": 0.0, "pools": {}})
            st["recall"][key] = recall_at_m(ids, rel)
            st["totals"][key] = len(rel)
            st["precision_sum"] += candidate_precision(ids, rel)
            st["pools"][key] = ids

        # Downstream ranking: cf_ranker_cf_pool vs cf_ranker_best_pool isolates the retrieval
        # effect (same ranker, different pools); enrichment_ranker_best_pool vs cf_ranker_best_pool
        # isolates the ranking effect (same pool, different rankers). Rankers can only score
        # CF-evidence items; non-CF pool members are unrankable by the existing rankers.
        evidence = cf_tuned_gen.model.evidence(ur, movies.keys() - ur.keys())
        cf_scores = {
            movie_id: min(5.0, max(0.5, mean + num / (den + 300.0)))
            for movie_id, (mean, num, den) in evidence.items()
        }
        pool_for_ranking = pools[best_family_key]
        cf_pool = pools["cf@200"]
        # Enrichment-tuned ranker: CF + enriched content, normalized within the pool (Phase 6).
        enriched_feats = diagnostics.enriched_content_scores(ur, movies, vectors, pool_for_ranking, threshold)
        enr_scores = diagnostics.rerank_pool_scores(pool_for_ranking, cf_scores, enriched_feats, 0.7)

        rank_tasks = {
            "cf_ranker_cf_pool": {mid: cf_scores[mid] for mid in cf_pool if mid in cf_scores},
            "cf_ranker_best_pool": {mid: cf_scores[mid] for mid in pool_for_ranking if mid in cf_scores},
            "enrichment_ranker_best_pool": enr_scores,
        }
        for ranker_name, score_map in rank_tasks.items():
            recs = [mid for mid, _ in sorted(score_map.items(), key=lambda row: (-row[1], row[0]))][:k]
            metrics = evaluate_user_phase7(recs, rel, k, vectors, popularity_counts)
            per_user_ranking.setdefault(ranker_name, {})[key] = metrics
            coverage_sets.setdefault(ranker_name, set()).update(recs)
            ranked_lists.setdefault(ranker_name, {})[key] = recs

        pool_records[key] = {
            "pool_sizes": {name: len(ids) for name, ids in pools.items()},
            "cf_scored_fraction": len([mid for mid in pool_for_ranking if mid in cf_scores]) / len(pool_for_ranking) if pool_for_ranking else 0.0,
            "best_pool_hit": len(set(pool_for_ranking) & rel),
        }

    wall = time.perf_counter() - t0

    retrieval_summary = {}
    for name, st in retrieval_stats.items():
        recall_by_user = st["recall"]
        totals = st["totals"]
        n = len(recall_by_user)
        div_sample = {}
        sample_keys = sorted(st["pools"])[:300]
        div_pair = genre_diversity(st["pools"][sample_keys[0]], movies) if sample_keys else {"pairwise_genre_dissimilarity": 0.0, "unique_genres": 0}
        overlap = pool_overlap_sample({key: ids for key, ids in st["pools"].items()})
        retrieval_summary[name] = {
            "mean_recall": sum(recall_by_user.values()) / n if n else 0.0,
            "user_coverage": user_retrieval_coverage(recall_by_user),
            "mean_candidate_precision": st["precision_sum"] / n if n else 0.0,
            "exclusion": exclusion_summary(recall_by_user, totals),
            "genre_diversity_sample": div_pair,
            "pool_overlap": overlap,
            "users": n,
        }

    ranking_summary = {
        name: summarize_users(vals, len(coverage_sets[name]), len(movies))
        for name, vals in per_user_ranking.items()
    }
    return {
        "retrieval": retrieval_summary,
        "retrieval_per_user": {name: st["recall"] for name, st in retrieval_stats.items()},
        "pool_records": pool_records,
        "ranking_per_user": per_user_ranking,
        "ranking_summary": ranking_summary,
        "ranked_lists": ranked_lists,
        "wall_seconds": wall,
        "evaluated_users": len(per_user_ranking.get("cf_ranker_cf_pool", {})),
    }


def segment_analysis(outer_train, retrieval_per_user, ranking, eval_m=EVAL_M):
    """Stage 7a: training-only history-size segments for retrieval and ranking.

    `retrieval_per_user` maps method name -> {user_key: recall}.
    """
    sizes = {}
    for user_id, movie_id, rating, ts in outer_train:
        sizes[user_id] = sizes.get(user_id, 0) + 1
    segments = {}
    for seg in SEGMENT_NAMES:
        keys = {
            f"user:{uid}"
            for uid, size in sizes.items()
            if segment_name(size) == seg and f"user:{uid}" in ranking.get("cf_ranker_cf_pool", {})
        }
        seg_doc = {"users": len(keys)}
        for name, recall in retrieval_per_user.items():
            sub = [recall[k2] for k2 in keys if k2 in recall]
            seg_doc[name] = {
                "mean_recall": sum(sub) / len(sub) if sub else None,
                "user_coverage": user_retrieval_coverage({k2: recall[k2] for k2 in keys if k2 in recall}),
            }
        for ranker, rows in ranking.items():
            sub = [rows[k2] for k2 in keys if k2 in rows]
            seg_doc[ranker] = {
                "ndcg": sum(r["ndcg"] for r in sub) / len(sub) if sub else None,
                "hr": sum(r["hr"] for r in sub) / len(sub) if sub else None,
                "ild": sum(r["ild"] for r in sub) / len(sub) if sub else None,
            }
        segments[seg] = seg_doc
    return segments


def bootstrap_comparisons(retrieval_per_user, ranking_per_user, resamples=2000, confidence=0.95, seed=2026):
    """Stage 7b: paired user-level bootstrap on recall and downstream NDCG."""
    comparisons = {}
    method_keys = [name for name in retrieval_per_user if name != "cf@200"]
    cf_recall = retrieval_per_user.get("cf@200", {})
    for name in method_keys:
        other = retrieval_per_user[name]
        recall_left = {k2: {"recall": v2} for k2, v2 in other.items()}
        recall_right = {k2: {"recall": v2} for k2, v2 in cf_recall.items()}
        comparisons[f"retrieval_{name}_vs_cf200_recall"] = paired_user_bootstrap(
            recall_left, recall_right, metric="recall", resamples=resamples, confidence=confidence, seed=seed,
        )
    if "cf_ranker_best_pool" in ranking_per_user and "cf_ranker_cf_pool" in ranking_per_user:
        comparisons["ranking_cf_ranker_best_pool_vs_cf_ranker_cf_pool_ndcg"] = paired_user_bootstrap(
            ranking_per_user["cf_ranker_best_pool"], ranking_per_user["cf_ranker_cf_pool"], metric="ndcg",
            resamples=resamples, confidence=confidence, seed=seed,
        )
    if "enrichment_ranker_best_pool" in ranking_per_user and "cf_ranker_best_pool" in ranking_per_user:
        comparisons["ranking_enrichment_ranker_best_pool_vs_cf_ranker_best_pool_ndcg"] = paired_user_bootstrap(
            ranking_per_user["enrichment_ranker_best_pool"], ranking_per_user["cf_ranker_best_pool"], metric="ndcg",
            resamples=resamples, confidence=confidence, seed=seed,
        )
    return comparisons
