"""Phase 10 staged experiments: shared pools, relevance rankers, diversity layers, selection, outer eval."""

import time

from .. import content as content_mod
from ..data import group_by_user
from ..phase7.metrics import evaluate_user_phase7, summarize_users
from ..phase9.fusion import merge_candidates
from ..phase9.generators import CFCandidateGenerator, CandidateList, ContentCandidateGenerator, PopularityCandidateGenerator
from .diversity import genre_constraint_order, genre_redundancy_scores, mmr_scores, popularity_penalty_scores, rank_candidates
from .metrics import novelty_stats, popularity_concentration, unique_genre_representation
from .scoring import extract_components, normalize_components
from ..diagnostics import cf_top_m_candidates

POOL_M = 200
RETRIEVAL_MS = (100, 200, 500)
SEGMENT_CUTOFFS = (30, 80)
SEGMENT_NAMES = ("history_sparse", "history_medium", "history_dense")

RELEVANCE_CONFIGS = (
    {"name": "cf_only", "method": "cf_only", "pool": "cf"},
    {"name": "cf_only_pop", "method": "cf_only", "pool": "popularity"},
    {"name": "cf_only_cfpop", "method": "cf_only", "pool": "cf_popularity"},
    {"name": "popularity_only", "method": "popularity_only", "pool": "popularity"},
    {"name": "cf_pop_fallback", "method": "cf_pop_fallback", "pool": "popularity"},
    {"name": "combined_a25", "method": "combined", "params": {"alpha": 0.25}, "pool": "popularity"},
    {"name": "combined_a50", "method": "combined", "params": {"alpha": 0.50}, "pool": "popularity"},
    {"name": "combined_a75", "method": "combined", "params": {"alpha": 0.75}, "pool": "popularity"},
    {"name": "rrf_cf_pop", "method": "rrf", "params": {"rrf_k": 60}, "pool": "popularity"},
    {"name": "three_w1", "method": "three_signal", "params": {"weights": {"cf": 0.6, "popularity": 0.2, "content": 0.2}}, "pool": "cf_popularity"},
    {"name": "three_w2", "method": "three_signal", "params": {"weights": {"cf": 0.5, "popularity": 0.3, "content": 0.2}}, "pool": "cf_popularity"},
    {"name": "three_w3", "method": "three_signal", "params": {"weights": {"cf": 0.4, "popularity": 0.2, "content": 0.4}}, "pool": "cf_popularity"},
)

DIVERSITY_CONFIGS = (
    {"name": "none", "layer": None},
    {"name": "pop_penalty_l10", "layer": "popularity_penalty", "params": {"lam": 0.1}},
    {"name": "pop_penalty_l30", "layer": "popularity_penalty", "params": {"lam": 0.3}},
    {"name": "pop_penalty_l50", "layer": "popularity_penalty", "params": {"lam": 0.5}},
    {"name": "mmr_l75", "layer": "mmr", "params": {"lam": 0.75}},
    {"name": "mmr_l50", "layer": "mmr", "params": {"lam": 0.50}},
    {"name": "genre_div_l30", "layer": "genre_redundancy", "params": {"lam": 0.3}},
    {"name": "genre_div_l50", "layer": "genre_redundancy", "params": {"lam": 0.5}},
    {"name": "genre_cap_3", "layer": "genre_constraint", "params": {"max_per_primary": 3}},
    {"name": "genre_cap_5", "layer": "genre_constraint", "params": {"max_per_primary": 5}},
)


def segment_name(history_size):
    if history_size <= SEGMENT_CUTOFFS[0]:
        return "history_sparse"
    if history_size <= SEGMENT_CUTOFFS[1]:
        return "history_medium"
    return "history_dense"


def build_pools(train_ratings, movies, vectors, user_ratings, settings, pool_m=POOL_M, cf_gen=None, pop_gen=None, cont_gen=None):
    """Build the shared, frozen candidate pools for one user (identical IDs for every ranker)."""
    if cf_gen is None:
        cf_gen = CFCandidateGenerator(
            train_ratings,
            neighbors=settings.get("cf_tuning", {}).get("neighbors", 100),
            shrinkage=settings["cf"]["shrinkage"],
            prediction_shrinkage=settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0),
        )
    if pop_gen is None:
        pop_gen = PopularityCandidateGenerator(train_ratings)
    if cont_gen is None:
        cont_gen = ContentCandidateGenerator(movies, vectors)
    cf_list = cf_gen.generate(user_ratings, movies, pool_m)
    pop_list = pop_gen.generate(user_ratings, movies, pool_m)
    cont_list = cont_gen.generate(user_ratings, movies, pool_m)
    cf_pop_pool = merge_candidates([cf_list, pop_list], pool_m, mode="union", quotas={"cf": 0.5, "popularity": 0.5}).ids()
    pools = {
        "cf": cf_list.ids(),
        "popularity": pop_list.ids(),
        "cf_popularity": cf_pop_pool,
    }
    return pools, {"cf": cf_list, "popularity": pop_list, "content": cont_list}


def _score_pool(pool_ids, user_ratings, cf_scores, popularity_counts, vectors, movies):
    content_scores = content_mod.score(
        content_mod.user_profile(vectors, user_ratings) or {}, vectors, set(pool_ids)
    ) if user_ratings else {}
    components, coverage = extract_components(user_ratings, pool_ids, cf_scores, popularity_counts, content_scores, movies)
    norm = normalize_components(components)
    return components, norm, coverage


def evaluate_configurations(
    train_ratings,
    relevant_by_user,
    movies,
    vectors,
    settings,
    user_filter=None,
    relevance_configs=RELEVANCE_CONFIGS,
    diversity_configs=DIVERSITY_CONFIGS,
    pool_m=POOL_M,
    diversity_base=None,
):
    """Stage 2/3/6: build shared pools once per user; score every ranker on identical pools;
    apply diversity layers to the configured base relevance ranker. Returns per-user metrics."""
    cf_gen = CFCandidateGenerator(
        train_ratings,
        neighbors=settings.get("cf_tuning", {}).get("neighbors", 100),
        shrinkage=settings["cf"]["shrinkage"],
        prediction_shrinkage=settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0),
    )
    pop_gen = PopularityCandidateGenerator(train_ratings)
    cont_gen = ContentCandidateGenerator(movies, vectors)
    popularity_counts = dict(pop_gen.positive_counts)
    train_users = group_by_user(train_ratings)

    users = sorted(relevant_by_user)
    if user_filter:
        users = [u for u in users if u in user_filter]

    per_user = {}
    coverage_sets = {}
    scoring_diag = {}
    ranked_lists = {}
    diversity_base = diversity_base or {"name": "combined_a50", "method": "combined", "params": {"alpha": 0.5}, "pool": "popularity"}
    base_name = diversity_base["name"]

    t0 = time.perf_counter()
    pred_shrinkage = settings.get("cf_tuning", {}).get("prediction_shrinkage", 300.0)
    for user_id in users:
        rel = relevant_by_user[user_id]
        ur = train_users.get(user_id, {})
        if not ur:
            continue
        key = f"user:{user_id}"
        evidence = cf_gen.model.evidence(ur, movies.keys() - ur.keys())
        cf_scores = {
            mid: min(5.0, max(0.5, mean + num / (den + pred_shrinkage)))
            for mid, (mean, num, den) in evidence.items()
        }
        cf_items = sorted(cf_scores.items(), key=lambda row: (-row[1], row[0]))[:pool_m]
        cf_list = CandidateList("cf", tuple(cf_items))
        pop_list = pop_gen.generate(ur, movies, pool_m)
        cf_pop_pool = merge_candidates([cf_list, pop_list], pool_m, mode="union", quotas={"cf": 0.5, "popularity": 0.5}).ids()
        pools = {
            "cf": cf_list.ids(),
            "popularity": pop_list.ids(),
            "cf_popularity": cf_pop_pool,
        }

        pool_norms = {}
        pool_covs = {}
        for pool_name, pool_ids in pools.items():
            components, norm, coverage = _score_pool(pool_ids, ur, cf_scores, popularity_counts, vectors, movies)
            pool_norms[pool_name] = norm
            pool_covs[pool_name] = coverage
            acc = scoring_diag.setdefault(f"pool:{pool_name}", {
                "pool": pool_name, "users": 0, "candidates": 0, "cf_scored": 0,
                "popularity_scored": 0, "content_scored": 0, "missing_content": 0,
                "cf_raw_sum": 0.0, "cf_raw_count": 0, "pop_raw_sum": 0.0, "content_raw_sum": 0.0,
            })
            acc["users"] += 1
            acc["candidates"] += coverage["candidates"]
            acc["cf_scored"] += coverage["cf_scored"]
            acc["popularity_scored"] += coverage["popularity_scored"]
            acc["content_scored"] += coverage["content_scored"]
            acc["missing_content"] += coverage["missing_content"]
            acc["cf_raw_sum"] += sum(components["cf"].values())
            acc["cf_raw_count"] += len(components["cf"])
            acc["pop_raw_sum"] += sum(components["popularity"].values())
            acc["content_raw_sum"] += sum(components["content"].values())

        ranked_lists_this_user = {}
        for cfg in relevance_configs:
            pool_ids = pools[cfg["pool"]]
            recs, scores, diag = rank_candidates(pool_ids, pool_norms[cfg["pool"]], cfg["method"], cfg.get("params"))
            ranked_lists_this_user[cfg["name"]] = recs
            per_user.setdefault(cfg["name"], {})[key] = evaluate_user_phase7(recs, rel, 10, vectors, popularity_counts)
            coverage_sets.setdefault(cfg["name"], set()).update(recs)
            macc = scoring_diag.setdefault(cfg["name"], {
                "pool": cfg["pool"], "users": 0, "candidates": 0, "ranked_candidates": 0,
                "fallback_candidates": 0, "dropped_candidates": 0,
            })
            macc["users"] += 1
            macc["candidates"] += pool_covs[cfg["pool"]]["candidates"]
            macc["ranked_candidates"] += diag.get("ranked_candidates", 0)
            macc["fallback_candidates"] += diag.get("fallback_candidates", 0)
            macc["dropped_candidates"] += diag.get("dropped_candidates", 0)

        # Diversity layers applied to the configured base relevance ranker's full ordering.
        base_cfg = next(c for c in relevance_configs if c["name"] == base_name)
        base_pool = pools[base_cfg["pool"]]
        _, base_scores, _ = rank_candidates(base_pool, pool_norms[base_cfg["pool"]], base_cfg["method"], base_cfg.get("params"))

        for cfg in diversity_configs:
            if cfg["layer"] is None:
                recs = _order(base_scores, k=10)
            elif cfg["layer"] == "popularity_penalty":
                adjusted = popularity_penalty_scores(base_scores, popularity_counts, base_pool, cfg["params"]["lam"])
                recs = _order(adjusted, k=10)
            elif cfg["layer"] == "mmr":
                _, recs = mmr_scores(base_scores, base_pool, movies, cfg["params"]["lam"], k=10)
            elif cfg["layer"] == "genre_redundancy":
                adjusted = genre_redundancy_scores(base_scores, base_pool, movies, cfg["params"]["lam"], k=10)
                recs = list(adjusted.keys())[:10]
            elif cfg["layer"] == "genre_constraint":
                recs = genre_constraint_order(base_scores, movies, cfg["params"]["max_per_primary"], k=10)
            else:
                raise ValueError(cfg["layer"])
            name = f"{base_name}+{cfg['name']}"
            ranked_lists_this_user[name] = recs
            per_user.setdefault(name, {})[key] = evaluate_user_phase7(recs, rel, 10, vectors, popularity_counts)
            coverage_sets.setdefault(name, set()).update(recs)

        for name, recs in ranked_lists_this_user.items():
            ranked_lists.setdefault(name, []).append(recs)

    wall = time.perf_counter() - t0
    ranking_summary = {
        name: summarize_users(vals, len(coverage_sets[name]), len(movies))
        for name, vals in per_user.items()
    }
    novelty = {
        name: novelty_stats(lists, popularity_counts)
        for name, lists in ranked_lists.items()
    }
    genre_repr = {
        name: unique_genre_representation(lists, movies)
        for name, lists in ranked_lists.items()
    }
    pop_conc = {
        name: popularity_concentration(lists, popularity_counts)
        for name, lists in ranked_lists.items()
    }
    final_diag = {}
    for k_diag, acc in scoring_diag.items():
        cands = acc["candidates"] or 1
        if k_diag.startswith("pool:"):
            final_diag[k_diag] = {
                "pool": acc["pool"],
                "users": acc["users"],
                "total_candidates": acc["candidates"],
                "cf_scored_pct": acc["cf_scored"] / cands,
                "popularity_scored_pct": acc["popularity_scored"] / cands,
                "content_scored_pct": acc["content_scored"] / cands,
                "missing_cf_pct": (acc["candidates"] - acc["cf_scored"]) / cands,
                "missing_content_pct": acc["missing_content"] / cands,
                "mean_raw_cf": acc["cf_raw_sum"] / (acc["cf_raw_count"] or 1),
                "mean_raw_popularity": acc["pop_raw_sum"] / cands,
                "mean_raw_content": acc["content_raw_sum"] / (acc["content_scored"] or 1),
            }
        else:
            final_diag[k_diag] = {
                "pool": acc["pool"],
                "users": acc["users"],
                "total_candidates": acc["candidates"],
                "ranked_pct": acc["ranked_candidates"] / cands,
                "fallback_pct": acc["fallback_candidates"] / cands,
                "dropped_pct": acc["dropped_candidates"] / cands,
            }
    return {
        "per_user": per_user,
        "summary": ranking_summary,
        "novelty": novelty,
        "genre_representation": genre_repr,
        "popularity_concentration": pop_conc,
        "scoring_diagnostics": final_diag,
        "wall_seconds": wall,
        "users": len(per_user.get(base_name, {})),
        "popularity_counts": popularity_counts,
        "ranked_lists": ranked_lists,
    }


def order_from_scores(scores, k=10):
    return _order(scores, k)


def _order(scores, k=10):
    return [mid for mid, _ in sorted(scores.items(), key=lambda row: (-row[1], row[0]))][:k]


def selection_on_inner(inner_results, relevance_configs=RELEVANCE_CONFIGS, diversity_configs=DIVERSITY_CONFIGS):
    """Stage 4: select best-accuracy and best-balanced configs using inner validation only."""
    summary = inner_results["summary"]
    novelty = inner_results["novelty"]
    accuracy_rank = sorted(
        (name for name in summary),
        key=lambda name: -summary[name]["ndcg"],
    )
    best_accuracy = accuracy_rank[0] if accuracy_rank else None

    def zscores(names, values):
        vals = [values[name] for name in names]
        mean = sum(vals) / len(vals)
        var = sum((v - mean) ** 2 for v in vals) / len(vals)
        std = var ** 0.5 or 1.0
        return {name: (values[name] - mean) / std for name in names}

    names = list(summary)
    z_ndcg = zscores(names, {name: summary[name]["ndcg"] for name in names})
    z_cov = zscores(names, {name: summary[name]["coverage"] for name in names})
    z_nov = zscores(names, {name: novelty[name]["novelty"] for name in names})
    balanced = {
        name: (z_ndcg[name] + z_cov[name] + z_nov[name]) / 3.0
        for name in names
    }
    best_balanced = max(names, key=lambda name: balanced[name])
    return {
        "best_accuracy": best_accuracy,
        "best_balanced": best_balanced,
        "balanced_scores": balanced,
        "accuracy_ranking": accuracy_rank,
        "criterion": "best-accuracy = max inner NDCG@10; best-balanced = max mean z-score of (inner NDCG@10, recommendation catalog coverage, novelty)",
    }


def segment_analysis(outer_train, per_user):
    sizes = {}
    for user_id, movie_id, rating, ts in outer_train:
        sizes[user_id] = sizes.get(user_id, 0) + 1
    segments = {}
    reference = next(iter(per_user))
    for seg in SEGMENT_NAMES:
        keys = {
            f"user:{uid}"
            for uid, size in sizes.items()
            if segment_name(size) == seg and f"user:{uid}" in per_user[reference]
        }
        seg_doc = {"users": len(keys)}
        for name, rows in per_user.items():
            sub = [rows[k] for k in keys if k in rows]
            seg_doc[name] = {
                "ndcg": sum(r["ndcg"] for r in sub) / len(sub) if sub else None,
                "map": sum(r["map"] for r in sub) / len(sub) if sub else None,
                "hr": sum(r["hr"] for r in sub) / len(sub) if sub else None,
                "ild": sum(r["ild"] for r in sub) / len(sub) if sub else None,
            }
        segments[seg] = seg_doc
    return segments
