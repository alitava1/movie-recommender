"""Phase 10 audit of Phase 9 source-contribution claims using actual per-user candidate sets."""

import statistics

from ..data import group_by_user
from ..phase9.generators import CFCandidateGenerator, PopularityCandidateGenerator
from ..phase9.fusion import merge_candidates

POOL_M = 200


def _relevant_counts(pool_ids, relevant):
    hits = set(pool_ids) & relevant
    return len(hits)


def audit_phase9_sources(
    outer_train,
    movies,
    settings,
    inner_fraction=0.2,
    threshold=4.0,
    pool_m=POOL_M,
    user_filter=None,
    cf_neighbors=100,
    cf_prediction_shrinkage=300.0,
):
    """Per-user set-intersection audit of CF vs popularity retrieval on inner validation.

    Rebuilds the exact Phase 9 generators (same signals, same truncation semantics) on
    inner-train only and evaluates against the same inner-validation relevant items.
    """
    from ..phase7.temporal import split_inner_chronological

    inner_train, validation = split_inner_chronological(outer_train, inner_fraction)
    inner_users = group_by_user(inner_train)
    relevant = {}
    for user_id, movie_id, rating, _ in validation:
        if rating >= threshold:
            relevant.setdefault(user_id, set()).add(movie_id)
    eligible = {u: rel for u, rel in relevant.items() if inner_users.get(u)}
    if user_filter:
        eligible = {u: rel for u, rel in eligible.items() if u in user_filter}

    cf_gen = CFCandidateGenerator(
        inner_train, neighbors=cf_neighbors, shrinkage=settings["cf"]["shrinkage"],
        prediction_shrinkage=cf_prediction_shrinkage,
    )
    pop_gen = PopularityCandidateGenerator(inner_train)

    agg = {
        "users": 0,
        "relevant_cf": 0, "relevant_pop": 0, "relevant_intersection": 0,
        "relevant_pop_only": 0, "relevant_cf_only": 0, "relevant_union": 0,
        "relevant_quota_pool": 0,
        "pool_size_cf": 0, "pool_size_pop": 0, "pool_size_union": 0, "pool_size_quota": 0,
        "cf_scored_fraction_union_sum": 0.0,
        "cf_scored_fraction_pop_sum": 0.0,
        "cf_scored_fraction_pop_min": 1.0,
        "cf_scored_fraction_quota_sum": 0.0,
        "union_recall_values": [], "cf_recall_values": [], "pop_recall_values": [],
        "quota_recall_values": [],
    }
    pop_only_examples = []
    for user_id in sorted(eligible):
        rel = eligible[user_id]
        ur = inner_users[user_id]
        cf_list = cf_gen.generate(ur, movies, pool_m)
        pop_list = pop_gen.generate(ur, movies, pool_m)
        c_cf = set(cf_list.ids())
        c_pop = set(pop_list.ids())
        c_union = c_cf | c_pop
        fused = merge_candidates([cf_list, pop_list], pool_m, mode="union", quotas={"cf": 0.5, "popularity": 0.5})
        c_quota = set(fused.ids())

        r_cf, r_pop = c_cf & rel, c_pop & rel
        agg["users"] += 1
        agg["relevant_cf"] += len(r_cf)
        agg["relevant_pop"] += len(r_pop)
        agg["relevant_intersection"] += len(r_cf & r_pop)
        agg["relevant_pop_only"] += len(r_pop - r_cf)
        agg["relevant_cf_only"] += len(r_cf - r_pop)
        agg["relevant_union"] += len(c_union & rel)
        agg["relevant_quota_pool"] += len(c_quota & rel)
        agg["pool_size_cf"] += len(c_cf)
        agg["pool_size_pop"] += len(c_pop)
        agg["pool_size_union"] += len(c_union)
        agg["pool_size_quota"] += len(c_quota)
        ev_keys = set(cf_gen.model.evidence(ur, c_union).keys())
        agg["cf_scored_fraction_union_sum"] += len(c_union & ev_keys) / len(c_union) if c_union else 0.0
        pop_frac = len(c_pop & ev_keys) / len(c_pop) if c_pop else 0.0
        agg["cf_scored_fraction_pop_sum"] += pop_frac
        if c_pop and pop_frac < agg["cf_scored_fraction_pop_min"]:
            agg["cf_scored_fraction_pop_min"] = pop_frac
        agg["cf_scored_fraction_quota_sum"] += len(c_quota & ev_keys) / len(c_quota) if c_quota else 0.0
        n_rel = len(rel)
        agg["union_recall_values"].append(len(c_union & rel) / n_rel)
        agg["cf_recall_values"].append(len(r_cf) / n_rel)
        agg["pop_recall_values"].append(len(r_pop) / n_rel)
        agg["quota_recall_values"].append(len(c_quota & rel) / n_rel)
        if len(pop_only_examples) < 5 and (r_pop - r_cf):
            pop_only_examples.append({
                "user_id": user_id,
                "relevant_pop_only_movies": sorted(r_pop - r_cf)[:10],
                "cf_pool_size": len(c_cf),
                "pop_pool_size": len(c_pop),
            })

    n = agg["users"] or 1
    totals_rel = sum(len(r) for r in eligible.values())
    return {
        "protocol": "Phase 9 generators rebuilt on inner-train only; identical inner-validation relevant denominators; pool M=200",
        "pool_size_definition": "CandidateList.ids(M) truncates each generator's own full ranking to M BEFORE fusion; the 50/50 quota pool merges those truncated source lists and applies the M cap post-fusion, so quota != unrestricted union",
        "cf_scored_fraction_definition": "|pool items with CF evidence| / |pool| per user, reported for popularity, quota, and union pools",
        "users": agg["users"],
        "total_relevant_items": totals_rel,
        "mean_relevant_per_user": totals_rel / n,
        "relevant_cf": agg["relevant_cf"],
        "relevant_pop": agg["relevant_pop"],
        "relevant_intersection": agg["relevant_intersection"],
        "relevant_pop_only": agg["relevant_pop_only"],
        "relevant_cf_only": agg["relevant_cf_only"],
        "relevant_union": agg["relevant_union"],
        "relevant_quota_pool": agg["relevant_quota_pool"],
        "recall_union": statistics.mean(agg["union_recall_values"]),
        "recall_cf": statistics.mean(agg["cf_recall_values"]),
        "recall_pop": statistics.mean(agg["pop_recall_values"]),
        "recall_quota_pool": statistics.mean(agg["quota_recall_values"]),
        "mean_pool_size_cf": agg["pool_size_cf"] / n,
        "mean_pool_size_pop": agg["pool_size_pop"] / n,
        "mean_pool_size_union": agg["pool_size_union"] / n,
        "mean_pool_size_quota": agg["pool_size_quota"] / n,
        "mean_cf_scored_fraction_union_pool": agg["cf_scored_fraction_union_sum"] / n,
        "mean_cf_scored_fraction_pop_pool": agg["cf_scored_fraction_pop_sum"] / n,
        "min_cf_scored_fraction_pop_pool": agg["cf_scored_fraction_pop_min"] if agg["users"] else 0.0,
        "mean_cf_scored_fraction_quota_pool": agg["cf_scored_fraction_quota_sum"] / n,
        "pop_only_examples": pop_only_examples,
    }
