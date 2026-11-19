"""Phase 10 rankers (Methods A-G) and diversity rerankers (Methods H-K)."""

from .scoring import combined_score, normalize_components, rrf_score


def _order(score_map, k=10):
    return [mid for mid, _ in sorted(score_map.items(), key=lambda row: (-row[1], row[0]))][:k]


def rank_candidates(pool_ids, norm_components, method, params=None):
    """Common ranking interface. Returns (top_ids, full_order_scores, diagnostics).

    Every ranker scores the complete pool (see per-method missing-signal handling);
    `cf_only` is the historical exception that ranks only CF-scored candidates.
    """
    params = params or {}
    pool = list(dict.fromkeys(pool_ids))
    if method == "cf_only":
        scores = dict(norm_components.get("cf", {}))
        diag = {"ranked_candidates": len(scores), "dropped_candidates": len(pool) - len(scores)}
    elif method == "popularity_only":
        scores = {mid: norm_components["popularity"].get(mid, 0.0) for mid in pool}
        diag = {"ranked_candidates": len(scores), "dropped_candidates": 0}
    elif method == "cf_pop_fallback":
        cf = norm_components.get("cf", {})
        pop = norm_components["popularity"]
        scores = {mid: cf.get(mid, pop.get(mid, 0.0)) for mid in pool}
        diag = {"ranked_candidates": len(scores), "fallback_candidates": len(pool) - len(cf)}
    elif method == "combined":
        alpha = float(params.get("alpha", 0.5))
        scores = combined_score(
            pool,
            {"cf": norm_components.get("cf", {}), "popularity": norm_components["popularity"]},
            {"cf": alpha, "popularity": 1.0 - alpha},
        )
        diag = {"ranked_candidates": len(scores), "alpha": alpha}
    elif method == "three_signal":
        weights = params.get("weights", {"cf": 0.6, "popularity": 0.2, "content": 0.2})
        scores = combined_score(pool, norm_components, weights)
        diag = {"ranked_candidates": len(scores), "weights": dict(weights)}
    elif method == "rrf":
        k = int(params.get("rrf_k", 60))
        scores = rrf_score({name: vals for name, vals in norm_components.items() if vals}, rrf_k=k)
        scores = {mid: scores.get(mid, 0.0) for mid in pool}
        diag = {"ranked_candidates": len(scores), "rrf_k": k}
    else:
        raise ValueError(f"unknown ranking method {method!r}")
    return _order(scores), scores, diag


def popularity_penalty_scores(relevance_scores, popularity_counts, pool_ids, lam):
    """Method H: adjusted = relevance - lam * normalized log-popularity penalty."""
    import math

    logs = {mid: math.log1p(popularity_counts.get(mid, 0)) for mid in pool_ids}
    norm = _minmax_zero_one(logs)
    return {mid: relevance_scores[mid] - lam * norm[mid] for mid in pool_ids}


def genre_redundancy_scores(relevance_scores, pool_ids, movies, lam, selected=None, k=None):
    """Method I: greedy genre-diversity reranking with Jaccard redundancy penalty.

    Picks argmax(norm_relevance - lam * max_jaccard_to_selected) one item at a time;
    returns {movie_id: adjusted_score_at_pick} with the final selection order as dict order.
    """
    relevance_norm = _minmax_zero_one(relevance_scores)
    order = sorted(relevance_norm.items(), key=lambda row: (-row[1], row[0]))
    pool_order = [mid for mid, _ in order]
    gsets = {mid: (set(movies[mid].genres) if mid in movies else set()) for mid in pool_order}
    max_sim = {mid: 0.0 for mid in pool_order}
    for sel in (selected or []):
        sset = set(movies[sel].genres) if sel in movies else set()
        for mid in pool_order:
            gset = gsets[mid]
            union = gset | sset
            if union:
                sim = len(gset & sset) / len(union)
                if sim > max_sim[mid]:
                    max_sim[mid] = sim

    remaining = set(pool_order)
    limit = len(remaining) if k is None else min(k, len(remaining))
    out = {}
    for _ in range(limit):
        best_mid, best_key, best_score = None, None, None
        for mid in pool_order:
            if mid not in remaining:
                continue
            score = relevance_norm[mid] - lam * max_sim[mid]
            key = (-score, mid)
            if best_key is None or key < best_key:
                best_key, best_mid, best_score = key, mid, score
        out[best_mid] = best_score
        remaining.remove(best_mid)
        sset = gsets[best_mid]
        for mid in remaining:
            gset = gsets[mid]
            union = gset | sset
            if union:
                sim = len(gset & sset) / len(union)
                if sim > max_sim[mid]:
                    max_sim[mid] = sim
    return out


def mmr_scores(relevance_scores, pool_ids, movies, lam, k=None):
    """Method J: greedy MMR over genre sets. lam=1 -> relevance only; lam=0 -> diversity only."""
    if not 0.0 <= lam <= 1.0:
        raise ValueError("MMR lambda must be in [0, 1]")
    relevance_norm = _minmax_zero_one(relevance_scores)
    pool_order = sorted(pool_ids, key=lambda mid: (-relevance_norm[mid], mid))
    gsets = {mid: (set(movies[mid].genres) if mid in movies else set()) for mid in pool_order}
    max_sim = {mid: 0.0 for mid in pool_order}
    remaining = set(pool_order)
    limit = len(remaining) if k is None else min(k, len(remaining))
    selected = []
    for _ in range(limit):
        best_mid, best_key = None, None
        for mid in pool_order:
            if mid not in remaining:
                continue
            mmr = lam * relevance_norm[mid] - (1.0 - lam) * max_sim[mid]
            key = (-mmr, mid)
            if best_key is None or key < best_key:
                best_key, best_mid = key, mid
        selected.append(best_mid)
        remaining.remove(best_mid)
        sset = gsets[best_mid]
        for mid in remaining:
            gset = gsets[mid]
            union = gset | sset
            if union:
                sim = len(gset & sset) / len(union)
                if sim > max_sim[mid]:
                    max_sim[mid] = sim
    return {mid: relevance_scores[mid] for mid in selected}, selected


def genre_constraint_order(relevance_scores, movies, max_per_primary=3, k=None):
    """Method K: enforce at most `max_per_primary` recommendations per primary genre
    (the movie's first listed genre). Movies without genres bypass the constraint.
    Multi-label limitation: only the first genre is treated as primary; nothing is invented."""
    counts = {}
    order = []
    deferred = []
    for mid, _ in sorted(relevance_scores.items(), key=lambda row: (-row[1], row[0])):
        movie = movies.get(mid)
        primary = movie.genres[0] if movie and movie.genres else None
        if primary is None:
            order.append(mid)
            continue
        if counts.get(primary, 0) >= max_per_primary:
            deferred.append(mid)
            continue
        counts[primary] = counts.get(primary, 0) + 1
        order.append(mid)
    if k is not None:
        if len(order) < k:
            order.extend(deferred[: k - len(order)])
        return order[:k]
    return order


def _minmax_zero_one(values):
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    return {mid: ((v - lo) / span if span > 0 else 0.5) for mid, v in values.items()}
