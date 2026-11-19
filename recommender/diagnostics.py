"""Phase 5 hybrid diagnostics, CF-first candidate gating, enrichment, and MMR."""

import heapq
import math
import statistics
import time

from . import content, evaluation, hybrid, nested, popularity
from .data import group_by_user

DIAGNOSTIC_METRICS = ("ndcg", "map", "hr", "precision", "recall", "ild")
DEFAULT_CF_FIRST_WEIGHTS = (0.95, 0.90, 0.80, 0.70)
DEFAULT_MMR_LAMBDAS = (1.0, 0.85, 0.70, 0.50)
PRIMARY_MMR_LAMBDA = 0.85


def _pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    den = math.sqrt(dx * dy)
    return num / den if den > 0 else None


def _average_ranks(values):
    indexed = sorted(enumerate(values), key=lambda row: row[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j - 1) / 2.0 + 1.0
        for pos in range(i, j):
            ranks[indexed[pos][0]] = avg_rank
        i = j
    return ranks


def _spearman(xs, ys):
    if len(xs) < 2:
        return None
    return _pearson(_average_ranks(xs), _average_ranks(ys))


def _cosine(left, right):
    if not left or not right:
        return 0.0
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(term, 0.0) for term, value in left.items())


def _quantile(sorted_vals, prob):
    if not sorted_vals:
        return None
    pos = (len(sorted_vals) - 1) * prob
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return sorted_vals[lo]
    return sorted_vals[lo] * (hi - pos) + sorted_vals[hi] * (pos - lo)


def _summarize_scalar(values):
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "median": None, "max": None, "iqr": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": statistics.mean(ordered),
        "std": statistics.stdev(ordered) if len(ordered) > 1 else 0.0,
        "min": ordered[0],
        "median": _quantile(ordered, 0.5),
        "max": ordered[-1],
        "iqr": _quantile(ordered, 0.75) - _quantile(ordered, 0.25),
    }


def build_genre_preferences(user_ratings, movies):
    """Smoothed per-genre preference strength in [-1, 1] from a user's training ratings."""
    if not user_ratings:
        return {}
    mean_rating = sum(user_ratings.values()) / len(user_ratings)
    sums = {}
    counts = {}
    for movie_id, rating in user_ratings.items():
        movie = movies.get(movie_id)
        if not movie:
            continue
        dev = rating - mean_rating
        for genre in movie.genres:
            sums[genre] = sums.get(genre, 0.0) + dev
            counts[genre] = counts.get(genre, 0) + 1
    # Shrink toward 0 with prior count 2.0 and scale by rating half-width (2.0) for bounded affinity.
    return {
        genre: max(-1.0, min(1.0, (sums[genre] / (counts[genre] + 2.0)) / 2.0))
        for genre in sums
    }


def preferred_release_year(user_ratings, movies, threshold=4.0):
    """Weighted average release year from positively rated (or all rated) training items."""
    pos_weight = 0.0
    pos_sum = 0.0
    all_weight = 0.0
    all_sum = 0.0
    for movie_id, rating in user_ratings.items():
        movie = movies.get(movie_id)
        if not movie or movie.year is None:
            continue
        all_sum += movie.year
        all_weight += 1.0
        if rating >= threshold:
            weight = 1.0 + (rating - threshold)
            pos_sum += movie.year * weight
            pos_weight += weight
    if pos_weight > 0:
        return pos_sum / pos_weight
    if all_weight > 0:
        return all_sum / all_weight
    return None


def enriched_content_scores(user_ratings, movies, vectors, candidates, threshold=4.0):
    """Content-based feature enrichment over candidates using only metadata + train ratings.

    Combines title+genre TF-IDF cosine (0.50), smoothed genre preference affinity (0.35),
    and release-year proximity (0.15) normalized locally over the candidate pool.
    """
    if not user_ratings or not candidates:
        return {}
    profile = content.user_profile(vectors, user_ratings)
    tfidf_raw = content.score(profile, vectors, candidates) if profile else {m: 0.0 for m in candidates}
    genre_prefs = build_genre_preferences(user_ratings, movies)
    pref_year = preferred_release_year(user_ratings, movies, threshold)

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

    tfidf_n = hybrid.normalize(tfidf_raw)
    genre_n = hybrid.normalize(genre_raw)
    year_n = hybrid.normalize(year_raw)
    return {
        movie_id: 0.50 * tfidf_n.get(movie_id, 0.5) + 0.35 * genre_n.get(movie_id, 0.5) + 0.15 * year_n.get(movie_id, 0.5)
        for movie_id in candidates
    }


def cf_top_m_candidates(cf_predictions, top_m=100):
    """Select top-M candidate movie IDs by CF predicted rating with deterministic tie-breaking."""
    if not cf_predictions or top_m <= 0:
        return []
    return [
        movie_id
        for movie_id, _ in heapq.nsmallest(top_m, cf_predictions.items(), key=lambda row: (-row[1], row[0]))
    ]


def rerank_pool_scores(candidate_ids, cf_predictions, secondary_scores, cf_weight):
    """Locally normalize CF and secondary signals within candidate_ids and blend without missing-signal boost."""
    if not candidate_ids:
        return {}
    cf_pool = {movie_id: cf_predictions[movie_id] for movie_id in candidate_ids if movie_id in cf_predictions}
    if not cf_pool:
        return {}
    sec_pool = {movie_id: secondary_scores.get(movie_id, 0.0) for movie_id in cf_pool}
    cf_n = hybrid.normalize(cf_pool)
    sec_n = hybrid.normalize(sec_pool)
    sec_weight = 1.0 - cf_weight
    return {
        movie_id: cf_weight * cf_n[movie_id] + sec_weight * sec_n[movie_id]
        for movie_id in cf_pool
    }


def mmr_rerank(relevance_scores, vectors, k=10, lam=0.85):
    """Greedy Maximal Marginal Relevance reranking over precomputed relevance scores.

    `relevance_scores` are min-max normalized to [0, 1] over the candidate pool so `lam`
    balances normalized relevance against max pairwise TF-IDF cosine similarity.
    When `lam == 1.0`, exact relevance descending / ID ascending order is returned.
    """
    if not relevance_scores or k <= 0:
        return []
    if not 0.0 <= lam <= 1.0:
        raise ValueError("MMR lambda must be between 0 and 1")
    norm_rel = hybrid.normalize(relevance_scores)
    if lam == 1.0:
        return nested._top_ids(norm_rel, k)

    remaining = sorted(norm_rel.keys(), key=lambda mid: (-norm_rel[mid], mid))
    selected = [remaining.pop(0)]
    max_sim = {
        mid: _cosine(vectors.get(mid, {}), vectors.get(selected[0], {}))
        for mid in remaining
    }
    while remaining and len(selected) < k:
        best_idx = 0
        best_key = None
        for idx, mid in enumerate(remaining):
            score = lam * norm_rel[mid] - (1.0 - lam) * max_sim[mid]
            key = (-score, mid)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = idx
        chosen = remaining.pop(best_idx)
        selected.append(chosen)
        chosen_vec = vectors.get(chosen, {})
        for mid in remaining:
            sim = _cosine(vectors.get(mid, {}), chosen_vec)
            if sim > max_sim[mid]:
                max_sim[mid] = sim
    return selected


def audit_inner_fold(fold, outer_train, movies, settings, inner_fraction=0.2, threshold=4.0, k=10, top_m=100):
    """Audit normalization, score distributions, correlations, and hybrid intrusion on inner validation only."""
    inner_train, validation = nested.split_inner(outer_train, inner_fraction)
    pop_scores, vectors, cf_model = nested._model_parts(
        inner_train, movies, settings, settings["cf"]["neighbors"]
    )
    cf_model.prediction_shrinkage = settings["cf"].get("prediction_shrinkage", 100.0)
    inner_users = group_by_user(inner_train)
    relevant_by_user = {}
    for user_id, movie_id, rating, _ in validation:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)
    eligible = {u: rel for u, rel in relevant_by_user.items() if inner_users.get(u)}

    raw_spreads = {"cf": [], "content": [], "popularity": []}
    raw_mins = {"cf": [], "content": [], "popularity": []}
    raw_maxs = {"cf": [], "content": [], "popularity": []}
    raw_means = {"cf": [], "content": [], "popularity": []}
    raw_stds = {"cf": [], "content": [], "popularity": []}
    norm_means = {"cf": [], "content": [], "popularity": []}
    norm_stds = {"cf": [], "content": [], "popularity": []}

    corrs = {
        "pearson_cf_content": [], "spearman_cf_content": [],
        "pearson_cf_popularity": [], "spearman_cf_popularity": [],
        "pearson_content_popularity": [], "spearman_content_popularity": [],
    }
    rel_signal_means = {"cf": [], "content": [], "popularity": []}
    nonrel_signal_means = {"cf": [], "content": [], "popularity": []}

    all_cand_counts = []
    cf_cand_counts = []
    missing_cf_counts = []
    jaccard_10_cf_cont = []
    jaccard_10_cf_pop = []
    jaccard_100_cf_cont = []
    jaccard_100_cf_pop = []
    cf_top100_recall = []
    cf_all_recall = []

    weights = settings["hybrid_weights"]
    no_cf_slots = 0
    no_cf_hits = 0
    out_cf100_slots = 0
    out_cf100_hits = 0
    with_cf_added_slots = 0
    with_cf_added_hits = 0
    displaced_cf_slots = 0
    displaced_cf_hits = 0
    total_hyb_slots = 0

    hyb_ndcg_sum = 0.0
    hyb_no_renorm_ndcg_sum = 0.0
    cf_ndcg_sum = 0.0
    cf_gated_hyb_ndcg_sum = 0.0

    for user_id, relevant in eligible.items():
        user_ratings = inner_users[user_id]
        candidates, sigs = hybrid.user_signals(user_ratings, movies, pop_scores, cf_model, vectors)
        cf_s = sigs.get("cf", {})
        co_s = sigs.get("content", {})
        po_s = sigs.get("popularity", {})

        all_cand_counts.append(len(candidates))
        cf_cand_counts.append(len(cf_s))
        missing_cf_counts.append(len(candidates - cf_s.keys()))

        normed = {}
        for name, values in (("cf", cf_s), ("content", co_s), ("popularity", po_s)):
            if values:
                vals = list(values.values())
                lo, hi = min(vals), max(vals)
                raw_mins[name].append(lo)
                raw_maxs[name].append(hi)
                raw_spreads[name].append(hi - lo)
                raw_means[name].append(statistics.mean(vals))
                raw_stds[name].append(statistics.stdev(vals) if len(vals) > 1 else 0.0)
                nvals = hybrid.normalize(values)
                normed[name] = nvals
                nv = list(nvals.values())
                norm_means[name].append(statistics.mean(nv))
                norm_stds[name].append(statistics.stdev(nv) if len(nv) > 1 else 0.0)
                rel_vals = [nvals[m] for m in relevant if m in nvals]
                nonrel_vals = [v for m, v in nvals.items() if m not in relevant]
                if rel_vals:
                    rel_signal_means[name].append(statistics.mean(rel_vals))
                if nonrel_vals:
                    nonrel_signal_means[name].append(statistics.mean(nonrel_vals))
            else:
                normed[name] = {}

        common_cc = sorted(cf_s.keys() & co_s.keys())
        if len(common_cc) >= 5:
            xs = [cf_s[m] for m in common_cc]
            ys = [co_s[m] for m in common_cc]
            if (p := _pearson(xs, ys)) is not None:
                corrs["pearson_cf_content"].append(p)
            if (s := _spearman(xs, ys)) is not None:
                corrs["spearman_cf_content"].append(s)

        common_cp = sorted(cf_s.keys() & po_s.keys())
        if len(common_cp) >= 5:
            xs = [cf_s[m] for m in common_cp]
            ys = [po_s[m] for m in common_cp]
            if (p := _pearson(xs, ys)) is not None:
                corrs["pearson_cf_popularity"].append(p)
            if (s := _spearman(xs, ys)) is not None:
                corrs["spearman_cf_popularity"].append(s)

        common_op = sorted(co_s.keys() & po_s.keys())
        if len(common_op) >= 5:
            xs = [co_s[m] for m in common_op]
            ys = [po_s[m] for m in common_op]
            if (p := _pearson(xs, ys)) is not None:
                corrs["pearson_content_popularity"].append(p)
            if (s := _spearman(xs, ys)) is not None:
                corrs["spearman_content_popularity"].append(s)

        cf_top10_list = nested._top_ids(cf_s, k)
        cf_top100_list = cf_top_m_candidates(cf_s, top_m)
        co_top10_list = nested._top_ids(co_s, k)
        co_top100_list = nested._top_ids(co_s, top_m)
        po_top10_list = nested._top_ids(po_s, k)
        po_top100_list = nested._top_ids(po_s, top_m)

        cf10, co10, po10 = set(cf_top10_list), set(co_top10_list), set(po_top10_list)
        cf100, co100, po100 = set(cf_top100_list), set(co_top100_list), set(po_top100_list)
        if cf10 | co10:
            jaccard_10_cf_cont.append(len(cf10 & co10) / len(cf10 | co10))
        if cf10 | po10:
            jaccard_10_cf_pop.append(len(cf10 & po10) / len(cf10 | po10))
        if cf100 | co100:
            jaccard_100_cf_cont.append(len(cf100 & co100) / len(cf100 | co100))
        if cf100 | po100:
            jaccard_100_cf_pop.append(len(cf100 & po100) / len(cf100 | po100))

        cf_top100_recall.append(len(cf100 & relevant) / len(relevant))
        cf_all_recall.append(len(cf_s.keys() & relevant) / len(relevant))

        hyb_top10_list = nested._blend_ids(normed, weights, k)
        hyb10 = set(hyb_top10_list)
        total_hyb_slots += len(hyb_top10_list)

        for mid in hyb10 - cf10:
            if mid not in cf_s:
                no_cf_slots += 1
                if mid in relevant:
                    no_cf_hits += 1
            else:
                with_cf_added_slots += 1
                if mid in relevant:
                    with_cf_added_hits += 1
            if mid not in cf100:
                out_cf100_slots += 1
                if mid in relevant:
                    out_cf100_hits += 1
        for mid in cf10 - hyb10:
            displaced_cf_slots += 1
            if mid in relevant:
                displaced_cf_hits += 1

        # Ablation: zero missing CF instead of renormalizing weights, and restrict to CF-scored only.
        total_weight = sum(weights.values())
        no_renorm_scores = {
            mid: sum(weights[name] * normed[name].get(mid, 0.0) for name in weights) / total_weight
            for mid in candidates
        }
        hyb_no_renorm_top10 = nested._top_ids(no_renorm_scores, k)
        cf_gated_normed = {
            name: {mid: val for mid, val in values.items() if mid in cf_s}
            for name, values in normed.items()
        }
        cf_gated_top10 = nested._blend_ids(cf_gated_normed, weights, k)

        cf_ndcg_sum += evaluation.ndcg_at_k(cf_top10_list, relevant, k)
        hyb_ndcg_sum += evaluation.ndcg_at_k(hyb_top10_list, relevant, k)
        hyb_no_renorm_ndcg_sum += evaluation.ndcg_at_k(hyb_no_renorm_top10, relevant, k)
        cf_gated_hyb_ndcg_sum += evaluation.ndcg_at_k(cf_gated_top10, relevant, k)

    n_users = len(eligible)
    return {
        "fold": f"u{fold}",
        "evaluated_users": n_users,
        "candidate_pools": {
            "all_unseen": _summarize_scalar(all_cand_counts),
            "cf_scored": _summarize_scalar(cf_cand_counts),
            "missing_cf": _summarize_scalar(missing_cf_counts),
            "cf_top100_recall": statistics.mean(cf_top100_recall) if cf_top100_recall else 0.0,
            "cf_all_recall": statistics.mean(cf_all_recall) if cf_all_recall else 0.0,
            "jaccard_top10_cf_content": statistics.mean(jaccard_10_cf_cont) if jaccard_10_cf_cont else 0.0,
            "jaccard_top10_cf_popularity": statistics.mean(jaccard_10_cf_pop) if jaccard_10_cf_pop else 0.0,
            "jaccard_top100_cf_content": statistics.mean(jaccard_100_cf_cont) if jaccard_100_cf_cont else 0.0,
            "jaccard_top100_cf_popularity": statistics.mean(jaccard_100_cf_pop) if jaccard_100_cf_pop else 0.0,
        },
        "score_distributions": {
            name: {
                "raw_min_mean": statistics.mean(raw_mins[name]) if raw_mins[name] else None,
                "raw_max_mean": statistics.mean(raw_maxs[name]) if raw_maxs[name] else None,
                "raw_spread_mean": statistics.mean(raw_spreads[name]) if raw_spreads[name] else None,
                "raw_mean": statistics.mean(raw_means[name]) if raw_means[name] else None,
                "raw_std_mean": statistics.mean(raw_stds[name]) if raw_stds[name] else None,
                "norm_mean": statistics.mean(norm_means[name]) if norm_means[name] else None,
                "norm_std_mean": statistics.mean(norm_stds[name]) if norm_stds[name] else None,
                "norm_relevant_mean": statistics.mean(rel_signal_means[name]) if rel_signal_means[name] else None,
                "norm_nonrelevant_mean": statistics.mean(nonrel_signal_means[name]) if nonrel_signal_means[name] else None,
            }
            for name in ("cf", "content", "popularity")
        },
        "correlations": {key: _summarize_scalar(vals) for key, vals in corrs.items()},
        "hybrid_intrusion": {
            "total_hybrid_top10_slots": total_hyb_slots,
            "no_cf_slots": no_cf_slots,
            "no_cf_hits": no_cf_hits,
            "no_cf_precision": no_cf_hits / no_cf_slots if no_cf_slots else 0.0,
            "outside_cf100_slots": out_cf100_slots,
            "outside_cf100_hits": out_cf100_hits,
            "outside_cf100_precision": out_cf100_hits / out_cf100_slots if out_cf100_slots else 0.0,
            "with_cf_added_slots": with_cf_added_slots,
            "with_cf_added_hits": with_cf_added_hits,
            "with_cf_added_precision": with_cf_added_hits / with_cf_added_slots if with_cf_added_slots else 0.0,
            "displaced_cf_slots": displaced_cf_slots,
            "displaced_cf_hits": displaced_cf_hits,
            "displaced_cf_precision": displaced_cf_hits / displaced_cf_slots if displaced_cf_slots else 0.0,
        },
        "inner_ndcg_ablation": {
            "cf_default": cf_ndcg_sum / n_users if n_users else 0.0,
            "hybrid_default_renorm": hyb_ndcg_sum / n_users if n_users else 0.0,
            "hybrid_zero_missing_cf": hyb_no_renorm_ndcg_sum / n_users if n_users else 0.0,
            "hybrid_cf_scored_only": cf_gated_hyb_ndcg_sum / n_users if n_users else 0.0,
        },
    }


def select_phase5_parameters(
    outer_train,
    movies,
    settings,
    nested_settings,
    k=10,
    threshold=4.0,
    top_m=100,
    cf_first_weights=DEFAULT_CF_FIRST_WEIGHTS,
):
    """Select parameters for CF, hybrid, and CF-first architectures strictly on inner validation."""
    inner_train, validation = nested.split_inner(outer_train, nested_settings["inner_validation_fraction"])
    cf_grid = nested_settings["cf_grid"]
    hybrid_grid = nested_settings["hybrid_weight_grid"]
    neighbors_grid = list(dict.fromkeys(int(value) for value in cf_grid["neighbors"]))
    shrinkage_grid = list(dict.fromkeys(float(value) for value in cf_grid["prediction_shrinkage"]))
    cf_first_weights = tuple(float(w) for w in cf_first_weights)
    if not neighbors_grid or min(neighbors_grid) <= 0 or not shrinkage_grid or min(shrinkage_grid) < 0:
        raise ValueError("CF tuning grid must contain positive neighbors and non-negative prediction shrinkage")
    if not hybrid_grid or not cf_first_weights:
        raise ValueError("hybrid and CF-first weight grids must not be empty")

    max_neighbors = max(neighbors_grid)
    pop_scores, vectors, cf_model = nested._model_parts(inner_train, movies, settings, max_neighbors)
    inner_users = group_by_user(inner_train)
    relevant_by_user = {}
    for user_id, movie_id, rating, _ in validation:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)
    eligible = {user_id: rel for user_id, rel in relevant_by_user.items() if inner_users.get(user_id)}

    cf_configs = [(n, s) for n in neighbors_grid for s in shrinkage_grid]
    hybrid_configs = [(n, s, dict(w)) for n, s in cf_configs for w in hybrid_grid]
    cf_first_configs = [(n, s, w_cf) for n, s in cf_configs for w_cf in cf_first_weights]

    if not eligible:
        default_cf = {
            "neighbors": settings["cf"]["neighbors"],
            "prediction_shrinkage": settings["cf"].get("prediction_shrinkage", 100.0),
        }
        default_weights = dict(settings["hybrid_weights"])
        default_w_cf = cf_first_weights[0]
        return {
            "fallback": True,
            "evaluated_users": 0,
            "inner_train_interactions": len(inner_train),
            "inner_validation_interactions": len(validation),
            "cf_tuned": {"parameters": default_cf, "ndcg": None, "candidates": []},
            "hybrid_tuned": {"parameters": {**default_cf, "weights": default_weights}, "ndcg": None, "candidates": []},
            "cf_top100_content_reranking": {"parameters": {**default_cf, "cf_weight": default_w_cf}, "ndcg": None, "candidates": []},
            "cf_top100_content_enrichment": {"parameters": {**default_cf, "cf_weight": default_w_cf}, "ndcg": None, "candidates": []},
        }

    cf_scores = [0.0] * len(cf_configs)
    hybrid_scores = [0.0] * len(hybrid_configs)
    rerank_scores = [0.0] * len(cf_first_configs)
    enrich_scores = [0.0] * len(cf_first_configs)
    cf_views = {neighbors: nested._neighbor_view(cf_model, neighbors) for neighbors in neighbors_grid}

    for user_id, relevant in eligible.items():
        user_ratings = inner_users[user_id]
        _, base_signals = hybrid.user_signals(user_ratings, movies, pop_scores, vectors=vectors)
        candidates = movies.keys() - user_ratings.keys()
        base_normalized = {name: hybrid.normalize(values) for name, values in base_signals.items()}
        content_raw = base_signals.get("content", {})
        enriched_cache = {}

        for neighbor_index, neighbors in enumerate(neighbors_grid):
            evidence = cf_views[neighbors].evidence(user_ratings, candidates)
            for shrinkage_index, prediction_shrinkage in enumerate(shrinkage_grid):
                config_index = neighbor_index * len(shrinkage_grid) + shrinkage_index
                predictions = nested._cf_predictions(evidence, prediction_shrinkage)
                normalized_cf = hybrid.normalize(predictions)
                cf_scores[config_index] += evaluation.ndcg_at_k(nested._top_ids(normalized_cf, k), relevant, k)

                normalized = dict(base_normalized)
                normalized["cf"] = normalized_cf
                for weight_index, weights in enumerate(hybrid_grid):
                    hybrid_index = config_index * len(hybrid_grid) + weight_index
                    hybrid_scores[hybrid_index] += evaluation.ndcg_at_k(
                        nested._blend_ids(normalized, weights, k), relevant, k
                    )

                top_m_ids = cf_top_m_candidates(predictions, top_m)
                pool_key = tuple(top_m_ids)
                if pool_key not in enriched_cache:
                    enriched_cache[pool_key] = enriched_content_scores(
                        user_ratings, movies, vectors, top_m_ids, threshold
                    )
                enriched_pool = enriched_cache[pool_key]

                for w_index, w_cf in enumerate(cf_first_weights):
                    idx = config_index * len(cf_first_weights) + w_index
                    rerank_map = rerank_pool_scores(top_m_ids, predictions, content_raw, w_cf)
                    enrich_map = rerank_pool_scores(top_m_ids, predictions, enriched_pool, w_cf)
                    rerank_scores[idx] += evaluation.ndcg_at_k(nested._top_ids(rerank_map, k), relevant, k)
                    enrich_scores[idx] += evaluation.ndcg_at_k(nested._top_ids(enrich_map, k), relevant, k)

    count = len(eligible)
    cf_cands = [
        {"parameters": {"neighbors": n, "prediction_shrinkage": s}, "ndcg": sc / count}
        for (n, s), sc in zip(cf_configs, cf_scores)
    ]
    hyb_cands = [
        {"parameters": {"neighbors": n, "prediction_shrinkage": s, "weights": w}, "ndcg": sc / count}
        for (n, s, w), sc in zip(hybrid_configs, hybrid_scores)
    ]
    rerank_cands = [
        {"parameters": {"neighbors": n, "prediction_shrinkage": s, "cf_weight": w}, "ndcg": sc / count}
        for (n, s, w), sc in zip(cf_first_configs, rerank_scores)
    ]
    enrich_cands = [
        {"parameters": {"neighbors": n, "prediction_shrinkage": s, "cf_weight": w}, "ndcg": sc / count}
        for (n, s, w), sc in zip(cf_first_configs, enrich_scores)
    ]

    best_cf = max(cf_cands, key=lambda row: row["ndcg"])
    best_hyb = max(hyb_cands, key=lambda row: row["ndcg"])
    best_rerank = max(rerank_cands, key=lambda row: row["ndcg"])
    best_enrich = max(enrich_cands, key=lambda row: row["ndcg"])
    return {
        "fallback": False,
        "evaluated_users": count,
        "inner_train_interactions": len(inner_train),
        "inner_validation_interactions": len(validation),
        "cf_tuned": {**best_cf, "candidates": cf_cands},
        "hybrid_tuned": {**best_hyb, "candidates": hyb_cands},
        "cf_top100_content_reranking": {**best_rerank, "candidates": rerank_cands},
        "cf_top100_content_enrichment": {**best_enrich, "candidates": enrich_cands},
    }


def _summarize_extended(per_user, coverage_count, catalog_size):
    users = len(per_user)
    if not users:
        return {**{m: None for m in DIAGNOSTIC_METRICS}, "coverage": 0.0, "users": 0}
    summary = {
        metric: sum(row[metric] for row in per_user.values()) / users
        for metric in DIAGNOSTIC_METRICS
    }
    summary["coverage"] = coverage_count / catalog_size
    summary["users"] = users
    return summary


def evaluate_phase5_fold(
    fold,
    outer_train,
    outer_test,
    movies,
    settings,
    nested_settings,
    k=10,
    threshold=4.0,
    top_m=100,
    cf_first_weights=DEFAULT_CF_FIRST_WEIGHTS,
    mmr_lambdas=DEFAULT_MMR_LAMBDAS,
    primary_mmr_lambda=PRIMARY_MMR_LAMBDA,
):
    """Run inner audit, train-only selection, outer refit, and single outer test pass for one fold."""
    audit_started = time.perf_counter()
    audit = audit_inner_fold(
        fold, outer_train, movies, settings,
        inner_fraction=nested_settings["inner_validation_fraction"],
        threshold=threshold,
        k=k,
        top_m=top_m,
    )
    audit_seconds = time.perf_counter() - audit_started

    sel_started = time.perf_counter()
    selection = select_phase5_parameters(
        outer_train, movies, settings, nested_settings,
        k=nested_settings.get("selection_k", k),
        threshold=threshold,
        top_m=top_m,
        cf_first_weights=cf_first_weights,
    )
    selection_seconds = time.perf_counter() - sel_started

    cf_sel = selection["cf_tuned"]["parameters"]
    hyb_sel = selection["hybrid_tuned"]["parameters"]
    rerank_sel = selection["cf_top100_content_reranking"]["parameters"]
    enrich_sel = selection["cf_top100_content_enrichment"]["parameters"]

    max_neighbors = max(
        cf_sel["neighbors"],
        hyb_sel["neighbors"],
        rerank_sel["neighbors"],
        enrich_sel["neighbors"],
    )
    refit_started = time.perf_counter()
    pop_scores, vectors, cf_model = nested._model_parts(outer_train, movies, settings, max_neighbors)
    refit_seconds = time.perf_counter() - refit_started

    train_users = group_by_user(outer_train)
    relevant_by_user = {}
    for user_id, movie_id, rating, _ in outer_test:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)

    mmr_names = {lam: f"cf_top100_content_reranking_mmr_lam_{lam:.2f}" for lam in mmr_lambdas}
    cf_mmr_names = {lam: f"cf_tuned_top100_mmr_lam_{lam:.2f}" for lam in mmr_lambdas}
    model_names = [
        "cf_tuned",
        "hybrid_tuned",
        "cf_top100_content_enrichment",
        "cf_top100_content_reranking",
        "cf_top100_content_reranking_mmr",
        *mmr_names.values(),
        *cf_mmr_names.values(),
    ]
    per_user = {name: {} for name in model_names}
    coverage = {name: set() for name in model_names}

    needed_configs = (
        (cf_sel["neighbors"], cf_sel["prediction_shrinkage"]),
        (hyb_sel["neighbors"], hyb_sel["prediction_shrinkage"]),
        (rerank_sel["neighbors"], rerank_sel["prediction_shrinkage"]),
        (enrich_sel["neighbors"], enrich_sel["prediction_shrinkage"]),
    )
    cf_views = {n: nested._neighbor_view(cf_model, n) for n, _ in needed_configs}

    score_started = time.perf_counter()
    for user_id, relevant in relevant_by_user.items():
        user_ratings = train_users.get(user_id, {})
        if not user_ratings:
            continue
        _, base_signals = hybrid.user_signals(user_ratings, movies, pop_scores, vectors=vectors)
        candidates = movies.keys() - user_ratings.keys()
        preds_cache = {}
        for n, s in needed_configs:
            if (n, s) not in preds_cache:
                ev = cf_views[n].evidence(user_ratings, candidates)
                preds_cache[(n, s)] = nested._cf_predictions(ev, s)

        cf_preds = preds_cache[(cf_sel["neighbors"], cf_sel["prediction_shrinkage"])]
        hyb_preds = preds_cache[(hyb_sel["neighbors"], hyb_sel["prediction_shrinkage"])]
        rerank_preds = preds_cache[(rerank_sel["neighbors"], rerank_sel["prediction_shrinkage"])]
        enrich_preds = preds_cache[(enrich_sel["neighbors"], enrich_sel["prediction_shrinkage"])]

        # 1. CF tuned baseline
        cf_recs = nested._top_ids(cf_preds, k)
        # 2. Existing hybrid tuned
        hyb_signals = dict(base_signals)
        hyb_signals["cf"] = hyb_preds
        hyb_recs = [row["movie_id"] for row in hybrid.rank(movies, hyb_signals, k, hyb_sel["weights"])]
        # 3. CF top-100 + content enrichment
        enrich_pool_ids = cf_top_m_candidates(enrich_preds, top_m)
        enriched_feats = enriched_content_scores(user_ratings, movies, vectors, enrich_pool_ids, threshold)
        enrich_scores = rerank_pool_scores(enrich_pool_ids, enrich_preds, enriched_feats, enrich_sel["cf_weight"])
        enrich_recs = nested._top_ids(enrich_scores, k)
        # 4. CF top-100 + content reranking
        rerank_pool_ids = cf_top_m_candidates(rerank_preds, top_m)
        rerank_scores = rerank_pool_scores(
            rerank_pool_ids, rerank_preds, base_signals.get("content", {}), rerank_sel["cf_weight"]
        )
        rerank_recs = nested._top_ids(rerank_scores, k)

        rec_map = {
            "cf_tuned": cf_recs,
            "hybrid_tuned": hyb_recs,
            "cf_top100_content_enrichment": enrich_recs,
            "cf_top100_content_reranking": rerank_recs,
        }

        # 5. Independent MMR on CF top-100 + content reranking, and on CF tuned top-100
        cf_top_pool = {mid: cf_preds[mid] for mid in cf_top_m_candidates(cf_preds, top_m)}
        for lam in mmr_lambdas:
            rec_map[mmr_names[lam]] = mmr_rerank(rerank_scores, vectors, k, lam)
            rec_map[cf_mmr_names[lam]] = mmr_rerank(cf_top_pool, vectors, k, lam)
        rec_map["cf_top100_content_reranking_mmr"] = rec_map[mmr_names[primary_mmr_lambda]]

        uid_key = f"u{fold}:{user_id}"
        for name, recs in rec_map.items():
            per_user[name][uid_key] = evaluation.evaluate_user_extended(recs, relevant, k, vectors)
            coverage[name].update(recs)

    score_seconds = time.perf_counter() - score_started
    metrics = {
        name: _summarize_extended(values, len(coverage[name]), len(movies))
        for name, values in per_user.items()
    }
    return {
        "audit": audit,
        "selection": selection,
        "metrics": metrics,
        "per_user": per_user,
        "_coverage": coverage,
        "resources": {
            "audit_seconds": audit_seconds,
            "selection_seconds": selection_seconds,
            "refit_seconds": refit_seconds,
            "outer_scoring_seconds": score_seconds,
        },
    }


def aggregate_phase5_folds(folds, catalog_size, resamples=2000, confidence=0.95, seed=2026):
    """Aggregate per-fold metrics, sample SDs, pooled means, and paired bootstrap vs cf_tuned."""
    names = list(next(iter(folds.values()))["metrics"].keys())
    pooled = {}
    for name in names:
        fold_rows = [res["metrics"][name] for res in folds.values() if res["metrics"][name]["users"]]
        total_users = sum(row["users"] for row in fold_rows)
        pooled[name] = {
            metric: sum(row[metric] * row["users"] for row in fold_rows) / total_users if total_users else None
            for metric in DIAGNOSTIC_METRICS
        }
        all_cov = set().union(*(res["_coverage"][name] for res in folds.values()))
        pooled[name].update({"coverage": len(all_cov) / catalog_size, "users": total_users})
        pooled[name]["fold_mean_std"] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in fold_rows) if fold_rows else None,
                "std": statistics.stdev(row[metric] for row in fold_rows) if len(fold_rows) > 1 else 0.0 if fold_rows else None,
            }
            for metric in (*DIAGNOSTIC_METRICS, "coverage")
        }

    per_user = {
        name: {key: value for res in folds.values() for key, value in res["per_user"][name].items()}
        for name in names
    }
    comparisons = {}
    for candidate in names:
        if candidate == "cf_tuned":
            continue
        if per_user[candidate].keys() & per_user["cf_tuned"].keys():
            comparisons[f"{candidate}_vs_cf_tuned_ndcg"] = nested.paired_bootstrap(
                per_user[candidate], per_user["cf_tuned"], "ndcg", resamples, confidence, seed
            )
            comparisons[f"{candidate}_vs_cf_tuned_ild"] = nested.paired_bootstrap(
                per_user[candidate], per_user["cf_tuned"], "ild", resamples, confidence, seed
            )
    if per_user["cf_top100_content_reranking"].keys() & per_user["hybrid_tuned"].keys():
        comparisons["cf_top100_content_reranking_vs_hybrid_tuned_ndcg"] = nested.paired_bootstrap(
            per_user["cf_top100_content_reranking"], per_user["hybrid_tuned"], "ndcg", resamples, confidence, seed
        )
    if per_user["cf_top100_content_enrichment"].keys() & per_user["hybrid_tuned"].keys():
        comparisons["cf_top100_content_enrichment_vs_hybrid_tuned_ndcg"] = nested.paired_bootstrap(
            per_user["cf_top100_content_enrichment"], per_user["hybrid_tuned"], "ndcg", resamples, confidence, seed
        )
    return {"pooled": pooled, "comparisons": comparisons}


def aggregate_audit(folds):
    """Aggregate inner-fold score distribution, correlation, and intrusion diagnostics across folds."""
    fold_audits = [res["audit"] for res in folds.values()]
    intrusion = {
        key: sum(a["hybrid_intrusion"][key] for a in fold_audits)
        for key in (
            "total_hybrid_top10_slots",
            "no_cf_slots",
            "no_cf_hits",
            "outside_cf100_slots",
            "outside_cf100_hits",
            "with_cf_added_slots",
            "with_cf_added_hits",
            "displaced_cf_slots",
            "displaced_cf_hits",
        )
    }
    intrusion["no_cf_precision"] = intrusion["no_cf_hits"] / intrusion["no_cf_slots"] if intrusion["no_cf_slots"] else 0.0
    intrusion["outside_cf100_precision"] = intrusion["outside_cf100_hits"] / intrusion["outside_cf100_slots"] if intrusion["outside_cf100_slots"] else 0.0
    intrusion["with_cf_added_precision"] = intrusion["with_cf_added_hits"] / intrusion["with_cf_added_slots"] if intrusion["with_cf_added_slots"] else 0.0
    intrusion["displaced_cf_precision"] = intrusion["displaced_cf_hits"] / intrusion["displaced_cf_slots"] if intrusion["displaced_cf_slots"] else 0.0

    def _mean_path(*keys):
        vals = []
        for a in fold_audits:
            cur = a
            for k in keys:
                cur = cur[k]
            if cur is not None:
                vals.append(cur)
        return {
            "mean": statistics.mean(vals) if vals else None,
            "std": statistics.stdev(vals) if len(vals) > 1 else 0.0 if vals else None,
        }

    return {
        "per_fold": {a["fold"]: a for a in fold_audits},
        "pooled_intrusion": intrusion,
        "mean_inner_ndcg_ablation": {
            key: _mean_path("inner_ndcg_ablation", key)
            for key in ("cf_default", "hybrid_default_renorm", "hybrid_zero_missing_cf", "hybrid_cf_scored_only")
        },
        "mean_correlations": {
            key: _mean_path("correlations", key, "mean")
            for key in (
                "pearson_cf_content", "spearman_cf_content",
                "pearson_cf_popularity", "spearman_cf_popularity",
                "pearson_content_popularity", "spearman_content_popularity",
            )
        },
        "mean_candidate_pools": {
            "all_unseen": _mean_path("candidate_pools", "all_unseen", "mean"),
            "cf_scored": _mean_path("candidate_pools", "cf_scored", "mean"),
            "missing_cf": _mean_path("candidate_pools", "missing_cf", "mean"),
            "cf_top100_recall": _mean_path("candidate_pools", "cf_top100_recall"),
            "cf_all_recall": _mean_path("candidate_pools", "cf_all_recall"),
            "jaccard_top10_cf_content": _mean_path("candidate_pools", "jaccard_top10_cf_content"),
            "jaccard_top10_cf_popularity": _mean_path("candidate_pools", "jaccard_top10_cf_popularity"),
            "jaccard_top100_cf_content": _mean_path("candidate_pools", "jaccard_top100_cf_content"),
            "jaccard_top100_cf_popularity": _mean_path("candidate_pools", "jaccard_top100_cf_popularity"),
        },
        "mean_score_distributions": {
            signal: {
                stat: _mean_path("score_distributions", signal, stat)
                for stat in (
                    "raw_min_mean", "raw_max_mean", "raw_spread_mean",
                    "raw_mean", "raw_std_mean", "norm_mean", "norm_std_mean",
                    "norm_relevant_mean", "norm_nonrelevant_mean",
                )
            }
            for signal in ("cf", "content", "popularity")
        },
    }
