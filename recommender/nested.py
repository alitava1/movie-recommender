"""Train-only inner model selection and untouched outer-fold evaluation."""

import heapq
import math
import random
import statistics
import time

from . import content, evaluation, hybrid, popularity
from .collaborative import ItemItemCF
from .data import group_by_user

METRICS = ("precision", "recall", "ndcg", "map")


def split_inner(ratings, fraction=0.2):
    """Split each user's interactions chronologically; one-row histories stay in train."""
    if not 0 < fraction < 1:
        raise ValueError("inner validation fraction must be between 0 and 1")
    grouped = {}
    for row in ratings:
        grouped.setdefault(row[0], []).append(row)
    train, validation = [], []
    for rows in grouped.values():
        rows.sort(key=lambda row: (row[3], row[1]))
        count = min(len(rows) - 1, max(1, int(len(rows) * fraction))) if len(rows) > 1 else 0
        train.extend(rows[:-count] if count else rows)
        if count:
            validation.extend(rows[-count:])
    return train, validation


def _model_parts(ratings, movies, settings, max_neighbors):
    return (
        popularity.popularity_scores(ratings, settings["popularity"]["prior_count"]),
        content.build_tfidf(movies),
        ItemItemCF.build(
            ratings,
            neighbors=max_neighbors,
            shrinkage=settings["cf"]["shrinkage"],
            prediction_shrinkage=0.0,
        ),
    )


def _neighbor_view(model, neighbors):
    return ItemItemCF(
        neighbors,
        model.shrinkage,
        {movie_id: rows[:neighbors] for movie_id, rows in model.item_neighbors.items()},
        0.0,
    )


def _cf_predictions(evidence, prediction_shrinkage):
    return {
        movie_id: min(5.0, max(0.5, mean + numerator / (denominator + prediction_shrinkage)))
        for movie_id, (mean, numerator, denominator) in evidence.items()
    }


def _top_ids(values, k):
    """Top-k movie IDs from a signal using the engine's score/ID tie-break order."""
    return [movie_id for movie_id, _ in heapq.nsmallest(k, values.items(), key=lambda row: (-row[1], row[0]))]


def _blend_ids(normalized, weights, k):
    """Top-k from pre-normalized signals using the hybrid blend formula."""
    combined = {}
    for name, values in normalized.items():
        weight = weights[name]
        for movie_id, value in values.items():
            total, used = combined.get(movie_id, (0.0, 0.0))
            combined[movie_id] = total + weight * value, used + weight
    scored = ((movie_id, total / used) for movie_id, (total, used) in combined.items() if used)
    return [movie_id for movie_id, _ in heapq.nsmallest(k, scored, key=lambda row: (-row[1], row[0]))]


def _metrics(ids, relevant, k):
    return evaluation.evaluate_user(ids, relevant, k)


def select_parameters(outer_train, movies, settings, nested_settings, k, threshold):
    """Select CF and hybrid settings using only an inner split of outer_train."""
    inner_train, validation = split_inner(outer_train, nested_settings["inner_validation_fraction"])
    cf_grid = nested_settings["cf_grid"]
    hybrid_grid = nested_settings["hybrid_weight_grid"]
    neighbors_grid = list(dict.fromkeys(int(value) for value in cf_grid["neighbors"]))
    shrinkage_grid = list(dict.fromkeys(float(value) for value in cf_grid["prediction_shrinkage"]))
    if not neighbors_grid or min(neighbors_grid) <= 0 or not shrinkage_grid or min(shrinkage_grid) < 0:
        raise ValueError("CF tuning grid must contain positive neighbors and non-negative prediction shrinkage")
    if not hybrid_grid:
        raise ValueError("hybrid weight grid must not be empty")
    max_neighbors = max(neighbors_grid)
    popularity_scores, vectors, cf_model = _model_parts(inner_train, movies, settings, max_neighbors)
    inner_users = group_by_user(inner_train)
    relevant_by_user = {}
    for user_id, movie_id, rating, _ in validation:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)
    eligible = {
        user_id: relevant
        for user_id, relevant in relevant_by_user.items()
        if inner_users.get(user_id)
    }

    cf_configs = [(neighbors, shrinkage) for neighbors in neighbors_grid for shrinkage in shrinkage_grid]
    hybrid_configs = [
        (neighbors, shrinkage, dict(weights))
        for neighbors, shrinkage in cf_configs
        for weights in hybrid_grid
    ]
    cf_scores = [0.0] * len(cf_configs)
    hybrid_scores = [0.0] * len(hybrid_configs)
    cf_views = {neighbors: _neighbor_view(cf_model, neighbors) for neighbors in neighbors_grid}

    for user_id, relevant in eligible.items():
        user_ratings = inner_users[user_id]
        _, base_signals = hybrid.user_signals(user_ratings, movies, popularity_scores, vectors=vectors)
        candidates = movies.keys() - user_ratings.keys()
        base_normalized = {name: hybrid.normalize(values) for name, values in base_signals.items()}
        for neighbor_index, neighbors in enumerate(neighbors_grid):
            evidence = cf_views[neighbors].evidence(user_ratings, candidates)
            for shrinkage_index, prediction_shrinkage in enumerate(shrinkage_grid):
                config_index = neighbor_index * len(shrinkage_grid) + shrinkage_index
                predictions = _cf_predictions(evidence, prediction_shrinkage)
                normalized_cf = hybrid.normalize(predictions)
                cf_scores[config_index] += _metrics(_top_ids(normalized_cf, k), relevant, k)["ndcg"]
                normalized = dict(base_normalized)
                normalized["cf"] = normalized_cf
                for weight_index, weights in enumerate(hybrid_grid):
                    hybrid_index = config_index * len(hybrid_grid) + weight_index
                    hybrid_scores[hybrid_index] += _metrics(
                        _blend_ids(normalized, weights, k), relevant, k
                    )["ndcg"]

    if not eligible:
        default_cf = {
            "neighbors": settings["cf"]["neighbors"],
            "prediction_shrinkage": settings["cf"].get("prediction_shrinkage", 0.0),
        }
        default_weights = dict(settings["hybrid_weights"])
        return {
            "fallback": True,
            "evaluated_users": 0,
            "cf": {"parameters": default_cf, "ndcg": None, "candidates": []},
            "hybrid": {"parameters": {**default_cf, "weights": default_weights}, "ndcg": None, "candidates": []},
            "inner_train_interactions": len(inner_train),
            "inner_validation_interactions": len(validation),
        }

    count = len(eligible)
    cf_candidates = [
        {"parameters": {"neighbors": neighbors, "prediction_shrinkage": shrinkage}, "ndcg": score / count}
        for (neighbors, shrinkage), score in zip(cf_configs, cf_scores)
    ]
    hybrid_candidates = [
        {
            "parameters": {"neighbors": neighbors, "prediction_shrinkage": shrinkage, "weights": weights},
            "ndcg": score / count,
        }
        for (neighbors, shrinkage, weights), score in zip(hybrid_configs, hybrid_scores)
    ]
    # max() preserves the first grid entry on exact ties.
    best_cf = max(cf_candidates, key=lambda row: row["ndcg"])
    best_hybrid = max(hybrid_candidates, key=lambda row: row["ndcg"])
    return {
        "fallback": False,
        "evaluated_users": count,
        "cf": {**best_cf, "candidates": cf_candidates},
        "hybrid": {**best_hybrid, "candidates": hybrid_candidates},
        "inner_train_interactions": len(inner_train),
        "inner_validation_interactions": len(validation),
    }


def evaluate_fold(fold, outer_train, outer_test, movies, settings, nested_settings, k, threshold, selection_k=None):
    """Select on outer_train only, refit there, then score the untouched outer_test."""
    started = time.perf_counter()
    selection = select_parameters(outer_train, movies, settings, nested_settings, selection_k or k, threshold)
    selection_seconds = time.perf_counter() - started

    cf_selected = selection["cf"]["parameters"]
    hybrid_selected = selection["hybrid"]["parameters"]
    cf_default = {
        "neighbors": settings["cf"]["neighbors"],
        "prediction_shrinkage": settings["cf"].get("prediction_shrinkage", 0.0),
    }
    model_neighbors = max(
        cf_selected["neighbors"], hybrid_selected["neighbors"], cf_default["neighbors"]
    )
    refit_started = time.perf_counter()
    popularity_scores, vectors, cf_model = _model_parts(outer_train, movies, settings, model_neighbors)
    refit_seconds = time.perf_counter() - refit_started
    train_users = group_by_user(outer_train)
    relevant_by_user = {}
    for user_id, movie_id, rating, _ in outer_test:
        if rating >= threshold:
            relevant_by_user.setdefault(user_id, set()).add(movie_id)

    model_settings = {
        "popularity": (None, None, {"popularity": 1.0}),
        "content": (None, None, {"content": 1.0}),
        "cf_default": (cf_default, None, {"cf": 1.0}),
        "hybrid_default": (cf_default, settings["hybrid_weights"], None),
        "cf_tuned": (cf_selected, None, {"cf": 1.0}),
        "hybrid_tuned": (hybrid_selected, hybrid_selected["weights"], None),
    }
    per_user = {name: {} for name in model_settings}
    coverage = {name: set() for name in model_settings}
    score_started = time.perf_counter()
    cf_views = {
        neighbors: _neighbor_view(cf_model, neighbors)
        for neighbors in {cf_default["neighbors"], cf_selected["neighbors"], hybrid_selected["neighbors"]}
    }
    for user_id, relevant in relevant_by_user.items():
        user_ratings = train_users.get(user_id, {})
        if not user_ratings:
            continue
        _, base_signals = hybrid.user_signals(user_ratings, movies, popularity_scores, vectors=vectors)
        predictions_by_config = {}
        candidates = movies.keys() - user_ratings.keys()
        for config in (cf_default, cf_selected, hybrid_selected):
            key = (config["neighbors"], config["prediction_shrinkage"])
            if key not in predictions_by_config:
                evidence = cf_views[config["neighbors"]].evidence(user_ratings, candidates)
                predictions_by_config[key] = _cf_predictions(evidence, config["prediction_shrinkage"])

        for name, (config, weights, fixed_weights) in model_settings.items():
            if name == "popularity":
                signals, rank_weights = {"popularity": base_signals.get("popularity", {})}, fixed_weights
            elif name == "content":
                signals, rank_weights = {"content": base_signals.get("content", {})}, fixed_weights
            elif name.startswith("cf_"):
                key = (config["neighbors"], config["prediction_shrinkage"])
                signals, rank_weights = {"cf": predictions_by_config[key]}, fixed_weights
            else:
                key = (config["neighbors"], config["prediction_shrinkage"])
                signals = dict(base_signals)
                signals["cf"] = predictions_by_config[key]
                rank_weights = weights
            rows = hybrid.rank(movies, signals, k, rank_weights)
            recommended = [row["movie_id"] for row in rows]
            per_user[name][f"u{fold}:{user_id}"] = evaluation.evaluate_user(recommended, relevant, k)
            coverage[name].update(recommended)
    score_seconds = time.perf_counter() - score_started

    metrics = {}
    for name, values in per_user.items():
        metrics[name] = evaluation.summarize(values, len(coverage[name]) / len(movies)) if values else {
            **{metric: None for metric in METRICS}, "coverage": 0.0, "users": 0,
        }
    return {
        "selection": selection,
        "metrics": metrics,
        "per_user": per_user,
        "_coverage": coverage,
        "resources": {
            "selection_seconds": selection_seconds,
            "refit_seconds": refit_seconds,
            "outer_scoring_seconds": score_seconds,
            "outer_train_interactions": len(outer_train),
            "outer_test_interactions": len(outer_test),
            "inner_train_interactions": selection["inner_train_interactions"],
            "inner_validation_interactions": selection["inner_validation_interactions"],
        },
    }


def _percentile(sorted_values, probability):
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] * (upper - position) + sorted_values[upper] * (position - lower)


def paired_bootstrap(left, right, metric="ndcg", resamples=2000, confidence=0.95, seed=2026):
    """Paired percentile interval for left-minus-right on matching user-fold rows."""
    if resamples <= 0 or not 0 < confidence < 1:
        raise ValueError("resamples must be positive and confidence must be between 0 and 1")
    keys = sorted(left.keys() & right.keys())
    differences = [left[key][metric] - right[key][metric] for key in keys]
    if not differences:
        raise ValueError("paired bootstrap needs at least one matching user")
    estimate = sum(differences) / len(differences)
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(differences) for _ in differences) / len(differences) for _ in range(resamples))
    tail = (1 - confidence) / 2
    return {
        "metric": metric,
        "left_minus_right": estimate,
        "lower": _percentile(means, tail),
        "upper": _percentile(means, 1 - tail),
        "confidence": confidence,
        "resamples": resamples,
        "paired_user_folds": len(differences),
    }


def aggregate_folds(folds, catalog_size, resamples=2000, confidence=0.95, seed=2026):
    names = next(iter(folds.values()))["metrics"].keys()
    pooled = {}
    for name in names:
        fold_values = [result["metrics"][name] for result in folds.values() if result["metrics"][name]["users"]]
        total_users = sum(row["users"] for row in fold_values)
        pooled[name] = {
            metric: sum(row[metric] * row["users"] for row in fold_values) / total_users if total_users else None
            for metric in METRICS
        }
        all_coverage = set().union(*(result["_coverage"][name] for result in folds.values()))
        pooled[name].update({"coverage": len(all_coverage) / catalog_size, "users": total_users})
        pooled[name]["fold_mean_std"] = {
            metric: {
                "mean": statistics.mean(row[metric] for row in fold_values) if fold_values else None,
                "std": statistics.stdev(row[metric] for row in fold_values) if len(fold_values) > 1 else 0.0 if fold_values else None,
            }
            for metric in (*METRICS, "coverage")
        }
        if fold_values:
            pooled[name]["fold_mean_std"]["coverage"] = {
                "mean": statistics.mean(row["coverage"] for row in fold_values),
                "std": statistics.stdev(row["coverage"] for row in fold_values) if len(fold_values) > 1 else 0.0,
            }

    per_user = {
        name: {key: value for result in folds.values() for key, value in result["per_user"][name].items()}
        for name in names
    }
    comparisons = {}
    for candidate, baseline in (("cf_tuned", "cf_default"), ("hybrid_tuned", "hybrid_default"), ("hybrid_tuned", "cf_tuned")):
        comparisons[f"{candidate}_vs_{baseline}_ndcg"] = paired_bootstrap(
            per_user[candidate], per_user[baseline], "ndcg", resamples, confidence, seed
        ) if candidate in per_user and baseline in per_user and per_user[candidate].keys() & per_user[baseline].keys() else None
    return {"pooled": pooled, "comparisons": comparisons}
