"""Offline ranking evaluation on the official MovieLens 100K u1-u5 splits.

Every fold builds its models from uN.base only and scores uN.test, so no test
rating reaches similarity, profiles, or popularity. Percentages are pooled
per-user means across folds, never best-fold numbers.
"""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts, content, evaluation, hybrid, popularity
from recommender.collaborative import ItemItemCF
from recommender.data import group_by_user, load_movies, load_ratings

MODELS = {
    "popularity": {"popularity": 1.0},
    "content": {"content": 1.0},
    "cf": {"cf": 1.0},
    "hybrid": None,
}


def evaluate_fold(dataset_dir, fold, settings, k, threshold):
    train = load_ratings("ml-100k", dataset_dir / f"u{fold}.base")
    test = load_ratings("ml-100k", dataset_dir / f"u{fold}.test")
    movies = load_movies("ml-100k", dataset_dir / "u.item")
    train_users = group_by_user(train)

    started = time.perf_counter()
    popularity_scores = popularity.popularity_scores(train, settings["popularity"]["prior_count"])
    vectors = content.build_tfidf(movies)
    cf = ItemItemCF.build(train, **settings["cf"])
    build_seconds = time.perf_counter() - started

    relevants = {}
    for user_id, movie_id, rating, _ in test:
        if rating >= threshold:
            relevants.setdefault(user_id, set()).add(movie_id)

    per_user = {name: {} for name in MODELS}
    coverage = {name: set() for name in MODELS}
    started = time.perf_counter()
    for user_id, relevant in relevants.items():
        user_ratings = train_users.get(user_id, {})
        if not user_ratings:
            continue
        _, signals = hybrid.user_signals(user_ratings, movies, popularity_scores, cf, vectors)
        for name, weights in MODELS.items():
            rows = hybrid.rank(movies, signals, k, weights or settings["hybrid_weights"])
            recommended = [row["movie_id"] for row in rows]
            per_user[name][user_id] = evaluation.evaluate_user(recommended, relevant, k)
            coverage[name].update(recommended)
    score_seconds = time.perf_counter() - started

    results = {
        name: evaluation.summarize(values, len(coverage[name]) / len(movies))
        for name, values in per_user.items()
    }
    return results, coverage, len(movies), build_seconds, score_seconds


def main():
    parser = argparse.ArgumentParser(description="Evaluate popularity, content, CF, and hybrid ranking on MovieLens 100K.")
    parser.add_argument("--dataset", default="ml-100k", choices=["ml-100k"])
    parser.add_argument("--folds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts" / "evaluation" / "ml-100k.json")
    args = parser.parse_args()
    if len(set(args.folds)) != len(args.folds):
        parser.error("--folds must not contain duplicates")
    if any(fold not in range(1, 6) for fold in args.folds):
        parser.error("--folds values must be 1-5")
    if not args.output.resolve().is_relative_to(ROOT):
        parser.error("--output must be inside the project directory")

    experiment = json.loads((ROOT / "configs" / "experiment.json").read_text(encoding="utf-8"))
    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    k = args.k if args.k is not None else experiment["evaluation"]["k"]
    if k <= 0:
        parser.error("--k must be positive")
    threshold = experiment["evaluation"]["relevant_threshold"]
    dataset_dir = ROOT / "data" / "raw" / args.dataset

    print(f"Evaluation: {args.dataset}, folds {args.folds}, k={k}, relevant if rating >= {threshold}")
    folded = {}
    coverage_sets = {name: set() for name in MODELS}
    catalog_sizes = set()
    started = time.perf_counter()
    for fold in args.folds:
        results, recommended, catalog_size, build_seconds, score_seconds = evaluate_fold(dataset_dir, fold, settings, k, threshold)
        folded[f"u{fold}"] = results
        for name in MODELS:
            coverage_sets[name].update(recommended[name])
        catalog_sizes.add(catalog_size)
        users = results["hybrid"]["users"]
        print(f"  u{fold}: {users} users, build {build_seconds:.1f}s, score {score_seconds:.1f}s")
    elapsed = time.perf_counter() - started

    # Pool per-user metric means across folds; weight each user once, not each fold equally.
    if len(catalog_sizes) != 1:
        raise ValueError("catalog size changed across folds")
    catalog_size = next(iter(catalog_sizes))
    pooled = {}
    for name in MODELS:
        users = sum(fold_results[name]["users"] for fold_results in folded.values())
        pooled[name] = {
            metric: sum(fold_results[name][metric] * fold_results[name]["users"] for fold_results in folded.values()) / users
            for metric in ("precision", "recall", "ndcg", "map")
        }
        pooled[name]["coverage"] = len(coverage_sets[name]) / catalog_size
        pooled[name]["users"] = users

    header = f"{'model':<12}{'P@'+str(k):>9}{'R@'+str(k):>9}{'NDCG@'+str(k):>9}{'MAP@'+str(k):>9}{'coverage':>10}{'users':>8}"
    print(header)
    for name, values in pooled.items():
        print(
            f"{name:<12}{values['precision']:>9.4f}{values['recall']:>9.4f}"
            f"{values['ndcg']:>9.4f}{values['map']:>9.4f}{values['coverage']:>10.4f}{values['users']:>8}"
        )
    print(f"Total wall time: {elapsed:.1f}s")

    document = {
        "dataset": args.dataset,
        "k": k,
        "relevant_threshold": threshold,
        "weights": settings["hybrid_weights"],
        "elapsed_seconds": round(elapsed, 1),
        "pooled": pooled,
        "folds": folded,
    }
    artifacts.save(args.output, document, "evaluation")
    print(f"Results written to {args.output}")


if __name__ == "__main__":
    main()
