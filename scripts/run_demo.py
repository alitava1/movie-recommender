import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts, content, hybrid
from recommender.collaborative import ARTIFACT_KIND as CF_KIND, ItemItemCF
from recommender.content import ARTIFACT_KIND as CONTENT_KIND
from recommender.data import Movie, group_by_user, load_ratings


def main():
    parser = argparse.ArgumentParser(description="Print hybrid recommendations for a MovieLens user.")
    parser.add_argument("--dataset", default="ml-100k", choices=["ml-100k", "ml-1m"])
    parser.add_argument("--user", type=int, default=1)
    parser.add_argument("--ratings-file", help="Optional training ratings file; defaults to the full ratings file")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts", help="Artifact root directory")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()
    if args.user <= 0 or args.k <= 0:
        parser.error("--user and -k must be positive")
    if args.ratings_file and Path(args.ratings_file).name != args.ratings_file:
        parser.error("--ratings-file must be a file name inside the selected dataset directory")
    if not args.artifacts.resolve().is_relative_to(ROOT):
        parser.error("--artifacts must be inside the project directory")

    dataset_dir = ROOT / "data" / "raw" / args.dataset
    model_dir = args.artifacts / args.dataset
    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    ratings_file = args.ratings_file or ("u.data" if args.dataset == "ml-100k" else "ratings.dat")
    ratings = load_ratings(args.dataset, dataset_dir / ratings_file)
    user_ratings = group_by_user(ratings).get(args.user, {})

    popularity_scores = {int(movie_id): score for movie_id, score in artifacts.load(model_dir / "popularity.json", "popularity").items()}
    content_payload = artifacts.load(model_dir / "content.json", CONTENT_KIND)
    movies = {
        int(movie_id): Movie(int(movie_id), value["title"], value["year"], tuple(value["genres"]))
        for movie_id, value in content_payload["movies"].items()
    }
    vectors = content.load_vectors(content_payload["vectors"])
    cf = ItemItemCF.from_payload(artifacts.load(model_dir / "cf.json", CF_KIND))
    results = hybrid.recommend(user_ratings, movies, popularity_scores, cf, vectors, args.k, settings["hybrid_weights"])

    print(f"Dataset: {args.dataset}; user: {args.user}; rated items: {len(user_ratings)}")
    for rank, row in enumerate(results, 1):
        parts = ", ".join(f"{name}={value:.3f}" for name, value in row["components"].items() if value is not None)
        print(f"{rank:2}. {row['title']} [movie {row['movie_id']}]  hybrid={row['score']:.3f}  {parts}")
    if not results:
        print("No eligible recommendations available.")


if __name__ == "__main__":
    main()
