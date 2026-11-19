import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from recommender import artifacts, content, popularity
from recommender.collaborative import ARTIFACT_KIND, ItemItemCF
from recommender.data import load_dataset


def main():
    parser = argparse.ArgumentParser(description="Build popularity, TF-IDF, and collaborative-filtering artifacts.")
    parser.add_argument("--dataset", default="ml-100k", choices=["ml-100k", "ml-1m"])
    parser.add_argument("--ratings-file", help="Optional ratings file name, e.g. u1.base")
    parser.add_argument("--artifacts", type=Path, default=ROOT / "artifacts", help="Artifact root directory")
    args = parser.parse_args()
    if args.ratings_file and Path(args.ratings_file).name != args.ratings_file:
        parser.error("--ratings-file must be a file name inside the selected dataset directory")
    if not args.artifacts.resolve().is_relative_to(ROOT):
        parser.error("--artifacts must be inside the project directory")

    settings = json.loads((ROOT / "configs" / "model.json").read_text(encoding="utf-8"))
    data_dir = ROOT / "data" / "raw" / args.dataset
    output = args.artifacts / args.dataset
    start = time.perf_counter()
    ratings, movies = load_dataset(args.dataset, data_dir, args.ratings_file)
    print(f"Loaded {len(ratings)} ratings, {len(movies)} movies in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    scores = popularity.popularity_scores(ratings, settings["popularity"]["prior_count"])
    artifacts.save(output / "popularity.json", scores, "popularity")
    print(f"Popularity artifact: {len(scores)} movies in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    vectors = content.build_tfidf(movies)
    artifacts.save(output / "content.json", {
        "movies": {str(movie_id): {"title": movie.title, "year": movie.year, "genres": list(movie.genres)} for movie_id, movie in movies.items()},
        "vectors": content.save_vectors(vectors),
    }, content.ARTIFACT_KIND)
    print(f"TF-IDF artifact: {len(vectors)} vectors in {time.perf_counter() - start:.2f}s")

    start = time.perf_counter()
    model = ItemItemCF.build(ratings, **settings["cf"])
    artifacts.save(output / "cf.json", model.to_payload(), ARTIFACT_KIND)
    print(f"CF artifact: {sum(len(value) for value in model.item_neighbors.values())} neighbor links in {time.perf_counter() - start:.2f}s")
    print(f"Artifacts written under {output}")


if __name__ == "__main__":
    main()
