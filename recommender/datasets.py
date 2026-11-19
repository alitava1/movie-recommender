"""Read-only, streaming validation for MovieLens 100K and 1M."""

import sqlite3
import tempfile
from collections import Counter
from pathlib import Path

SUPPORTED = {"ml-100k", "ml-1m"}


class DatasetValidationError(ValueError):
    pass


def _parse_line(dataset, line, line_number):
    try:
        if dataset == "ml-100k":
            fields = line.rstrip("\r\n").split("\t")
            if len(fields) != 4:
                raise ValueError("expected 4 tab-separated fields")
            user, movie, rating, timestamp = fields
        else:
            fields = line.rstrip("\r\n").split("::")
            if len(fields) != 4:
                raise ValueError("expected 4 ::-separated fields")
            user, movie, rating, timestamp = fields
        if not all(value.strip() for value in (user, movie, rating, timestamp)):
            raise ValueError("missing field")
        user_id, movie_id = int(user), int(movie)
        rating_value, timestamp_value = float(rating), int(timestamp)
        if user_id <= 0 or movie_id <= 0:
            raise ValueError("user and movie IDs must be positive")
        if not 0.5 <= rating_value <= 5.0:
            raise ValueError("rating must be between 0.5 and 5")
        if timestamp_value < 0:
            raise ValueError("timestamp must be non-negative")
        return user_id, movie_id, rating_value
    except (ValueError, OverflowError) as exc:
        raise DatasetValidationError(f"ratings line {line_number}: {exc}") from exc


def _read_genres(dataset, path):
    counts = Counter()
    movie_ids = set()
    with path.open("r", encoding="latin-1", newline="") as source:
        for line_number, line in enumerate(source, 1):
            if dataset == "ml-100k":
                fields = line.rstrip("\r\n").split("|")
                if len(fields) != 24 or not fields[0].strip() or not fields[1].strip():
                    raise DatasetValidationError(f"movies line {line_number}: malformed or missing metadata")
                try:
                    movie_id = int(fields[0])
                    flags = [int(value) for value in fields[5:24]]
                except ValueError as exc:
                    raise DatasetValidationError(f"movies line {line_number}: invalid ID or genre flag") from exc
                if movie_id <= 0 or any(value not in (0, 1) for value in flags):
                    raise DatasetValidationError(f"movies line {line_number}: invalid ID or genre flag")
                for genre, enabled in zip(("unknown", "action", "adventure", "animation", "children", "comedy", "crime", "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery", "romance", "sci-fi", "thriller", "war", "western"), flags):
                    if enabled:
                        counts[genre] += 1
            else:
                # rsplit keeps any "::" inside a title in the title portion.
                head, genre_field = line.rstrip("\r\n").rsplit("::", 1)
                id_field, title_field = head.split("::", 1)
                if not id_field.strip() or not title_field.strip() or not genre_field.strip():
                    raise DatasetValidationError(f"movies line {line_number}: malformed or missing metadata")
                try:
                    movie_id = int(id_field)
                except ValueError as exc:
                    raise DatasetValidationError(f"movies line {line_number}: invalid movie ID") from exc
                if movie_id <= 0:
                    raise DatasetValidationError(f"movies line {line_number}: invalid movie ID")
                for genre in genre_field.split("|"):
                    if not genre.strip():
                        raise DatasetValidationError(f"movies line {line_number}: empty genre")
                    counts[genre.strip().lower()] += 1
            if movie_id in movie_ids:
                raise DatasetValidationError(f"movies line {line_number}: duplicate movie ID {movie_id}")
            movie_ids.add(movie_id)
    return movie_ids, counts


def validate_dataset(dataset, dataset_dir, temp_root):
    if dataset not in SUPPORTED:
        raise ValueError(f"unsupported dataset {dataset!r}; choose one of {sorted(SUPPORTED)}")
    directory = Path(dataset_dir)
    ratings_path = directory / ("u.data" if dataset == "ml-100k" else "ratings.dat")
    movies_path = directory / ("u.item" if dataset == "ml-100k" else "movies.dat")
    for path in (ratings_path, movies_path):
        if not path.is_file():
            raise FileNotFoundError(f"required file missing: {path}")

    movie_ids, genre_counts = _read_genres(dataset, movies_path)
    users, rated_movies, rating_counts = set(), set(), Counter()
    interactions = duplicates = 0
    Path(temp_root).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="validate-", dir=temp_root) as scratch:
        connection = sqlite3.connect(Path(scratch) / "seen.sqlite3")
        try:
            connection.execute("CREATE TABLE interactions (user_id INTEGER, movie_id INTEGER, PRIMARY KEY (user_id, movie_id))")
            with ratings_path.open("r", encoding="ascii", newline="") as source:
                for line_number, line in enumerate(source, 1):
                    user_id, movie_id, rating = _parse_line(dataset, line, line_number)
                    interactions += 1
                    users.add(user_id)
                    rated_movies.add(movie_id)
                    rating_counts[f"{rating:g}"] += 1
                    cursor = connection.execute("INSERT OR IGNORE INTO interactions VALUES (?, ?)", (user_id, movie_id))
                    duplicates += cursor.rowcount == 0
            connection.commit()
        finally:
            connection.close()

    possible = len(users) * len(movie_ids)
    return {
        "dataset": dataset,
        "ratings_file": ratings_path.name,
        "movies_file": movies_path.name,
        "interactions": interactions,
        "duplicate_user_movie_pairs": duplicates,
        "users": len(users),
        "rated_movies": len(rated_movies),
        "catalog_movies": len(movie_ids),
        "rating_distribution": dict(sorted(rating_counts.items(), key=lambda item: float(item[0]))),
        "genre_distribution": dict(sorted(genre_counts.items())),
        "interaction_sparsity": 1 - interactions / possible if possible else 0,
        "valid": duplicates == 0 and rated_movies <= movie_ids,
        "movies_without_metadata": len(rated_movies - movie_ids),
    }
