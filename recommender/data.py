"""Read-only dataset loaders for model building."""

import re
from dataclasses import dataclass
from pathlib import Path

from .datasets import DatasetValidationError

GENRES_100K = (
    "unknown", "action", "adventure", "animation", "children", "comedy", "crime",
    "documentary", "drama", "fantasy", "film-noir", "horror", "musical", "mystery",
    "romance", "sci-fi", "thriller", "war", "western",
)
_YEAR = re.compile(r"\((\d{4})\)\s*$")


@dataclass(frozen=True)
class Movie:
    movie_id: int
    title: str
    year: int | None
    genres: tuple[str, ...]


def _parse_rating_line(dataset, line, line_number):
    fields = line.rstrip("\r\n").split("\t" if dataset == "ml-100k" else "::")
    try:
        if len(fields) != 4:
            raise ValueError(f"expected 4 fields, got {len(fields)}")
        if not all(field.strip() for field in fields):
            raise ValueError("missing field")
        user_id, movie_id = int(fields[0]), int(fields[1])
        rating, timestamp = float(fields[2]), int(fields[3])
        if user_id <= 0 or movie_id <= 0:
            raise ValueError("user and movie IDs must be positive")
        if not 0.5 <= rating <= 5.0:
            raise ValueError("rating must be between 0.5 and 5")
        if timestamp < 0:
            raise ValueError("timestamp must be non-negative")
        return user_id, movie_id, rating, timestamp
    except (ValueError, OverflowError) as exc:
        raise DatasetValidationError(f"ratings line {line_number}: {exc}") from exc


def load_ratings(dataset, path):
    """Return (user_id, movie_id, rating, timestamp) tuples in file order."""
    if dataset not in {"ml-100k", "ml-1m"}:
        raise ValueError(f"unsupported dataset {dataset!r}")
    ratings = []
    with Path(path).open("r", encoding="ascii", newline="") as source:
        for line_number, line in enumerate(source, 1):
            ratings.append(_parse_rating_line(dataset, line, line_number))
    return ratings


def load_movies(dataset, path):
    if dataset not in {"ml-100k", "ml-1m"}:
        raise ValueError(f"unsupported dataset {dataset!r}")
    movies = {}
    with Path(path).open("r", encoding="latin-1", newline="") as source:
        for line_number, line in enumerate(source, 1):
            line = line.rstrip("\r\n")
            if dataset == "ml-100k":
                fields = line.split("|")
                if len(fields) != 24 or not all(field.strip() for field in (fields[0], fields[1])):
                    raise DatasetValidationError(f"movies line {line_number}: malformed or missing metadata")
                try:
                    movie_id = int(fields[0])
                    flags = [int(value) for value in fields[5:24]]
                except ValueError as exc:
                    raise DatasetValidationError(f"movies line {line_number}: invalid ID or genre flag") from exc
                if movie_id <= 0 or any(flag not in (0, 1) for flag in flags):
                    raise DatasetValidationError(f"movies line {line_number}: invalid ID or genre flag")
                title = fields[1]
                genres = tuple(genre for genre, flag in zip(GENRES_100K, flags) if flag)
            else:
                head, genre_field = line.rsplit("::", 1)
                id_field, title = head.split("::", 1)
                if not id_field.strip() or not title.strip() or not genre_field.strip():
                    raise DatasetValidationError(f"movies line {line_number}: malformed or missing metadata")
                try:
                    movie_id = int(id_field)
                except ValueError as exc:
                    raise DatasetValidationError(f"movies line {line_number}: invalid movie ID") from exc
                if movie_id <= 0:
                    raise DatasetValidationError(f"movies line {line_number}: invalid movie ID")
                genres = tuple(genre.strip().lower() for genre in genre_field.split("|"))
            if movie_id in movies:
                raise DatasetValidationError(f"movies line {line_number}: duplicate movie ID {movie_id}")
            match = _YEAR.search(title)
            movies[movie_id] = Movie(movie_id, title, int(match.group(1)) if match else None, genres)
    return movies


def load_dataset(dataset, data_dir, ratings_name=None):
    """Load ratings and metadata from an already-validated dataset directory."""
    directory = Path(data_dir)
    ratings_name = ratings_name or ("u.data" if dataset == "ml-100k" else "ratings.dat")
    movies_name = "u.item" if dataset == "ml-100k" else "movies.dat"
    ratings = load_ratings(dataset, directory / ratings_name)
    movies = load_movies(dataset, directory / movies_name)
    missing = {movie_id for _, movie_id, _, _ in ratings} - movies.keys()
    if missing:
        raise DatasetValidationError(f"{len(missing)} rated movies lack metadata")
    return ratings, movies


def group_by_user(ratings):
    grouped = {}
    for user_id, movie_id, rating, _ in ratings:
        grouped.setdefault(user_id, {})[movie_id] = rating
    return grouped
