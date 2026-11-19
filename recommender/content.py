"""TF-IDF over movie titles and genres, with cosine user profiles."""

import math
import re
from collections import Counter

ARTIFACT_KIND = "tfidf-content"
_TOKEN = re.compile(r"[^\W_]+", re.UNICODE)
_STOPWORDS = {"the", "a", "an", "of", "and", "in", "on", "to", "for", "de", "la", "le"}


def tokens(title):
    words = [word for word in _TOKEN.findall(title.lower()) if word not in _STOPWORDS]
    return [word for word in words if not word.isdigit()]


def movie_terms(movie):
    return tokens(movie.title) + [f"genre:{genre}" for genre in movie.genres]


def build_tfidf(movies):
    """Return normalized TF-IDF vectors keyed by movie ID."""
    documents = {movie_id: Counter(movie_terms(movie)) for movie_id, movie in movies.items()}
    document_count = len(documents)
    frequency = Counter()
    for counts in documents.values():
        frequency.update(counts.keys())
    vectors = {}
    for movie_id, counts in documents.items():
        total = sum(counts.values()) or 1
        weighted = {
            term: (count / total) * (math.log((1 + document_count) / (1 + frequency[term])) + 1)
            for term, count in counts.items()
        }
        norm = math.sqrt(sum(value * value for value in weighted.values()))
        vectors[movie_id] = {term: value / norm for term, value in weighted.items()} if norm else {}
    return vectors


def user_profile(vectors, user_ratings):
    """Rating-deviation-weighted centroid; empty when preferences carry no signal."""
    if not user_ratings:
        return {}
    mean = sum(user_ratings.values()) / len(user_ratings)
    profile = {}
    for movie_id, rating in user_ratings.items():
        for term, value in vectors.get(movie_id, {}).items():
            profile[term] = profile.get(term, 0.0) + (rating - mean) * value
    norm = math.sqrt(sum(value * value for value in profile.values()))
    return {term: value / norm for term, value in profile.items()} if norm else {}


def score(profile, vectors, candidates=None):
    """Cosine similarity of the user profile against movie vectors."""
    if not profile:
        return {}
    candidates = candidates if candidates is not None else vectors
    return {
        movie_id: sum(profile.get(term, 0.0) * value for term, value in vectors[movie_id].items())
        for movie_id in candidates if vectors.get(movie_id)
    }


def save_vectors(vectors):
    return {str(key): value for key, value in vectors.items()}


def load_vectors(payload):
    return {int(key): value for key, value in payload.items()}
