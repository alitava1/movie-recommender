"""Phase 14 application service: Phase 10 hybrid engine (three_w3) over a local dataset.

Default dataset ml-100k (fast startup); `--dataset ml-1m` for the research-scale catalog.
Personality is an explicit, disabled-by-default experiment stub (Phase 13 Outcome D):
it is never loaded and never influences recommendations.
"""

import math
import threading
import time

from .. import content as content_mod
from . import metadata as metadata_mod
from ..data import load_dataset
from ..phase10.diversity import rank_candidates
from ..phase10.scoring import extract_components, normalize_components
from ..phase7.cf_adapter import build_cf

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # -I isolation mode: experimental personality mode requires numpy
    np = None
    HAVE_NUMPY = False

def _mm(values):
    if not values:
        return {}
    lo, hi = min(values.values()), max(values.values())
    span = hi - lo
    return {m: ((v - lo) / span if span > 0 else 0.5) for m, v in values.items()}

POSITIVE_THRESHOLD = 4.0
POOL_M = 200
HYBRID_WEIGHTS = {"cf": 0.4, "popularity": 0.2, "content": 0.4}  # Phase 10 three_w3


class RecommendationService:
    """Loads a MovieLens dataset once, exposes browse/search/details/recommendations."""

    def __init__(self, dataset="ml-100k", data_dir=None, neighbors=100):
        self.dataset = dataset
        t0 = time.time()
        data_dir = data_dir or f"data/raw/{dataset}"
        self.ratings, self.movies = load_dataset(dataset, data_dir)
        self.catalog = sorted(self.movies)
        self.item_index = {mid: i for i, mid in enumerate(self.catalog)}
        self.vectors = content_mod.build_tfidf(self.movies)
        self.pop_counts = {}
        self.item_sum = {}
        for _, mid, rating, _ in self.ratings:
            self.pop_counts[mid] = self.pop_counts.get(mid, 0) + 1
            self.item_sum[mid] = self.item_sum.get(mid, 0.0) + rating
        self.pop_ranked = sorted(self.catalog, key=lambda m: (-self.pop_counts.get(m, 0), m))
        self.cf_model = build_cf(self.ratings, neighbors=neighbors, shrinkage=10.0, prediction_shrinkage=0.0)
        self.prediction_shrinkage = 300.0
        self._cf_cache = {}
        self._pers_cache = {}
        self.metadata = None  # optional TMDBMetadataProvider wired by create_server
        self.load_seconds = round(time.time() - t0, 1)
        self.personality_enabled = False  # Phase 13 Outcome D: never enabled by the app
        self._lock = threading.Lock()

    # -- dataset-side helpers -------------------------------------------------

    def _cf_scores(self, user_ratings, pool):
        key = tuple(sorted(user_ratings.items()))
        with self._lock:
            if key in self._cf_cache:
                return self._cf_cache[key]
        model = self.cf_model
        u_mean = sum(user_ratings.values()) / len(user_ratings)
        votes = {}
        for rated_mid, rating in user_ratings.items():
            for nb_mid, sim in model.item_neighbors.get(rated_mid, []):
                if nb_mid in pool:
                    votes.setdefault(nb_mid, []).append((sim, rating))
        scores = {}
        for mid, v in votes.items():
            num = sum(sim * (r - u_mean) for sim, r in v)
            den = sum(abs(sim) for sim, _ in v)
            mean = sum(r for _, r in v) / len(v)
            scores[mid] = min(5.0, max(0.5, mean + num / (den + self.prediction_shrinkage)))
        with self._lock:
            self._cf_cache[key] = scores
        return scores

    def popularity_pool(self, seen, m=POOL_M):
        return [mid for mid in self.pop_ranked if mid not in seen][:m]

    # -- public API -----------------------------------------------------------

    def search(self, query, limit=50):
        q = (query or "").strip().lower()
        if not q:
            return []
        hits = [m for m in self.catalog if q in self.movies[m].title.lower()]
        hits.sort(key=lambda mid: (-self.pop_counts.get(mid, 0), mid))
        return hits[:limit]

    def browse(self, offset=0, limit=50, genre=None, sort="popular", year_min=None, year_max=None, min_rating=None):
        items = self.catalog
        if genre:
            g = genre.lower()
            items = [mid for mid in items if g in self.movies[mid].genres]
        if year_min is not None:
            items = [mid for mid in items if self.movies[mid].year is not None and self.movies[mid].year >= year_min]
        if year_max is not None:
            items = [mid for mid in items if self.movies[mid].year is not None and self.movies[mid].year <= year_max]
        if min_rating is not None:
            items = [mid for mid in items
                     if self.pop_counts.get(mid, 0) > 0 and (self.item_sum[mid] / self.pop_counts[mid]) >= min_rating]

        if sort == "highest_rated":
            min_support = 20 if self.dataset == "ml-1m" else 15
            items = [mid for mid in items if self.pop_counts.get(mid, 0) >= min_support]
            items.sort(key=lambda mid: (-(self.item_sum[mid] / self.pop_counts[mid]), -self.pop_counts[mid], mid))
        elif sort == "year_desc":
            items.sort(key=lambda mid: (-(self.movies[mid].year or 0), -self.pop_counts.get(mid, 0), mid))
        elif sort == "year_asc":
            items.sort(key=lambda mid: (self.movies[mid].year or 9999, -self.pop_counts.get(mid, 0), mid))
        elif sort == "title":
            items.sort(key=lambda mid: (self.movies[mid].title.lower(), mid))
        else:
            items = sorted(items, key=lambda mid: (-self.pop_counts.get(mid, 0), mid))
        window = items[offset:offset + limit]
        return window, len(items)

    def highest_rated(self, exclude=(), limit=12):
        skip = set(exclude)
        min_support = 30 if self.dataset == "ml-1m" else 20
        cands = [
            (self.item_sum[mid] / self.pop_counts[mid], self.pop_counts[mid], mid)
            for mid in self.catalog
            if mid not in skip and self.pop_counts.get(mid, 0) >= min_support
        ]
        cands.sort(key=lambda row: (-row[0], -row[1], row[2]))
        return self._enrich_details([self.movie_details(mid) for _, _, mid in cands[:limit]])

    def recent_releases(self, exclude=(), limit=10):
        skip = set(exclude)
        min_support = 10
        cands = [
            (self.movies[mid].year, self.pop_counts.get(mid, 0), mid)
            for mid in self.catalog
            if mid not in skip and self.movies[mid].year is not None and self.pop_counts.get(mid, 0) >= min_support
        ]
        cands.sort(key=lambda row: (-row[0], -row[1], row[2]))
        return self._enrich_details([self.movie_details(mid) for _, _, mid in cands[:limit]])

    def all_genres(self):
        genres = sorted({g for m in self.movies.values() for g in m.genres})
        return genres

    def similar_movies(self, mid, exclude=(), limit=8):
        """Real TF-IDF cosine neighbors using existing Phase 2 movie vectors."""
        vec = self.vectors.get(mid)
        if not vec:
            return []
        skip = set(exclude) | {mid}
        scored = content_mod.score(vec, self.vectors, self.movies.keys() - skip)
        ordered = sorted(scored.items(), key=lambda kv: (-kv[1], -self.pop_counts.get(kv[0], 0), kv[0]))[:limit]
        out = []
        for other_id, sim in ordered:
            d = self.movie_details(other_id)
            if d:
                d["similarity"] = round(sim, 3)
                out.append(d)
        return self._enrich_details(out)

    def hidden_gems(self, exclude=(), limit=12):
        """Acclaimed films with lower mainstream rating volume (real dataset statistics)."""
        skip = set(exclude)
        min_n = 25 if self.dataset == "ml-1m" else 15
        max_n = 250 if self.dataset == "ml-1m" else 80
        cands = []
        for mid in self.catalog:
            if mid in skip:
                continue
            n = self.pop_counts.get(mid, 0)
            if min_n <= n <= max_n:
                mean = self.item_sum[mid] / n
                if mean >= 3.8:
                    cands.append((mean, n, mid))
        cands.sort(key=lambda row: (-row[0], -row[1], row[2]))
        return self._enrich_details([self.movie_details(mid) for _, _, mid in cands[:limit]])

    def because_you_liked(self, user_ratings, limit=8):
        """If the user has rated >= 4.0 movies, find TF-IDF neighbors for their top anchor film."""
        positives = [(r, self.pop_counts.get(mid, 0), mid)
                     for mid, r in user_ratings.items() if r >= POSITIVE_THRESHOLD and mid in self.movies]
        if not positives:
            return None, []
        positives.sort(key=lambda row: (-row[0], -row[1], row[2]))
        anchor_id = positives[0][2]
        anchor = self.movie_details(anchor_id)
        items = self.similar_movies(anchor_id, exclude=user_ratings.keys(), limit=limit)
        return anchor, items

    def discover(self, user_ratings):
        """Curated editorial payload using real dataset facts and Phase 10 recommendations."""
        seen = set(user_ratings)
        recs = self._enrich_details(self.recommend(user_ratings, top_n=10))
        popular_ids = self.popularity_pool(seen, 12)
        popular = self._enrich_details([self.movie_details(mid) for mid in popular_ids])
        acclaimed = self.highest_rated(exclude=seen, limit=10)
        recent = self.recent_releases(exclude=seen, limit=10)
        anchor, similar = self.because_you_liked(user_ratings, limit=10)
        gems = self.hidden_gems(exclude=seen, limit=10)
        # Featured film: top personalized recommendation when available, else top acclaimed popular film.
        featured = None
        if recs:
            featured = self.movie_details(recs[0]["movie_id"], fetch_remote=True)
            if featured:
                featured["explanation"] = recs[0].get("explanation")
                featured["score"] = recs[0].get("score")
        elif popular:
            featured = self.movie_details(popular[0]["movie_id"], fetch_remote=True)
        return {
            "featured": featured,
            "recommendations": recs,
            "because_you_liked": {"anchor": anchor, "movies": similar} if anchor and similar else None,
            "popular": popular,
            "highest_rated": acclaimed,
            "recent_releases": recent,
            "hidden_gems": gems,
            "genres": self.all_genres(),
            "dataset": self.dataset,
            "catalog_size": len(self.catalog),
            "metadata_status": self.metadata.status_summary() if self.metadata else {"provider": "TMDB", "configured": False},
        }

    def _attach_meta(self, details, meta):
        if not details or not meta or meta.get("status") != "found":
            return details
        use_img_proxy = bool(self.metadata and self.metadata.proxy_url)
        for k in ("poster_url", "backdrop_url", "overview", "release_date", "runtime",
                  "original_language", "tmdb_rating", "tmdb_votes", "director", "cast", "tmdb_id"):
            if meta.get(k) is not None:
                val = meta[k]
                if use_img_proxy and k in ("poster_url", "backdrop_url"):
                    val = metadata_mod.rewrite_tmdb_image_url(val)
                details[k] = val
        return details

    def _enrich_details(self, details_list):
        """Batch-attach already-cached metadata without making any external API requests."""
        clean = [d for d in details_list if d]
        if not clean or not self.metadata or not self.metadata.store:
            return clean
        cached = self.metadata.store.get_metadata_batch([d["movie_id"] for d in clean])
        for d in clean:
            meta = cached.get(d["movie_id"])
            if meta:
                self._attach_meta(d, meta)
        return clean

    def movie_details(self, mid, fetch_remote=False):
        movie = self.movies.get(mid)
        if not movie:
            return None
        n = self.pop_counts.get(mid, 0)
        out = {
            "movie_id": mid,
            "title": movie.title,
            "year": movie.year,
            "genres": list(movie.genres),
            "rating_count": n,
            "mean_rating": round(self.item_sum[mid] / n, 2) if n else None,
        }
        if self.metadata:
            meta = self.metadata.get_metadata(mid, movie.title, movie.year, allow_remote=fetch_remote)
            self._attach_meta(out, meta)
        return out

    def recommend(self, user_ratings, top_n=10, exclude_rated=True, pool_m=POOL_M):
        """Phase 10 three_w3 hybrid over a fixed popularity pool; honest explanations."""
        seen = set(user_ratings)
        pool = self.popularity_pool(seen if exclude_rated else set(), pool_m)
        if not pool:
            return []
        if not user_ratings:
            recs = pool[:top_n]
            return [self._explain(mid, None, None, "popular with all users (new-user popularity fallback)") for mid in recs]

        cf_scores = self._cf_scores(user_ratings, set(pool))
        profile = content_mod.user_profile(self.vectors, user_ratings)
        content_scores = content_mod.score(profile, self.vectors, set(pool)) if profile else {}
        components, _cov = extract_components(user_ratings, pool, cf_scores, self.pop_counts, content_scores, {})
        norm = normalize_components(components)
        _recs, full_scores, _diag = rank_candidates(pool, norm, "three_signal", {"weights": HYBRID_WEIGHTS})
        top = sorted(full_scores.items(), key=lambda r: (-r[1], r[0]))[:top_n]

        top_genres = self._user_top_genres(user_ratings)
        results = []
        for mid, score in top:
            cf_v = norm["cf"].get(mid)
            pop_v = norm["popularity"].get(mid, 0.0)
            cont_v = norm["content"].get(mid)
            signal, text = self._top_signal(cf_v, pop_v, cont_v, mid, top_genres, user_ratings)
            results.append(self._explain(mid, score, {"cf": cf_v, "popularity": pop_v, "content": cont_v}, text, signal))
        return results

    def _explain(self, mid, score, components, text, signal=None):
        details = self.movie_details(mid)
        out = dict(details or {"movie_id": mid, "title": f"Movie {mid}", "year": None, "genres": []})
        out["score"] = round(score, 4) if score is not None else None
        out["explanation"] = text
        out["explanation_signal"] = signal
        if components is not None:
            out["components"] = {k: (round(v, 3) if v is not None else None) for k, v in components.items()}
        return out

    def _user_top_genres(self, user_ratings, k=2):
        genre_scores = {}
        for mid, rating in user_ratings.items():
            if rating < POSITIVE_THRESHOLD or mid not in self.movies:
                continue
            for g in self.movies[mid].genres:
                genre_scores[g] = genre_scores.get(g, 0.0) + rating
        return [g for g, _ in sorted(genre_scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]]

    def _top_signal(self, cf_v, pop_v, cont_v, mid, top_genres, user_ratings):
        parts = []
        if cf_v is not None:
            parts.append(("collaborative", cf_v, "viewers with similar rating histories also liked this movie"))
        if cont_v is not None:
            parts.append(("content", cont_v, "its title/genre profile matches movies you rated highly"))
        parts.append(("popularity", pop_v, "it is popular across the whole community"))
        signal, _v, text = max(parts, key=lambda p: p[1])
        if top_genres and self.movies.get(mid):
            matched = [g for g in self.movies[mid].genres if g in top_genres]
            if matched:
                text += f"; matches your preferred {', '.join(matched)} genre(s)"
        return signal, text

    def rebuild(self):
        """Explicit expensive recomputation (never triggered per rating)."""
        self.__init__(dataset=self.dataset,
                      data_dir=f"data/raw/{self.dataset}",
                      neighbors=100)
        return {"rebuilt": True, "load_seconds": self.load_seconds}

    # -- experimental personality mode (Phase 15; off by default; Phase 13 Outcome D) --

    MIN_PROFILED_USERS = 5
    DEFAULT_EXPERIMENTAL_WEIGHT = 0.15  # Phase 13 diagnostic value; NOT validated — exposed, not auto-tuned

    def experimental_recommend(self, store, user_id, top_n=10, weight=None):
        """Experimental personality-blended ranking (Phase 11 architecture, Phase 13 recipe).

        Requires: the requesting user's own self-reported profile, and at least
        MIN_PROFILED_USERS profiled application users with ratings to fit the Ridge model on.
        Returns (recs, meta) or raises ValueError with an honest blocker message.
        """
        profile = store.get_personality(user_id)
        if not HAVE_NUMPY:
            raise ValueError("experimental personality mode requires numpy (run without -I isolation)")
        if not profile:
            raise ValueError("no personality profile saved; save a profile first (settings section)")
        weight = self.DEFAULT_EXPERIMENTAL_WEIGHT if weight is None else float(weight)
        if not 0.0 <= weight <= 1.0:
            raise ValueError("weight must be in [0, 1]")
        profiled = store.profiled_users()
        rated_profiled = [u for u in profiled if store.user_ratings(u)]
        if len(rated_profiled) < self.MIN_PROFILED_USERS:
            raise ValueError(
                f"experimental mode unavailable: {len(rated_profiled)} profiled user(s) with ratings, "
                f"but the personality model needs at least {self.MIN_PROFILED_USERS} to fit")

        user_ratings = store.user_ratings(user_id)
        seen = set(user_ratings)
        pool = self.popularity_pool(seen, POOL_M)
        if not pool:
            return [], {"weight": weight}

        # Lazily fit the Ridge model on consenting profiled users only; cache until data changes.
        TRAITS_ORDER = ("openness", "conscientiousness", "extraversion", "agreeableness", "neuroticism")
        fit_key = tuple(sorted(rated_profiled)) + tuple(sorted((u, tuple(sorted(store.user_ratings(u).items())))
                                                               for u in rated_profiled))
        model_info = self._pers_cache.get("key")
        if model_info != fit_key:
            from ..phase13.engine import build_attr_matrix, fit_ridge
            train_rows = []
            traits_raw = {}
            for u in rated_profiled:
                hist = store.user_ratings(u)
                traits_raw[u] = store.get_personality(u)
                for mid, rating in hist.items():
                    train_rows.append((u, mid, rating, 0))
            self._pers_attr, _ = build_attr_matrix(train_rows, self.catalog, self.item_index)
            traits_norm = {u: {t: (tr[t] - 1.0) / 6.0 for t in TRAITS_ORDER} for u, tr in traits_raw.items()}
            self._pers_weights = fit_ridge(train_rows, traits_norm, self.item_index, self._pers_attr)
            self._pers_cache = {"key": fit_key}

        w = np.asarray(self._pers_weights)
        t = np.array([(profile[t] - 1.0) / 6.0 for t in ("openness", "conscientiousness", "extraversion",
                                                         "agreeableness", "neuroticism")])
        u_const = w[0] + float(w[1:6] @ t)
        eff = 2.0 * (w[6:12] + w[12:42].reshape(5, 6).T @ t)
        pers_raw = {}
        for mid in pool:
            i = self.item_index[mid]
            a = self._pers_attr[i]
            pers_raw[mid] = u_const + float(a @ eff) - float(eff @ np.full(6, 0.5))

        cf_scores = self._cf_scores(user_ratings, set(pool))
        profile_c = content_mod.user_profile(self.vectors, user_ratings)
        content_scores = content_mod.score(profile_c, self.vectors, set(pool)) if profile_c else {}
        components, _ = extract_components(user_ratings, pool, cf_scores, self.pop_counts, content_scores, {})
        norm = normalize_components(components)
        pers_n = _mm(pers_raw)
        blended = {}
        for m in pool:
            num, den = 0.0, 0.0
            parts = {"cf": norm["cf"].get(m), "popularity": norm["popularity"].get(m, 0.0),
                     "content": norm["content"].get(m), "personality": pers_n.get(m)}
            for name, v in (("cf", parts["cf"]), ("popularity", parts["popularity"]),
                            ("content", parts["content"])):
                if v is not None:
                    num += weight * 0.0  # base keeps its own weights below
            # base three_w3 blend, then personality adjustment on top
            base_v = 0.0
            wsum = 0.0
            for name, wgt in HYBRID_WEIGHTS.items():
                v = parts[name if name != "cf" else "cf"]
                if v is not None:
                    base_v += wgt * v
                    wsum += wgt
            base_v = base_v / wsum if wsum else 0.0
            blended[m] = (1.0 - weight) * base_v + weight * parts["personality"]
        top = sorted(blended.items(), key=lambda r: (-r[1], r[0]))[:top_n]

        top_genres = self._user_top_genres(user_ratings)
        results = []
        for mid, score in top:
            details = self.movie_details(mid)
            matched = [g for g in (details or {}).get("genres", []) if g in top_genres]
            signal = "personality" if weight > 0 and abs(pers_n.get(mid, 0.0)) > 0 else "base"
            text = (f"experimental: blended with a self-reported personality profile "
                    f"(weight {weight}); " + self._top_signal(norm["cf"].get(mid), norm["popularity"].get(mid, 0.0),
                                                              norm["content"].get(mid), mid, top_genres, user_ratings)[1])
            if matched:
                text += f"; matches your preferred {', '.join(matched)} genre(s)"
            out = dict(details or {"movie_id": mid, "title": f"Movie {mid}", "year": None, "genres": []})
            out.update({"score": round(score, 4), "explanation": text, "explanation_signal": "experimental_personality",
                        "experimental": True})
            results.append(out)
        meta = {"weight": weight, "model": "Phase 11 Ridge (42 bilinear features) fit on opt-in app users",
                "profiled_users": len(rated_profiled),
                "warning": "experimental and not proven to improve recommendation accuracy (Phase 13 Outcome D)"}
        return results, meta
