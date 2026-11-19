"""Phase 7 CF adapter: exact co-rating-only similarity via chunked numpy, pure-Python fallback.

Both paths return the standard ``ItemItemCF`` structure so all Phase 2-6 ranking semantics
(evidence, prediction shrinkage, reverse neighbor index) are preserved unchanged.
"""

from math import sqrt

from ..collaborative import ItemItemCF
from ..data import group_by_user

CHUNK_COLUMNS = 512


def _numpy_build(ratings, neighbors, shrinkage):
    """Exact rebuild of ItemItemCF math using chunked numpy matmuls (co-rating users only)."""
    import numpy as np

    by_user = group_by_user(ratings)
    users = sorted(by_user)
    u_idx = {user: i for i, user in enumerate(users)}
    movie_ids = sorted({movie for items in by_user.values() for movie in items})
    m_idx = {movie: i for i, movie in enumerate(movie_ids)}
    n_users, n_movies = len(users), len(movie_ids)

    dev = np.zeros((n_users, n_movies), dtype=np.float64)
    ind = np.zeros((n_users, n_movies), dtype=np.float64)
    for user, items in by_user.items():
        mean = sum(items.values()) / len(items)
        ui = u_idx[user]
        for movie, rating in items.items():
            mi = m_idx[movie]
            dev[ui, mi] = rating - mean
            ind[ui, mi] = 1

    dev_sq = dev * dev
    # Each item's neighbor list is exactly its own column's global top-N over co-rated pairs,
    # matching the pure-Python builder's full-pair accumulation + trim (the similarity formula
    # is symmetric, so the value computed from either endpoint's column is the same).
    trimmed = {}
    for start in range(0, n_movies, CHUNK_COLUMNS):
        end = min(start + CHUNK_COLUMNS, n_movies)
        d_chunk = dev[:, start:end]
        dot = dev.T @ d_chunk                          # [n_movies, chunk]
        left_sq = dev_sq.T @ ind[:, start:end]         # co-rated sum of source deviations squared
        right_sq = ind.T @ dev_sq[:, start:end]        # co-rated sum of target deviations squared
        co = ind.T @ ind[:, start:end]                 # co-rating counts

        for c_local in range(end - start):
            target = start + c_local
            col_dot = dot[:, c_local]
            col_lsq = left_sq[:, c_local]
            col_rsq = right_sq[:, c_local]
            col_co = co[:, c_local].copy()
            col_co[target] = 0  # exclude self-pair
            norms_sq = col_lsq * col_rsq
            with np.errstate(divide="ignore", invalid="ignore"):
                raw = col_dot / np.sqrt(norms_sq) * col_co / (col_co + shrinkage)
            sims = np.where((col_co > 0) & (col_dot > 0) & (norms_sq > 0), raw, 0.0)
            k = min(neighbors, int(np.count_nonzero(sims > 0)))
            if k == 0:
                trimmed[movie_ids[target]] = []
                continue
            # Include every source tied with the k-th largest similarity so the Python
            # (-sim, movie_id) tie-break reproduces the pure-Python global sort-and-trim.
            boundary = np.partition(sims, len(sims) - k)[len(sims) - k]
            cands = [
                (movie_ids[int(src)], float(sims[src]))
                for src in np.nonzero(sims >= boundary)[0]
            ]
            cands.sort(key=lambda item: (-item[1], item[0]))
            trimmed[movie_ids[target]] = cands[:neighbors]
    return trimmed


def build_cf(ratings, neighbors, shrinkage=10.0, prediction_shrinkage=100.0):
    """Build ItemItemCF with exact Phase 2-6 math; numpy fast path when available."""
    try:
        import numpy  # noqa: F401
    except ImportError:
        return ItemItemCF.build(
            ratings, neighbors=neighbors, shrinkage=shrinkage, prediction_shrinkage=prediction_shrinkage
        )
    item_neighbors = _numpy_build(ratings, neighbors, shrinkage)
    return ItemItemCF(neighbors, shrinkage, item_neighbors, prediction_shrinkage)
