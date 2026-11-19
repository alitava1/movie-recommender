"""Phase 13 engine: strata splits, item attributes, Ridge fit, pools, ranking, metrics."""

import math

try:
    import numpy as np
    HAVE_NUMPY = True
except ImportError:  # -I isolation mode: pure-Python fallbacks below
    np = None
    HAVE_NUMPY = False

from .. import evaluation
from ..personality.preference_model import MOVIE_ATTRS, RidgePersonalityModel, _solve_linear, featurize_pair
from ..phase7.metrics import novelty_at_k
from ..phase11.ranking import rank_with_personality as _rwp
from .data import TRAITS
from .profiles import make_profile

K_VALUES = (1, 5, 10, 20)
WEIGHT_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)
POOL_M = 200
RELEVANCE = 4.0
MIN_TRAIN_HISTORY = 5


# ---------- strata ----------

def per_user_ordered(rows):
    """rows already sorted globally by (tstamp, movie); group per user preserving order."""
    by_user = {}
    for uid, mid, rating, ts in rows:
        by_user.setdefault(uid, []).append((mid, rating, ts))
    return by_user


def warm_stratum(rows):
    """Latest eligible (rating >= RELEVANCE) interaction per user is the target; earlier rows train.

    Users with < MIN_TRAIN_HISTORY training rows are excluded (counted separately).
    """
    by_user = per_user_ordered(rows)
    train, targets, excluded = [], {}, []
    for uid, items in by_user.items():
        target_idx = None
        for i in range(len(items) - 1, -1, -1):
            if items[i][1] >= RELEVANCE:
                target_idx = i
                break
        if target_idx is None or target_idx < MIN_TRAIN_HISTORY:
            excluded.append(uid)
            continue
        targets[uid] = {items[target_idx][0]}
        train.extend((uid, mid, rating, ts) for mid, rating, ts in items[:target_idx])
    return train, targets, excluded


def sparse_stratum(rows, k):
    """First k interactions are the permitted history; later eligible rows are targets."""
    by_user = per_user_ordered(rows)
    train, targets, excluded = [], {}, []
    for uid, items in by_user.items():
        hist = items[:k]
        later = [(mid, rating, ts) for mid, rating, ts in items[k:] if rating >= RELEVANCE]
        if not later:
            excluded.append(uid)
            continue
        targets[uid] = {mid for mid, _, _ in later}
        train.extend((uid, mid, rating, ts) for mid, rating, ts in hist)
    return train, targets, excluded


def participant_folds(n_users, n_folds=10):
    """Deterministic participant-disjoint folds over sorted user indices."""
    users = list(range(n_users))
    size = math.ceil(n_users / n_folds)
    return [users[i * size:(i + 1) * size] for i in range(n_folds) if users[i * size:(i + 1) * size]]


# ---------- training-only item attributes (frozen definitions) ----------

def item_statistics(train_rows, items):
    """Bayesian-smoothed item mean, popularity count/rank/decile from training rows only."""
    counts, sums = {}, {}
    total, n = 0.0, 0
    for _, mid, rating, _ in train_rows:
        counts[mid] = counts.get(mid, 0) + 1
        sums[mid] = sums.get(mid, 0.0) + rating
        total += rating
        n += 1
    global_mean = total / max(1, n)
    m = 10.0
    bayes = {mid: (sums.get(mid, 0.0) + m * global_mean) / (counts.get(mid, 0) + m) for mid in items}
    ordered = sorted(items, key=lambda mid: (-counts.get(mid, 0), mid))
    rank = {mid: i / max(1, len(ordered) - 1) for i, mid in enumerate(ordered)}
    decile = {mid: (math.floor(rank[mid] * 10) + 0.5) / 10.0 for mid in items}
    return {
        "counts": counts,
        "global_mean": global_mean,
        "bayes_norm": {mid: (bayes[mid] - 0.5) / 4.5 for mid in items},
        "pop_rank": rank,
        "pop_decile": decile,
    }


def factor_coordinates(train_rows, item_index, n_items, n_factors=3, iters=10):
    """Top-k power-iteration coordinates of the training co-rating count matrix (deterministic).

    Matvec C·v = A^T (A·v) with A the binary user-item matrix; numpy when available,
    identical pure-Python matvec otherwise.
    """
    by_user = per_user_ordered(train_rows)
    if HAVE_NUMPY:
        user_items = {u: np.array([item_index[mid] for mid, _, _ in items], dtype=np.int64)
                      for u, items in by_user.items() if items}
        start = (np.arange(n_items) * 7919 % 1000).astype(np.float64) / 1000.0 + 0.5
        coords = []
        n_users = (max(by_user) + 1) if by_user else 1

        def matvec(vec):
            Av = np.zeros(n_users)
            for u, idx in user_items.items():
                Av[u] = vec[idx].sum()
            Cv = np.zeros(n_items)
            for u, idx in user_items.items():
                Cv[idx] += Av[u]
            return Cv

        for _ in range(n_factors):
            v = start - sum((float(start @ u)) * u for u in coords)
            for _ in range(iters):
                cv = matvec(v)
                norm = math.sqrt(float(cv @ cv))
                if norm > 0:
                    v = cv / norm
            coords.append(v.copy())
        return np.stack(coords, axis=1) if coords else np.zeros((n_items, 0))

    user_items = {u: [item_index[mid] for mid, _, _ in items] for u, items in by_user.items() if items}
    start = [(((i * 7919) % 1000) / 1000.0 + 0.5) for i in range(n_items)]
    coords = []

    def matvec_list(vec):
        av = {}
        for u, idx in user_items.items():
            av[u] = sum(vec[i] for i in idx)
        cv = [0.0] * n_items
        for u, idx in user_items.items():
            a = av[u]
            for i in idx:
                cv[i] += a
        return cv

    for _ in range(n_factors):
        v = list(start)
        for _ in range(iters):
            cv = matvec_list(v)
            norm = math.sqrt(sum(x * x for x in cv))
            if norm > 0:
                v = [x / norm for x in cv]
        coords.append(list(v))
    # Transpose to per-item rows (matching the numpy (n_items x n_factors) layout).
    return [[coords[k][i] for k in range(len(coords))] for i in range(n_items)]


def build_attr_matrix(train_rows, items, item_index, power_iters=10):
    """ATTR matrix (n_items x 6) in MOVIE_ATTRS order; all statistics training-only.

    numpy path returns an ndarray; the pure-Python fallback returns list-of-lists with an
    identical layout (indexed by position, columns in MOVIE_ATTRS order).
    """
    stats = item_statistics(train_rows, items)
    fac = factor_coordinates(train_rows, item_index, len(items), iters=power_iters)
    if HAVE_NUMPY:
        lo = fac.min(axis=0) if fac.size else np.zeros(0)
        hi = fac.max(axis=0) if fac.size else np.zeros(0)
        span = np.where(hi - lo > 0, hi - lo, 1.0)
        fac_norm = (fac - lo) / span if fac.size else fac
        n = len(items)
        attr = np.zeros((n, 6))
        for i, mid in enumerate(items):
            attr[i, 0] = stats["bayes_norm"][mid]
            attr[i, 1] = stats["pop_rank"][mid]
            attr[i, 2] = stats["pop_decile"][mid]
            for j in range(min(3, fac_norm.shape[1])):
                attr[i, 3 + j] = fac_norm[i, j]
        return attr, stats
    lo = [min(col) for col in zip(*fac)] if fac else []
    hi = [max(col) for col in zip(*fac)] if fac else []
    span = [(h - l) or 1.0 for l, h in zip(lo, hi)]
    fac_norm = [[(row[j] - lo[j]) / span[j] for j in range(len(span))] for row in fac]
    attr = []
    for i, mid in enumerate(items):
        row = [stats["bayes_norm"][mid], stats["pop_rank"][mid], stats["pop_decile"][mid]]
        fac_row = fac_norm[i] if fac_norm else [0.5, 0.5, 0.5]
        row.extend(fac_row[:3])
        while len(row) < 6:
            row.append(0.5)
        attr.append(row)
    return attr, stats


def attr_dicts(items, item_index, attr):
    """Lazy per-item attribute dicts keyed by MOVIE_ATTRS names (for RidgePersonalityModel)."""
    cache = {}

    def get(mid):
        if mid not in cache:
            i = item_index[mid]
            cache[mid] = {name: float(attr[i, j]) for j, name in enumerate(MOVIE_ATTRS)}
        return cache[mid]

    return get


# ---------- Ridge fit (numpy chunked; identical math to Phase 11) ----------

def fit_ridge(train_rows, traits_norm, item_index, attr, alpha=5.0):
    """Fit the 42-feature bilinear Ridge on training rows; returns weights list (bias first).

    numpy chunked accumulation when available; identical pure-Python accumulation otherwise.
    """
    dim = 1 + len(TRAITS) + len(MOVIE_ATTRS) + len(TRAITS) * len(MOVIE_ATTRS)
    user_sums, user_counts = {}, {}
    for uid, _, rating, _ in train_rows:
        user_sums[uid] = user_sums.get(uid, 0.0) + rating
        user_counts[uid] = user_counts.get(uid, 0) + 1
    by_user = per_user_ordered(train_rows)
    if not HAVE_NUMPY:
        xtx = [[0.0] * dim for _ in range(dim)]
        xty = [0.0] * dim
        for uid, items in by_user.items():
            t_map = traits_norm[uid]
            u_mean = user_sums[uid] / user_counts[uid]
            for mid, rating, _ in items:
                attr_dict = dict(zip(MOVIE_ATTRS, attr[item_index[mid]]))
                target = rating - u_mean
                x = featurize_pair(t_map, attr_dict)
                for i in range(dim):
                    xi = x[i]
                    if xi == 0.0:
                        continue
                    xty[i] += xi * target
                    row_i = xtx[i]
                    for j in range(i, dim):
                        row_i[j] += xi * x[j]
        for i in range(dim):
            for j in range(i):
                xtx[i][j] = xtx[j][i]
            if i > 0:
                xtx[i][i] += alpha
        return _solve_linear(xtx, xty)

    xtx = np.zeros((dim, dim))
    xty = np.zeros(dim)
    for uid, items in by_user.items():
        t = np.array([traits_norm[uid][t_name] for t_name in TRAITS])
        idx = np.array([item_index[mid] for mid, _, _ in items])
        a = np.asarray(attr)[idx]
        y = np.array([rating for _, rating, _ in items]) - user_sums[uid] / user_counts[uid]
        n = len(items)
        ones = np.ones((n, 1))
        inter = (a[:, None, :] * t[None, :, None]).reshape(n, len(TRAITS) * len(MOVIE_ATTRS))
        x = np.hstack([ones, np.tile(t, (n, 1)), a, inter])
        xtx += x.T @ x
        xty += x.T @ y

    reg = np.eye(dim) * alpha
    reg[0, 0] = 0.0  # bias not regularized (Phase 11 convention)
    weights = np.linalg.solve(xtx + reg, xty)
    return weights.tolist()


def ridge_model(weights):
    return RidgePersonalityModel(weights=list(weights))


def ridge_scores(weights_model, profile, attr_get, pool_ids):
    feats = {mid: attr_get(mid) for mid in pool_ids}
    return weights_model.score_candidates(profile, feats, list(pool_ids))


# ---------- pools and ranking ----------

def popularity_pool(ranked_items, seen, m=POOL_M):
    """Top-m unseen items from a precomputed popularity ranking (early-exit scan)."""
    out = []
    for mid in ranked_items:
        if mid in seen:
            continue
        out.append(mid)
        if len(out) >= m:
            break
    return out


def hybrid_base_scores(pool_ids, ur, cf_model, pop_counts, prediction_shrinkage, alpha=0.5):
    """Phase 10-style CF+popularity combined scores over a fixed pool (missing CF renormalized).

    cf_model=None short-circuits to popularity-only (used by zero-history folds where the
    held-out user has no permitted history, so CF evidence is empty by construction).
    """
    from ..phase10.scoring import extract_components, normalize_components
    from ..phase10.diversity import rank_candidates

    evidence = cf_model.evidence(ur, set(pool_ids)) if (cf_model is not None and ur) else {}
    cf_scores = {
        mid: min(5.0, max(0.5, mean + num / (den + prediction_shrinkage)))
        for mid, (mean, num, den) in evidence.items()
    }
    components, _cov = extract_components(ur, pool_ids, cf_scores, pop_counts, {}, {})
    norm = normalize_components(components)
    _, scores, _diag = rank_candidates(pool_ids, norm, "combined", {"alpha": alpha})
    return scores, len(cf_scores)


def rank_with_personality13(pool_ids, base_scores, traits_norm, user_key, weights_model, attr_get, weight, enabled, k=10):
    profile = make_profile(user_key, traits_norm) if (enabled and traits_norm is not None) else None
    feats = {mid: attr_get(mid) for mid in pool_ids}
    return _rwp(pool_ids, base_scores, profile, feats, weights_model,
                personality_weight=weight, enabled=enabled, k=k)


# ---------- metrics ----------

def build_item_bitsets(train_rows):
    """item -> integer bitset over user indices (ILD via co-rating Jaccard)."""
    bitsets = {}
    for uid, mid, _, _ in train_rows:
        bitsets[mid] = bitsets.get(mid, 0) | (1 << uid)
    return bitsets


def ild_at_10(recs, bitsets, k=10):
    items = recs[:k]
    if len(items) < 2:
        return 0.0
    total = pairs = 0
    for i in range(len(items)):
        a = bitsets.get(items[i], 0)
        for j in range(i + 1, len(items)):
            b = bitsets.get(items[j], 0)
            inter = (a & b).bit_count()
            union = a.bit_count() + b.bit_count() - inter
            total += 1.0 - (inter / union if union else 0.0)
            pairs += 1
    return total / pairs if pairs else 0.0


def evaluate(recs, relevant, pop_counts, bitsets, k=10, recall_k=50):
    row = evaluation.evaluate_user(recs, relevant, k)
    row["hr"] = evaluation.hit_rate_at_k(recs, relevant, k)
    row["ild"] = ild_at_10(recs, bitsets, k)
    row["novelty"] = novelty_at_k(recs, pop_counts, k)
    row["recall50"] = evaluation.recall_at_k(recs[:recall_k], relevant, recall_k)
    return row


def summarize(per_user, rec_lists, catalog_size, pop_counts):
    users = len(per_user)
    summary = {
        name: sum(row[name] for row in per_user.values()) / users
        for name in ("precision", "recall", "ndcg", "map", "hr", "ild", "novelty", "recall50")
    } if users else {name: 0.0 for name in ("precision", "recall", "ndcg", "map", "hr", "ild", "novelty", "recall50")}
    distinct = set()
    tail_threshold = _tail_threshold(pop_counts)
    tail = head = total = 0
    ordered = sorted(pop_counts.values(), reverse=True)
    head_cut = ordered[max(1, int(len(ordered) * 0.1)) - 1] if ordered else 0
    for recs in rec_lists:
        distinct.update(recs)
        for mid in recs:
            c = pop_counts.get(mid, 0)
            tail += 1 if c < tail_threshold else 0
            head += 1 if c >= head_cut else 0
            total += 1
    summary["coverage"] = len(distinct) / catalog_size if catalog_size else 0.0
    summary["long_tail_share"] = tail / total if total else 0.0
    summary["popularity_concentration"] = head / total if total else 0.0
    summary["users"] = users
    return summary


def _tail_threshold(pop_counts):
    ordered = sorted(pop_counts.values())
    if not ordered:
        return 0.0
    pos = 0.8 * (len(ordered) - 1)
    lo, hi = int(math.floor(pos)), int(math.ceil(pos))
    return ordered[lo] if lo == hi else ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def pool_recall(pool_fn, targets_by_user, m_values=(100, 200)):
    """Candidate-generation Recall@M and precision diagnostics (denominator = all targets)."""
    out = {}
    for m in m_values:
        recall_sum, prec_sum, covered = 0.0, 0.0, 0
        n = 0
        for uid, relevant in targets_by_user.items():
            pool = pool_fn(uid)[:m]
            hits = len(set(pool) & relevant)
            recall_sum += hits / len(relevant) if relevant else 0.0
            prec_sum += hits / len(pool) if pool else 0.0
            covered += 1 if hits else 0
            n += 1
        out[f"recall_at_{m}"] = recall_sum / max(1, n)
        out[f"candidate_precision_at_{m}"] = prec_sum / max(1, n)
        out[f"user_retrieval_coverage_at_{m}"] = covered / max(1, n)
    return out


def pop_pool_fn(pop_counts, hist_by_user, catalog_items, m=POOL_M):
    ranked = sorted(catalog_items, key=lambda mid: (-pop_counts.get(mid, 0), mid))
    return lambda uid: popularity_pool(ranked, set(hist_by_user.get(uid, ())), m)
