"""Lightweight regularized personality-preference model (pure Python Ridge regression)."""

from . import TRAITS
from .simulator import MOVIE_ATTRS


def _solve_linear(matrix, rhs):
    """Solve dense linear system A x = b via Gaussian elimination with partial pivoting."""
    n = len(rhs)
    aug = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        diag = aug[col][col]
        if abs(diag) < 1e-12:
            continue
        inv = 1.0 / diag
        for j in range(col, n + 1):
            aug[col][j] *= inv
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                factor = aug[r][col]
                for j in range(col, n + 1):
                    aug[r][j] -= factor * aug[col][j]
    return [aug[i][n] for i in range(n)]


def featurize_pair(user_traits, movie_attrs):
    """Build scaled bilinear [bias, 2*(t-0.5), 2*(a-0.5), 4*(t-0.5)*(a-0.5)] feature vector."""
    t_vals = [2.0 * (user_traits.get(t, 0.5) - 0.5) for t in TRAITS]
    a_vals = [2.0 * (movie_attrs.get(a, 0.5) - 0.5) for a in MOVIE_ATTRS]
    feats = [1.0, *t_vals, *a_vals]
    for tv in t_vals:
        for av in a_vals:
            feats.append(tv * av)
    return feats


class RidgePersonalityModel:
    """Regularized Ridge regression mapping questionnaire trait estimates + item attrs to rating affinity."""

    def __init__(self, alpha=5.0, weights=None):
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        self.alpha = alpha
        self.weights = list(weights) if weights is not None else []

    def fit(self, train_ratings, observed_profiles, movie_features, center_by_user=True):
        """Fit weights strictly from training interactions and questionnaire-derived profiles."""
        if not train_ratings:
            self.weights = [0.0] * (1 + len(TRAITS) + len(MOVIE_ATTRS) + len(TRAITS) * len(MOVIE_ATTRS))
            return self

        user_sums = {}
        user_counts = {}
        global_sum = 0.0
        for uid, _, rating, _ in train_ratings:
            user_sums[uid] = user_sums.get(uid, 0.0) + rating
            user_counts[uid] = user_counts.get(uid, 0) + 1
            global_sum += rating
        global_mean = global_sum / len(train_ratings)
        user_means = {uid: user_sums[uid] / user_counts[uid] for uid in user_sums}

        dim = 1 + len(TRAITS) + len(MOVIE_ATTRS) + len(TRAITS) * len(MOVIE_ATTRS)
        xtx = [[0.0] * dim for _ in range(dim)]
        xty = [0.0] * dim

        for uid, mid, rating, _ in train_ratings:
            prof = observed_profiles.get(uid)
            attrs = movie_features.get(mid)
            if prof is None or attrs is None:
                continue
            target = rating - (user_means[uid] if center_by_user else global_mean)
            x = featurize_pair(prof.traits, attrs)
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
            # Do not regularize bias term (index 0); regularize all trait/attr/interaction weights.
            if i > 0:
                xtx[i][i] += self.alpha

        self.weights = _solve_linear(xtx, xty)
        return self

    def personality_score(self, user_profile, movie_features):
        """Score a single movie from questionnaire-derived user profile and movie attributes."""
        if not self.weights:
            return 0.0
        traits = user_profile.traits if hasattr(user_profile, "traits") else user_profile
        feats = featurize_pair(traits, movie_features)
        return sum(w * f for w, f in zip(self.weights, feats))

    def score_candidates(self, user_profile, movie_features_by_id, candidates):
        """Return {movie_id: personality_score} over candidates via factored user weights."""
        if not self.weights:
            return {}
        traits = user_profile.traits if hasattr(user_profile, "traits") else user_profile
        t_vals = [2.0 * (traits.get(t, 0.5) - 0.5) for t in TRAITS]
        w = self.weights
        n_t = len(TRAITS)
        n_a = len(MOVIE_ATTRS)
        u_const = w[0] + sum(w[1 + ti] * t_vals[ti] for ti in range(n_t))
        eff = [
            2.0 * (w[1 + n_t + ai] + sum(w[1 + n_t + n_a + ti * n_a + ai] * t_vals[ti] for ti in range(n_t)))
            for ai in range(n_a)
        ]
        e0, e1, e2, e3, e4, e5 = eff
        a0, a1, a2, a3, a4, a5 = MOVIE_ATTRS
        return {
            mid: (
                u_const
                + e0 * (attrs[a0] - 0.5)
                + e1 * (attrs[a1] - 0.5)
                + e2 * (attrs[a2] - 0.5)
                + e3 * (attrs[a3] - 0.5)
                + e4 * (attrs[a4] - 0.5)
                + e5 * (attrs[a5] - 0.5)
            )
            for mid in candidates
            if (attrs := movie_features_by_id.get(mid)) is not None
        }


def personality_score(user_profile, movie_features, model=None):
    """Transparent public scoring interface for a user profile and movie attribute dict."""
    if model is None:
        return 0.0
    return model.personality_score(user_profile, movie_features)
