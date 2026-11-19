"""Phase 13 personality profiles: normalization, negative controls, and score-level perturbation."""

import random

from ..personality.profile import ObservedProfile
from .data import TRAITS


def normalize_traits(traits_raw, ref_users):
    """Min-max normalize traits to [0, 1] using statistics of `ref_users` only (leak-safe)."""
    bounds = {}
    for t in TRAITS:
        vals = [traits_raw[u][t] for u in ref_users]
        bounds[t] = (min(vals), max(vals))
    out = {}
    for uid, row in traits_raw.items():
        out[uid] = {
            t: (row[t] - bounds[t][0]) / (bounds[t][1] - bounds[t][0]) if bounds[t][1] > bounds[t][0] else 0.5
            for t in TRAITS
        }
    return out, bounds


def make_profile(user_key, traits_norm, source="published_questionnaire_scores"):
    return ObservedProfile(user_id=user_key, traits=dict(traits_norm), completeness=1.0, source=source)


def shuffled_profiles(traits_by_user, seed):
    """Deranged permutation across the same participants (no fixed points when n >= 2)."""
    users = sorted(traits_by_user)
    rng = random.Random(seed)
    donors = list(users)
    rng.shuffle(donors)
    for i in range(len(users)):
        if donors[i] == users[i]:
            j = (i + 1) % len(users)
            donors[i], donors[j] = donors[j], donors[i]
    return {u: dict(traits_by_user[d]) for u, d in zip(users, donors)}


def randomized_profiles(traits_by_user, seed):
    """Resample each trait from the empirical marginal of the reference participants."""
    users = sorted(traits_by_user)
    rng = random.Random(seed)
    marginals = {t: [traits_by_user[u][t] for u in users] for t in TRAITS}
    return {
        u: {t: marginals[t][rng.randrange(len(marginals[t]))] for t in TRAITS}
        for u in users
    }


def noisy_profiles(traits_by_user, sigma, seed):
    """Score-level Gaussian perturbation: sigma is on the original 1-7 scale, applied as
    sigma/6 on the normalized [0, 1] scale, clamped to [0, 1]. NOT measured questionnaire noise."""
    users = sorted(traits_by_user)
    rng = random.Random(seed)
    s = sigma / 6.0
    return {
        u: {t: min(1.0, max(0.0, traits_by_user[u][t] + rng.gauss(0.0, s))) for t in TRAITS}
        for u in users
    }


def missing_profiles(traits_by_user, n_missing, seed, ref_users=None):
    """Mask n traits per participant to the reference-participant median (leak-safe)."""
    if not 0 <= n_missing <= len(TRAITS):
        raise ValueError("n_missing out of range")
    users = sorted(traits_by_user)
    ref = list(ref_users) if ref_users else users
    rng = random.Random(seed)
    medians = {}
    for t in TRAITS:
        vals = sorted(traits_by_user[u][t] for u in ref)
        m = len(vals) // 2
        medians[t] = vals[m] if m % 2 or len(vals) % 2 else (vals[m - 1] + vals[m]) / 2.0
        if not vals:
            medians[t] = 0.5
    if n_missing == 0:
        return {u: dict(traits_by_user[u]) for u in users}
    out = {}
    for u in users:
        dropped = set(rng.sample(TRAITS, n_missing))
        out[u] = {t: (medians[t] if t in dropped else traits_by_user[u][t]) for t in TRAITS}
    return out


def ablate_single_trait(traits_by_user, trait, ref_users=None):
    """Set one trait for every participant to the reference median (trait ablation)."""
    users = sorted(traits_by_user)
    ref = list(ref_users) if ref_users else users
    vals = sorted(traits_by_user[u][trait] for u in ref)
    med = vals[len(vals) // 2] if vals else 0.5
    return {u: {t: (med if t == trait else traits_by_user[u][t]) for t in TRAITS} for u in users}
