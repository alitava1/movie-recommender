"""Phase 11 observable personality profiles, leakage guards, and negative/ablation controls."""

import random

from ..personality import TRAITS
from ..personality.profile import ObservedProfile, generate_population_profiles, randomize_profiles, shuffle_profiles
from ..personality.questionnaire import build_schema


def assert_observed_profile(profile):
    """Guard: ranking models may only consume ObservedProfile instances, never latent dicts."""
    if profile is None:
        return None
    if not isinstance(profile, ObservedProfile):
        raise TypeError(f"personality input must be ObservedProfile, got {type(profile).__name__}")
    return profile


def mask_dimensions(observed_profiles, n_missing_dims, seed=2026):
    """Mask `n_missing_dims` OCEAN traits per user to neutral 0.5 and adjust completeness."""
    if not 0 <= n_missing_dims <= len(TRAITS):
        raise ValueError(f"n_missing_dims must be in [0, {len(TRAITS)}]")
    if n_missing_dims == 0:
        return dict(observed_profiles)
    rng = random.Random(seed)
    masked = {}
    for uid in sorted(observed_profiles):
        prof = observed_profiles[uid]
        dropped = set(rng.sample(TRAITS, n_missing_dims))
        traits = {t: (0.5 if t in dropped else prof.traits[t]) for t in TRAITS}
        frac_kept = (len(TRAITS) - n_missing_dims) / len(TRAITS)
        masked[uid] = ObservedProfile(
            user_id=uid,
            traits=traits,
            completeness=round(prof.completeness * frac_kept, 4),
            source=f"masked_{n_missing_dims}_dims",
        )
    return masked


def build_control_profiles(user_ids, schema, seed, latent_override=None, base_profiles=None):
    """Build correct, noisy (sigma=1.0), missing (2 dims), shuffled, and random ObservedProfiles."""
    if base_profiles is None:
        latent, _, correct = generate_population_profiles(
            user_ids, schema, seed=seed, response_noise=0.45, latent_override=latent_override,
        )
    else:
        latent = latent_override
        correct = base_profiles
    _, _, noisy = generate_population_profiles(
        user_ids, schema, seed=seed + 1000, response_noise=1.0, latent_override=latent,
    )
    missing = mask_dimensions(correct, n_missing_dims=2, seed=seed + 2000)
    shuffled = shuffle_profiles(correct, seed=seed + 3000)
    random_p = randomize_profiles(user_ids, seed=seed + 4000)
    return {
        "correct": correct,
        "noisy": noisy,
        "missing": missing,
        "shuffled": shuffled,
        "random": random_p,
    }
