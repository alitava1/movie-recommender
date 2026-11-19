"""Controlled synthetic preference simulator (Environment B).

Kept strictly separate from recommendation models so ground-truth latent traits
and non-personality user factors never leak into model inputs.
"""

from dataclasses import dataclass
import math
import random

from . import TRAITS
from .profile import generate_population_profiles
from .questionnaire import build_schema, sample_latent_traits

MOVIE_ATTRS = (
    "complexity",
    "novelty",
    "emotional_intensity",
    "social_orientation",
    "predictability",
    "release_norm",
)

# Controlled simulation coefficients linking centered traits (t - 0.5) to centered movie attrs (a - 0.5).
# These are arbitrary synthetic experimental ground-truth weights, NOT psychological facts.
GROUND_TRUTH_MATRIX = {
    "openness": {"complexity": 1.6, "novelty": 1.4, "predictability": -1.2},
    "conscientiousness": {"predictability": 1.3, "complexity": 0.6, "novelty": -0.7},
    "extraversion": {"social_orientation": 1.7, "emotional_intensity": 0.6},
    "agreeableness": {"social_orientation": 1.1, "emotional_intensity": -0.8},
    "neuroticism": {"emotional_intensity": 1.5, "predictability": -0.6},
}

SCENARIOS = {
    "A_null": {"strength": 0.0, "confound": 0.0, "description": "Null personality effect (exact zero contribution)."},
    "B_weak": {"strength": 0.25, "confound": 0.0, "description": "Weak controlled personality effect."},
    "C_moderate": {"strength": 0.60, "confound": 0.0, "description": "Moderate controlled personality effect."},
    "D_strong": {"strength": 1.10, "confound": 0.0, "description": "Strong controlled personality effect."},
    "E_confounded": {"strength": 0.35, "confound": 0.70, "description": "Confounded preference factors correlated with latent personality."},
}


@dataclass(frozen=True)
class SyntheticDataset:
    scenario: str
    seed: int
    movie_features: dict
    observed_profiles: dict
    warm_train: list
    warm_val: list
    warm_test: list
    cold_test_ratings: list
    warm_users: tuple
    cold_test_users: tuple
    # Ground truth below is for simulator diagnostics/evaluation ONLY; never pass to models.
    _latent_traits: dict
    _user_factors: dict


def build_movie_attributes(movies, seed=2026):
    """Build deterministic [0, 1] item attributes over MovieLens catalog (explicitly synthetic)."""
    rng = random.Random(seed)
    years = [m.year for m in movies.values() if m.year is not None]
    min_y = min(years) if years else 1920
    max_y = max(years) if years else 1998
    span_y = max(1, max_y - min_y)

    features = {}
    for movie_id in sorted(movies):
        movie = movies[movie_id]
        gset = set(movie.genres)
        release_norm = (movie.year - min_y) / span_y if movie.year is not None else 0.5
        # Weak structural offsets + synthetic variation so attributes are not identical to genres.
        complexity = max(0.0, min(1.0, 0.35 + (0.25 if "drama" in gset or "mystery" in gset or "sci-fi" in gset else 0.0) + rng.uniform(-0.25, 0.35)))
        novelty = max(0.0, min(1.0, 0.30 + 0.30 * release_norm + rng.uniform(-0.25, 0.35)))
        emotional = max(0.0, min(1.0, 0.35 + (0.25 if "horror" in gset or "thriller" in gset or "war" in gset or "romance" in gset else 0.0) + rng.uniform(-0.25, 0.35)))
        social = max(0.0, min(1.0, 0.35 + (0.25 if "comedy" in gset or "adventure" in gset or "musical" in gset else 0.0) + rng.uniform(-0.25, 0.35)))
        predictability = max(0.0, min(1.0, 0.55 - 0.25 * novelty + rng.uniform(-0.25, 0.25)))
        features[movie_id] = {
            "complexity": complexity,
            "novelty": novelty,
            "emotional_intensity": emotional,
            "social_orientation": social,
            "predictability": predictability,
            "release_norm": release_norm,
        }
    return features


def compute_personality_effect(latent_traits, movie_attrs, strength):
    """Deterministic simulator equation for personality contribution."""
    if strength == 0.0:
        return 0.0
    total = 0.0
    for trait, attr_weights in GROUND_TRUTH_MATRIX.items():
        t_centered = 2.0 * (latent_traits[trait] - 0.5)
        for attr, coef in attr_weights.items():
            total += coef * t_centered * (2.0 * (movie_attrs[attr] - 0.5))
    return strength * total


def generate_synthetic_dataset(
    movies,
    scenario_name,
    seed,
    n_users=360,
    interactions_per_user=50,
    schema=None,
    response_noise=0.45,
    missing_rate=0.0,
    consistency=1.0,
    user_test_fraction=0.25,
    rating_noise=0.40,
):
    """Generate a controlled synthetic dataset with warm interaction and cold user-held-out splits."""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"unknown scenario {scenario_name!r}")
    if interactions_per_user < 5 or interactions_per_user > len(movies):
        raise ValueError("interactions_per_user must be between 5 and catalog size")
    if not 0.0 < user_test_fraction < 0.5:
        raise ValueError("user_test_fraction must be in (0.0, 0.5)")

    cfg = SCENARIOS[scenario_name]
    strength = cfg["strength"]
    confound = cfg["confound"]
    schema = schema or build_schema()
    rng = random.Random(seed)
    movie_ids = sorted(movies)
    movie_features = build_movie_attributes(movies, seed=seed + 17)

    # Latent non-personality item factors (4-dim) and item popularity bias.
    item_factors = {
        mid: [rng.gauss(0.0, 0.45) for _ in range(4)]
        for mid in movie_ids
    }
    item_bias = {mid: rng.gauss(0.0, 0.40) for mid in movie_ids}
    # Popularity skew creates realistic co-rating density for CF (power-law-like over catalog).
    exposure_weights = [math.exp(item_bias[mid] * 2.5) for mid in movie_ids]

    user_ids = list(range(1, n_users + 1))
    latent_override = {uid: sample_latent_traits(rng) for uid in user_ids}
    latent_traits, _, observed_profiles = generate_population_profiles(
        user_ids, schema, seed=seed + 99,
        response_noise=response_noise, missing_rate=missing_rate,
        consistency=consistency, latent_override=latent_override,
    )

    user_factors = {}
    user_bias = {}
    for uid in user_ids:
        lt = latent_traits[uid]
        raw_f = [rng.gauss(0.0, 0.6) for _ in range(4)]
        if confound > 0.0:
            # Couple latent non-personality taste factors with latent personality traits.
            trait_vec = [lt[t] - 0.5 for t in TRAITS[:4]]
            raw_f = [
                (1.0 - confound) * f + confound * (2.5 * tv)
                for f, tv in zip(raw_f, trait_vec)
            ]
        user_factors[uid] = raw_f
        user_bias[uid] = rng.gauss(0.0, 0.25)

    shuffled_users = list(user_ids)
    rng.shuffle(shuffled_users)
    n_cold = max(1, int(round(n_users * user_test_fraction)))
    cold_users = tuple(sorted(shuffled_users[:n_cold]))
    warm_users = tuple(sorted(shuffled_users[n_cold:]))

    weight_by_mid = dict(zip(movie_ids, exposure_weights))

    warm_train, warm_val, warm_test, cold_test_ratings = [], [], [], []
    base_ts = 880_000_000
    cold_set = set(cold_users)

    for uid in user_ids:
        chosen = sorted(
            movie_ids,
            key=lambda mid: -(rng.random() ** (1.0 / weight_by_mid[mid])),
        )[:interactions_per_user]
        rng.shuffle(chosen)
        rows = []
        uf = user_factors[uid]
        ub = user_bias[uid]
        lt = latent_traits[uid]
        for step, mid in enumerate(chosen):
            it_f = item_factors[mid]
            user_pref = sum(a * b for a, b in zip(uf, it_f))
            pers_eff = compute_personality_effect(lt, movie_features[mid], strength)
            raw = 3.25 + ub + item_bias[mid] + user_pref + pers_eff + rng.gauss(0.0, rating_noise)
            rating = float(max(1, min(5, int(round(raw)))))
            rows.append((uid, mid, rating, base_ts + uid * 1000 + step))

        if uid in cold_set:
            cold_test_ratings.extend(rows)
        else:
            n_val = max(1, int(len(rows) * 0.2))
            n_test = max(1, int(len(rows) * 0.2))
            n_train = len(rows) - n_val - n_test
            warm_train.extend(rows[:n_train])
            warm_val.extend(rows[n_train:n_train + n_val])
            warm_test.extend(rows[n_train + n_val:])

    return SyntheticDataset(
        scenario=scenario_name,
        seed=seed,
        movie_features=movie_features,
        observed_profiles=observed_profiles,
        warm_train=warm_train,
        warm_val=warm_val,
        warm_test=warm_test,
        cold_test_ratings=cold_test_ratings,
        warm_users=warm_users,
        cold_test_users=cold_users,
        _latent_traits=latent_traits,
        _user_factors=user_factors,
    )
