"""Questionnaire-derived user personality profiles and negative-control permutations."""

from dataclasses import dataclass
import random

from . import TRAITS
from .questionnaire import sample_latent_traits, score_responses, simulate_responses


@dataclass(frozen=True)
class ObservedProfile:
    """Questionnaire-derived personality profile exposed to recommenders (no latent truth)."""

    user_id: int
    traits: dict
    completeness: float
    source: str = "synthetic_questionnaire"


def generate_population_profiles(
    user_ids,
    schema,
    seed,
    response_noise=0.45,
    missing_rate=0.0,
    consistency=1.0,
    latent_override=None,
):
    """Independently generate latent traits, Likert responses, and observed profiles."""
    rng = random.Random(seed)
    latent_by_user = {}
    responses_by_user = {}
    observed_by_user = {}
    for user_id in sorted(user_ids):
        latent = dict(latent_override[user_id]) if latent_override and user_id in latent_override else sample_latent_traits(rng)
        answers = simulate_responses(
            latent, schema, rng,
            response_noise=response_noise,
            missing_rate=missing_rate,
            consistency=consistency,
        )
        scored = score_responses(answers, schema)
        latent_by_user[user_id] = latent
        responses_by_user[user_id] = answers
        observed_by_user[user_id] = ObservedProfile(
            user_id=user_id,
            traits=scored["traits"],
            completeness=scored["completeness"],
        )
    return latent_by_user, responses_by_user, observed_by_user


def shuffle_profiles(observed_profiles, seed):
    """Permute observed profiles across the exact same user population (deranged when n >= 2)."""
    user_ids = sorted(observed_profiles)
    if len(user_ids) < 2:
        return dict(observed_profiles)
    rng = random.Random(seed)
    donor_ids = list(user_ids)
    rng.shuffle(donor_ids)
    # Break any fixed points deterministically so no user keeps their own profile.
    for i in range(len(user_ids)):
        if donor_ids[i] == user_ids[i]:
            j = (i + 1) % len(user_ids)
            donor_ids[i], donor_ids[j] = donor_ids[j], donor_ids[i]
    return {
        uid: ObservedProfile(
            user_id=uid,
            traits=dict(observed_profiles[donor].traits),
            completeness=observed_profiles[donor].completeness,
            source="shuffled_control",
        )
        for uid, donor in zip(user_ids, donor_ids)
    }


def randomize_profiles(user_ids, seed):
    """Generate uniform [0.0, 1.0] random profiles completely unrelated to any user."""
    rng = random.Random(seed)
    return {
        uid: ObservedProfile(
            user_id=uid,
            traits={trait: rng.random() for trait in TRAITS},
            completeness=1.0,
            source="random_control",
        )
        for uid in sorted(user_ids)
    }
