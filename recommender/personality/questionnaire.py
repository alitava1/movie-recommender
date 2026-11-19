"""Synthetic Big Five self-report questionnaire simulation and scoring.

All items use explicit synthetic placeholder identifiers. This is NOT a
validated psychological instrument and reproduces no proprietary wording.
"""

from . import TRAITS

DEFAULT_SCALE_MIN = 1
DEFAULT_SCALE_MAX = 5


def build_schema(items_per_trait=4, scale_min=DEFAULT_SCALE_MIN, scale_max=DEFAULT_SCALE_MAX):
    """Build a balanced synthetic Big Five questionnaire schema with reverse-keyed items."""
    if items_per_trait < 2:
        raise ValueError("items_per_trait must be at least 2")
    if scale_max <= scale_min:
        raise ValueError("scale_max must exceed scale_min")
    items = []
    for trait_idx, trait in enumerate(TRAITS):
        for item_num in range(1, items_per_trait + 1):
            keyed = 1 if item_num % 2 == 1 else -1
            # Spread synthetic item difficulties deterministically across [-0.30, +0.30].
            frac = (item_num - 1) / (items_per_trait - 1)
            difficulty = round(-0.30 + 0.60 * frac, 4)
            items.append({
                "item_id": f"SYN_{trait[:3].upper()}_{item_num}",
                "trait": trait,
                "trait_index": trait_idx,
                "keyed": keyed,
                "difficulty": difficulty,
                "prompt": f"SYNTHETIC_PLACEHOLDER_{trait.upper()}_{item_num} (simulated self-report item; not a psychological instrument)",
            })
    return {
        "schema_type": "synthetic_big_five_placeholder",
        "validated_instrument": False,
        "scale_min": scale_min,
        "scale_max": scale_max,
        "items_per_trait": items_per_trait,
        "traits": list(TRAITS),
        "items": items,
    }


def sample_latent_traits(rng):
    """Sample 5 continuous latent traits on [0.0, 1.0] from Beta(2, 2)."""
    return {trait: rng.betavariate(2.0, 2.0) for trait in TRAITS}


def simulate_responses(
    latent_traits,
    schema,
    rng,
    response_noise=0.45,
    missing_rate=0.0,
    consistency=1.0,
):
    """Generate ordinal Likert self-report responses from latent traits."""
    if not 0.0 <= missing_rate < 1.0:
        raise ValueError("missing_rate must be in [0.0, 1.0)")
    if response_noise < 0.0 or consistency <= 0.0:
        raise ValueError("response_noise must be >= 0 and consistency > 0")
    for trait in TRAITS:
        val = latent_traits.get(trait)
        if val is None or not 0.0 <= val <= 1.0:
            raise ValueError(f"latent trait {trait!r} must be in [0.0, 1.0]")

    smin = schema["scale_min"]
    smax = schema["scale_max"]
    mid = (smin + smax) / 2.0
    half_span = (smax - smin) / 2.0

    responses = {}
    for item in schema["items"]:
        item_id = item["item_id"]
        if missing_rate > 0.0 and rng.random() < missing_rate:
            responses[item_id] = None
            continue
        latent = latent_traits[item["trait"]]
        z = consistency * (latent - 0.5) - item["difficulty"] + rng.gauss(0.0, response_noise)
        positive_score = int(round(mid + (half_span + 0.6) * z))
        positive_score = max(smin, min(smax, positive_score))
        if item["keyed"] == -1:
            responses[item_id] = smin + smax - positive_score
        else:
            responses[item_id] = positive_score
    return responses


def score_responses(responses, schema):
    """Validate, reverse-score, and normalize Likert responses into [0.0, 1.0] trait estimates."""
    items_by_id = {item["item_id"]: item for item in schema["items"]}
    unknown = set(responses) - set(items_by_id)
    if unknown:
        raise ValueError(f"unknown questionnaire item IDs: {sorted(unknown)}")

    smin = schema["scale_min"]
    smax = schema["scale_max"]
    span = float(smax - smin)

    sums = {trait: 0.0 for trait in TRAITS}
    counts = {trait: 0 for trait in TRAITS}
    expected_counts = {trait: 0 for trait in TRAITS}
    for item in schema["items"]:
        expected_counts[item["trait"]] += 1

    for item_id, raw in responses.items():
        if raw is None:
            continue
        if isinstance(raw, bool) or not isinstance(raw, int) or not (smin <= raw <= smax):
            raise ValueError(f"invalid Likert response for {item_id!r}: {raw!r}; expected int in [{smin}, {smax}] or None")
        item = items_by_id[item_id]
        adjusted = (smin + smax - raw) if item["keyed"] == -1 else raw
        trait = item["trait"]
        sums[trait] += (adjusted - smin) / span
        counts[trait] += 1

    traits = {}
    trait_completeness = {}
    total_answered = 0
    total_items = len(schema["items"])
    for trait in TRAITS:
        c = counts[trait]
        exp_c = expected_counts[trait]
        total_answered += c
        trait_completeness[trait] = c / exp_c if exp_c else 0.0
        traits[trait] = (sums[trait] / c) if c > 0 else 0.5

    return {
        "traits": traits,
        "completeness": total_answered / total_items if total_items else 0.0,
        "trait_completeness": trait_completeness,
    }
