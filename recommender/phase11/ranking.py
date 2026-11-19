"""Phase 11 personality-aware ranker extending Phase 10 three_w3 without changing it."""

import math

from .. import hybrid
from .personality import assert_observed_profile


def rank_with_personality(
    pool_ids,
    base_scores,
    user_profile,
    movie_features,
    personality_model,
    personality_weight=0.0,
    enabled=True,
    k=10,
):
    """Extend Phase 10 baseline scores over a FIXED candidate pool with a bounded personality signal.

    When enabled=False, personality_weight=0.0, user_profile is None, or personality_model is None,
    returns `base_scores` untouched so top-k ranking is bit-identical to Phase 10 `three_w3`.
    """
    pool = list(dict.fromkeys(pool_ids))
    if not pool:
        return [], {}, {"personality_active": False, "candidates": 0}

    if not enabled or personality_weight == 0.0 or user_profile is None or personality_model is None:
        recs = [mid for mid, _ in sorted(base_scores.items(), key=lambda row: (-row[1], row[0]))][:k]
        return recs, dict(base_scores), {"personality_active": False, "candidates": len(pool)}

    if not 0.0 <= personality_weight <= 1.0:
        raise ValueError("personality_weight must be in [0.0, 1.0]")

    profile = assert_observed_profile(user_profile)
    pers_raw = personality_model.score_candidates(profile, movie_features, pool)
    if not pers_raw:
        recs = [mid for mid, _ in sorted(base_scores.items(), key=lambda row: (-row[1], row[0]))][:k]
        return recs, dict(base_scores), {"personality_active": False, "candidates": len(pool)}

    # Guard against numerical invalidity before normalization.
    for val in pers_raw.values():
        if math.isnan(val) or math.isinf(val):
            raise ValueError("personality score produced NaN or Inf")

    base_sub = {mid: base_scores[mid] for mid in pool if mid in base_scores}
    base_n = hybrid.normalize(base_sub)
    pers_pool = {mid: pers_raw.get(mid, 0.0) for mid in base_sub}
    pers_n = hybrid.normalize(pers_pool)
    w_base = 1.0 - personality_weight
    blended = {
        mid: w_base * base_n[mid] + personality_weight * pers_n[mid]
        for mid in base_sub
    }
    recs = [mid for mid, _ in sorted(blended.items(), key=lambda row: (-row[1], row[0]))][:k]
    return recs, blended, {"personality_active": True, "candidates": len(base_sub), "weight": personality_weight}
