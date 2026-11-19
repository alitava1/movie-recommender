"""Modular personality-aware ranking wrapper preserving the Phase 6 baseline."""

from .. import diagnostics, hybrid, nested


ARCHITECTURES = (
    "cf_content_temporal",
    "personality_only",
    "cf_content_temporal_personality",
    "cf_content_temporal_random_personality",
    "cf_content_temporal_shuffled_personality",
    "cf_content_temporal_personality_disabled",
)

PERSONALITY_WEIGHT_GRID = (0.0, 0.05, 0.10, 0.20, 0.30, 0.40)


def baseline_cf_content_temporal_scores(
    user_ratings,
    movies,
    vectors,
    cf_model,
    prediction_shrinkage,
    cf_weight,
    top_m=100,
    threshold=4.0,
    popularity_scores=None,
):
    """Compute Phase 6 cf_top100_content_enrichment scores (or popularity fallback if cold user)."""
    candidates = movies.keys() - user_ratings.keys()
    if not user_ratings:
        if popularity_scores:
            return {mid: popularity_scores[mid]["bayesian"] for mid in candidates if mid in popularity_scores}
        return {}
    evidence = cf_model.evidence(user_ratings, candidates)
    preds = nested._cf_predictions(evidence, prediction_shrinkage)
    top_ids = diagnostics.cf_top_m_candidates(preds, top_m)
    if not top_ids:
        return {}
    enriched = diagnostics.enriched_content_scores(user_ratings, movies, vectors, top_ids, threshold)
    return diagnostics.rerank_pool_scores(top_ids, preds, enriched, cf_weight)


def blend_with_personality(base_scores, personality_scores, personality_weight=0.0, enabled=True):
    """Combine normalized baseline scores with normalized personality scores over identical candidates.

    When disabled or personality_weight == 0.0, returns base_scores untouched so ranking
    reproduces the non-personality baseline bit-for-bit.
    """
    if not enabled or personality_weight == 0.0:
        return dict(base_scores)
    if not 0.0 <= personality_weight <= 1.0:
        raise ValueError("personality_weight must be in [0.0, 1.0]")
    if not base_scores:
        return {}
    if not personality_scores:
        return dict(base_scores)
    base_n = hybrid.normalize(base_scores)
    pers_pool = {mid: personality_scores.get(mid, 0.0) for mid in base_scores}
    pers_n = hybrid.normalize(pers_pool)
    w_base = 1.0 - personality_weight
    return {
        mid: w_base * base_n[mid] + personality_weight * pers_n[mid]
        for mid in base_scores
    }


def rank_architecture(
    architecture,
    base_scores,
    user_profile,
    movie_features,
    personality_model,
    personality_weight=0.0,
    all_candidates=None,
    k=10,
):
    """Rank top-k movie IDs for any of the six Phase 8 experimental configurations A-F."""
    if architecture not in ARCHITECTURES:
        raise ValueError(f"unknown architecture {architecture!r}")

    if architecture == "cf_content_temporal":
        return nested._top_ids(base_scores, k)

    if architecture == "cf_content_temporal_personality_disabled":
        # Explicitly disabled: pass base_scores directly without any personality influence.
        scores = blend_with_personality(base_scores, {}, personality_weight=personality_weight, enabled=False)
        return nested._top_ids(scores, k)

    if architecture == "personality_only":
        pool = all_candidates if all_candidates is not None else movie_features.keys()
        if user_profile is None or personality_model is None:
            return []
        pers = personality_model.score_candidates(user_profile, movie_features, pool)
        return nested._top_ids(pers, k)

    # C, D, E: blend baseline scores over base_scores' candidate pool with personality scores.
    if user_profile is None or personality_model is None or not base_scores:
        return nested._top_ids(base_scores, k)
    pers = personality_model.score_candidates(user_profile, movie_features, base_scores.keys())
    blended = blend_with_personality(base_scores, pers, personality_weight=personality_weight, enabled=True)
    return nested._top_ids(blended, k)
