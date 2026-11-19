"""Phase 10 candidate feature extraction, normalization, and the common scoring interface."""

import math

from .. import content as content_mod
from ..hybrid import normalize as minmax_normalize


def extract_components(user_ratings, pool_ids, cf_scores, popularity_counts, content_scores, movies):
    """Per-candidate component scores for one pool.

    Returns (components, coverage) where components maps component name -> {movie_id: raw score}
    and coverage records how many candidates each component can score.

    - cf: predicted rating (missing where no CF evidence)
    - popularity: positive-interaction count from training (always available for catalog items
      with at least one positive training rating; 0.0 otherwise)
    - content: TF-IDF user-profile cosine (missing where the movie vector is empty)
    """
    pool = list(dict.fromkeys(pool_ids))
    cf = {mid: cf_scores[mid] for mid in pool if mid in cf_scores}
    pop = {mid: float(popularity_counts.get(mid, 0)) for mid in pool}
    cont = {mid: content_scores[mid] for mid in pool if mid in content_scores}
    missing_content = [mid for mid in pool if mid not in cont]
    components = {"cf": cf, "popularity": pop, "content": cont}
    coverage = {
        "candidates": len(pool),
        "cf_scored": len(cf),
        "popularity_scored": len(pop),
        "content_scored": len(cont),
        "missing_content": len(missing_content),
    }
    return components, coverage


def normalize_components(components):
    """Per-pool min-max normalization to [0, 1]; constant components map to neutral 0.5.

    Popularity is log-compressed before normalization so that head items do not pin the scale:
    norm_pop uses log1p(count) min-max scaled per pool. Missing values stay missing.
    """
    norm = {}
    for name, scores in components.items():
        if name == "popularity":
            scores = {mid: math.log1p(value) for mid, value in scores.items()}
        norm[name] = minmax_normalize(scores) if scores else {}
    return norm


def combined_score(pool_ids, norm_components, weights, missing_policy="renormalize"):
    """Weighted blend of normalized components over the pool.

    weights: {component: weight}; all weights must be >= 0 and at least one > 0.
    missing_policy="renormalize": candidates lacking a component are scored with the remaining
    weights renormalized (never silently dropped). "zero": missing components contribute 0.
    """
    if missing_policy not in ("renormalize", "zero"):
        raise ValueError("missing_policy must be 'renormalize' or 'zero'")
    total_weight = sum(weights.values())
    if total_weight <= 0 or any(w < 0 for w in weights.values()):
        raise ValueError("weights must be non-negative with a positive sum")
    out = {}
    for mid in pool_ids:
        num = 0.0
        den = 0.0
        for name, weight in weights.items():
            value = norm_components.get(name, {}).get(mid)
            if value is None:
                if missing_policy == "zero":
                    den += weight
                continue
            num += weight * value
            den += weight
        out[mid] = num / den if den > 0 else 0.0
    return out


def rrf_score(norm_components, rrf_k=60):
    """Rank-based fusion over the ranked order of each normalized component (k=60 default)."""
    fused = {}
    for name, scores in norm_components.items():
        ranked = sorted(scores.items(), key=lambda row: (-row[1], row[0]))
        for rank, (mid, _) in enumerate(ranked, 1):
            fused[mid] = fused.get(mid, 0.0) + 1.0 / (rrf_k + rank)
    return fused


def content_scores_for_pool(user_ratings, vectors, pool_ids):
    """TF-IDF cosine scores for pool items; items with empty vectors are absent (missing)."""
    if not user_ratings:
        return {}
    profile = content_mod.user_profile(vectors, user_ratings)
    if not profile:
        return {}
    return content_mod.score(profile, vectors, set(pool_ids))
