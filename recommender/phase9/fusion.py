"""Phase 9 candidate fusion: union, Reciprocal Rank Fusion, and source quotas."""

from dataclasses import dataclass

RRF_K = 60


@dataclass(frozen=True)
class FusionResult:
    source: str
    items: tuple                 # ((movie_id, fused_score), ...) sorted by (-score, movie_id)
    attribution: dict            # per-source diagnostics for the final pool

    def ids(self):
        return [movie_id for movie_id, _ in self.items]


def _scores_map(candidate_list, max_candidates):
    rows = candidate_list.items[:max_candidates] if max_candidates else candidate_list.items
    return {movie_id: score for movie_id, score in rows}


def merge_candidates(lists, max_candidates, mode="union", quotas=None, rrf_k=RRF_K):
    """Fuse candidate lists into one pool capped at max_candidates.

    mode="union": fused score = max normalized source score (ties by movie_id).
    mode="rrf": fused score = sum of 1 / (rrf_k + rank) over sources containing the item.
    quotas: optional {source: fraction}; each source contributes at most that share of the cap
    (taken in its own deterministic order), remaining slots filled by the fused ranking.
    """
    if not lists:
        return FusionResult("empty", (), {})
    if mode not in ("union", "rrf"):
        raise ValueError("mode must be 'union' or 'rrf'")
    if any(not 0.0 <= q for q in (quotas or {}).values()):
        raise ValueError("quotas must be non-negative")

    score_maps = {cl.source: _scores_map(cl, max_candidates) for cl in lists}
    rank_maps = {
        source: {movie_id: rank + 1 for rank, movie_id in enumerate(scores.keys())}
        for source, scores in score_maps.items()
    }
    all_ids = set().union(*(scores.keys() for scores in score_maps.values()))

    if mode == "rrf":
        fused = {
            movie_id: sum(1.0 / (rrf_k + rank_maps[source][movie_id]) for source in rank_maps if movie_id in rank_maps[source])
            for movie_id in all_ids
        }
    else:
        norm_maps = {}
        for source, scores in score_maps.items():
            if scores:
                lo, hi = min(scores.values()), max(scores.values())
                span = hi - lo
                norm_maps[source] = {
                    movie_id: ((value - lo) / span if span > 0 else 0.5)
                    for movie_id, value in scores.items()
                }
            else:
                norm_maps[source] = {}
        fused = {
            movie_id: max(norm_maps[source][movie_id] for source in norm_maps if movie_id in norm_maps[source])
            for movie_id in all_ids
        }

    ordered = sorted(fused.items(), key=lambda row: (-row[1], row[0]))
    pool_ids = {movie_id for movie_id, _ in ordered[:max_candidates]}

    if quotas:
        quota_ids = set()
        for source, fraction in quotas.items():
            if source not in score_maps:
                continue
            allowance = int(round(max_candidates * fraction))
            ranked = sorted(score_maps[source].items(), key=lambda row: (-row[1], row[0]))[:allowance]
            quota_ids.update(movie_id for movie_id, _ in ranked)
        # Quota picks first, then fill any remaining slots by fused ranking; trim to the cap.
        for movie_id, _ in ordered:
            if len(quota_ids) >= max_candidates:
                break
            quota_ids.add(movie_id)
        pool_ids = set(list(quota_ids)[:max_candidates])

    final_items = tuple(sorted(((mid, fused[mid]) for mid in pool_ids), key=lambda row: (-row[1], row[0])))
    final_id_set = {mid for mid, _ in final_items}

    attribution = {}
    for cl in lists:
        source_ids = set(rank_maps[cl.source].keys())
        others = set().union(*(keys for src, keys in rank_maps.items() if src != cl.source)) if len(rank_maps) > 1 else set()
        attribution[cl.source] = {
            "contributed_candidates": len(source_ids),
            "overlapping_with_other_sources": len(source_ids & others),
            "unique_to_source_before_cap": len(source_ids - others),
            "in_final_pool": len(source_ids & final_id_set),
        }
    return FusionResult("fusion", final_items, attribution)
