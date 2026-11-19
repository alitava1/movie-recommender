"""Phase 11 preliminary audit and Phase 10 non-personality baseline reproduction."""

import hashlib
import json
from pathlib import Path

from .. import content
from ..data import load_movies, load_ratings
from ..phase7.temporal import split_outer_temporal
from ..phase10.experiments import RELEVANCE_CONFIGS, evaluate_configurations

REFERENCE_SYSTEMS = ("cf_only", "popularity_only", "rrf_cf_pop", "combined_a50", "three_w3")


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def audit_inputs(root):
    """Verify integrity and provenance of ML-1M data, Phase 8, and Phase 10 artifacts."""
    root = Path(root)
    ml1m = root / "data" / "raw" / "ml-1m"
    p10_res = root / "artifacts" / "evaluation" / "phase10-candidate-aware-ranking" / "ranking_results.json"
    p8_res = root / "artifacts" / "evaluation" / "phase8-personality" / "scenario_results.json"
    files = {
        "ml1m_ratings": ml1m / "ratings.dat",
        "ml1m_movies": ml1m / "movies.dat",
        "phase10_ranking_results": p10_res,
        "phase8_scenario_results": p8_res,
    }
    missing = [name for name, path in files.items() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing required Phase 11 prerequisite files: {missing}")

    ratings = load_ratings("ml-1m", ml1m / "ratings.dat")
    movies = load_movies("ml-1m", ml1m / "movies.dat")
    outer_train, outer_test, excluded = split_outer_temporal(ratings)
    return {
        "checksums_sha256": {name: sha256_file(path) for name, path in files.items()},
        "ml1m": {
            "ratings": len(ratings),
            "catalog_movies": len(movies),
            "outer_train_rows": len(outer_train),
            "outer_test_targets": len(outer_test),
            "excluded_users": len(excluded),
        },
        "phase10_reference_systems": list(REFERENCE_SYSTEMS),
    }


def reproduce_phase10_baselines(root, outer_train, outer_test_rows, movies, vectors, settings, user_filter=None, tol=1e-12):
    """Re-run Phase 10's exact 5 reference baselines and compare to stored Phase 10 outer results."""
    p10_path = Path(root) / "artifacts" / "evaluation" / "phase10-candidate-aware-ranking" / "ranking_results.json"
    expected_summary = json.loads(p10_path.read_text(encoding="utf-8"))["payload"]["outer_summary"]

    relevant_outer = {}
    for uid, mid, rating, _ in outer_test_rows:
        if rating >= 4.0:
            relevant_outer.setdefault(uid, set()).add(mid)

    ref_cfgs = [c for c in RELEVANCE_CONFIGS if c["name"] in REFERENCE_SYSTEMS]
    reproduced = evaluate_configurations(
        outer_train, relevant_outer, movies, vectors, settings,
        user_filter=user_filter,
        relevance_configs=ref_cfgs,
        diversity_configs=({"name": "none", "layer": None},),
        diversity_base=next(c for c in ref_cfgs if c["name"] == "three_w3"),
    )

    comparison = {}
    all_ok = user_filter is None
    for name in REFERENCE_SYSTEMS:
        exp = expected_summary[name]
        rep = reproduced["summary"][name]
        diff = {m: rep[m] - exp[m] for m in ("ndcg", "map", "hr", "coverage", "ild")}
        matched = max(abs(v) for v in diff.values()) <= tol
        if user_filter is None and not matched:
            all_ok = False
        comparison[name] = {
            "expected": {m: exp[m] for m in ("ndcg", "map", "hr", "coverage", "ild", "users")},
            "reproduced": {m: rep[m] for m in ("ndcg", "map", "hr", "coverage", "ild", "users")},
            "diff": diff,
            "matched_within_tol": matched if user_filter is None else "smoke_subset",
        }
    return {
        "reproduced": all_ok if user_filter is None else "smoke_subset",
        "tolerance": tol,
        "wall_seconds": reproduced["wall_seconds"],
        "systems": comparison,
        "baseline_three_w3_per_user": reproduced["per_user"]["three_w3"],
    }
