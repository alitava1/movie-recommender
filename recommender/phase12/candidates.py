"""Phase 12 structured candidate records, validation, and transparent scoring rubric."""

import csv
import json
from pathlib import Path

from .sources import CANDIDATES  # re-exported for convenience

# Participant-linkage classification categories (Section 5 of the mission).
VERIFIED_SAME = "verified_same_participants"
VERIFIED_RESEARCH = "verified_research_linkage"
PARTIAL = "partial_linkage"
UNVERIFIED = "unverified_linkage"
UNLINKED = "unlinked_datasets"

# Personality-instrument classification categories (Section 6).
INST_SCORES_ONLY = "final_scores_from_validated_inventory"
INST_RESPONSES = "raw_questionnaire_responses"
INST_INFERRED = "inferred_from_text_or_behavior"
INST_PREDICTED = "predicted_by_ml_model"
INST_SYNTHETIC = "synthetic_profiles"
INST_UNKNOWN = "unknown"

# License-status classification categories (Section 7).
LIC_OPEN_RESEARCH = "open_research_license"
LIC_NONCOMMERCIAL = "non_commercial_research_access"
LIC_APPLICATION = "application_or_agreement_required"
LIC_UNAVAILABLE = "unavailable"
LIC_UNKNOWN = "unknown"

REQUIRED_FIELDS = (
    "name", "source_url", "publication", "publication_year", "availability",
    "license", "license_status", "instrument", "instrument_class", "dimensions",
    "instrument_validity", "participants", "preference_records", "preference_domain",
    "linkage", "linkage_confidence", "timestamps", "cold_start_suitability",
    "missing_data", "demographics", "consent", "privacy", "reproducibility",
    "main_limitations", "overall_assessment",
)

# Transparent scoring rubric: 0-2 per dimension; None = unknown (never scored as positive).
RUBRIC_DIMENSIONS = (
    "personality_measurement_validity",
    "participant_linkage_reliability",
    "preference_data_suitability",
    "cold_start_suitability",
    "sample_size",
    "temporal_evaluation_suitability",
    "reproducibility",
    "legal_access_feasibility",
    "ethical_feasibility",
)

_SCORE_FOR = {
    # personality measurement validity
    ("personality_measurement_validity", INST_RESPONSES): 2,
    ("personality_measurement_validity", INST_SCORES_ONLY): 1,
    ("personality_measurement_validity", INST_INFERRED): 0,
    ("personality_measurement_validity", INST_PREDICTED): 0,
    ("personality_measurement_validity", INST_SYNTHETIC): 0,
    # participant linkage reliability
    ("participant_linkage_reliability", VERIFIED_SAME): 2,
    ("participant_linkage_reliability", VERIFIED_RESEARCH): 2,
    ("participant_linkage_reliability", PARTIAL): 1,
    ("participant_linkage_reliability", UNVERIFIED): 0,
    ("participant_linkage_reliability", UNLINKED): 0,
    # legal access feasibility
    ("legal_access_feasibility", LIC_OPEN_RESEARCH): 2,
    ("legal_access_feasibility", LIC_NONCOMMERCIAL): 2,
    ("legal_access_feasibility", LIC_APPLICATION): 1,
    ("legal_access_feasibility", LIC_UNAVAILABLE): 0,
    ("legal_access_feasibility", LIC_UNKNOWN): 0,
}


def validate_candidate(record):
    """Validate required metadata; unknown values are preserved, never converted into evidence."""
    missing = [f for f in REQUIRED_FIELDS if f not in record]
    if missing:
        raise ValueError(f"candidate {record.get('name', '<unnamed>')!r} missing fields: {missing}")
    if record["linkage_confidence"] == VERIFIED_SAME and record["linkage"] not in (VERIFIED_SAME, VERIFIED_RESEARCH):
        raise ValueError(f"candidate {record['name']!r}: linkage_confidence VERIFIED_SAME requires verified linkage class")
    return record


def score_candidate(record):
    """Score each rubric dimension 0-2 (None when evidence is unknown) plus an overall grade.

    Overall grade is gated: any hard blocker (unverified/unlinked personality linkage or
    unavailable license) caps the overall recommendation regardless of other strengths.
    """
    dims = {}
    for dim in RUBRIC_DIMENSIONS:
        if dim == "personality_measurement_validity":
            dims[dim] = _SCORE_FOR.get((dim, record.get("instrument_class")), None)
        elif dim == "participant_linkage_reliability":
            dims[dim] = _SCORE_FOR.get((dim, record.get("linkage")), None)
        elif dim == "legal_access_feasibility":
            dims[dim] = _SCORE_FOR.get((dim, record.get("license_status")), None)
        else:
            dims[dim] = record.get(f"score_{dim}")

    hard_blockers = []
    if dims["participant_linkage_reliability"] in (0, None):
        hard_blockers.append("participant_linkage_unverified_or_unknown")
    if dims["personality_measurement_validity"] in (0, None):
        hard_blockers.append("personality_instrument_not_validated_or_unknown")
    if dims["legal_access_feasibility"] in (0, None):
        hard_blockers.append("license_unavailable_or_unknown")

    scored = [v for v in dims.values() if v is not None]
    total = sum(scored) if scored else 0
    max_total = 2 * len(RUBRIC_DIMENSIONS)

    if hard_blockers:
        overall = "unsuitable"
    elif total >= 15:
        overall = "strong"
    elif total >= 11:
        overall = "moderate"
    else:
        overall = "weak"

    return {
        "dimensions": dims,
        "total_score": total,
        "max_score": max_total,
        "hard_blockers": hard_blockers,
        "overall": overall,
    }


def load_candidates(path):
    """Load candidate records from JSON (list) with validation."""
    records = json.loads(Path(path).read_text(encoding="utf-8"))
    return [validate_candidate(r) for r in records]


def save_candidates(path, records):
    Path(path).write_text(
        json.dumps([validate_candidate(r) for r in records], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def candidates_to_csv(records, path):
    fields = list(REQUIRED_FIELDS) + [f"score_{d}" for d in RUBRIC_DIMENSIONS if not d.startswith(("personality_measurement", "participant_linkage", "legal_access"))]
    fields += ["total_score", "overall"]
    with Path(path).open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for rec in records:
            sc = score_candidate(rec)
            writer.writerow({**{k: rec.get(k) for k in REQUIRED_FIELDS}, "total_score": sc["total_score"], "overall": sc["overall"]})
