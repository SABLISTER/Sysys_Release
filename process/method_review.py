"""Study design classification and span validation."""
import json
import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

ALLOWED_METHOD_TYPES = {
    "randomized_controlled_trial",
    "cohort_observational",
    "case_control",
    "cross_sectional",
    "longitudinal",
    "systematic_review_meta_analysis",
    "narrative_review",
    "case_report_case_series",
    "preclinical_animal",
    "in_vitro_cellular",
    "imaging_ml_classifier",
    "methods_protocol",
    "genetic_association",
    "mixed_methods",
    "unknown",
}


@dataclass
class SpanValidation:
    field: str
    quote: str
    start_char: int = -1
    end_char: int = -1
    valid: bool = False
    reason: str = ""


def extraction_prompt(title: str, abstract: str) -> str:
    """Build the LLM prompt for study design classification."""
    return f"""Classify the study design and extract method details from this paper.

Title: {title}
Abstract: {abstract}

Respond with ONLY a JSON object:
{{
    "study_method_type": "<one of: randomized_controlled_trial, cohort_observational, case_control, cross_sectional, longitudinal, systematic_review_meta_analysis, narrative_review, case_report_case_series, preclinical_animal, in_vitro_cellular, imaging_ml_classifier, methods_protocol, genetic_association, mixed_methods, unknown>",
    "confidence": <0.0-1.0>,
    "sample_size": <integer or -1 if unknown>,
    "has_control_group": <true/false>,
    "is_longitudinal": <true/false>,
    "design_notes": "<brief note about the design>",
    "evidence_spans": ["<key quote 1>", "<key quote 2>"]
}}"""


def parse_json_object(text: str) -> dict | None:
    """Parse JSON from LLM response, stripping markdown fences if needed."""
    from process.llm import parse_llm_json

    result = parse_llm_json(text)
    if result is None:
        logger.debug("Could not parse JSON from LLM response: %s",
                     (text or "")[:200])
    return result


def validate_spans(abstract: str, spans: list) -> list[SpanValidation]:
    """Check each quote span is a real substring of the abstract.

    For each span: try exact match, then case-insensitive, then fuzzy.
    Record start_char/end_char if found.
    """
    results = []
    if not abstract or not spans:
        return results

    abstract_lower = abstract.lower()

    for span_text in spans:
        if not isinstance(span_text, str) or not span_text.strip():
            continue

        sv = SpanValidation(field="evidence_span", quote=span_text)

        # 1. Exact match
        idx = abstract.find(span_text)
        if idx >= 0:
            sv.start_char = idx
            sv.end_char = idx + len(span_text)
            sv.valid = True
            sv.reason = "exact_match"
            results.append(sv)
            continue

        # 2. Case-insensitive match
        span_lower = span_text.lower()
        idx = abstract_lower.find(span_lower)
        if idx >= 0:
            sv.start_char = idx
            sv.end_char = idx + len(span_text)
            sv.valid = True
            sv.reason = "case_insensitive_match"
            results.append(sv)
            continue

        # 3. Fuzzy match — normalize whitespace and try again.
        # Offsets are set to -1 because norm_abstract positions do not map
        # back to character positions in the original abstract.
        norm_abstract = re.sub(r"\s+", " ", abstract_lower)
        norm_span = re.sub(r"\s+", " ", span_lower.strip())
        idx = norm_abstract.find(norm_span)
        if idx >= 0:
            # Offsets against the normalized string do not map reliably back to
            # the original abstract, so start_char and end_char remain unset
            # (defaulting to None) for fuzzy whitespace matches.
            sv.start_char = -1
            sv.end_char = -1
            sv.valid = True
            sv.reason = "fuzzy_whitespace_match"
            results.append(sv)
            continue

        # 4. Check if a large substring (>60% of span) appears in abstract
        words = norm_span.split()
        if len(words) >= 4:
            # Try progressively shorter sub-spans from the start
            for end in range(len(words), max(len(words) // 2, 3) - 1, -1):
                sub = " ".join(words[:end])
                if sub in norm_abstract:
                    sv.valid = True
                    sv.reason = f"partial_match ({end}/{len(words)} words)"
                    break

        if not sv.valid:
            sv.reason = "not_found"

        results.append(sv)

    return results


def validate_method_profile(
    abstract: str,
    parsed: dict | None,
    confidence_threshold: float = 0.5,
) -> dict:
    """Normalize parsed LLM output: enforce allowed method types, validate spans.

    Args:
        abstract: The paper abstract used for span validation.
        parsed: Raw LLM output dict, or None if the LLM returned nothing.
        confidence_threshold: Minimum confidence below which a record is flagged
            for manual review.  Defaults to 0.5; callers should pass the value
            from ``Config.get_method_review_config().get("confidence_threshold")``
            so the threshold stays consistent with ``research_config.yaml``.
    """
    default = {
        "study_method_type": "unknown",
        "confidence": 0.0,
        "sample_size": -1,
        "has_control_group": False,
        "is_longitudinal": False,
        "design_notes": "",
        "evidence_spans": [],
        "span_validations": [],
        "needs_manual_review": True,
        "manual_review_reason": "no_llm_response",
    }

    if parsed is None:
        return default

    profile = dict(default)

    # Extract and normalize study_method_type
    raw_type = str(parsed.get("study_method_type", "unknown")).strip().lower()
    raw_type = raw_type.replace(" ", "_").replace("-", "_")
    if raw_type in ALLOWED_METHOD_TYPES:
        profile["study_method_type"] = raw_type
    else:
        profile["study_method_type"] = "unknown"
        logger.debug("Unrecognized method type '%s', defaulting to unknown", raw_type)

    # Confidence
    try:
        profile["confidence"] = float(parsed.get("confidence", 0.0))
    except (ValueError, TypeError):
        profile["confidence"] = 0.0

    # Sample size
    try:
        profile["sample_size"] = int(parsed.get("sample_size", -1))
    except (ValueError, TypeError):
        profile["sample_size"] = -1

    # Boolean fields
    profile["has_control_group"] = bool(parsed.get("has_control_group", False))
    profile["is_longitudinal"] = bool(parsed.get("is_longitudinal", False))
    profile["design_notes"] = str(parsed.get("design_notes", ""))

    # Evidence spans + validation
    raw_spans = parsed.get("evidence_spans", [])
    if isinstance(raw_spans, list):
        profile["evidence_spans"] = [s for s in raw_spans if isinstance(s, str)]
    else:
        profile["evidence_spans"] = []

    if abstract and profile["evidence_spans"]:
        validations = validate_spans(abstract, profile["evidence_spans"])
        profile["span_validations"] = [
            {
                "quote": sv.quote,
                "start_char": sv.start_char,
                "end_char": sv.end_char,
                "valid": sv.valid,
                "reason": sv.reason,
            }
            for sv in validations
        ]

    # Flag for manual review
    needs_review = False
    review_reasons = []
    if profile["confidence"] < confidence_threshold:
        needs_review = True
        review_reasons.append(f"low_confidence ({profile['confidence']:.2f})")
    if profile["study_method_type"] == "unknown":
        needs_review = True
        review_reasons.append("unknown_type")
    profile["needs_manual_review"] = needs_review
    profile["manual_review_reason"] = "; ".join(review_reasons) if review_reasons else ""

    return profile


def summarize_method_types(records: list[dict]) -> dict:
    """Produce summary: count by method type, % flagged, total."""
    type_counts: dict[str, int] = {}
    flagged = 0
    total = len(records)

    for rec in records:
        mp = rec.get("method_profile", {})
        mtype = mp.get("study_method_type", "unknown")
        type_counts[mtype] = type_counts.get(mtype, 0) + 1
        if mp.get("needs_manual_review", False):
            flagged += 1

    return {
        "total_papers": total,
        "flagged_for_review": flagged,
        "flagged_pct": round(flagged / total * 100, 1) if total else 0.0,
        "type_counts": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
    }
