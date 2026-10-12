"""
GRADE evidence certainty assessment for the Sysys pipeline.

Implements the Grading of Recommendations Assessment, Development and
Evaluation (GRADE) framework, rating certainty of evidence across five
domains: risk of bias, inconsistency, indirectness, imprecision, and
publication bias.  Each hypothesis component receives an overall certainty
rating (High / Moderate / Low / Very Low) derived from the domain ratings.

The module also generates a Summary of Findings (SoF) table suitable for
display in the GUI and export to Markdown / JSON.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from core.claims import claim_list, claim_value
from core.io import load_json as _load_json

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class GradeConcern(Enum):
    NO_CONCERN = "no_concern"
    SERIOUS = "serious"
    VERY_SERIOUS = "very_serious"


class GradeCertainty(Enum):
    HIGH = "High"
    MODERATE = "Moderate"
    LOW = "Low"
    VERY_LOW = "Very Low"


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class DomainRating:
    domain: str  # risk_of_bias / inconsistency / indirectness / imprecision / publication_bias
    concern: GradeConcern = GradeConcern.NO_CONCERN
    rationale: str = ""
    downgrade_levels: int = 0  # 0, 1, or 2
    data_points: dict = field(default_factory=dict)


@dataclass
class ComponentGrade:
    component_id: str
    component_label: str
    n_studies: int = 0
    starting_certainty: str = "Low"  # "High" or "Low"
    domains: list = field(default_factory=list)  # list[DomainRating]
    overall_certainty: GradeCertainty = GradeCertainty.VERY_LOW
    summary: str = ""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HIGH_QUALITY_DESIGNS = {
    "randomized_controlled_trial",
    "systematic_review_meta_analysis",
    "rct",
    "meta-analysis",
    "systematic review",
}

_CERTAINTY_LADDER = [
    GradeCertainty.HIGH,
    GradeCertainty.MODERATE,
    GradeCertainty.LOW,
    GradeCertainty.VERY_LOW,
]


def _evidence_tier(study_method_type: str) -> str:
    """Map a study design label to an evidence tier.

    Covers the method-type vocabulary emitted by
    ``process.method_review.ALLOWED_METHOD_TYPES`` (plus legacy aliases);
    anything unrecognized falls through to "unknown".
    """
    t = study_method_type.lower()
    if t in ("rct", "meta-analysis", "systematic review",
             "randomized_controlled_trial", "systematic_review_meta_analysis"):
        return "high"
    if t in ("cohort", "case-control",
             "cohort_observational", "longitudinal", "case_control"):
        return "moderate"
    if t in ("case report", "expert opinion",
             "case_report_case_series", "cross_sectional", "narrative_review",
             "preclinical_animal", "in_vitro_cellular", "imaging_ml_classifier",
             "genetic_association", "mixed_methods", "methods_protocol"):
        return "low"
    return "unknown"


# ---------------------------------------------------------------------------
# Domain assessors
# ---------------------------------------------------------------------------

def assess_risk_of_bias(
    method_profiles: list[dict],
    component_studies: list[str],
) -> DomainRating:
    """Assess risk of bias based on study-design quality.

    Logic:
    - Count studies by evidence_tier for this component's articles.
    - >50 % "low" tier  -> very serious concern (downgrade 2)
    - >50 % "moderate"  -> serious concern (downgrade 1)
    - >50 % "high"      -> no concern
    - Also factor in flagged_for_review count and has_control_group rate.
    """
    relevant = [
        mp for mp in method_profiles
        if mp.get("article_id") in set(component_studies)
    ]
    n = len(relevant)
    if n == 0:
        return DomainRating(
            domain="risk_of_bias",
            concern=GradeConcern.VERY_SERIOUS,
            rationale="No method-profile data available for component studies.",
            downgrade_levels=2,
            data_points={"n_profiles": 0},
        )

    tier_counts: dict[str, int] = {"high": 0, "moderate": 0, "low": 0, "unknown": 0}
    flagged = 0
    control_count = 0
    for mp in relevant:
        tier = _evidence_tier(mp.get("study_method_type", "unknown"))
        tier_counts[tier] = tier_counts.get(tier, 0) + 1
        if mp.get("flagged_for_review"):
            flagged += 1
        if mp.get("has_control_group"):
            control_count += 1

    low_pct = (tier_counts["low"] + tier_counts["unknown"]) / n
    mod_pct = tier_counts["moderate"] / n
    control_rate = control_count / n
    flagged_rate = flagged / n

    # Determine concern level
    if low_pct > 0.5 or flagged_rate > 0.4:
        concern = GradeConcern.VERY_SERIOUS
        downgrade = 2
    elif mod_pct > 0.5 or low_pct > 0.3:
        concern = GradeConcern.SERIOUS
        downgrade = 1
    else:
        concern = GradeConcern.NO_CONCERN
        downgrade = 0

    # Extra penalty if very few studies have controls
    if control_rate < 0.3 and downgrade < 2:
        concern = GradeConcern.SERIOUS if concern == GradeConcern.NO_CONCERN else GradeConcern.VERY_SERIOUS
        downgrade = min(downgrade + 1, 2)

    parts = [
        f"{n} studies evaluated.",
        f"Tier breakdown: {tier_counts['high']} high, {tier_counts['moderate']} moderate, "
        f"{tier_counts['low']} low, {tier_counts['unknown']} unknown.",
    ]
    if flagged:
        parts.append(f"{flagged} flagged for review.")
    if control_rate < 0.5:
        parts.append(f"Only {control_rate:.0%} have a control group.")

    return DomainRating(
        domain="risk_of_bias",
        concern=concern,
        rationale=" ".join(parts),
        downgrade_levels=downgrade,
        data_points={
            "n_profiles": n,
            "tier_counts": tier_counts,
            "flagged": flagged,
            "control_rate": round(control_rate, 3),
        },
    )


def assess_inconsistency(
    interrater_records: list[dict],
    robustness_data: dict,
    component_studies: list[str],
) -> DomainRating:
    """Assess inconsistency (heterogeneity across studies) for one component.

    Uses per-component interrater records derived from Stage 8.  Each record
    gives an original and second-rater score; we derive a per-paper agreement
    score and standard deviation, then average them across the component's
    articles.  Robustness stability is folded in if a global stability score
    is available.
    """
    signals: list[str] = []
    worst_concern = GradeConcern.NO_CONCERN
    worst_downgrade = 0

    study_set = set(component_studies)
    component_records = [
        r for r in interrater_records
        if isinstance(r, dict) and r.get("article_id") in study_set
    ]

    agreement = None
    score_std = None
    if component_records:
        scores = []
        stds = []
        for r in component_records:
            orig = r.get("original_score")
            second = r.get("second_score")
            try:
                orig_i = int(orig) if orig is not None else None
                second_i = int(second) if second is not None else None
            except (ValueError, TypeError):
                orig_i = None
                second_i = None
            if orig_i is not None and second_i is not None:
                diff = abs(orig_i - second_i)
                scores.append(1.0 - diff / 5.0)
                stds.append(diff / 5.0)
        if scores:
            agreement = sum(scores) / len(scores)
        if stds:
            score_std = sum(stds) / len(stds)

    if agreement is not None:
        if agreement < 0.4:
            worst_concern = GradeConcern.VERY_SERIOUS
            worst_downgrade = 2
            signals.append(f"Mean agreement {agreement:.2f} (<0.4 — very low).")
        elif agreement < 0.7:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1
            signals.append(f"Mean agreement {agreement:.2f} (moderate).")
        else:
            signals.append(f"Mean agreement {agreement:.2f} (good).")
    else:
        signals.append("Interrater data not available for this component.")
        worst_concern = GradeConcern.SERIOUS
        worst_downgrade = 1

    if score_std is not None and score_std > 0.3:
        signals.append(f"High score variability (std={score_std:.2f}).")
        if worst_downgrade < 1:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1

    # --- robustness stability (global, available only if present) ---
    stability = None
    if isinstance(robustness_data, dict):
        stability = robustness_data.get("stability_score")

    if stability is not None:
        if stability < 0.5:
            signals.append(f"Robustness stability {stability:.2f} (low).")
            if worst_downgrade < 2:
                worst_concern = GradeConcern.VERY_SERIOUS if worst_downgrade >= 1 else GradeConcern.SERIOUS
                worst_downgrade = min(worst_downgrade + 1, 2)
        elif stability < 0.7:
            signals.append(f"Robustness stability {stability:.2f} (moderate).")
        else:
            signals.append(f"Robustness stability {stability:.2f} (good).")

    return DomainRating(
        domain="inconsistency",
        concern=worst_concern,
        rationale=" ".join(signals) if signals else "Insufficient data.",
        downgrade_levels=worst_downgrade,
        data_points={
            "n_component_records": len(component_records),
            "mean_agreement": round(agreement, 3) if agreement is not None else None,
            "mean_score_std": round(score_std, 3) if score_std is not None else None,
            "stability_score": stability,
        },
    )


def assess_indirectness(
    claims: list[dict],
    component: dict,
) -> DomainRating:
    """Assess indirectness — how directly evidence addresses the hypothesis.

    Checks what fraction of claims mention the component's keywords and
    whether conditions/key entities in claims match the hypothesis focus.
    """
    keywords = [kw.lower() for kw in component.get("keywords", [])]
    comp_label = component.get("label", "").lower()

    if not claims:
        return DomainRating(
            domain="indirectness",
            concern=GradeConcern.VERY_SERIOUS,
            rationale="No claims available to assess directness.",
            downgrade_levels=2,
            data_points={"n_claims": 0, "keyword_match_rate": 0},
        )

    keyword_hits = 0
    for claim in claims:
        # Combine all text fields for matching
        text = " ".join([
            " ".join(claim_list(claim, "findings")),
            " ".join(claim_list(claim, "mechanisms")),
            " ".join(claim_list(claim, "conditions")),
            " ".join(claim_list(claim, "key_entities")),
            str(claim_value(claim, "methodology", "") or ""),
        ]).lower()

        if comp_label and comp_label in text:
            keyword_hits += 1
        elif any(kw in text for kw in keywords):
            keyword_hits += 1

    match_rate = keyword_hits / len(claims) if claims else 0

    if match_rate < 0.2:
        concern = GradeConcern.VERY_SERIOUS
        downgrade = 2
        rationale = (
            f"Only {match_rate:.0%} of {len(claims)} claims directly reference "
            f"the component keywords. Most evidence is about related but not "
            f"identical constructs."
        )
    elif match_rate < 0.5:
        concern = GradeConcern.SERIOUS
        downgrade = 1
        rationale = (
            f"{match_rate:.0%} of {len(claims)} claims reference the component "
            f"keywords. Evidence is partially indirect."
        )
    else:
        concern = GradeConcern.NO_CONCERN
        downgrade = 0
        rationale = (
            f"{match_rate:.0%} of {len(claims)} claims directly reference "
            f"the component keywords."
        )

    return DomainRating(
        domain="indirectness",
        concern=concern,
        rationale=rationale,
        downgrade_levels=downgrade,
        data_points={
            "n_claims": len(claims),
            "keyword_match_rate": round(match_rate, 3),
        },
    )


def assess_imprecision(
    robustness_data: dict,
    n_studies: int,
    component_id: str = "",
) -> DomainRating:
    """Assess imprecision (wide confidence intervals, small samples).

    - Bootstrap CI width: >0.5 very serious, 0.3-0.5 serious, <0.3 no concern
    - n_studies < 3  -> very serious
    - n_studies < 5  -> serious

    The bootstrap CI is currently computed across the whole corpus (Stage 9),
    so it is reported as a global reference while the sample-size concern is
    component-specific.
    """
    signals: list[str] = []
    worst_concern = GradeConcern.NO_CONCERN
    worst_downgrade = 0

    # --- CI width ---
    ci_width = None
    if isinstance(robustness_data, dict):
        bootstrap = robustness_data.get("bootstrap") or {}
        if isinstance(bootstrap, dict):
            ci_lower = bootstrap.get("ci_lower")
            ci_upper = bootstrap.get("ci_upper")
            if ci_lower is not None and ci_upper is not None:
                ci_width = ci_upper - ci_lower

    if ci_width is not None:
        if ci_width > 0.5:
            worst_concern = GradeConcern.VERY_SERIOUS
            worst_downgrade = 2
            signals.append(f"Bootstrap CI width {ci_width:.2f} (very wide).")
        elif ci_width > 0.3:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1
            signals.append(f"Bootstrap CI width {ci_width:.2f} (wide).")
        else:
            signals.append(f"Bootstrap CI width {ci_width:.2f} (acceptable).")
    else:
        signals.append("No bootstrap CI data available.")

    # --- sample size ---
    if n_studies < 3:
        signals.append(f"Only {n_studies} studies (very small evidence base).")
        if worst_downgrade < 2:
            worst_concern = GradeConcern.VERY_SERIOUS
            worst_downgrade = 2
    elif n_studies < 5:
        signals.append(f"Only {n_studies} studies (small evidence base).")
        if worst_downgrade < 1:
            worst_concern = GradeConcern.SERIOUS
            worst_downgrade = 1
    else:
        signals.append(f"{n_studies} studies.")

    if component_id:
        signals.append(f"Component: {component_id}.")
    if ci_width is not None:
        signals.append("Bootstrap CI is corpus-global; component-specific CI not available.")

    return DomainRating(
        domain="imprecision",
        concern=worst_concern,
        rationale=" ".join(signals),
        downgrade_levels=worst_downgrade,
        data_points={
            "ci_width": round(ci_width, 3) if ci_width is not None else None,
            "n_studies": n_studies,
            "component_id": component_id,
        },
    )


def assess_publication_bias(
    n_studies: int,
    bias_test_results: Optional[dict] = None,
) -> DomainRating:
    """Assess publication bias using Egger's test, trim-and-fill, and fail-safe N.

    When actual bias test results are available (from process.meta_analysis),
    uses Egger's p-value as the primary signal, with trim-and-fill adjustment
    magnitude and fail-safe N as supporting evidence.  Falls back to study-count
    heuristic when no formal tests have been run.

    Accepts both legacy key ``egger_p_value`` and the new ``eggers_p`` key
    produced by ``meta_analysis.run_bias_tests``.
    """
    if bias_test_results and isinstance(bias_test_results, dict):
        if bias_test_results.get("skipped"):
            pass  # fall through to heuristic
        else:
            # Accept both key conventions
            p_value = bias_test_results.get("eggers_p",
                        bias_test_results.get("egger_p_value"))
            trimfill_n = bias_test_results.get("trimfill_n_imputed", 0)
            failsafe = bias_test_results.get("failsafe_n", 0)
            trimfill_adj = bias_test_results.get("trimfill_adjusted_effect")
            trimfill_orig = bias_test_results.get("trimfill_original_effect")

            if p_value is not None:
                data_points: dict = {
                    "egger_p": p_value,
                    "n_studies": n_studies,
                    "tested": True,
                    "trimfill_n_imputed": trimfill_n,
                    "failsafe_n": failsafe,
                }

                # Build detailed rationale
                parts: list[str] = []

                if p_value < 0.05:
                    concern = GradeConcern.VERY_SERIOUS
                    downgrade = 2
                    parts.append(
                        f"Egger's test significant (p={p_value:.3f}), "
                        f"suggesting publication bias."
                    )
                elif p_value < 0.10:
                    concern = GradeConcern.SERIOUS
                    downgrade = 1
                    parts.append(f"Egger's test borderline (p={p_value:.3f}).")
                else:
                    concern = GradeConcern.NO_CONCERN
                    downgrade = 0
                    parts.append(f"Egger's test non-significant (p={p_value:.3f}).")

                # Trim-and-fill supporting evidence
                if trimfill_n > 0 and trimfill_adj is not None and trimfill_orig is not None:
                    shift = abs(trimfill_adj - trimfill_orig)
                    parts.append(
                        f"Trim-and-fill imputed {trimfill_n} studies "
                        f"(adjusted effect shift: {shift:.3f})."
                    )
                    if shift > 0.2 and concern == GradeConcern.NO_CONCERN:
                        concern = GradeConcern.SERIOUS
                        downgrade = 1

                # Fail-safe N supporting evidence
                if failsafe > 0:
                    threshold = 5 * n_studies + 10
                    if failsafe < threshold:
                        parts.append(
                            f"Fail-safe N = {failsafe} (below {threshold} threshold)."
                        )
                        if concern == GradeConcern.NO_CONCERN:
                            concern = GradeConcern.SERIOUS
                            downgrade = 1
                    else:
                        parts.append(
                            f"Fail-safe N = {failsafe} (above {threshold} threshold)."
                        )

                return DomainRating(
                    domain="publication_bias",
                    concern=concern,
                    rationale=" ".join(parts),
                    downgrade_levels=downgrade,
                    data_points=data_points,
                )

    # Heuristic fallback — no formal test available
    if n_studies < 10:
        return DomainRating(
            domain="publication_bias",
            concern=GradeConcern.SERIOUS,
            rationale=(
                f"Only {n_studies} studies — too few to assess publication bias. "
                f"Serious concern assumed by default."
            ),
            downgrade_levels=1,
            data_points={"n_studies": n_studies, "tested": False},
        )

    return DomainRating(
        domain="publication_bias",
        concern=GradeConcern.SERIOUS,
        rationale=(
            f"{n_studies} studies available but no formal bias test performed. "
            f"Publication bias untested; serious concern assumed."
        ),
        downgrade_levels=1,
        data_points={"n_studies": n_studies, "tested": False},
    )


# ---------------------------------------------------------------------------
# Overall certainty
# ---------------------------------------------------------------------------

def compute_overall_certainty(
    starting: str,
    domains: list[DomainRating],
) -> GradeCertainty:
    """Compute overall GRADE certainty.

    Start at HIGH (for RCTs/SRs) or LOW (for observational).
    Downgrade by the sum of ``downgrade_levels`` across all 5 domains.
    """
    start_idx = 0 if starting == "High" else 2  # HIGH=0, LOW=2
    total_downgrade = sum(d.downgrade_levels for d in domains)
    final_idx = min(start_idx + total_downgrade, len(_CERTAINTY_LADDER) - 1)
    return _CERTAINTY_LADDER[final_idx]


# ---------------------------------------------------------------------------
# Component-level GRADE
# ---------------------------------------------------------------------------

def grade_component(
    component: dict,
    claims: list[dict],
    method_profiles: list[dict],
    interrater_data: list[dict],
    robustness_data: dict,
    component_studies: list[str],
    bias_test_results: Optional[dict] = None,
) -> ComponentGrade:
    """Run full GRADE assessment for one hypothesis component."""
    n_studies = len(component_studies)

    # Determine starting certainty from study designs
    high_design_count = 0
    for mp in method_profiles:
        if mp.get("article_id") in set(component_studies):
            smt = mp.get("study_method_type", "unknown").lower()
            if smt in _HIGH_QUALITY_DESIGNS:
                high_design_count += 1

    starting = "High" if n_studies > 0 and high_design_count / max(n_studies, 1) > 0.5 else "Low"

    # Assess all 5 domains
    rob = assess_risk_of_bias(method_profiles, component_studies)
    incon = assess_inconsistency(interrater_data, robustness_data, component_studies)
    indir = assess_indirectness(claims, component)
    imprec = assess_imprecision(robustness_data, n_studies, component_id=component.get("id", ""))
    pub_bias = assess_publication_bias(n_studies, bias_test_results)

    domains = [rob, incon, indir, imprec, pub_bias]
    overall = compute_overall_certainty(starting, domains)

    # Summary text
    concern_parts = []
    for d in domains:
        if d.concern != GradeConcern.NO_CONCERN:
            level = "serious" if d.concern == GradeConcern.SERIOUS else "very serious"
            concern_parts.append(f"{d.domain.replace('_', ' ')} ({level})")

    if concern_parts:
        summary = (
            f"{overall.value} certainty ({starting} start). "
            f"Downgraded for: {', '.join(concern_parts)}."
        )
    else:
        summary = f"{overall.value} certainty ({starting} start). No concerns identified."

    return ComponentGrade(
        component_id=component.get("id", ""),
        component_label=component.get("label", ""),
        n_studies=n_studies,
        starting_certainty=starting,
        domains=domains,
        overall_certainty=overall,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Pipeline-level GRADE
# ---------------------------------------------------------------------------

def grade_all_components(config) -> list[ComponentGrade]:
    """Load all pipeline data and run GRADE for every hypothesis component.

    Reads from pipeline output files (same pattern as ``collect_prisma_data``).
    Returns a list of :class:`ComponentGrade` objects.
    """
    grades: list[ComponentGrade] = []

    # --- load pipeline outputs ------------------------------------------------
    session = _load_json(config.output_dir / "analysis_session.json")
    if not isinstance(session, dict):
        session = {}

    # Hypothesis components
    components_raw = session.get("hypothesis_components", [])
    if not components_raw:
        logger.warning("No hypothesis components found in session data.")
        return grades

    # Method profiles
    method_profiles_raw: list[dict] = []
    mp_dict = session.get("method_profiles", {})
    if isinstance(mp_dict, dict):
        for aid, mp in mp_dict.items():
            if isinstance(mp, dict):
                mp.setdefault("article_id", aid)
                method_profiles_raw.append(mp)

    # Interrater results
    interrater_raw = _load_json(config.clusters_dir / "interrater_results.json")
    if not isinstance(interrater_raw, list):
        # Fall back to legacy/session storage formats
        fallback = session.get("interrater_results", {})
        interrater_raw = fallback if isinstance(fallback, list) else []

    # Robustness result
    robustness_raw = session.get("robustness_result")
    if not isinstance(robustness_raw, dict):
        robustness_raw = {}

    # Claims
    claims_raw = _load_json(config.claims_dir / "claims_filtered.json")
    if not isinstance(claims_raw, list):
        # Fall back to claim_data in session
        cd = session.get("claim_data", {})
        if isinstance(cd, dict):
            claims_raw = list(cd.values()) if cd else []
        else:
            claims_raw = []

    # Evidence links — used to map articles to components
    evidence_links = session.get("evidence_links", [])

    # --- build component→article mapping ---
    comp_article_map: dict[str, set[str]] = {}
    for link in evidence_links:
        if isinstance(link, dict):
            cid = link.get("component_id", "")
            aid = link.get("article_id", "")
            if cid and aid:
                comp_article_map.setdefault(cid, set()).add(aid)

    # If no evidence links, try to infer from claims + keyword matching
    if not comp_article_map and claims_raw:
        for comp in components_raw:
            cid = comp.get("id", "")
            keywords = [kw.lower() for kw in comp.get("keywords", [])]
            comp_label = comp.get("label", "").lower()
            matched_articles: set[str] = set()
            for claim in claims_raw:
                text = " ".join([
                    " ".join(claim.get("findings", [])),
                    " ".join(claim.get("conditions", [])),
                ]).lower()
                if comp_label and comp_label in text:
                    matched_articles.add(claim.get("article_id", ""))
                elif any(kw in text for kw in keywords):
                    matched_articles.add(claim.get("article_id", ""))
            matched_articles.discard("")
            if matched_articles:
                comp_article_map[cid] = matched_articles

    # Build component→interrater record mapping for per-component GRADE
    interrater_records: list[dict] = interrater_raw if isinstance(interrater_raw, list) else []
    interrater_by_component: dict[str, list[dict]] = {
        cid: [] for cid in comp_article_map
    }
    for record in interrater_records:
        if not isinstance(record, dict):
            continue
        aid = record.get("article_id", "")
        if not aid:
            continue
        for cid, articles in comp_article_map.items():
            if aid in articles:
                interrater_by_component[cid].append(record)

    # --- grade each component ------------------------------------------------
    for comp in components_raw:
        cid = comp.get("id", "")
        component_studies = list(comp_article_map.get(cid, set()))

        # Filter claims to those from component's articles
        comp_claims = [
            c for c in claims_raw
            if c.get("article_id") in set(component_studies)
        ]

        grade = grade_component(
            component=comp,
            claims=comp_claims,
            method_profiles=method_profiles_raw,
            interrater_data=interrater_by_component.get(cid, []),
            robustness_data=robustness_raw,
            component_studies=component_studies,
        )
        grades.append(grade)

    logger.info("GRADE assessment completed for %d components.", len(grades))
    return grades


# ---------------------------------------------------------------------------
# Summary of Findings table
# ---------------------------------------------------------------------------

def _concern_label(concern: GradeConcern) -> str:
    if concern == GradeConcern.NO_CONCERN:
        return "No concern"
    if concern == GradeConcern.SERIOUS:
        return "Serious"
    return "Very serious"


def generate_sof_table(grades: list[ComponentGrade]) -> list[dict]:
    """Generate a GRADE Summary of Findings table.

    Returns a list of row dicts with columns: component, n_studies,
    risk_of_bias, inconsistency, indirectness, imprecision,
    publication_bias, overall_certainty.  Each domain cell contains
    the concern level and a brief rationale.
    """
    rows: list[dict] = []
    for g in grades:
        domain_map = {d.domain: d for d in g.domains}
        row: dict[str, Any] = {
            "component": g.component_label,
            "component_id": g.component_id,
            "n_studies": g.n_studies,
        }
        for domain_key in ["risk_of_bias", "inconsistency", "indirectness",
                           "imprecision", "publication_bias"]:
            dr = domain_map.get(domain_key)
            if dr:
                row[domain_key] = {
                    "concern": dr.concern.value,
                    "label": _concern_label(dr.concern),
                    "rationale": dr.rationale,
                    "downgrade": dr.downgrade_levels,
                }
            else:
                row[domain_key] = {
                    "concern": "no_concern",
                    "label": "Not assessed",
                    "rationale": "",
                    "downgrade": 0,
                }
        row["overall_certainty"] = g.overall_certainty.value
        row["summary"] = g.summary
        rows.append(row)
    return rows

