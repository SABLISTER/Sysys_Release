"""
NeuroQuery API bridge for cross-referencing hypothesis components
with predicted brain activation patterns.

Queries the NeuroQuery API, computes convergence scores between predicted
and observed brain regions, and generates component-level reports.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from core.claims import claim_list

logger = logging.getLogger(__name__)

# ── Region normalisation helpers ──────────────────────────────────────

_REGION_ALIASES: dict[str, str] = {
    "dlpfc": "dorsolateral prefrontal cortex",
    "vmpfc": "ventromedial prefrontal cortex",
    "ofc": "orbitofrontal cortex",
    "acc": "anterior cingulate cortex",
    "pcc": "posterior cingulate cortex",
    "sma": "supplementary motor area",
    "stg": "superior temporal gyrus",
    "ifg": "inferior frontal gyrus",
    "mfg": "middle frontal gyrus",
    "sfg": "superior frontal gyrus",
    "ipl": "inferior parietal lobule",
    "spl": "superior parietal lobule",
    "mtg": "middle temporal gyrus",
    "itg": "inferior temporal gyrus",
    "tpj": "temporoparietal junction",
    "pfc": "prefrontal cortex",
    "vlpfc": "ventrolateral prefrontal cortex",
}


def _normalise_region(name: str) -> str:
    """Lowercase, strip whitespace, expand common abbreviations."""
    name = name.strip().lower()
    name = re.sub(r'\s+', ' ', name)
    return _REGION_ALIASES.get(name, name)


def _regions_match(a: str, b: str) -> bool:
    """Fuzzy match two region names (substring containment after normalisation)."""
    a_norm = _normalise_region(a)
    b_norm = _normalise_region(b)
    if a_norm == b_norm:
        return True
    if a_norm in b_norm or b_norm in a_norm:
        return True
    return False


# ── NeuroQuery API ────────────────────────────────────────────────────

def query_neuroquery(terms: str) -> dict:
    """
    Query the NeuroQuery API.

    POST to https://neuroquery.org/api/query with {"text": terms}
    Returns parsed JSON response or empty dict on failure.
    Timeout: 5 seconds. Graceful fallback on any error.
    """
    try:
        import requests
    except ImportError:
        logger.warning("requests library not available; cannot query NeuroQuery")
        return {}

    url = "https://neuroquery.org/api/query"
    try:
        resp = requests.post(
            url,
            json={"text": terms},
            timeout=5,
            headers={"Accept": "application/json"},
        )
        resp.raise_for_status()
        data = resp.json()
        return data
    except requests.exceptions.Timeout:
        logger.warning("NeuroQuery API timed out for query: %s", terms[:80])
        return {}
    except requests.exceptions.ConnectionError:
        logger.warning("NeuroQuery API unreachable for query: %s", terms[:80])
        return {}
    except Exception as exc:
        logger.warning("NeuroQuery API error for query '%s': %s", terms[:80], exc)
        return {}


# ── Convergence scoring ───────────────────────────────────────────────

def compute_convergence_score(
    predicted_regions: list[dict],
    observed_regions: list[str],
) -> dict:
    """
    Compare predicted activation regions (from NeuroQuery) with observed
    mentions (from extracted claims).

    Parameters
    ----------
    predicted_regions : list[dict]
        From NeuroQuery: [{"region": "amygdala", "weight": 0.8}, ...]
    observed_regions : list[str]
        From claims: ["amygdala", "prefrontal cortex", ...]

    Returns
    -------
    dict with convergence_score, matched/unmatched region lists, and counts.
    """
    if not predicted_regions:
        return {
            "convergence_score": 0.0,
            "observed_not_predicted": list(observed_regions),
            "predicted_not_observed": [],
            "matched_regions": [],
            "total_predicted": 0,
            "total_observed": len(observed_regions),
        }

    matched: list[str] = []
    predicted_not_observed: list[str] = []

    for pred in predicted_regions:
        pred_name = pred.get("region", "")
        if not pred_name:
            continue
        found = False
        for obs in observed_regions:
            if _regions_match(pred_name, obs):
                matched.append(pred_name)
                found = True
                break
        if not found:
            predicted_not_observed.append(pred_name)

    # Observed regions not in any prediction
    observed_not_predicted: list[str] = []
    for obs in observed_regions:
        found = False
        for pred in predicted_regions:
            if _regions_match(obs, pred.get("region", "")):
                found = True
                break
        if not found:
            observed_not_predicted.append(obs)

    total_predicted = len(predicted_regions)
    score = len(matched) / total_predicted if total_predicted > 0 else 0.0

    return {
        "convergence_score": round(score, 3),
        "observed_not_predicted": observed_not_predicted,
        "predicted_not_observed": predicted_not_observed,
        "matched_regions": matched,
        "total_predicted": total_predicted,
        "total_observed": len(observed_regions),
    }


# ── Component-level convergence report ────────────────────────────────

def generate_convergence_report(
    components: list[dict],
    claims: list[dict],
    config: Any = None,
) -> dict:
    """
    For each hypothesis component:
    1. Build a query string from component keywords
    2. Query NeuroQuery
    3. Collect all brain_regions mentioned in claims for papers linked
       to this component
    4. Compute convergence score

    Parameters
    ----------
    components : list[dict]
        Hypothesis components with 'id', 'label', 'keywords' fields.
    claims : list[dict]
        Extracted claims with 'key_entities' (or legacy 'brain_regions') and optionally
        'component_id' fields.
    config : optional
        If provided and has output_dir, saves results to
        config.output_dir / "neuroquery_convergence.json".

    Returns
    -------
    dict mapping component_id -> convergence results.
    """
    report: dict[str, Any] = {}

    for comp in components:
        comp_id = comp.get("id", comp.get("label", "unknown"))
        label = comp.get("label", comp_id)
        keywords = comp.get("keywords", [])
        if isinstance(keywords, list):
            query_str = " ".join(keywords) if keywords else label
        else:
            query_str = str(keywords) if keywords else label

        # Query NeuroQuery
        nq_result = query_neuroquery(query_str)
        predicted_regions = nq_result.get("brain_regions", [])

        # Collect observed brain regions from claims linked to this component
        observed: list[str] = []
        for claim in claims:
            claim_comp = claim.get("component_id", "")
            brain_regions = claim_list(claim, "key_entities")
            if not brain_regions:
                continue
            # Include if claim matches this component or no component filter
            if not claim_comp or str(claim_comp) == str(comp_id):
                observed.extend(brain_regions)

        # De-duplicate observed
        observed = list(dict.fromkeys(observed))

        convergence = compute_convergence_score(predicted_regions, observed)
        convergence["component_id"] = comp_id
        convergence["component_label"] = label
        convergence["query"] = query_str
        convergence["neuroquery_available"] = bool(nq_result)

        report[str(comp_id)] = convergence

    # Save to disk if config provided
    if config is not None:
        output_dir = getattr(config, "output_dir", None)
        if output_dir:
            out_path = Path(output_dir) / "neuroquery_convergence.json"
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            logger.info("Saved convergence report to %s", out_path)

    return report
