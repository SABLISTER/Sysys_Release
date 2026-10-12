"""Load hard_mode_config.yaml into a plain dict with defaults."""

from __future__ import annotations

import copy
import re
from pathlib import Path
from typing import Any

import yaml

from process.queries import get_natural_language_query

DEFAULT_PATH_NAME = "hard_mode_config.yaml"


def _domain_name_to_id(name: str, index: int) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", name.strip().lower()).strip("_")
    return slug or f"domain_{index}"


def domains_to_query_families(
    search: dict[str, Any],
    hm_defaults: dict[str, Any] | None,
    *,
    providers: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Build hard-mode query_families from a research_config-style `search` block."""
    base_terms = [str(t) for t in (search.get("base_terms") or []) if str(t).strip()]
    domains = search.get("domains") or []
    per_source = int((hm_defaults or {}).get("per_source_limit", 50))
    prov = providers or ["s2", "openalex", "pubmed", "wos", "scopus"]
    out: list[dict[str, Any]] = []
    for i, dom in enumerate(domains):
        if not isinstance(dom, dict):
            continue
        name = str(dom.get("name") or f"domain_{i}")
        fid = _domain_name_to_id(name, i)
        terms = [str(t) for t in (dom.get("terms") or []) if str(t).strip()]
        require = [str(t) for t in (dom.get("require_also") or []) if str(t).strip()]
        sem = list(dict.fromkeys([*base_terms, *terms, *require]))
        out.append(
            {
                "id": fid,
                "label": name,
                "semantic_terms": sem,
                "pubmed_query": None,
                "boolean_pubmed": None,
                "wos_query": None,
                "scopus_query": None,
                "per_source_limit": per_source,
                "providers": list(prov),
                "use_elicit": False,
            }
        )
    return out


def _fill_missing_nested(dst: dict[str, Any], src: dict[str, Any]) -> None:
    """Fill keys missing in dst from src (research_config overlay)."""
    for k, v in src.items():
        if k not in dst or dst[k] is None:
            dst[k] = copy.deepcopy(v)
        elif isinstance(dst[k], dict) and isinstance(v, dict):
            _fill_missing_nested(dst[k], v)
        elif isinstance(dst[k], list) and isinstance(v, list) and len(dst[k]) == 0 and len(v) > 0:
            dst[k] = copy.deepcopy(v)


def _normalize_hard_mode_config(merged: dict[str, Any]) -> None:
    """Align with research_config: NLQ, optional query_families from search.domains."""
    search = merged.get("search")
    if not isinstance(search, dict):
        search = {}
        merged["search"] = search
    relevance = merged.get("relevance")
    if not isinstance(relevance, dict):
        relevance = {}
        merged["relevance"] = relevance

    nl = (merged.get("natural_language_question") or "").strip()
    if not nl:
        merged["natural_language_question"] = get_natural_language_query(merged)

    derive = bool(merged.get("derive_query_families_from_search"))
    families = merged.get("query_families")
    has_families = isinstance(families, list) and len(families) > 0
    domains = search.get("domains") if isinstance(search.get("domains"), list) else []

    if derive and domains:
        merged["query_families"] = domains_to_query_families(search, merged.get("defaults"))
    elif (not has_families) and domains:
        merged["query_families"] = domains_to_query_families(search, merged.get("defaults"))


def merge_research_config_overlay(hm: dict[str, Any], research: dict[str, Any] | None) -> dict[str, Any]:
    """Fill missing name/description/search/relevance (and paths) from a research_config dict."""
    if research:
        for key in ("name", "description", "paths"):
            if key in research and research[key] is not None:
                if key not in hm or hm[key] is None:
                    hm[key] = copy.deepcopy(research[key])
        if isinstance(research.get("search"), dict):
            if "search" not in hm or not isinstance(hm.get("search"), dict):
                hm["search"] = copy.deepcopy(research["search"])
            else:
                _fill_missing_nested(hm["search"], research["search"])
        if isinstance(research.get("relevance"), dict):
            if "relevance" not in hm or not isinstance(hm.get("relevance"), dict):
                hm["relevance"] = copy.deepcopy(research["relevance"])
            else:
                _fill_missing_nested(hm["relevance"], research["relevance"])
    _normalize_hard_mode_config(hm)
    return hm


# Canonical per-section defaults (single source of truth). The big defaults
# dict in load_hard_mode_config() references these, and hard_mode/triage.py
# (_cheap_triage_cfg) / hard_mode/fulltext.py (fulltext_cfg) build on them
# instead of re-declaring their own setdefault trees.
CHEAP_TRIAGE_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "use_sentence_transformers": True,
    "embedding_model": None,
    "semantic_weight": 0.45,
    "tfidf_weight": 0.35,
    "token_overlap_weight": 0.15,
    "retrieval_hit_weight": 0.05,
    "high_threshold": 0.5,
    "medium_threshold": 0.28,
    "pass3_include_bands": ["high", "medium"],
    "pass3_low_band_min": 5,
    "pass3_low_band_max": 25,
    "pass3_low_band_fraction": 0.1,
    "pass3_random_seed": 13,
    "compare_embedding_model": None,
    "compare_subset_size": 100,
    "compare_random_seed": 13,
}

FULL_TEXT_DEFAULTS: dict[str, Any] = {
    "enabled": True,
    "prefer_for_cheap_triage": True,
    "prefer_for_pass3": True,
    "keep_only_full_text": False,
    "triage_max_chars": 20000,
    "score_max_chars": None,
}


def load_hard_mode_config(
    path: Path | None = None,
    project_root: Path | None = None,
    research: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Load YAML; merge shallow defaults for missing keys.

    When ``research`` is set (e.g. ``research_config.yaml``), overlapping keys
    (``name``, ``description``, ``search``, ``relevance``, ``paths``) fill gaps
    in the hard-mode file so both configs stay in sync.

    Same top-level keys as ``research_config.yaml`` are supported: ``search``,
    ``relevance``. If ``query_families`` is empty and ``search.domains`` is set,
    families are derived automatically.
    """
    root = project_root or Path(__file__).resolve().parent.parent
    cfg_path = path or (root / DEFAULT_PATH_NAME)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Hard mode config not found: {cfg_path}")
    with open(cfg_path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    defaults = {
        "name": "hard-mode",
        "description": "",
        "natural_language_question": "",
        "search": {"base_terms": [], "domains": []},
        "relevance": {"threshold": 3, "research_question": "", "rubric": {}},
        "defaults": {"per_source_limit": 50},
        "cheap_triage": dict(CHEAP_TRIAGE_DEFAULTS),
        "full_text": dict(FULL_TEXT_DEFAULTS),
        "query_families": [],
        "axes": [],
        "overlap_rules": {
            "axis_score_threshold": 0.35,
            "axis_ids_for_overlap": ["axis_a", "axis_b", "axis_c", "axis_d"],
        },
        "tier_rules": {
            "tier1_min_axes_strong": 3,
            "axis_strong_threshold": 0.55,
            "tier1_min_integration": 0.5,
            "tier1_min_confidence": 0.45,
            "tier2_min_confidence": 0.3,
        },
        "evidence_types": ["empirical", "review", "theory", "computational"],
        "stopping": {
            "max_retrieval_rounds": 12,
            "min_new_reviewed_relevant_per_round": 1,
        },
        "seeds": {
            "min_human_axes_for_bridge_seed": 3,
            "require_reviewer_relevant": True,
        },
        "dive": {"citation_relation_limit": 20, "max_seed_papers": 15},
        "graph": {"top_keyword_suggestions": 12, "min_reviewed_for_graph": 5},
        "llm_scoring": {
            "abstract_max_chars": None,
            "title_max_chars": None,
            "max_output_tokens": None,
            "use_reasoning": False,
        },
    }
    merged = copy.deepcopy(defaults)
    merged.update({k: v for k, v in raw.items() if v is not None})
    if "defaults" in raw and isinstance(raw["defaults"], dict):
        merged["defaults"] = {**defaults["defaults"], **raw["defaults"]}
    for key in (
        "search",
        "relevance",
        "cheap_triage",
        "full_text",
        "overlap_rules",
        "tier_rules",
        "stopping",
        "seeds",
        "dive",
        "graph",
        "llm_scoring",
    ):
        if key in raw and isinstance(raw[key], dict):
            merged[key] = {
                **defaults[key],
                **{k: v for k, v in raw[key].items() if v is not None},
            }
    merge_research_config_overlay(merged, research)
    return merged
