"""Build hypothesis components from the loaded research configuration."""

from __future__ import annotations

import re
from typing import Any

from core.models import HypothesisComponent


def _clean_text(value: Any) -> str:
    return " ".join(str(value or "").split()).strip()


def _unique_terms(values: list[Any], limit: int | None = None) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = _clean_text(value)
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
        if limit is not None and len(terms) >= limit:
            break
    return terms


def _component_id(label: str, fallback: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", label.strip().lower()).strip("_")
    return slug or fallback


def _search_config(config: Any) -> dict[str, Any]:
    try:
        search = config.get_search_config()
    except AttributeError:
        search = {}
    return search if isinstance(search, dict) else {}


def _research_config(config: Any) -> dict[str, Any]:
    research = getattr(config, "research_config", {})
    return research if isinstance(research, dict) else {}


def _research_question(config: Any, research: dict[str, Any]) -> str:
    relevance_raw = research.get("relevance")
    relevance = relevance_raw if isinstance(relevance_raw, dict) else {}
    for value in (
        research.get("natural_language_question"),
        relevance.get("research_question"),
        getattr(config, "hypothesis_text", ""),
        research.get("hypothesis"),
        research.get("question"),
        research.get("description"),
    ):
        text = _clean_text(value)
        if text:
            return text
    return "What does the scholarly literature report for this research question?"


def build_hypothesis_components_from_config(config: Any) -> list[HypothesisComponent]:
    """Create GUI/search components from current-schema config values.

    The component list feeds Search-tab keyword construction, so this helper
    uses the configured search terms first and only falls back to plain
    hypothesis text when structured search domains are absent.
    """
    research = _research_config(config)
    search = _search_config(config)
    base_terms = _unique_terms(search.get("base_terms") or [])
    domains_raw = search.get("domains")
    domains = domains_raw if isinstance(domains_raw, list) else []
    research_question = _research_question(config, research)

    components: list[HypothesisComponent] = []
    domain_terms: list[str] = []

    for index, domain in enumerate(domains):
        if not isinstance(domain, dict):
            continue
        label = _clean_text(domain.get("name")) or f"Domain {index + 1}"
        terms = _unique_terms([
            *(domain.get("terms") or []),
            *(domain.get("require_also") or []),
            *base_terms[:2],
        ])
        if not terms:
            continue
        domain_terms.extend(terms)
        components.append(
            HypothesisComponent(
                id=_component_id(label, f"domain_{index + 1}"),
                label=label,
                description=(
                    _clean_text(domain.get("description"))
                    or f"Search domain for the configured review question: {label}"
                ),
                keywords=terms,
                weight=1.0,
            )
        )

    core_keywords = _unique_terms([*base_terms, *domain_terms], limit=16)
    if research_question or core_keywords:
        components.insert(
            0,
            HypothesisComponent(
                id="research_question",
                label="Research Question",
                description=research_question,
                keywords=core_keywords,
                weight=1.0,
            ),
        )

    existing_labels = {component.label.lower() for component in components}
    for index, condition in enumerate(getattr(config, "co_occurring_conditions", []) or []):
        label = _clean_text(condition)
        if not label or label.lower() in existing_labels:
            continue
        components.append(
            HypothesisComponent(
                id=_component_id(label, f"context_{index + 1}"),
                label=label,
                description=f"{label} as a configured construct, population, or contextual factor",
                keywords=_unique_terms([label, *base_terms[:2]]),
                weight=1.0,
            )
        )

    return components
