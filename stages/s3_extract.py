"""Stage 3: Structured claim extraction.

Reads relevant.json from Stage 2, extracts structured claims from each paper
using LLM + heuristic fallback, outputs claims.json.
"""
import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional

if TYPE_CHECKING:
    from core.config import Config

logger = logging.getLogger(__name__)

DEFAULT_INPUT = Path("data/claims/relevant.json")
DEFAULT_OUTPUT = Path("data/claims/claims.json")

from core.claims import claim_list
from process.domain_patterns import (
    contains_term, extraction_patterns, research_config as _research_config,
)


def _clean_terms(values: list[Any]) -> list[str]:
    terms: list[str] = []
    seen: set[str] = set()
    for value in values:
        term = " ".join(str(value or "").split()).strip()
        if not term:
            continue
        key = term.lower()
        if key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def _search_terms_from_config(config: Any) -> list[str]:
    research = _research_config(config)
    search_raw = research.get("search")
    search = search_raw if isinstance(search_raw, dict) else {}
    values: list[Any] = []
    values.extend(search.get("base_terms") or [])
    for domain in search.get("domains") or []:
        if not isinstance(domain, dict):
            continue
        values.append(domain.get("name"))
        values.extend(domain.get("terms") or [])
        values.extend(domain.get("require_also") or [])
    values.extend(getattr(config, "co_occurring_conditions", []) or [])
    return _clean_terms(values)


# ---------------------------------------------------------------------------
# Heuristic extraction (pure regex/keyword, no LLM)
# ---------------------------------------------------------------------------

def _heuristic_extract(paper: dict, config: Any = None, *, patterns=None,
                       search_terms=None) -> dict:
    """Regex/keyword-based extraction as fallback when LLM is unavailable."""
    text = (paper.get("abstract", "") + " " + paper.get("title", "")).lower()
    original_text = paper.get("abstract", "") + " " + paper.get("title", "")
    patterns = extraction_patterns(config) if patterns is None else patterns
    condition_patterns = patterns["conditions"]
    configured_terms = (search_terms if search_terms is not None
                        else _search_terms_from_config(config))
    mechanism_terms = patterns["mechanisms"]

    conditions = [
        c for c, patterns in condition_patterns.items()
        if any(re.search(p, original_text, re.IGNORECASE) for p in patterns)
    ]
    mechanisms = [m for m in mechanism_terms if contains_term(text, m)]
    key_entities = [e for e in patterns["key_entities"] if contains_term(text, e)]
    methodology = next((m for m in patterns["methodology_hints"] if contains_term(text, m)), "")
    network_effects = any(contains_term(text, t) for t in patterns["network_terms"])

    # Extract sentences that look like findings
    sentences = re.split(r"[.!?]+", paper.get("abstract", ""))
    findings = [
        s.strip() for s in sentences
        if any(contains_term(s, h) for h in patterns["finding_hints"])
    ][:5]
    supports_primary = bool(
        findings
        or any(contains_term(text, term) for term in configured_terms[:40])
    )

    claims = {
        "findings": findings,
        "mechanisms": mechanisms,
        "conditions": conditions,
        "key_entities": key_entities,
        "methodology": methodology,
        "supports_primary_question": supports_primary,
        "network_effects_described": network_effects,
        "relationships": "",
        "key_quote": findings[0] if findings else "",
        "error": None,
    }
    # Opt-in legacy flag: only a pack or config that defines
    # criticality_terms produces it.
    if patterns["criticality_terms"]:
        claims["supports_criticality"] = any(
            contains_term(text, t) for t in patterns["criticality_terms"]
        )
    return claims


def _merge_heuristic_into_claims(claims: dict, heuristic: dict) -> dict:
    """Fill any empty fields in LLM claims with heuristic results."""
    merged = dict(claims)
    # Fill empty list fields
    for field in ("findings", "mechanisms", "conditions", "key_entities"):
        # claim_list also sees legacy names (brain_regions) from older schemas.
        if not claim_list(merged, field):
            merged[field] = heuristic.get(field, [])
    # Fill empty string fields
    for field in ("methodology", "relationships", "key_quote"):
        if not merged.get(field):
            merged[field] = heuristic.get(field, "")
    return merged


# ---------------------------------------------------------------------------
# Stage entry point
# ---------------------------------------------------------------------------

def run(
    config: "Optional[Config]" = None,
    cancel_event: Any = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    **kwargs,
) -> dict:
    """Run Stage 3 extraction.

    Strategy:
    1. Try LLM extraction via process.claim_extractor (if LLM is configured)
    2. Fall back to heuristic extraction for any papers that fail
    3. Always run heuristic as a supplement to fill in any empty fields

    Returns a summary dict with paper count and output path.
    """
    patterns = extraction_patterns(config)  # Validate before LLM calls or output writes.
    search_terms = _search_terms_from_config(config)

    # Resolve paths
    input_path = kwargs.get("input_path") or kwargs.get("relevant_path") or (
        config.claims_dir / "relevant.json" if config else DEFAULT_INPUT)
    output_path = kwargs.get("output_path") or (
        config.claims_dir / "claims.json" if config else DEFAULT_OUTPUT)
    input_path = Path(input_path)
    output_path = Path(output_path)

    logger.info("Stage 3 — Claim Extraction")
    logger.info("  Input:  %s", input_path)
    logger.info("  Output: %s", output_path)

    # Load relevant papers
    if not input_path.exists():
        msg = f"Input file not found: {input_path}"
        logger.error(msg)
        raise FileNotFoundError(msg)

    with open(input_path, encoding="utf-8") as f:
        papers = json.load(f)

    total = len(papers)
    logger.info("  Loaded %d relevant papers", total)
    if progress_callback:
        progress_callback(f"Loaded {total} relevant papers")

    if total == 0:
        logger.warning("No papers to extract — writing empty claims.json")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump([], f, indent=2)
        return {"total": 0, "extracted": 0, "output_path": str(output_path)}

    # Determine if LLM extraction is available
    use_llm = False
    model = ""
    max_tokens = kwargs.get("max_tokens", 4096)
    use_thinking = kwargs.get("use_thinking", False)

    if config and getattr(config, "llm_provider", ""):
        model = getattr(config, "llm_model", "")
        if model:
            use_llm = True
            logger.info("  LLM extraction: provider=%s, model=%s",
                        config.llm_provider, model)
        else:
            logger.info("  No LLM model configured — heuristic-only mode")
    else:
        logger.info("  No LLM config — heuristic-only mode")

    extracted_papers = []
    was_cancelled = False

    if use_llm:
        # LLM extraction with checkpoint/resume
        from process.claim_extractor import extract_all_claims
        from process.llm import set_config
        assert config is not None  # use_llm is only set True when config is truthy
        set_config(config)

        checkpoint_dir = config.claims_dir
        extracted_papers, was_cancelled = extract_all_claims(
            papers,
            model=model,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every=1,
            cancel_event=cancel_event,
            progress_callback=progress_callback,
            use_thinking=use_thinking,
            max_tokens=max_tokens,
            research_config=getattr(config, "research_config", None),
            config=config,
        )

        if was_cancelled:
            logger.info("LLM extraction was cancelled")
            return {"total": total, "extracted": len(extracted_papers),
                    "cancelled": True, "output_path": str(output_path)}

        # For papers where LLM failed, fall back to heuristic
        for paper in extracted_papers:
            claims = paper.get("claims", {})
            if claims.get("error"):
                logger.info("[Extract] LLM failed for '%s' — using heuristic",
                            paper.get("title", "?")[:60])
                heuristic = _heuristic_extract(paper, config, patterns=patterns,
                                               search_terms=search_terms)
                paper["claims"] = heuristic
                paper["claims"]["extraction_method"] = "heuristic_fallback"
            else:
                # Supplement LLM results with heuristic to fill gaps
                heuristic = _heuristic_extract(paper, config, patterns=patterns,
                                               search_terms=search_terms)
                paper["claims"] = _merge_heuristic_into_claims(claims, heuristic)
                paper["claims"]["extraction_method"] = "llm+heuristic"
    else:
        # Pure heuristic extraction (no LLM)
        for i, paper in enumerate(papers):
            if cancel_event is not None:
                is_set = (cancel_event.is_set()
                          if hasattr(cancel_event, "is_set")
                          else bool(cancel_event))
                if is_set:
                    was_cancelled = True
                    logger.info("Heuristic extraction cancelled at %d/%d",
                                i + 1, total)
                    break

            title = paper.get("title", "?")[:80]
            progress_msg = f"[{i + 1}/{total}] Extracting (heuristic): {title}"
            logger.info(progress_msg)
            if progress_callback:
                progress_callback(progress_msg)

            claims = _heuristic_extract(paper, config, patterns=patterns,
                                        search_terms=search_terms)
            claims["extraction_method"] = "heuristic_only"
            extracted_paper = {**paper, "claims": claims}
            extracted_papers.append(extracted_paper)

            n_f = len(claims.get("findings", []))
            n_m = len(claims.get("mechanisms", []))
            n_e = len(claim_list(claims, "key_entities"))
            result_msg = (f"[{i + 1}/{total}] {title} → "
                          f"{n_f} findings, {n_m} mechanisms, {n_e} entities")
            logger.info(result_msg)
            if progress_callback:
                progress_callback(result_msg)

        if was_cancelled:
            return {"total": total, "extracted": len(extracted_papers),
                    "cancelled": True, "output_path": str(output_path)}

    # Save claims.json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(extracted_papers, f, indent=2, default=str)
    logger.info("Saved %d extracted papers to %s", len(extracted_papers),
                output_path)

    # Summary
    n_with_findings = sum(
        1 for p in extracted_papers
        if p.get("claims", {}).get("findings"))
    n_with_mechanisms = sum(
        1 for p in extracted_papers
        if p.get("claims", {}).get("mechanisms"))
    n_with_entities = sum(
        1 for p in extracted_papers if claim_list(p, "key_entities"))

    summary = {
        "total": total,
        "extracted": len(extracted_papers),
        "with_findings": n_with_findings,
        "with_mechanisms": n_with_mechanisms,
        "with_key_entities": n_with_entities,
        "output_path": str(output_path),
    }
    logger.info("Extraction summary: %s", summary)
    if progress_callback:
        progress_callback(
            f"Extraction complete: {len(extracted_papers)} papers, "
            f"{n_with_findings} with findings, "
            f"{n_with_mechanisms} with mechanisms")

    return summary
