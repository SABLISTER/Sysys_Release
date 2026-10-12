"""Citation expansion from reviewed seed papers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
from typing import Any

from acquire.semantic_scholar import SemanticScholarClient
from core.config import Config
from hard_mode.paths import ensure_run_dir
from hard_mode.records import append_hit, ensure_spine

logger = logging.getLogger(__name__)


def _normalize_doi(value: str | None) -> str:
    doi = str(value or "").strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi


def _normalize_pmid(value: str | None) -> str:
    return str(value or "").strip()


def _normalize_title(value: str | None) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _normalize_s2_paper(paper: dict[str, Any]) -> dict[str, Any]:
    ext_ids = paper.get("externalIds") or {}
    open_access_pdf = paper.get("openAccessPdf") or {}
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract"),
        "year": paper.get("year"),
        "doi": ext_ids.get("DOI") or paper.get("doi"),
        "pmid": ext_ids.get("PubMed"),
        "s2_id": paper.get("paperId"),
        "citation_count": paper.get("citationCount"),
        "tldr": (paper.get("tldr") or {}).get("text"),
        "authors": [a.get("name", "") for a in (paper.get("authors") or [])],
        "fields_of_study": paper.get("fieldsOfStudy"),
        "s2_open_access_pdf_url": open_access_pdf.get("url") or "",
        "source": "s2",
    }


def _candidate_matches_seed(seed: dict[str, Any], candidate: dict[str, Any]) -> bool:
    ext_ids = candidate.get("externalIds") or {}
    seed_doi = _normalize_doi(seed.get("doi"))
    cand_doi = _normalize_doi(ext_ids.get("DOI") or candidate.get("doi"))
    if seed_doi and cand_doi and seed_doi == cand_doi:
        return True

    seed_pmid = _normalize_pmid(seed.get("pmid"))
    cand_pmid = _normalize_pmid(ext_ids.get("PubMed") or candidate.get("pmid"))
    if seed_pmid and cand_pmid and seed_pmid == cand_pmid:
        return True

    seed_title = _normalize_title(seed.get("title"))
    cand_title = _normalize_title(candidate.get("title"))
    seed_year = str(seed.get("year") or "").strip()
    cand_year = str(candidate.get("year") or "").strip()
    if seed_title and cand_title and seed_title == cand_title:
        return (not seed_year) or (not cand_year) or (seed_year == cand_year)
    return False


async def _resolve_seed_s2_id(s2: SemanticScholarClient, seed: dict[str, Any]) -> str:
    existing = str(seed.get("s2_id") or "").strip()
    if existing:
        return existing

    doi = _normalize_doi(seed.get("doi"))
    pmid = _normalize_pmid(seed.get("pmid"))
    title = str(seed.get("title") or "").strip()

    for alias in (f"DOI:{doi}" if doi else "", f"PMID:{pmid}" if pmid else ""):
        if not alias:
            continue
        record = await s2.get_paper(alias)
        if record and record.get("paperId"):
            seed["s2_id"] = record["paperId"]
            return str(record["paperId"])

    for query in (doi, pmid, title):
        if not query:
            continue
        try:
            candidates = await s2.search(str(query), max_results=5)
        except Exception as exc:
            logger.warning("Seed resolution search failed for %s: %s", seed.get("title", "")[:80], exc)
            continue
        for candidate in candidates:
            if _candidate_matches_seed(seed, candidate) and candidate.get("paperId"):
                seed["s2_id"] = candidate["paperId"]
                return str(candidate["paperId"])

    return ""


def select_seeds(
    papers: list[dict[str, Any]],
    hm: dict[str, Any],
) -> list[dict[str, Any]]:
    seeds_cfg = hm.get("seeds") or {}
    require_rel = bool(seeds_cfg.get("require_reviewer_relevant", True))
    max_seeds = int((hm.get("dive") or {}).get("max_seed_papers", 15))
    out = []
    for p in papers:
        h = p.get("human") or {}
        if require_rel and h.get("reviewer_label") != "relevant":
            continue
        if not h.get("citation_seed_flag"):
            continue
        out.append(p)
    out.sort(
        key=lambda x: int(x.get("citation_count") or 0),
        reverse=True,
    )
    return out[:max_seeds]


async def run_citation_dive(
    config: Config,
    hm: dict[str, Any],
    run_id: str,
    seeds: list[dict[str, Any]],
    cancel_event: threading.Event | None = None,
) -> list[dict[str, Any]]:
    relation_limit = int((hm.get("dive") or {}).get("citation_relation_limit", 20))
    expanded: list[dict[str, Any]] = []
    if not seeds:
        return expanded

    s2 = SemanticScholarClient(api_key=config.s2_api_key)
    try:
        for idx, seed in enumerate(seeds, start=1):
            if cancel_event is not None and cancel_event.is_set():
                break
            paper_id = await _resolve_seed_s2_id(s2, seed)
            if not paper_id:
                logger.warning(
                    "Citation dive skipped unresolved seed: %s (doi=%s pmid=%s)",
                    seed.get("title", "")[:120],
                    seed.get("doi") or "",
                    seed.get("pmid") or "",
                )
                continue
            suid = str(seed.get("hard_mode_uid", ""))
            try:
                citations, references = await asyncio.gather(
                    s2.get_citations(paper_id, limit=relation_limit),
                    s2.get_references(paper_id, limit=relation_limit),
                )
            except Exception as e:
                logger.warning("Citation dive failed for seed %s: %s", paper_id, e)
                continue

            for rank, raw in enumerate(citations):
                if cancel_event is not None and cancel_event.is_set():
                    break
                p = ensure_spine(_normalize_s2_paper(raw))
                append_hit(
                    p,
                    provider="s2",
                    query_family_id="dive_citation_forward",
                    query_text=f"citations_of:{paper_id}",
                    rank=rank,
                )
                p.setdefault("dive", {})
                p["dive"] = {
                    "kind": "citation_forward",
                    "parent_seed_uid": suid,
                    "parent_s2_id": paper_id,
                }
                expanded.append(p)

            for rank, raw in enumerate(references):
                if cancel_event is not None and cancel_event.is_set():
                    break
                p = ensure_spine(_normalize_s2_paper(raw))
                append_hit(
                    p,
                    provider="s2",
                    query_family_id="dive_citation_backward",
                    query_text=f"references_of:{paper_id}",
                    rank=rank,
                )
                p.setdefault("dive", {})
                p["dive"] = {
                    "kind": "citation_backward",
                    "parent_seed_uid": suid,
                    "parent_s2_id": paper_id,
                }
                expanded.append(p)

            await asyncio.sleep(s2._delay)
            logger.info("Citation dive %d/%d done for %s", idx, len(seeds), paper_id)
    finally:
        await s2.close()

    run_dir = ensure_run_dir(config, run_id)
    path = run_dir / "candidates_dive.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(expanded, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Citation dive wrote %d papers to %s", len(expanded), path)
    return expanded
