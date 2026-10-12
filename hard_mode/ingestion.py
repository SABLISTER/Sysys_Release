"""Bounded per (query_family x provider) retrieval with provenance."""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Callable

from acquire.elicit import ElicitClient
from acquire.openalex import OpenAlexClient
from acquire.orchestrator import (
    _normalize_elicit_paper,
    _normalize_oa_paper,
    _normalize_pm_paper,
    _normalize_scopus_paper,
    _normalize_s2_paper,
    _normalize_wos_paper,
)
from acquire.pubmed import PubMedClient
from acquire.scopus import ScopusClient
from acquire.semantic_scholar import SemanticScholarClient
from acquire.web_of_science import WebOfScienceClient
from core.config import Config
from hard_mode.paths import ensure_run_dir
from hard_mode.records import append_hit, ensure_spine

logger = logging.getLogger(__name__)


def _build_s2_query(terms: list[str]) -> str:
    escaped = [f'"{t}"' if " " in t else t for t in terms if t.strip()]
    return " | ".join(escaped) if escaped else ""


def _build_pubmed_query(terms: list[str]) -> str:
    parts = [f'"{t}"' if " " in t else t for t in terms if t.strip()]
    if not parts:
        return ""
    joined = " OR ".join(parts)
    return f"({joined})[Title/Abstract]"


def _build_openalex_query(terms: list[str]) -> str:
    return " ".join(t.strip() for t in terms if t.strip())


def _build_wos_query(terms: list[str]) -> str:
    parts = [f'"{t}"' if " " in t else t for t in terms if t.strip()]
    if not parts:
        return ""
    return f"TS=({' OR '.join(parts)})"


def _build_scopus_query(terms: list[str]) -> str:
    parts = [f'"{t}"' if " " in t else t for t in terms if t.strip()]
    if not parts:
        return ""
    return f"TITLE-ABS-KEY({' OR '.join(parts)})"


def _family_limit(family: dict[str, Any], hm: dict[str, Any]) -> int:
    return int(
        family.get("per_source_limit")
        or (hm.get("defaults") or {}).get("per_source_limit")
        or 50,
    )


def _looks_structured_pubmed(text: str) -> bool:
    u = f" {text.upper()} "
    return (
        "[TITLE/ABSTRACT]" in u
        or " AND " in u
        or " OR " in u
        or "(" in text
    )


def _looks_structured_wos(text: str) -> bool:
    u = f" {text.upper()} "
    return (
        "TS=" in u
        or "TI=" in u
        or "AU=" in u
        or "DO=" in u
        or " AND " in u
        or " OR " in u
        or "(" in text
    )


def _looks_structured_scopus(text: str) -> bool:
    u = f" {text.upper()} "
    return (
        "TITLE-ABS-KEY(" in u
        or "TITLE(" in u
        or "ABS(" in u
        or "KEY(" in u
        or "AUTH(" in u
        or "DOI(" in u
        or " AND " in u
        or " OR " in u
        or "(" in text
    )


async def run_family_retrieval(
    config: Config,
    hm: dict[str, Any],
    run_id: str,
    cancel_event: threading.Event | None = None,
    progress_callback: Callable[[dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    """For each query family and each listed provider, fetch up to per_source_limit papers."""
    run_dir = ensure_run_dir(config, run_id)
    all_papers: list[dict[str, Any]] = []
    nlq = (hm.get("natural_language_question") or "").strip()

    families = hm.get("query_families") or []
    for fi, family in enumerate(families):
        if cancel_event is not None and cancel_event.is_set():
            break
        fid = str(family.get("id", f"family_{fi}"))
        label = str(family.get("label", fid))
        providers = family.get("providers") or ["s2", "openalex", "pubmed", "wos", "scopus"]
        limit = _family_limit(family, hm)
        sem = [str(t) for t in (family.get("semantic_terms") or []) if str(t).strip()]
        pubmed_override = family.get("pubmed_query") or family.get("boolean_pubmed")
        wos_override = family.get("wos_query")
        scopus_override = family.get("scopus_query")
        use_elicit = bool(family.get("use_elicit")) and bool(config.elicit_api_key) and bool(nlq)

        def _emit(**kw: Any) -> None:
            if progress_callback:
                progress_callback({"family_id": fid, **kw})

        if use_elicit:
            try:
                _emit(stage="elicit", message=f"{label}: Elicit")
                client = ElicitClient(api_key=config.elicit_api_key)
                try:
                    papers = await client.search(nlq, max_results=limit)
                    for rank, raw in enumerate(papers):
                        if cancel_event is not None and cancel_event.is_set():
                            break
                        p = ensure_spine(_normalize_elicit_paper(raw))
                        append_hit(
                            p,
                            provider="elicit",
                            query_family_id=fid,
                            query_text=nlq[:500],
                            rank=rank,
                        )
                        all_papers.append(p)
                finally:
                    await client.close()
            except Exception as exc:
                logger.error("Hard-mode elicit retrieval failed for %s: %s", label, exc)

        if "s2" in providers:
            try:
                q = _build_s2_query(sem)
                if q:
                    _emit(stage="s2", message=f"{label}: Semantic Scholar")
                    s2 = SemanticScholarClient(api_key=config.s2_api_key)
                    try:
                        papers = await s2.search(q, max_results=limit)
                        for rank, raw in enumerate(papers):
                            if cancel_event is not None and cancel_event.is_set():
                                break
                            p = ensure_spine(_normalize_s2_paper(raw))
                            append_hit(
                                p,
                                provider="s2",
                                query_family_id=fid,
                                query_text=q[:500],
                                rank=rank,
                            )
                            all_papers.append(p)
                    finally:
                        await s2.close()
                    await asyncio.sleep(0.2)
            except Exception as exc:
                logger.error("Hard-mode s2 retrieval failed for %s: %s", label, exc)

        if "openalex" in providers:
            try:
                q = _build_openalex_query(sem)
                if q:
                    _emit(stage="openalex", message=f"{label}: OpenAlex")
                    oa = OpenAlexClient(api_key=config.openalex_api_key)
                    try:
                        papers = await oa.search(q, max_results=limit)
                        for rank, raw in enumerate(papers):
                            if cancel_event is not None and cancel_event.is_set():
                                break
                            p = ensure_spine(_normalize_oa_paper(raw))
                            append_hit(
                                p,
                                provider="openalex",
                                query_family_id=fid,
                                query_text=q[:500],
                                rank=rank,
                            )
                            all_papers.append(p)
                    finally:
                        await oa.close()
                    await asyncio.sleep(0.2)
            except Exception as exc:
                logger.error("Hard-mode openalex retrieval failed for %s: %s", label, exc)

        if "pubmed" in providers:
            if pubmed_override and str(pubmed_override).strip():
                pq = str(pubmed_override).strip()
            else:
                pq = _build_pubmed_query(sem)
            if pq and not _looks_structured_pubmed(pq):
                pq = _build_pubmed_query(sem) or pq
            if pq:
                try:
                    _emit(stage="pubmed", message=f"{label}: PubMed")
                    pm = PubMedClient(
                        api_key=config.pubmed_api_key,
                        email=config.pubmed_email,
                        tool=config.pubmed_tool,
                    )
                    try:
                        papers = await pm.search(pq, max_results=min(limit, 1000))
                        for rank, raw in enumerate(papers):
                            if cancel_event is not None and cancel_event.is_set():
                                break
                            p = ensure_spine(_normalize_pm_paper(raw))
                            append_hit(
                                p,
                                provider="pubmed",
                                query_family_id=fid,
                                query_text=pq[:500],
                                rank=rank,
                            )
                            all_papers.append(p)
                    finally:
                        await pm.close()
                    await asyncio.sleep(0.2)
                except Exception as exc:
                    logger.error("Hard-mode pubmed retrieval failed for %s: %s", label, exc)

        if "wos" in providers:
            if not config.wos_api_key:
                logger.info("Skipping Web of Science for %s — WOS_API_KEY not configured", label)
            else:
                if wos_override and str(wos_override).strip():
                    wq = str(wos_override).strip()
                else:
                    wq = _build_wos_query(sem)
                if wq and not _looks_structured_wos(wq):
                    wq = _build_wos_query(sem) or wq
                if wq:
                    try:
                        _emit(stage="wos", message=f"{label}: Web of Science")
                        wos = WebOfScienceClient(api_key=config.wos_api_key)
                        try:
                            papers = await wos.search(wq, max_results=limit)
                            for rank, raw in enumerate(papers):
                                if cancel_event is not None and cancel_event.is_set():
                                    break
                                p = ensure_spine(_normalize_wos_paper(raw))
                                append_hit(
                                    p,
                                    provider="wos",
                                    query_family_id=fid,
                                    query_text=wq[:500],
                                    rank=rank,
                                )
                                all_papers.append(p)
                        finally:
                            await wos.close()
                        await asyncio.sleep(0.2)
                    except Exception as exc:
                        logger.error("Hard-mode wos retrieval failed for %s: %s", label, exc)

        if "scopus" in providers:
            if not config.scopus_api_key:
                logger.info("Skipping Scopus for %s — SCOPUS_API_KEY not configured", label)
            else:
                if scopus_override and str(scopus_override).strip():
                    sq = str(scopus_override).strip()
                else:
                    sq = _build_scopus_query(sem)
                if sq and not _looks_structured_scopus(sq):
                    sq = _build_scopus_query(sem) or sq
                if sq:
                    try:
                        _emit(stage="scopus", message=f"{label}: Scopus")
                        scopus = ScopusClient(
                            api_key=config.scopus_api_key,
                            insttoken=config.scopus_insttoken,
                        )
                        try:
                            papers = await scopus.search(sq, max_results=limit)
                            for rank, raw in enumerate(papers):
                                if cancel_event is not None and cancel_event.is_set():
                                    break
                                p = ensure_spine(_normalize_scopus_paper(raw))
                                append_hit(
                                    p,
                                    provider="scopus",
                                    query_family_id=fid,
                                    query_text=sq[:500],
                                    rank=rank,
                                )
                                all_papers.append(p)
                        finally:
                            await scopus.close()
                        await asyncio.sleep(0.2)
                    except Exception as exc:
                        logger.error("Hard-mode scopus retrieval failed for %s: %s", label, exc)

    raw_path = run_dir / "candidates_raw.json"
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(all_papers, f, indent=2, ensure_ascii=False, default=str)
    logger.info("Hard mode raw candidates: %d -> %s", len(all_papers), raw_path)
    return all_papers
