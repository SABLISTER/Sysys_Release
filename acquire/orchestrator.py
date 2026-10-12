"""Acquisition orchestrator — coordinates all provider searches.

Calls async provider clients, normalizes results to a common schema,
deduplicates, optionally runs citation snowballing, and saves raw JSON.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from core.config import Config
from core.io import atomic_write_json
from acquire.deduplicator import deduplicate_papers
from acquire.elicit import ElicitClient
from acquire.openalex import OpenAlexClient, invert_abstract
from acquire.pubmed import PubMedClient
from acquire.scopus import ScopusClient
from acquire.semantic_scholar import SemanticScholarClient
from acquire.web_of_science import WebOfScienceClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _normalize_external_id(value: str | None, prefixes: tuple[str, ...] = ()) -> str | None:
    if not value:
        return None
    text = str(value).strip()
    for prefix in prefixes:
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text.startswith("https://") or text.startswith("http://"):
        text = text.rstrip("/").split("/")[-1]
    return text or None


def _normalize_pmcid(value: str | None) -> str | None:
    pmcid = _normalize_external_id(
        value,
        prefixes=(
            "https://www.ncbi.nlm.nih.gov/pmc/articles/",
            "http://www.ncbi.nlm.nih.gov/pmc/articles/",
            "https://pmc.ncbi.nlm.nih.gov/articles/",
            "http://pmc.ncbi.nlm.nih.gov/articles/",
        ),
    )
    if not pmcid:
        return None
    if pmcid.isdigit():
        pmcid = f"PMC{pmcid}"
    return pmcid.upper()


def _normalize_pmid(value: str | None) -> str | None:
    return _normalize_external_id(
        value,
        prefixes=(
            "https://pubmed.ncbi.nlm.nih.gov/",
            "http://pubmed.ncbi.nlm.nih.gov/",
        ),
    )


def _location_url(location: dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return (
        location.get("pdf_url")
        or location.get("landing_page_url")
        or ""
    )


def _location_pdf_url(location: dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return location.get("pdf_url") or ""


def _location_host_type(location: dict | None) -> str:
    if not isinstance(location, dict):
        return ""
    return ((location.get("source") or {}).get("type") or "").strip()


def _first_nonempty(*values):
    """Return the first non-empty / non-None value (rejects [], {}, '')."""
    for value in values:
        if value not in (None, "", [], {}):
            return value
    return None


def _parse_year(value) -> int | None:
    """Coerce *value* to an integer year, handling date strings like '2024-01-15'."""
    if value in (None, ""):
        return None
    text = str(value).strip()
    if len(text) >= 4 and text[:4].isdigit():
        return int(text[:4])
    return int(text) if text.isdigit() else None


def _as_int(value) -> int | None:
    """Coerce a citation-count-like value to int, or None if it is not numeric."""
    if value is None or value == "":
        return None
    text = str(value).strip()
    return int(text) if text.isdigit() else None


def _iter_identifier_candidates(value) -> list[dict]:
    """Coerce varied identifier structures into a list of {type, value} dicts."""
    if isinstance(value, dict):
        if {"type", "value"} <= set(value):
            return [value]
        out = []
        for key, item in value.items():
            if isinstance(item, dict):
                candidate = dict(item)
                candidate.setdefault("type", key)
                out.append(candidate)
            else:
                out.append({"type": key, "value": item})
        return out
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    return []


def _identifier_value(value, *names: str) -> str | None:
    """Extract an identifier by type name from varied identifier structures."""
    wanted = {name.lower() for name in names}
    for candidate in _iter_identifier_candidates(value):
        ctype = str(
            candidate.get("type")
            or candidate.get("name")
            or candidate.get("@type")
            or ""
        ).lower()
        if ctype in wanted:
            return _first_nonempty(
                candidate.get("value"),
                candidate.get("id"),
                candidate.get("identifier"),
                candidate.get("content"),
            )
    if isinstance(value, dict):
        for name in names:
            direct = value.get(name) or value.get(name.lower()) or value.get(name.upper())
            if direct:
                return direct
    return None


def _extract_links(value) -> list[str]:
    """Extract URLs from a list of link objects or plain strings."""
    out = []
    items = value if isinstance(value, list) else [value]
    for item in items:
        if not isinstance(item, dict):
            continue
        href = item.get("url") or item.get("href") or item.get("@href")
        if href:
            out.append(href)
    return out


def _extract_wos_authors(value) -> list[str]:
    """Extract author names from various Web of Science response structures."""
    if isinstance(value, dict):
        value = value.get("authors") or value.get("author") or value.get("names") or []
    if not isinstance(value, list):
        return []
    authors = []
    for item in value:
        if not isinstance(item, dict):
            continue
        name = _first_nonempty(
            item.get("displayName"),
            item.get("fullName"),
            item.get("wosStandard"),
            item.get("name"),
        )
        if name:
            authors.append(str(name))
    return authors


def _extract_scopus_authors(value, creator: str | None = None) -> list[str]:
    """Extract author names from Scopus response structures."""
    if isinstance(value, dict):
        value = value.get("author") or []
    authors = []
    if isinstance(value, list):
        for item in value:
            if not isinstance(item, dict):
                continue
            name = _first_nonempty(
                item.get("authname"),
                item.get("ce:indexed-name"),
                item.get("preferred-name", {}).get("ce:indexed-name")
                if isinstance(item.get("preferred-name"), dict) else None,
            )
            if name:
                authors.append(str(name))
    if not authors and creator:
        authors.append(creator)
    return authors


# ---------------------------------------------------------------------------
# Normalizers — one per provider
# ---------------------------------------------------------------------------
def _normalize_elicit_paper(paper: dict) -> dict:
    return {
        "title": paper.get("title", ""),
        "abstract": paper.get("abstract"),
        "year": paper.get("year"),
        "doi": paper.get("doi"),
        "pmid": paper.get("pmid"),
        "elicit_id": paper.get("elicitId"),
        "citation_count": paper.get("citedByCount"),
        "tldr": None,
        "authors": paper.get("authors") or [],
        "fields_of_study": None,
        "venue": paper.get("venue"),
        "urls": paper.get("urls") or [],
        "source": "elicit",
    }


def _normalize_s2_paper(paper: dict) -> dict:
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


def _normalize_oa_paper(paper: dict) -> dict:
    doi = paper.get("doi")
    if doi and doi.startswith("https://doi.org/"):
        doi = doi[len("https://doi.org/"):]
    authors = []
    for authorship in (paper.get("authorships") or []):
        name = authorship.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)
    ids = paper.get("ids") or {}
    open_access = paper.get("open_access") or {}
    best_oa_location = paper.get("best_oa_location") or {}
    primary_location = paper.get("primary_location") or {}
    has_content = paper.get("has_content") or {}
    abstract = paper.get("abstract")
    if not abstract:
        abstract = invert_abstract(paper.get("abstract_inverted_index"))
    return {
        "title": paper.get("title", ""),
        "abstract": abstract,
        "year": paper.get("publication_year"),
        "doi": doi,
        "pmid": _normalize_pmid(ids.get("pmid")),
        "pmc": _normalize_pmcid(ids.get("pmcid")),
        "oa_id": paper.get("id"),
        "citation_count": paper.get("cited_by_count"),
        "tldr": None,
        "authors": authors,
        "fields_of_study": None,
        "openalex_is_oa": bool(open_access.get("is_oa")),
        "openalex_oa_status": open_access.get("oa_status") or "",
        "openalex_oa_url": open_access.get("oa_url") or "",
        "openalex_any_repository_has_fulltext": bool(
            open_access.get("any_repository_has_fulltext")
        ),
        "openalex_best_oa_url": _location_url(best_oa_location),
        "openalex_best_oa_pdf_url": _location_pdf_url(best_oa_location),
        "openalex_best_oa_host_type": _location_host_type(best_oa_location),
        "openalex_primary_url": _location_url(primary_location),
        "openalex_primary_pdf_url": _location_pdf_url(primary_location),
        "openalex_primary_host_type": _location_host_type(primary_location),
        "openalex_has_pdf_content": bool(has_content.get("pdf")),
        "openalex_has_grobid_xml": bool(has_content.get("grobid_xml")),
        "openalex_content_url": paper.get("content_url") or "",
        "source": "openalex",
    }


def _normalize_pm_paper(paper: dict) -> dict:
    """PubMed papers are already mostly normalized by PubMedClient."""
    paper.setdefault("source", "pubmed")
    paper.setdefault("citation_count", None)
    paper.setdefault("tldr", None)
    paper.setdefault("fields_of_study", None)
    paper.setdefault("s2_id", None)
    paper.setdefault("oa_id", None)
    return paper


def _normalize_wos_paper(paper: dict) -> dict:
    source_raw = paper.get("source")
    source = source_raw if isinstance(source_raw, dict) else {}
    identifiers = paper.get("identifiers") or {}
    links = paper.get("links") or []
    citations_raw = paper.get("citations")
    citation_count = _first_nonempty(
        paper.get("timesCited"),
        paper.get("citationsCount"),
        citations_raw.get("count") if isinstance(citations_raw, dict) else None,
    )
    names_raw = paper.get("names")
    names = names_raw if isinstance(names_raw, dict) else {}
    return {
        "title": _first_nonempty(
            paper.get("title"),
            paper.get("documentTitle"),
            names.get("title"),
        ) or "",
        "abstract": _first_nonempty(
            paper.get("abstract"),
            paper.get("abstractText"),
            paper.get("summary"),
        ),
        "year": _parse_year(
            _first_nonempty(
                source.get("publishYear"),
                source.get("publishedBiblioYear"),
                paper.get("publicationYear"),
                paper.get("year"),
            )
        ),
        "doi": _identifier_value(identifiers, "doi", "DOI"),
        "pmid": _normalize_pmid(_identifier_value(identifiers, "pmid", "pubmed", "PMID")),
        "wos_id": _first_nonempty(paper.get("uid"), paper.get("id")),
        "citation_count": _as_int(citation_count),
        "tldr": None,
        "authors": _extract_wos_authors(paper.get("names")),
        "fields_of_study": None,
        "venue": _first_nonempty(source.get("sourceTitle"), source.get("title")),
        "urls": _extract_links(links),
        "source": "wos",
    }


def _normalize_scopus_paper(paper: dict) -> dict:
    cover_date = _first_nonempty(
        paper.get("prism:coverDate"),
        paper.get("prism:coverDisplayDate"),
        paper.get("coverDate"),
    )
    citation_count = paper.get("citedby-count")
    creator = _first_nonempty(paper.get("dc:creator"), paper.get("creator"))
    return {
        "title": _first_nonempty(paper.get("dc:title"), paper.get("title")) or "",
        "abstract": _first_nonempty(
            paper.get("dc:description"),
            paper.get("description"),
            paper.get("subtypeDescription"),
        ),
        "year": _parse_year(cover_date),
        "doi": _first_nonempty(paper.get("prism:doi"), paper.get("doi")),
        "pmid": _normalize_pmid(_first_nonempty(paper.get("pubmed-id"), paper.get("pubmedId"))),
        "scopus_id": _first_nonempty(paper.get("eid"), paper.get("identifier")),
        "citation_count": _as_int(citation_count),
        "tldr": None,
        "authors": _extract_scopus_authors(paper.get("author"), creator=str(creator or "")),
        "fields_of_study": None,
        "venue": _first_nonempty(
            paper.get("prism:publicationName"),
            paper.get("publicationName"),
        ),
        "urls": _extract_links(paper.get("link")),
        "scopus_openaccess": str(paper.get("openaccess") or ""),
        "source": "scopus",
    }


# ---------------------------------------------------------------------------
# Save raw results (atomic write)
# ---------------------------------------------------------------------------
def _save_raw(path: Path, data: list[dict]) -> None:
    """Write JSON atomically and log the record count."""
    atomic_write_json(path, data)
    logger.info("Saved %d records to %s", len(data), path)


# ---------------------------------------------------------------------------
# Citation snowballing
# ---------------------------------------------------------------------------
def _select_snowball_seeds(corpus: list[dict], seed_limit: int) -> list[dict]:
    """Pick the top-cited papers with an S2 ID as snowball seeds."""
    seeds = [
        paper for paper in corpus
        if paper.get("s2_id") and paper.get("abstract") and paper.get("title")
    ]
    seeds.sort(
        key=lambda paper: (
            int(paper.get("citation_count") or 0),
            int(paper.get("year") or 0),
            len(paper.get("abstract") or ""),
        ),
        reverse=True,
    )
    return seeds[:max(0, int(seed_limit))]


async def _run_citation_snowball(config: Config, seed_papers: list[dict]) -> list[dict]:
    """Fetch citations + references for seed papers via S2 (parallel per-seed)."""
    if not seed_papers:
        return []

    relation_limit = max(1, int(config.citation_snowball_relation_limit))
    logger.info(
        "S2 snowball: expanding around %d seeds with %d citations and %d references each",
        len(seed_papers),
        relation_limit,
        relation_limit,
    )
    s2 = SemanticScholarClient(api_key=config.s2_api_key)
    expanded: list[dict] = []
    try:
        for index, seed in enumerate(seed_papers, start=1):
            paper_id = seed.get("s2_id")
            if not paper_id:
                continue
            try:
                citations, references = await asyncio.gather(
                    s2.get_citations(paper_id, limit=relation_limit),
                    s2.get_references(paper_id, limit=relation_limit),
                )
            except Exception as e:
                logger.warning(
                    "S2 snowball failed for %s (%d/%d): %s",
                    paper_id,
                    index,
                    len(seed_papers),
                    e,
                )
                continue

            expanded.extend(_normalize_s2_paper(paper) for paper in citations)
            expanded.extend(_normalize_s2_paper(paper) for paper in references)
            logger.info(
                "S2 snowball %d/%d: %s -> %d citations, %d references",
                index,
                len(seed_papers),
                paper_id,
                len(citations),
                len(references),
            )
            await asyncio.sleep(s2._delay)
    finally:
        await s2.close()

    return expanded


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------
async def run_acquisition(
    config: Config,
    search_config: dict | None = None,
    natural_language_query: str = "",
) -> list[dict]:
    """Run the full multi-provider acquisition pipeline.

    1. Build queries from search_config (or fall back to natural_language_query)
    2. Search all configured providers
    3. Normalize results to a common schema
    4. Optionally run citation snowballing
    5. Deduplicate
    6. Save corpus.json
    """
    from process.queries import (
        get_s2_queries,
        get_pubmed_queries,
        get_wos_queries,
        get_scopus_queries,
        get_openalex_filters,
    )

    config.ensure_dirs()
    all_papers: list[dict] = []
    # Provider-level failures. A provider that was attempted but errored
    # records itself here; if EVERY attempted provider fails, acquisition
    # raises instead of returning a silently empty corpus that would be
    # indistinguishable from a legitimate zero-result search.
    provider_errors: dict[str, str] = {}

    async def _acquire_elicit() -> list[dict]:
        elicit_query = " ".join((natural_language_query or "").split())
        elicit_papers: list[dict] = []
        if not config.elicit_api_key:
            logger.info("Skipping Elicit — ELICIT_API_KEY not configured")
        elif not elicit_query:
            logger.info("Skipping Elicit — no natural-language query available")
        else:
            elicit = ElicitClient(api_key=config.elicit_api_key)
            try:
                logger.info("Elicit query: %s...", elicit_query[:80])
                try:
                    papers = await elicit.search(elicit_query, max_results=500)
                    elicit_papers = [_normalize_elicit_paper(p) for p in papers]
                except Exception as exc:
                    logger.error("Elicit search failed: %s", exc)
                    provider_errors["elicit"] = str(exc)
            finally:
                await elicit.close()
            _save_raw(config.raw_dir / "elicit_raw.json", elicit_papers)
        return elicit_papers

    async def _acquire_s2() -> list[dict]:
        s2 = SemanticScholarClient(api_key=config.s2_api_key)
        s2_papers: list[dict] = []
        n_queries = n_failed = 0
        try:
            for query in get_s2_queries(search_config):
                n_queries += 1
                logger.info("S2 query: %s...", query[:80])
                try:
                    papers = await s2.search(query, max_results=500)
                    s2_papers.extend(_normalize_s2_paper(p) for p in papers)
                except Exception as exc:
                    n_failed += 1
                    logger.error("S2 query failed: %s", exc)
        except Exception as exc:
            logger.error("Semantic Scholar acquisition failed: %s", exc)
            provider_errors["s2"] = str(exc)
        finally:
            await s2.close()
        if n_queries and n_failed == n_queries:
            provider_errors.setdefault("s2", f"all {n_queries} queries failed")
        _save_raw(config.raw_dir / "s2_raw.json", s2_papers)
        return s2_papers

    async def _acquire_openalex() -> list[dict]:
        oa = OpenAlexClient(api_key=config.openalex_api_key)
        oa_papers: list[dict] = []
        n_queries = n_failed = 0
        try:
            for query_config in get_openalex_filters(search_config):
                n_queries += 1
                logger.info("OpenAlex query: %s...", query_config["search"][:80])
                try:
                    papers = await oa.search(
                        query_config["search"],
                        extra_filter=query_config.get("filter", ""),
                        max_results=500,
                    )
                    oa_papers.extend(_normalize_oa_paper(p) for p in papers)
                except Exception as exc:
                    n_failed += 1
                    logger.error("OpenAlex query failed: %s", exc)
        except Exception as exc:
            logger.error("OpenAlex acquisition failed: %s", exc)
            provider_errors["openalex"] = str(exc)
        finally:
            await oa.close()
        if n_queries and n_failed == n_queries:
            provider_errors.setdefault("openalex", f"all {n_queries} queries failed")
        _save_raw(config.raw_dir / "openalex_raw.json", oa_papers)
        return oa_papers

    async def _acquire_pubmed() -> list[dict]:
        pm = PubMedClient(
            api_key=config.pubmed_api_key,
            email=config.pubmed_email,
            tool=config.pubmed_tool,
        )
        pm_papers: list[dict] = []
        n_queries = n_failed = 0
        try:
            for query in get_pubmed_queries(search_config):
                n_queries += 1
                logger.info("PubMed query: %s...", query[:80])
                try:
                    papers = await pm.search(query, max_results=1000)
                    pm_papers.extend(_normalize_pm_paper(p) for p in papers)
                except Exception as exc:
                    n_failed += 1
                    logger.error("PubMed query failed: %s", exc)
        except Exception as exc:
            logger.error("PubMed acquisition failed: %s", exc)
            provider_errors["pubmed"] = str(exc)
        finally:
            await pm.close()
        if n_queries and n_failed == n_queries:
            provider_errors.setdefault("pubmed", f"all {n_queries} queries failed")
        _save_raw(config.raw_dir / "pubmed_raw.json", pm_papers)
        return pm_papers

    async def _acquire_wos() -> list[dict]:
        wos_papers: list[dict] = []
        if not config.wos_api_key:
            logger.info("Skipping Web of Science — WOS_API_KEY not configured")
            return wos_papers
        wos = WebOfScienceClient(api_key=config.wos_api_key)
        n_queries = n_failed = 0
        try:
            for query in get_wos_queries(search_config):
                n_queries += 1
                logger.info("Web of Science query: %s...", query[:80])
                try:
                    papers = await wos.search(query, max_results=500)
                    wos_papers.extend(_normalize_wos_paper(p) for p in papers)
                except Exception as exc:
                    n_failed += 1
                    logger.error("Web of Science query failed: %s", exc)
        except Exception as exc:
            logger.error("Web of Science acquisition failed: %s", exc)
            provider_errors["wos"] = str(exc)
        finally:
            await wos.close()
        if n_queries and n_failed == n_queries:
            provider_errors.setdefault("wos", f"all {n_queries} queries failed")
        _save_raw(config.raw_dir / "wos_raw.json", wos_papers)
        return wos_papers

    async def _acquire_scopus() -> list[dict]:
        scopus_papers: list[dict] = []
        if not config.scopus_api_key:
            logger.info("Skipping Scopus — SCOPUS_API_KEY not configured")
            return scopus_papers
        scopus = ScopusClient(
            api_key=config.scopus_api_key,
            insttoken=config.scopus_insttoken,
        )
        n_queries = n_failed = 0
        try:
            for query in get_scopus_queries(search_config):
                n_queries += 1
                logger.info("Scopus query: %s...", query[:80])
                try:
                    papers = await scopus.search(query, max_results=500)
                    scopus_papers.extend(_normalize_scopus_paper(p) for p in papers)
                except Exception as exc:
                    n_failed += 1
                    logger.error("Scopus query failed: %s", exc)
        except Exception as exc:
            logger.error("Scopus acquisition failed: %s", exc)
            provider_errors["scopus"] = str(exc)
        finally:
            await scopus.close()
        if n_queries and n_failed == n_queries:
            provider_errors.setdefault("scopus", f"all {n_queries} queries failed")
        _save_raw(config.raw_dir / "scopus_raw.json", scopus_papers)
        return scopus_papers

    provider_batches = await asyncio.gather(
        _acquire_elicit(),
        _acquire_s2(),
        _acquire_openalex(),
        _acquire_pubmed(),
        _acquire_wos(),
        _acquire_scopus(),
    )

    # Distinguish "every provider failed" from "legitimate zero results".
    attempted = {
        "elicit": bool(config.elicit_api_key
                       and " ".join((natural_language_query or "").split())),
        "s2": True,
        "openalex": True,
        "pubmed": True,
        "wos": bool(config.wos_api_key),
        "scopus": bool(config.scopus_api_key),
    }
    n_attempted = sum(1 for was in attempted.values() if was)
    failed = [name for name, was in attempted.items()
              if was and name in provider_errors]
    if n_attempted and len(failed) == n_attempted:
        details = "; ".join(f"{name}: {provider_errors[name]}" for name in failed)
        raise RuntimeError(
            f"All {n_attempted} acquisition providers failed ({details}). "
            "Refusing to continue with an empty corpus: this is a provider "
            "outage, not a zero-result search."
        )

    for batch in provider_batches:
        all_papers.extend(batch)

    logger.info("Total papers before dedup: %d", len(all_papers))

    # Deduplicate
    corpus = deduplicate_papers(all_papers)

    # Filter out papers without abstracts
    corpus = [p for p in corpus if p.get("abstract")]
    logger.info("Corpus after dedup + abstract filter: %d", len(corpus))

    # Citation snowballing
    if config.citation_snowball_enabled:
        seed_limit = max(0, int(config.citation_snowball_seed_limit))
        seed_papers = _select_snowball_seeds(corpus, seed_limit)
        if seed_papers:
            logger.info(
                "S2 snowball: selected %d seed papers from the initial corpus",
                len(seed_papers),
            )
            snowball_papers = await _run_citation_snowball(config, seed_papers)
            if snowball_papers:
                _save_raw(config.raw_dir / "s2_snowball_raw.json", snowball_papers)
                all_papers.extend(snowball_papers)
                corpus = deduplicate_papers(all_papers)
                corpus = [p for p in corpus if p.get("abstract")]
                logger.info(
                    "Corpus after S2 snowball expansion: %d papers",
                    len(corpus),
                )
            else:
                logger.info("S2 snowball produced no additional papers")

    # Save corpus
    corpus_path = config.corpus_dir / "corpus.json"
    _save_raw(corpus_path, corpus)

    return corpus
