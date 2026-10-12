"""Bridge between the GUI and the async search pipeline."""

import asyncio
import json
import logging
import threading
from enum import Enum
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class ProviderChoice(Enum):
    ALL = "All Providers"
    ELICIT = "Elicit"
    S2 = "Semantic Scholar"
    OPENALEX = "OpenAlex"
    PUBMED = "PubMed"
    WOS = "Web of Science"
    SCOPUS = "Scopus"


_PROVIDER_ORDER = (
    ProviderChoice.PUBMED,
    ProviderChoice.S2,
    ProviderChoice.OPENALEX,
    ProviderChoice.ELICIT,
    ProviderChoice.SCOPUS,
    ProviderChoice.WOS,
)


def normalize_provider_choices(provider) -> set[ProviderChoice]:
    """Return concrete search engines, expanding ALL to every provider."""
    if provider is None:
        raw = {ProviderChoice.ALL}
    elif isinstance(provider, (list, tuple, set)):
        raw = set(provider)
    else:
        raw = {provider}

    if not raw or ProviderChoice.ALL in raw:
        return set(_PROVIDER_ORDER)

    selected = {choice for choice in raw if choice in _PROVIDER_ORDER}
    return selected or set(_PROVIDER_ORDER)


def selected_search_provider_count(provider) -> int:
    """Count concrete search engines selected for a Stage 1 search."""
    return len(normalize_provider_choices(provider))


def provider_result_limits(max_results: int, provider) -> dict[ProviderChoice, int]:
    """Split a total result cap across selected search engines.

    The returned budgets sum to ``max_results``. Remainders follow the UI order
    so common default searches give the first selected engines the extra slots.
    """
    selected = normalize_provider_choices(provider)
    total_limit = max(1, int(max_results or 1))
    base, remainder = divmod(total_limit, len(selected))

    budgets: dict[ProviderChoice, int] = {}
    for choice in _PROVIDER_ORDER:
        if choice not in selected:
            continue
        budgets[choice] = base + (1 if remainder > 0 else 0)
        remainder = max(0, remainder - 1)
    return budgets


class AsyncBridge:
    """Runs an asyncio event loop on a background daemon thread.

    Allows the Qt main thread to submit coroutines without blocking.
    """

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="async-bridge",
        )
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def submit(self, coro):
        """Submit a coroutine; returns a concurrent.futures.Future."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def shutdown(self):
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)


# ── Query builders ──────────────────────────────────────────


def build_pubmed_query(terms: list[str]) -> str:
    """Build a PubMed query from user terms (OR'd, Title/Abstract field)."""
    parts = [f'"{t}"' if " " in t else t for t in terms]
    joined = " OR ".join(parts)
    return f"({joined})[Title/Abstract]"


def build_wos_query(terms: list[str]) -> str:
    """Build a Web of Science Starter topic query."""
    parts = [f'"{t}"' if " " in t else t for t in terms]
    return f"TS=({' OR '.join(parts)})"


def build_scopus_query(terms: list[str]) -> str:
    """Build a Scopus TITLE-ABS-KEY query."""
    parts = [f'"{t}"' if " " in t else t for t in terms]
    return f"TITLE-ABS-KEY({' OR '.join(parts)})"


def build_elicit_query(question: str, semantic_terms: list[str]) -> str:
    """Build an Elicit query, preferring the explicit natural-language question."""
    clean_question = " ".join((question or "").split())
    if clean_question:
        return clean_question
    return " ".join(t.strip() for t in semantic_terms if t.strip())


def _per_query_limit(max_results: int, query_count: int) -> int:
    """Spread a provider-level result cap across multiple query lines."""
    if query_count <= 0:
        return max(1, max_results)
    return max(1, (max_results + query_count - 1) // query_count)


# ── Single-query-per-provider builders ─────────────────────
#
# Each provider issues exactly ONE query. When an explicit Boolean query is
# supplied we keep its AND/OR structure and only wrap it in the provider's field
# code if it isn't already provider-specific. Otherwise we OR the term list into
# a single combined query. This avoids the previous one-query-per-term fan-out.


def _has_pubmed_field_tag(text: str) -> bool:
    upper = text.upper()
    return any(tag in upper for tag in (
        "[TITLE/ABSTRACT]", "[TIAB]", "[TW]", "[TITLE]", "[MESH", "[ALL FIELDS]",
    ))


def _has_wos_field_code(text: str) -> bool:
    upper = text.upper()
    return any(code in upper for code in (
        "TS=", "TI=", "AU=", "DO=", "AB=", "AK=", "ALL=",
    ))


def _has_scopus_field_code(text: str) -> bool:
    upper = text.upper()
    return any(code in upper for code in (
        "TITLE-ABS-KEY(", "TITLE(", "ABS(", "KEY(", "AUTH(", "DOI(", "ALL(",
    ))


def _build_pubmed_search(raw_boolean_query: str, terms: list[str]) -> str:
    """Return a single PubMed query, preferring an explicit Boolean query."""
    raw = (raw_boolean_query or "").strip()
    if raw:
        return raw if _has_pubmed_field_tag(raw) else f"({raw})[Title/Abstract]"
    return build_pubmed_query(terms)


def _build_wos_search(raw_boolean_query: str, terms: list[str]) -> str:
    """Return a single Web of Science topic query."""
    raw = (raw_boolean_query or "").strip()
    if raw:
        return raw if _has_wos_field_code(raw) else f"TS=({raw})"
    return build_wos_query(terms)


def _build_scopus_search(raw_boolean_query: str, terms: list[str]) -> str:
    """Return a single Scopus TITLE-ABS-KEY query."""
    raw = (raw_boolean_query or "").strip()
    if raw:
        return raw if _has_scopus_field_code(raw) else f"TITLE-ABS-KEY({raw})"
    return build_scopus_query(terms)


# ── Main search coroutine ──────────────────────────────────


async def run_gui_search(
    natural_language_question: str,
    semantic_terms: list[str],
    boolean_terms: list[str],
    provider: "ProviderChoice | list[ProviderChoice]",
    config,
    max_results: int = 500,
    min_results: int = 1,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[dict]:
    """Run a search across selected providers and return deduplicated corpus.

    *natural_language_question* feeds Elicit (preferred).
    *semantic_terms* feed Semantic Scholar and OpenAlex (natural-language).
    *boolean_terms* feed PubMed, Web of Science, and Scopus.
    *provider* can be a single ProviderChoice or a list of them.

    Results are accumulated additively: any existing corpus is loaded first,
    new papers are appended, and the combined set is deduplicated so repeated
    searches grow the corpus without duplicates.
    """
    # Lazy imports so the module loads fast and paths are already set up
    from acquire.elicit import ElicitClient
    from acquire.semantic_scholar import SemanticScholarClient
    from acquire.openalex import OpenAlexClient
    from acquire.pubmed import PubMedClient
    from acquire.scopus import ScopusClient
    from acquire.web_of_science import WebOfScienceClient
    from acquire.orchestrator import (
        _normalize_elicit_paper,
        _normalize_oa_paper,
        _normalize_pm_paper,
        _normalize_scopus_paper,
        _normalize_s2_paper,
        _normalize_wos_paper,
        _save_raw,
    )
    from acquire.deduplicator import deduplicate_papers

    # Normalize provider to concrete search engines and split the requested
    # total cap before each provider fans out across its query families.
    _provider_set = normalize_provider_choices(provider)
    provider_limits = provider_result_limits(max_results, _provider_set)
    logger.info(
        "Paper result cap: total=%d, providers=%d, per-provider=%s",
        max(1, int(max_results or 1)),
        len(_provider_set),
        {choice.value: provider_limits.get(choice, 0) for choice in _PROVIDER_ORDER
         if choice in _provider_set},
    )

    def _progress(msg: str) -> None:
        if progress_callback:
            progress_callback(msg)

    def _raise_if_cancelled() -> None:
        if cancel_event is not None and cancel_event.is_set():
            raise asyncio.CancelledError("Search cancelled by user")

    config.ensure_dirs()
    _raise_if_cancelled()
    _progress(
        "Paper limit: up to "
        f"{max(1, int(max_results or 1))} total across {len(_provider_set)} providers"
    )

    # ── Load existing corpus so searches are additive ────────
    output_path = config.corpus_dir / "corpus.json"
    existing_papers: list[dict] = []
    if output_path.exists():
        try:
            with open(output_path) as f:
                existing_papers = json.load(f)
            logger.info(
                f"Loaded existing corpus with {len(existing_papers)} papers from {output_path}"
            )
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Could not load existing corpus, starting fresh: {e}")
            existing_papers = []

    all_papers: list[dict] = []

    # Build ONE query per provider family so each engine issues a single API
    # query instead of one query per term. Fragmenting the question into many
    # per-term queries caused noisy recall and dozens of redundant API calls.
    elicit_query = build_elicit_query(natural_language_question, semantic_terms)

    # Free-text query for relevance-ranked engines (S2, OpenAlex, Elicit).
    semantic_query = " ".join((natural_language_question or "").split()).strip()
    if not semantic_query:
        semantic_query = " ".join(
            str(t).strip() for t in (semantic_terms or []) if str(t).strip()
        ).strip()

    # Inputs for the single combined Boolean query (PubMed, Scopus, WoS).
    boolean_source_terms = [
        str(t).strip() for t in (semantic_terms or []) if str(t).strip()
    ]
    raw_boolean_query = ""
    for candidate in (boolean_terms or []):
        if str(candidate).strip():
            raw_boolean_query = str(candidate).strip()
            break

    # Elicit (uses the natural-language research question)
    if ProviderChoice.ELICIT in _provider_set:
        _raise_if_cancelled()
        provider_limit = provider_limits.get(ProviderChoice.ELICIT, 0)
        if provider_limit <= 0:
            logger.info("Skipping Elicit — no share of paper limit")
        elif not config.elicit_api_key:
            logger.info("Skipping Elicit — ELICIT_API_KEY not configured")
        elif not elicit_query:
            logger.info("Skipping Elicit — no natural-language question provided")
        else:
            logger.info(f"Elicit query: {elicit_query[:80]}...")
            _progress("Searching Elicit...")
            elicit = ElicitClient(api_key=config.elicit_api_key)
            elicit_papers = []
            try:
                papers = await elicit.search(elicit_query, max_results=provider_limit)
                elicit_papers = [_normalize_elicit_paper(p) for p in papers][:provider_limit]
                all_papers.extend(elicit_papers)
                logger.info(f"Elicit returned {len(elicit_papers)} papers")
            except Exception as e:
                logger.error(f"Elicit error: {e}")
            finally:
                await elicit.close()

            _save_raw(config.raw_dir / "gui_elicit_raw.json", elicit_papers)

    # Semantic Scholar (single natural-language query)
    if ProviderChoice.S2 in _provider_set:
        _raise_if_cancelled()
        provider_limit = provider_limits.get(ProviderChoice.S2, 0)
        if provider_limit <= 0:
            logger.info("Skipping Semantic Scholar — no share of paper limit")
        elif not semantic_query:
            logger.info("Skipping Semantic Scholar — no semantic query provided")
        else:
            _progress("Searching Semantic Scholar...")
            s2 = SemanticScholarClient(api_key=config.s2_api_key)
            s2_papers = []
            try:
                logger.info(f"S2 query: {semantic_query[:80]}...")
                papers = await s2.search(semantic_query, max_results=provider_limit)
                s2_papers = [_normalize_s2_paper(p) for p in papers][:provider_limit]
                all_papers.extend(s2_papers)
                logger.info(f"Semantic Scholar returned {len(s2_papers)} papers")
            except Exception as e:
                logger.error(f"Semantic Scholar error: {e}")
            finally:
                await s2.close()

            _save_raw(config.raw_dir / "gui_s2_raw.json", s2_papers)

    # OpenAlex (single natural-language query)
    if ProviderChoice.OPENALEX in _provider_set:
        _raise_if_cancelled()
        provider_limit = provider_limits.get(ProviderChoice.OPENALEX, 0)
        if provider_limit <= 0:
            logger.info("Skipping OpenAlex — no share of paper limit")
        elif not semantic_query:
            logger.info("Skipping OpenAlex — no semantic query provided")
        else:
            _progress("Searching OpenAlex...")
            oa = OpenAlexClient(api_key=config.openalex_api_key)
            oa_papers = []
            try:
                logger.info(f"OpenAlex query: {semantic_query[:80]}...")
                papers = await oa.search(semantic_query, max_results=provider_limit)
                oa_papers = [_normalize_oa_paper(p) for p in papers][:provider_limit]
                all_papers.extend(oa_papers)
                logger.info(f"OpenAlex returned {len(oa_papers)} papers")
            except Exception as e:
                logger.error(f"OpenAlex error: {e}")
            finally:
                await oa.close()

            _save_raw(config.raw_dir / "gui_openalex_raw.json", oa_papers)

    # PubMed (single combined Boolean query)
    if ProviderChoice.PUBMED in _provider_set:
        _raise_if_cancelled()
        provider_limit = provider_limits.get(ProviderChoice.PUBMED, 0)
        if provider_limit <= 0:
            logger.info("Skipping PubMed — no share of paper limit")
        elif not (raw_boolean_query or boolean_source_terms):
            logger.info("Skipping PubMed — no query terms provided")
        else:
            _progress("Searching PubMed...")
            pm = PubMedClient(
                api_key=config.pubmed_api_key,
                email=config.pubmed_email,
                tool=config.pubmed_tool,
            )
            pm_papers = []
            try:
                query = _build_pubmed_search(raw_boolean_query, boolean_source_terms)
                logger.info(f"PubMed query: {query[:80]}...")
                papers = await pm.search(query, max_results=provider_limit)
                pm_papers = [_normalize_pm_paper(p) for p in papers][:provider_limit]
                all_papers.extend(pm_papers)
                logger.info(f"PubMed returned {len(pm_papers)} papers")
            except Exception as e:
                logger.error(f"PubMed error: {e}")
            finally:
                await pm.close()

            _save_raw(config.raw_dir / "gui_pubmed_raw.json", pm_papers)

    # Web of Science (single combined topic query)
    if ProviderChoice.WOS in _provider_set:
        _raise_if_cancelled()
        provider_limit = provider_limits.get(ProviderChoice.WOS, 0)
        if provider_limit <= 0:
            logger.info("Skipping Web of Science — no share of paper limit")
        elif not config.wos_api_key:
            logger.info("Skipping Web of Science — WOS_API_KEY not configured")
        elif not (raw_boolean_query or boolean_source_terms):
            logger.info("Skipping Web of Science — no query terms provided")
        else:
            _progress("Searching Web of Science...")
            wos = WebOfScienceClient(api_key=config.wos_api_key)
            wos_papers = []
            try:
                query = _build_wos_search(raw_boolean_query, boolean_source_terms)
                logger.info(f"Web of Science query: {query[:80]}...")
                papers = await wos.search(query, max_results=provider_limit)
                wos_papers = [_normalize_wos_paper(p) for p in papers][:provider_limit]
                all_papers.extend(wos_papers)
                logger.info(f"Web of Science returned {len(wos_papers)} papers")
            except Exception as e:
                logger.error(f"Web of Science error: {e}")
            finally:
                await wos.close()

            _save_raw(config.raw_dir / "gui_wos_raw.json", wos_papers)

    # Scopus (single combined TITLE-ABS-KEY query)
    if ProviderChoice.SCOPUS in _provider_set:
        _raise_if_cancelled()
        provider_limit = provider_limits.get(ProviderChoice.SCOPUS, 0)
        if provider_limit <= 0:
            logger.info("Skipping Scopus — no share of paper limit")
        elif not config.scopus_api_key:
            logger.info("Skipping Scopus — SCOPUS_API_KEY not configured")
        elif not (raw_boolean_query or boolean_source_terms):
            logger.info("Skipping Scopus — no query terms provided")
        else:
            _progress("Searching Scopus...")
            scopus = ScopusClient(
                api_key=config.scopus_api_key,
                insttoken=config.scopus_insttoken,
            )
            scopus_papers = []
            try:
                query = _build_scopus_search(raw_boolean_query, boolean_source_terms)
                logger.info(f"Scopus query: {query[:80]}...")
                papers = await scopus.search(query, max_results=provider_limit)
                scopus_papers = [_normalize_scopus_paper(p) for p in papers][:provider_limit]
                all_papers.extend(scopus_papers)
                logger.info(f"Scopus returned {len(scopus_papers)} papers")
            except Exception as e:
                logger.error(f"Scopus error: {e}")
            finally:
                await scopus.close()

            _save_raw(config.raw_dir / "gui_scopus_raw.json", scopus_papers)

    logger.info(f"New papers fetched this run: {len(all_papers)}")
    _raise_if_cancelled()
    _progress(f"Deduplicating {len(all_papers)} new papers...")

    # ── Merge with existing corpus and deduplicate ───────────
    combined = existing_papers + all_papers
    logger.info(
        f"Merging: {len(existing_papers)} existing + {len(all_papers)} new = "
        f"{len(combined)} total before dedup"
    )

    # Deduplicate across the full combined set
    corpus = deduplicate_papers(combined)

    # Filter papers without abstracts
    corpus = [p for p in corpus if p.get("abstract")]
    net_new = len(corpus) - len(existing_papers)
    logger.info(
        f"Corpus after dedup + abstract filter: {len(corpus)} "
        f"(+{net_new} net new papers)"
    )

    # Check minimum results threshold
    if len(corpus) < min_results:
        logger.warning(
            f"Corpus has {len(corpus)} papers, below minimum threshold of {min_results}. "
            f"Consider broadening search terms or increasing the paper limit."
        )

    # Save results (additive — overwrites with the merged superset)
    _raise_if_cancelled()
    _save_raw(output_path, corpus)

    return corpus
