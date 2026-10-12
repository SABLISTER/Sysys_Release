"""Stage 1: Literature Search.

Search and retrieve papers from Elicit, Semantic Scholar, OpenAlex,
PubMed, Web of Science, and Scopus.  Normalize, deduplicate, and
filter to papers with abstracts.
"""
import asyncio
import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


async def run(
    config,
    research: dict | None = None,
    progress_callback: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[asyncio.Event] = None,
    **kw,
) -> list[dict]:
    """Execute Stage 1 — multi-provider literature search.

    Parameters
    ----------
    config : Config
        Pipeline configuration (provides paths, API keys, etc.).
    research : dict | None
        Full research config; the ``"search"`` key drives query generation.
    progress_callback : callable, optional
        Called with a status string as each provider completes.
    cancel_event : asyncio.Event, optional
        Set this event to request cancellation mid-search.

    Returns
    -------
    list[dict]
        Deduplicated corpus as a list of paper dicts.
    """
    from acquire.orchestrator import run_acquisition
    from process.queries import get_natural_language_query

    def _progress(msg: str) -> None:
        logger.info(msg)
        if progress_callback:
            try:
                progress_callback(msg)
            except Exception:
                pass

    _progress("═" * 60)
    _progress("STAGE 1: LITERATURE SEARCH")
    _progress("═" * 60)

    if cancel_event and cancel_event.is_set():
        _progress("Stage 1 cancelled before start.")
        return []

    search_config = None
    if research:
        search_config = research.get("search")

    nl_query = get_natural_language_query(research)
    _progress(f"  NL query: {nl_query[:120]}")

    corpus = await run_acquisition(
        config,
        search_config=search_config,
        natural_language_query=nl_query,
    )

    if cancel_event and cancel_event.is_set():
        _progress("Stage 1 cancelled after acquisition.")
        return corpus

    _progress(f"Stage 1 complete: {len(corpus)} papers in corpus")
    return corpus


def check_output(config) -> dict:
    """Check whether Stage 1 output exists."""
    path = config.corpus_dir / "corpus.json"
    if not path.exists():
        return {"exists": False}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return {"exists": True, "paper_count": len(data), "path": str(path)}
    except Exception:
        return {"exists": False}
