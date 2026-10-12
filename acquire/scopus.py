"""Async Scopus API client."""
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

SCOPUS_BASE = "https://api.elsevier.com/content"


class ScopusClient:
    """Async client for the Scopus Search API."""

    def __init__(self, api_key: str = "", insttoken: str = "", rate_limit_delay: float = 0.35):
        headers: dict[str, str] = {"Accept": "application/json"}
        if api_key:
            headers["X-ELS-APIKey"] = api_key
        if insttoken:
            headers["X-ELS-Insttoken"] = insttoken
        self._client = httpx.AsyncClient(
            base_url=SCOPUS_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay

    async def search(self, query: str, max_results: int = 200) -> list[dict]:
        """Search Scopus and return raw result entries."""
        if not query:
            return []

        papers: list[dict] = []
        start = 0
        page_size = min(25, max_results)

        while start < max_results:
            params = {
                "query": query,
                "count": str(page_size),
                "start": str(start),
            }
            logger.info("Scopus search: start=%d query='%s'", start, query[:80])
            await asyncio.sleep(self._delay)

            try:
                resp = await self._client.get("/search/scopus", params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("Scopus API error: %s", exc)
                break

            results = data.get("search-results", {})
            entries = results.get("entry", [])
            total = int(results.get("opensearch:totalResults", 0))

            if not entries:
                break

            papers.extend(entries)
            start += len(entries)
            if start >= total or start >= max_results:
                break

        logger.info("Scopus search complete: %d papers", len(papers))
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
