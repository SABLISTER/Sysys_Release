"""Async Web of Science Starter API client."""
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

WOS_BASE = "https://api.clarivate.com/apis/wos-starter/v1"


class WebOfScienceClient:
    """Async client for the Web of Science Starter API."""

    def __init__(self, api_key: str = "", rate_limit_delay: float = 0.35):
        headers: dict[str, str] = {}
        if api_key:
            headers["X-ApiKey"] = api_key
        self._client = httpx.AsyncClient(
            base_url=WOS_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay

    async def search(self, query: str, max_results: int = 200) -> list[dict]:
        """Search WoS and return raw hit dicts."""
        if not query:
            return []

        papers: list[dict] = []
        page = 1
        per_page = min(50, max_results)

        while len(papers) < max_results:
            params = {
                "q": query,
                "limit": str(per_page),
                "page": str(page),
            }
            logger.info("WoS search: page=%d query='%s'", page, query[:80])
            await asyncio.sleep(self._delay)

            try:
                resp = await self._client.get("/documents", params=params)
                resp.raise_for_status()
                data = resp.json()
            except Exception as exc:
                logger.error("WoS API error: %s", exc)
                break

            hits = data.get("hits", [])
            total = data.get("metadata", {}).get("total", 0)

            if not hits:
                break

            papers.extend(hits)
            page += 1
            if len(papers) >= total or len(papers) >= max_results:
                break

        logger.info("WoS search complete: %d papers", len(papers))
        return papers[:max_results]

    async def close(self):
        await self._client.aclose()
