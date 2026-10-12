"""Async Elicit API client."""
import logging

import httpx

logger = logging.getLogger(__name__)

ELICIT_BASE = "https://elicit.com/api/v2"
ELICIT_MAX_RESULTS = 100


class ElicitClient:
    """Thin async wrapper around the Elicit search API."""

    def __init__(self, api_key: str = ""):
        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(
            base_url=ELICIT_BASE,
            headers=headers,
            timeout=60.0,
        )

    async def search(self, query: str, max_results: int = 100) -> list[dict]:
        """Search Elicit and return raw paper dicts."""
        if not query:
            return []

        limit = min(max_results, ELICIT_MAX_RESULTS)
        logger.info("Elicit search: query='%s' max_results=%d", query[:80], limit)

        try:
            resp = await self._client.post(
                "/search/papers",
                json={"query": query, "maxResults": limit},
            )
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("papers", data.get("results", []))
            logger.info("Elicit: returned %d papers", len(papers))
            return papers[:max_results]
        except httpx.HTTPStatusError as exc:
            logger.error("Elicit API error %d: %s", exc.response.status_code, exc)
            return []
        except Exception as exc:
            logger.error("Elicit error: %s", exc)
            return []

    async def close(self):
        await self._client.aclose()
