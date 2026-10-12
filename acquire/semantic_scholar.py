"""Async Semantic Scholar Graph API client."""
import asyncio
import logging

import httpx

logger = logging.getLogger(__name__)

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_FIELDS = (
    "paperId,title,abstract,year,citationCount,externalIds,"
    "authors,fieldsOfStudy,publicationTypes,tldr,openAccessPdf"
)
S2_FALLBACK_FIELDS = (
    "paperId,title,abstract,year,citationCount,externalIds,"
    "authors,fieldsOfStudy,publicationTypes"
)


class SemanticScholarClient:
    """Async client for the Semantic Scholar Academic Graph API."""

    def __init__(
        self,
        api_key: str = "",
        rate_limit_delay: float = 1.1,
        max_retries: int = 5,
        initial_backoff: float = 5.0,
        max_backoff: float = 5.0,
    ):
        headers: dict[str, str] = {}
        if api_key:
            headers["x-api-key"] = api_key
        self._client = httpx.AsyncClient(
            base_url=S2_BASE,
            headers=headers,
            timeout=30.0,
        )
        self._delay = rate_limit_delay
        self._max_retries = max_retries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff

    @staticmethod
    def _parse_retry_after(value: str | None) -> float | None:
        """Parse a Retry-After header value (delta-seconds form) into seconds."""
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            return None

    async def _request_with_backoff(self, path: str, params: dict) -> httpx.Response:
        """Issue a GET, retrying on 429 with exponential backoff.

        Honors the ``Retry-After`` header when present. After exhausting
        ``max_retries`` the final (still-429) response is returned so the
        caller can decide how to handle it.
        """
        backoff = self._initial_backoff
        resp: httpx.Response | None = None
        for attempt in range(self._max_retries + 1):
            await asyncio.sleep(self._delay)
            resp = await self._client.get(path, params=params)
            if resp.status_code != 429:
                return resp
            if attempt >= self._max_retries:
                logger.error(
                    "S2: still rate limited (429) after %d retries; giving up",
                    self._max_retries,
                )
                return resp
            wait = self._parse_retry_after(resp.headers.get("Retry-After"))
            if wait is None:
                wait = backoff
            # Enforce the 5s cap even when the server's Retry-After is larger.
            wait = min(wait, self._max_backoff)
            logger.warning(
                "S2: rate limited (429), waiting %.1fs (attempt %d/%d)...",
                wait, attempt + 1, self._max_retries,
            )
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, self._max_backoff)
        return resp  # type: ignore[return-value]

    async def _get_with_field_fallback(
        self,
        path: str,
        params: dict,
        *,
        fields: str,
        fallback_fields: str | None = None,
    ) -> httpx.Response:
        """GET with automatic field-set fallback on 400 and 429 backoff."""
        params["fields"] = fields
        resp = await self._request_with_backoff(path, params)
        if resp.status_code == 400 and fallback_fields:
            logger.warning("S2: field set rejected, retrying with fallback fields")
            params["fields"] = fallback_fields
            resp = await self._request_with_backoff(path, params)
        return resp

    async def search(self, query: str, max_results: int = 1000) -> list[dict]:
        """Search for papers. Returns raw S2 paper dicts."""
        if not query:
            return []

        papers: list[dict] = []
        offset = 0
        page_size = 100

        while offset < max_results:
            limit = min(page_size, max_results - offset)
            params = {"query": query, "limit": str(limit), "offset": str(offset)}

            logger.info("S2 search: offset=%d limit=%d query='%s'", offset, limit, query[:80])
            resp = await self._get_with_field_fallback(
                "/paper/search", params, fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )

            resp.raise_for_status()
            data = resp.json()

            total_available = data.get("total", 0)
            batch = data.get("data", [])
            logger.info("S2: batch=%d total_available=%d", len(batch), total_available)

            if not batch:
                break

            papers.extend(batch)
            offset += len(batch)
            if offset >= total_available:
                break

        logger.info("S2 search complete: %d papers", len(papers))
        return papers[:max_results]

    async def get_paper(self, paper_id: str) -> dict | None:
        """Fetch a single paper by S2 ID, DOI, or other external ID."""
        try:
            resp = await self._get_with_field_fallback(
                f"/paper/{paper_id}", {}, fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            logger.warning("S2 get_paper(%s) failed: %s", paper_id, exc)
            return None

    async def get_citations(self, paper_id: str, limit: int = 100) -> list[dict]:
        """Get papers that cite the given paper."""
        try:
            params = {"limit": str(limit)}
            resp = await self._get_with_field_fallback(
                f"/paper/{paper_id}/citations", params,
                fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )
            resp.raise_for_status()
            return [c.get("citingPaper", {}) for c in resp.json().get("data", [])]
        except Exception as exc:
            logger.warning("S2 get_citations(%s) failed: %s", paper_id, exc)
            return []

    async def get_references(self, paper_id: str, limit: int = 100) -> list[dict]:
        """Get papers referenced by the given paper."""
        try:
            params = {"limit": str(limit)}
            resp = await self._get_with_field_fallback(
                f"/paper/{paper_id}/references", params,
                fields=S2_FIELDS, fallback_fields=S2_FALLBACK_FIELDS,
            )
            resp.raise_for_status()
            return [r.get("citedPaper", {}) for r in resp.json().get("data", [])]
        except Exception as exc:
            logger.warning("S2 get_references(%s) failed: %s", paper_id, exc)
            return []

    async def close(self):
        await self._client.aclose()
