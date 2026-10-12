"""Elicit client hits the documented v2 search endpoint (regression test).

The client previously posted to a hallucinated ``/api/v1/search`` endpoint
with a ``limit`` body field; the documented API is
``POST https://elicit.com/api/v2/search/papers`` with ``maxResults``.
"""
from __future__ import annotations

import asyncio
import json

import httpx

from acquire.elicit import ElicitClient


def _client_with_captured_transport(captured: dict):
    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content.decode())
        captured["auth"] = request.headers.get("authorization")
        return httpx.Response(
            200,
            json={"papers": [{"title": "Example paper", "doi": "10.1/x"}]},
        )

    client = ElicitClient(api_key="sek_test")
    client._client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler),
        base_url="https://elicit.com/api/v2",
        headers={"Authorization": "Bearer sek_test"},
    )
    return client


def test_elicit_search_posts_to_v2_search_papers():
    captured: dict = {}

    async def run() -> None:
        client = _client_with_captured_transport(captured)
        papers = await client.search("test query", max_results=25)
        assert papers == [{"title": "Example paper", "doi": "10.1/x"}]

    asyncio.run(run())
    assert captured["url"] == "https://elicit.com/api/v2/search/papers"
    assert captured["body"] == {"query": "test query", "maxResults": 25}
    assert captured["auth"] == "Bearer sek_test"


def test_elicit_search_empty_query_short_circuits():
    async def run() -> None:
        client = ElicitClient(api_key="sek_test")
        assert await client.search("") == []

    asyncio.run(run())
