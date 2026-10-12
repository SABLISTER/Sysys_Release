"""OpenAlex auth handling: mailto vs premium API key (regression test)."""
from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from acquire.openalex import OpenAlexClient


def _run(coro):
    return asyncio.run(coro)


def _ok_response():
    resp = MagicMock()
    resp.status_code = 200
    resp.raise_for_status = MagicMock()
    return resp


def test_email_api_key_uses_mailto_only():
    async def run() -> None:
        client = OpenAlexClient(api_key="user@example.com", rate_limit_delay=0)
        assert client._premium_key == ""
        assert client._client.headers["User-Agent"] == "mailto:user@example.com"

        get_mock = AsyncMock(return_value=_ok_response())
        client._client.get = get_mock
        with patch("acquire.openalex.asyncio.sleep", new=AsyncMock()):
            await client._get_with_select_fallback("/works", {})
        sent_params = get_mock.await_args_list[0].kwargs["params"]
        assert "api_key" not in sent_params

    _run(run())


def test_premium_api_key_sent_as_query_param():
    async def run() -> None:
        client = OpenAlexClient(api_key="premium-key-123", rate_limit_delay=0)
        assert client._premium_key == "premium-key-123"
        assert "mailto:" not in client._client.headers.get("User-Agent", "")

        get_mock = AsyncMock(return_value=_ok_response())
        client._client.get = get_mock
        with patch("acquire.openalex.asyncio.sleep", new=AsyncMock()):
            await client._get_with_select_fallback("/works", {})
        sent_params = get_mock.await_args_list[0].kwargs["params"]
        assert sent_params["api_key"] == "premium-key-123"

    _run(run())


def test_explicit_mailto_param():
    async def run() -> None:
        client = OpenAlexClient(
            api_key="premium-key-123", mailto="user@example.com", rate_limit_delay=0
        )
        assert client._client.headers["User-Agent"] == "mailto:user@example.com"
        assert client._premium_key == "premium-key-123"

    _run(run())
