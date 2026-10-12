from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

from acquire.openalex import (
    OPENALEX_FALLBACK_SELECT_FIELDS,
    OPENALEX_SELECT_FIELDS,
    OpenAlexClient,
)


def test_http_500_triggers_select_fallback_before_server_retry():
    async def run() -> None:
        client = OpenAlexClient(api_key="user@example.com", rate_limit_delay=0)

        primary_response = MagicMock()
        primary_response.status_code = 500
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()

        get_mock = AsyncMock(side_effect=[primary_response, success_response])
        client._client.get = get_mock

        with patch("acquire.openalex.asyncio.sleep", new=AsyncMock()):
            resp = await client._get_with_select_fallback(
                "/works", {"filter": "has_abstract:true"}
            )

        assert resp is success_response
        assert get_mock.await_count == 2
        second_params = get_mock.await_args_list[1].kwargs["params"]
        assert second_params["select"] == OPENALEX_FALLBACK_SELECT_FIELDS

    asyncio.run(run())


def test_http_500_with_fallback_select_retries_as_server_error():
    async def run() -> None:
        client = OpenAlexClient(rate_limit_delay=0)

        server_error = MagicMock()
        server_error.status_code = 500
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()

        get_mock = AsyncMock(side_effect=[server_error, server_error, success_response])
        client._client.get = get_mock

        params = {"select": OPENALEX_FALLBACK_SELECT_FIELDS}
        with patch("acquire.openalex.asyncio.sleep", new=AsyncMock()):
            resp = await client._get_with_select_fallback("/works", params)

        assert resp is success_response
        assert get_mock.await_count == 3
        for call in get_mock.await_args_list:
            assert call.kwargs["params"]["select"] == OPENALEX_FALLBACK_SELECT_FIELDS

    asyncio.run(run())


def test_http_400_triggers_select_fallback():
    async def run() -> None:
        client = OpenAlexClient(rate_limit_delay=0)

        bad_request = MagicMock()
        bad_request.status_code = 400
        success_response = MagicMock()
        success_response.status_code = 200
        success_response.raise_for_status = MagicMock()

        get_mock = AsyncMock(side_effect=[bad_request, success_response])
        client._client.get = get_mock

        params = {"select": OPENALEX_SELECT_FIELDS}
        with patch("acquire.openalex.asyncio.sleep", new=AsyncMock()):
            resp = await client._get_with_select_fallback("/works", params)

        assert resp is success_response
        assert get_mock.await_args_list[1].kwargs["params"]["select"] == OPENALEX_FALLBACK_SELECT_FIELDS

    asyncio.run(run())
