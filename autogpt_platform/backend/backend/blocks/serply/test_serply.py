"""Unit tests for the Serply search block.

Complements the ``test_mock`` harness in the block (exercised by
``test_available_blocks``) with direct coverage of the request shape and the
error boundaries.
"""

from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.serply._api import (
    SERPLY_API_URL,
    SerplyClient,
    results_from_response,
)
from backend.blocks.serply.search import SerplySearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="serply",
    api_key=SecretStr("mock-serply-api-key"),
    title="Mock Serply API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


async def _collect(block, input_data: dict) -> dict:
    """Run ``block.execute`` and gather its yielded outputs into a dict."""
    outputs = {}
    async for name, value in block.execute(
        input_data,
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs[name] = value
    return outputs


def _row(position: int = 1, link: str = "https://agpt.co") -> dict:
    return {
        "title": "AutoGPT",
        "link": link,
        "description": "AutoGPT is a platform for building AI agents.",
        "position": position,
    }


def test_results_from_response_maps_fields():
    [r] = results_from_response({"results": [_row()]})
    assert r.title == "AutoGPT"
    assert r.url == "https://agpt.co"
    assert r.description == "AutoGPT is a platform for building AI agents."
    assert r.position == 1


def test_results_from_response_drops_non_url_links():
    rows = [_row(1, "CAESfQHrOzAV"), _row(2)]
    results = results_from_response({"results": rows})
    assert [r.url for r in results] == ["https://agpt.co"]


def test_results_from_response_missing_results():
    assert results_from_response({}) == []
    assert results_from_response({"results": None}) == []


@pytest.mark.asyncio
async def test_client_search_sends_key_and_user_agent():
    client = SerplyClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = True
    resp.json.return_value = {"results": []}
    client.requests.get = mock.AsyncMock(return_value=resp)

    out = await client.search({"q": "q", "num": 10})

    assert out == {"results": []}
    args, kwargs = client.requests.get.call_args
    assert args[0] == f"{SERPLY_API_URL}/v1/search"
    assert kwargs["params"] == {"q": "q", "num": 10}
    assert client.requests.extra_headers == {
        "X-Api-Key": "mock-serply-api-key",
        "User-Agent": "AutoGPT",
    }


@pytest.mark.asyncio
async def test_client_non_ok_status_raises():
    """raise_for_status only rejects >=400; a stranded 3xx must not reach
    response.json()."""
    client = SerplyClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = False
    resp.status = 302
    client.requests.get = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 302"):
        await client.search({"q": "q"})


@pytest.mark.asyncio
async def test_block_request_params_and_trim():
    block = SerplySearchBlock()
    seen = {}

    async def fake_search(credentials, params):
        seen.update(params)
        return {"results": [_row(i + 1) for i in range(4)]}

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(side_effect=fake_search)
    ):
        outputs = await _collect(
            block,
            {
                "credentials": TEST_CREDENTIALS_INPUT,
                "query": "site:go.dev generics",
                "max_results": 2,
                "country": "gb",
                "time_range": "week",
            },
        )

    # Always requests the API ceiling and trims client-side.
    assert seen == {"q": "site:go.dev generics", "num": 10, "gl": "gb", "tbs": "qdr:w"}
    assert len(outputs["results"]) == 2
    assert "[AutoGPT](https://agpt.co)" in outputs["context"]


@pytest.mark.asyncio
async def test_block_omits_unset_optional_params():
    block = SerplySearchBlock()
    seen = {}

    async def fake_search(credentials, params):
        seen.update(params)
        return {"results": [_row()]}

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(side_effect=fake_search)
    ):
        await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})

    assert seen == {"q": "q", "num": 10}


@pytest.mark.asyncio
async def test_block_wraps_api_errors():
    block = SerplySearchBlock()

    async def fake_search(credentials, params):
        raise ValueError("Serply search request failed with HTTP 401")

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(side_effect=fake_search)
    ):
        with pytest.raises(BlockExecutionError, match="HTTP 401"):
            await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})


@pytest.mark.asyncio
async def test_block_wraps_malformed_results():
    block = SerplySearchBlock()

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(return_value={"results": [None]})
    ):
        with pytest.raises(BlockExecutionError, match="Search failed"):
            await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})
