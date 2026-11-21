"""Unit tests for the SearchApi search block.

Complements the ``test_mock`` harness in the block (exercised by
``test_available_blocks``) with direct coverage of the request shape, the
response shapes SearchApi actually returns and the error boundaries.
"""

from unittest import mock

import pytest
from pydantic import SecretStr

from backend.blocks.searchapi._api import (
    SEARCHAPI_API_URL,
    SearchApiClient,
    results_from_response,
)
from backend.blocks.searchapi.search import SearchApiSearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="searchapi",
    api_key=SecretStr("mock-searchapi-api-key"),
    title="Mock SearchApi API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}

# The shape a result page with no hits has: HTTP 200, an ``error`` string and
# no ``organic_results``.
EMPTY_PAGE = {
    "search_metadata": {"status": "Success"},
    "error": "Google didn't return any results.",
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
        "position": position,
        "title": "AutoGPT",
        "link": link,
        "snippet": "AutoGPT is a platform for building AI agents.",
    }


def _response(status: int, body: dict | None = None) -> mock.Mock:
    resp = mock.Mock()
    resp.status = status
    resp.ok = 200 <= status < 300
    resp.reason = {401: "Unauthorized", 302: "Found"}.get(status, "")
    if body is None:
        resp.json.side_effect = ValueError("not JSON")
    else:
        resp.json.return_value = body
    return resp


def test_client_sends_bearer_key_and_reads_error_bodies():
    client = SearchApiClient(TEST_CREDENTIALS)
    assert client.requests.extra_headers == {
        "Authorization": "Bearer mock-searchapi-api-key",
        "User-Agent": "AutoGPT",
    }
    assert client.requests.raise_for_status is False


def test_results_from_response_maps_fields():
    [r] = results_from_response({"organic_results": [_row()]})
    assert r.title == "AutoGPT"
    assert r.url == "https://agpt.co"
    assert r.description == "AutoGPT is a platform for building AI agents."
    assert r.position == 1


def test_results_from_response_keeps_google_redirect_links():
    """Some rows link through google.com/goto; they are still http(s) URLs
    and are passed through untouched."""
    goto = "https://www.google.com/goto?url=https://agpt.co"
    [r] = results_from_response({"organic_results": [_row(1, goto)]})
    assert r.url == goto


def test_results_from_response_drops_non_url_links():
    rows = [_row(1, "CAESfQHrOzAV"), _row(2)]
    results = results_from_response({"organic_results": rows})
    assert [r.url for r in results] == ["https://agpt.co"]


@pytest.mark.parametrize(
    "response",
    [{}, {"organic_results": None}, EMPTY_PAGE],
    ids=["no-key", "null", "empty-page-with-error-string"],
)
def test_results_from_response_without_results(response: dict):
    assert results_from_response(response) == []


@pytest.mark.asyncio
async def test_client_search_sends_params_to_search_endpoint():
    client = SearchApiClient(TEST_CREDENTIALS)
    client.requests.get = mock.AsyncMock(
        return_value=_response(200, {"organic_results": []})
    )

    out = await client.search({"engine": "google", "q": "q", "num": 10})

    assert out == {"organic_results": []}
    args, kwargs = client.requests.get.call_args
    assert args[0] == f"{SEARCHAPI_API_URL}/api/v1/search"
    assert kwargs["params"] == {"engine": "google", "q": "q", "num": 10}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status", "body", "expected"),
    [
        (401, {"error": "Invalid API key."}, "HTTP 401: Invalid API key."),
        (302, None, "HTTP 302: Found"),
    ],
    ids=["api-error-body", "non-json-body"],
)
async def test_client_non_ok_status_raises_with_api_message(
    status: int, body: dict | None, expected: str
):
    client = SearchApiClient(TEST_CREDENTIALS)
    client.requests.get = mock.AsyncMock(return_value=_response(status, body))

    with pytest.raises(ValueError, match=expected):
        await client.search({"engine": "google", "q": "q"})


@pytest.mark.asyncio
async def test_block_request_params_and_trim():
    block = SearchApiSearchBlock()
    seen = {}

    async def fake_search(credentials, params):
        seen.update(params)
        return {"organic_results": [_row(i + 1) for i in range(4)]}

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

    assert seen == {
        "engine": "google",
        "q": "site:go.dev generics",
        "num": 2,
        "gl": "gb",
        "time_period": "last_week",
    }
    # A page can carry more rows than asked for; the block trims client-side.
    assert len(outputs["results"]) == 2
    assert "[AutoGPT](https://agpt.co)" in outputs["context"]


@pytest.mark.asyncio
async def test_block_omits_unset_optional_params():
    block = SearchApiSearchBlock()
    seen = {}

    async def fake_search(credentials, params):
        seen.update(params)
        return {"organic_results": [_row()]}

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(side_effect=fake_search)
    ):
        await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})

    assert seen == {"engine": "google", "q": "q", "num": 5}


@pytest.mark.asyncio
async def test_block_empty_page_yields_no_results():
    block = SearchApiSearchBlock()

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(return_value=EMPTY_PAGE)
    ):
        outputs = await _collect(
            block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"}
        )

    assert outputs["results"] == []
    assert outputs["context"] == ""
    assert "result" not in outputs


@pytest.mark.asyncio
async def test_block_wraps_api_errors():
    block = SearchApiSearchBlock()

    async def fake_search(credentials, params):
        raise ValueError(
            "SearchApi search request failed with HTTP 401: Invalid API key."
        )

    with mock.patch.object(
        block, "_search", new=mock.AsyncMock(side_effect=fake_search)
    ):
        with pytest.raises(BlockExecutionError, match="HTTP 401: Invalid API key."):
            await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})


@pytest.mark.asyncio
async def test_block_wraps_malformed_results():
    block = SearchApiSearchBlock()

    with mock.patch.object(
        block,
        "_search",
        new=mock.AsyncMock(return_value={"organic_results": [None]}),
    ):
        with pytest.raises(BlockExecutionError, match="Search failed"):
            await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})
