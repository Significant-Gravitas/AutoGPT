"""Unit tests for the Perplexity Search block.

Complements the ``test_mock`` harness in the block (exercised by
``test_available_blocks``) with direct coverage of the request shape,
the attribution headers, and the error boundaries.
"""

import logging
import traceback
from typing import Any
from unittest import mock

import pytest
from aiohttp import ClientConnectorError
from pydantic import SecretStr

from backend.blocks.perplexity_search import _api
from backend.blocks.perplexity_search._api import (
    FAST_SEARCH_COST_USD,
    PERPLEXITY_API_URL,
    WEB_SEARCH_COST_USD,
    PerplexitySearchResponse,
    PerplexitySearchResult,
    search,
)
from backend.blocks.perplexity_search.search import PerplexitySearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="perplexity",
    api_key=SecretStr("mock-perplexity-api-key"),
    title="Mock Perplexity API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.title,
}


def _row(title: str = "AutoGPT", url: str = "https://agpt.co", **extra) -> dict:
    return {
        "title": title,
        "url": url,
        "snippet": "AutoGPT is a platform for building AI agents.",
        **extra,
    }


def _search_response(results: list[dict]) -> PerplexitySearchResponse:
    return PerplexitySearchResponse.model_validate(
        {"id": "search-id", "results": results}
    )


class FakeSearch:
    """Stands in for the API call and records the payload each run sent."""

    def __init__(
        self, response: PerplexitySearchResponse, error: Exception | None = None
    ):
        self.response = response
        self.error = error
        self.payloads: list[dict[str, Any]] = []

    async def __call__(self, credentials, payload):
        self.payloads.append(payload)
        if self.error:
            raise self.error
        return self.response


async def _run(block: PerplexitySearchBlock, fake: FakeSearch, **inputs) -> dict:
    block._search = fake
    outputs: dict[str, Any] = {}
    async for name, value in block.execute(
        {"credentials": TEST_CREDENTIALS_INPUT, **inputs},
        credentials=TEST_CREDENTIALS,
        execution_context=ExecutionContext(),
    ):
        outputs.setdefault(name, []).append(value)
    return outputs


def _exception_chain_text(exc: BaseException) -> str:
    """Everything a logger would print for the exception: its str(), repr(),
    and the formatted traceback including any chained cause or context."""
    return f"{exc!r}\n{exc}\n" + "".join(traceback.format_exception(exc))


def _mock_response(status: int, body: bytes = b"", ok: bool | None = None):
    resp = mock.Mock()
    resp.status = status
    resp.ok = ok if ok is not None else status < 400
    resp.content = body
    resp.text.return_value = body.decode()
    resp.json.return_value = {"id": "search-id", "results": [_row()]}
    return resp


def test_response_tolerates_missing_snippet_and_dates():
    parsed = PerplexitySearchResponse.model_validate(
        {"results": [{"title": "t", "url": "https://agpt.co"}]}
    )
    assert parsed.results == [PerplexitySearchResult(title="t", url="https://agpt.co")]
    assert parsed.results[0].snippet == ""
    assert parsed.results[0].date is None
    assert parsed.results[0].last_updated is None


@pytest.mark.asyncio
async def test_search_posts_to_search_endpoint_with_attribution_headers():
    resp = _mock_response(200)
    with mock.patch.object(_api, "Requests") as requests_cls:
        requests_cls.return_value.post = mock.AsyncMock(return_value=resp)
        out = await search(TEST_CREDENTIALS, {"query": "q", "max_results": 1})

    assert out.results[0].url == "https://agpt.co"
    init_kwargs = requests_cls.call_args.kwargs
    assert init_kwargs["trusted_origins"] == [PERPLEXITY_API_URL]
    assert init_kwargs["extra_headers"] == {
        "Authorization": "Bearer mock-perplexity-api-key",
        "X-Pplx-Integration": "autogpt",
    }
    args, kwargs = requests_cls.return_value.post.call_args
    assert args[0] == f"{PERPLEXITY_API_URL}/search"
    assert kwargs["json"] == {"query": "q", "max_results": 1}
    assert kwargs["allow_redirects"] is False


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "status, body, expected",
    [
        (
            401,
            b'{"error":{"message":"Invalid API key provided.","type":"invalid_api_key","code":401}}',
            "HTTP 401: Invalid API key provided.",
        ),
        (
            422,
            b'{"detail":[{"loc":["body","max_results"],"msg":"Input should be less than or equal to 20","type":"less_than_equal"}]}',
            "HTTP 422: Input should be less than or equal to 20",
        ),
        (429, b"Too Many Requests", "HTTP 429: no error details were provided"),
        (301, b"", "HTTP 301: no error details were provided"),
    ],
)
async def test_search_non_ok_status_raises(status: int, body: bytes, expected: str):
    resp = _mock_response(status, body, ok=False)
    with mock.patch.object(_api, "Requests") as requests_cls:
        requests_cls.return_value.post = mock.AsyncMock(return_value=resp)
        with pytest.raises(ValueError) as exc_info:
            await search(TEST_CREDENTIALS, {"query": "q"})
    assert expected in str(exc_info.value)


@pytest.mark.asyncio
async def test_search_malformed_body_raises_without_echoing_it():
    resp = _mock_response(200)
    resp.json.return_value = {"results": [{"title": "no url", "secret": "s3cr3t"}]}
    with mock.patch.object(_api, "Requests") as requests_cls:
        requests_cls.return_value.post = mock.AsyncMock(return_value=resp)
        with pytest.raises(ValueError, match="could not parse") as exc_info:
            await search(TEST_CREDENTIALS, {"query": "q"})
    assert "s3cr3t" not in _exception_chain_text(exc_info.value)


@pytest.mark.asyncio
async def test_search_non_json_error_body_is_not_echoed():
    body = b"<html>Bad gateway for Bearer mock-perplexity-api-key</html>"
    resp = _mock_response(502, body, ok=False)
    with mock.patch.object(_api, "Requests") as requests_cls:
        requests_cls.return_value.post = mock.AsyncMock(return_value=resp)
        with pytest.raises(ValueError) as exc_info:
            await search(TEST_CREDENTIALS, {"query": "q"})
    text = _exception_chain_text(exc_info.value)
    assert "HTTP 502: no error details were provided" in text
    assert "<html>" not in text
    assert "mock-perplexity-api-key" not in text


@pytest.mark.asyncio
async def test_search_truncates_long_provider_error_message():
    body = b'{"error":{"message":"' + b"x" * 2000 + b'"}}'
    resp = _mock_response(400, body, ok=False)
    with mock.patch.object(_api, "Requests") as requests_cls:
        requests_cls.return_value.post = mock.AsyncMock(return_value=resp)
        with pytest.raises(ValueError) as exc_info:
            await search(TEST_CREDENTIALS, {"query": "q"})
    assert len(str(exc_info.value)) < 600


@pytest.mark.asyncio
async def test_search_transport_error_text_is_not_echoed():
    # Worst case: a transport error that quotes the request headers.
    leaky = ClientConnectorError(
        mock.Mock(), OSError("Authorization: Bearer mock-perplexity-api-key")
    )
    with mock.patch.object(_api, "Requests") as requests_cls:
        requests_cls.return_value.post = mock.AsyncMock(side_effect=leaky)
        with pytest.raises(ValueError, match="Could not reach Perplexity") as exc_info:
            await search(TEST_CREDENTIALS, {"query": "q"})
    text = _exception_chain_text(exc_info.value)
    assert "ClientConnectorError" in text
    assert "mock-perplexity-api-key" not in text
    assert exc_info.value.__cause__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "key",
    ["pplx-abc\r\nX-Injected: 1", "pplx-abc\n", "pplx-abc\x00", ""],
    ids=["crlf", "trailing-newline", "nul", "empty"],
)
async def test_key_that_cannot_be_a_header_fails_before_sending(key: str):
    # aiohttp rejects control characters in header values at write time
    # (ValueError "Forbidden control character detected in headers"); the
    # block refuses such a key up front so no transport error is involved.
    credentials = TEST_CREDENTIALS.model_copy(update={"api_key": SecretStr(key)})
    with (
        mock.patch.object(_api, "Requests") as requests_cls,
        pytest.raises(ValueError, match="cannot be sent in an HTTP header"),
    ):
        await search(credentials, {"query": "q"})
    requests_cls.assert_not_called()


@pytest.mark.asyncio
async def test_key_with_newline_never_reaches_error_output_or_logs(caplog):
    key = "pplx-abc\nX-Injected: 1"
    credentials = TEST_CREDENTIALS.model_copy(update={"api_key": SecretStr(key)})
    block = PerplexitySearchBlock()
    caplog.set_level(logging.DEBUG)

    with pytest.raises(BlockExecutionError) as exc_info:
        async for _ in block.execute(
            {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"},
            credentials=credentials,
            execution_context=ExecutionContext(),
        ):
            pass

    text = _exception_chain_text(exc_info.value)
    assert "cannot be sent in an HTTP header" in text
    assert "pplx-abc" not in text
    assert "pplx-abc" not in caplog.text
    assert "Authorization" not in text
    assert "Authorization" not in caplog.text


@pytest.mark.asyncio
async def test_search_happy_path_yields_each_result():
    fake = FakeSearch(
        _search_response(
            [_row(date="2024-01-01"), _row("Docs", "https://docs.agpt.co")]
        )
    )
    outputs = await _run(PerplexitySearchBlock(), fake, query="q", max_results=2)

    assert fake.payloads == [{"query": "q", "max_results": 2, "search_type": "web"}]
    assert len(outputs["results"]) == 1
    assert [r.url for r in outputs["results"][0]] == [
        "https://agpt.co",
        "https://docs.agpt.co",
    ]
    assert [r.url for r in outputs["result"]] == [
        "https://agpt.co",
        "https://docs.agpt.co",
    ]
    assert outputs["results"][0][0].date == "2024-01-01"
    assert outputs["results"][0][1].date is None
    assert "Title: Docs" in outputs["context"][0]
    assert "error" not in outputs


@pytest.mark.asyncio
async def test_search_forwards_optional_filters():
    fake = FakeSearch(_search_response([]))
    block = PerplexitySearchBlock()
    outputs = await _run(
        block,
        fake,
        query="q",
        search_type="fast",
        search_domain_filter=["perplexity.ai", "-reddit.com"],
        search_recency_filter="week",
    )

    assert fake.payloads == [
        {
            "query": "q",
            "max_results": 5,
            "search_type": "fast",
            "search_domain_filter": ["perplexity.ai", "-reddit.com"],
            "search_recency_filter": "week",
        }
    ]
    assert outputs["results"] == [[]]
    assert "result" not in outputs
    assert outputs["context"] == [""]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "search_type, cost",
    [("web", WEB_SEARCH_COST_USD), ("fast", FAST_SEARCH_COST_USD)],
)
async def test_search_records_provider_cost(search_type: str, cost: float):
    block = PerplexitySearchBlock()
    await _run(
        block, FakeSearch(_search_response([])), query="q", search_type=search_type
    )

    assert block.execution_stats.provider_cost == cost
    assert block.execution_stats.provider_cost_type == "cost_usd"


@pytest.mark.asyncio
async def test_search_api_error_raises_block_error():
    block = PerplexitySearchBlock()
    fake = FakeSearch(
        _search_response([]),
        error=ValueError("Perplexity Search request failed with HTTP 401"),
    )

    with pytest.raises(BlockExecutionError, match="HTTP 401"):
        await _run(block, fake, query="q")

    assert block.execution_stats.provider_cost is None


@pytest.mark.parametrize("max_results", [0, 21])
def test_max_results_outside_the_api_range_fails_the_schema_check(max_results):
    # The executor checks inputs against the JSON schema before charging the
    # run, so an out-of-range max_results never reaches the API.
    error = PerplexitySearchBlock.Input.validate_data(
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "query": "q",
            "max_results": max_results,
        }
    )
    assert error is not None


def test_max_results_inside_the_api_range_passes_the_schema_check():
    error = PerplexitySearchBlock.Input.validate_data(
        {"credentials": TEST_CREDENTIALS_INPUT, "query": "q", "max_results": 20}
    )
    assert error is None
