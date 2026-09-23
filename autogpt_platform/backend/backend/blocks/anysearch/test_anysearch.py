"""Unit tests for the AnySearch blocks.

Complements the ``test_mock`` harness in each block (exercised by
``test_available_blocks``) with direct coverage of the error boundaries:
API-level error envelopes, malformed upstream payloads, and per-query
failure isolation in the parallel block.
"""

import inspect
from unittest import mock

import pytest
from pydantic import SecretStr, ValidationError

from backend.blocks.anysearch._api import (
    ANYSEARCH_API_URL,
    AnySearchClient,
    AnySearchDomain,
    result_from_dict,
    unwrap_envelope,
)
from backend.blocks.anysearch.extract import AnySearchExtractBlock
from backend.blocks.anysearch.parallel_search import AnySearchParallelSearchBlock
from backend.blocks.anysearch.search import AnySearchBlock
from backend.data.execution import ExecutionContext
from backend.sdk import APIKeyCredentials
from backend.util.exceptions import BlockExecutionError

TEST_CREDENTIALS = APIKeyCredentials(
    id="01234567-89ab-cdef-0123-456789abcdef",
    provider="anysearch",
    api_key=SecretStr("mock-anysearch-api-key"),
    title="Mock Anysearch API key",
    expires_at=None,
)
TEST_CREDENTIALS_INPUT = {
    "provider": TEST_CREDENTIALS.provider,
    "id": TEST_CREDENTIALS.id,
    "type": TEST_CREDENTIALS.type,
    "title": TEST_CREDENTIALS.type,
}


def _mock_block(block, mocks: dict):
    """Apply mocks to a block's methods, wrapping sync mocks as async."""
    for name, mock_fn in mocks.items():
        original = getattr(block, name)
        if inspect.iscoroutinefunction(original):

            async def async_mock(*args, _fn=mock_fn, **kwargs):
                return _fn(*args, **kwargs)

            setattr(block, name, async_mock)
        else:
            setattr(block, name, mock_fn)


def _raise(exc: Exception):
    """Return a callable which raises the given exception."""

    def _raiser(*args, **kwargs):
        raise exc

    return _raiser


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


def _row(title: str = "AutoGPT", url: str = "https://agpt.co") -> dict:
    return {
        "title": title,
        "url": url,
        "snippet": "AutoGPT is a platform for building AI agents.",
        "content": None,
    }


def _search_response(results) -> dict:
    return {"code": 0, "message": "success", "data": {"results": results}}


def test_unwrap_envelope_success():
    data = unwrap_envelope({"code": 0, "message": "success", "data": {"results": []}})
    assert data == {"results": []}


def test_unwrap_envelope_api_error():
    with pytest.raises(ValueError, match="invalid_api_key"):
        unwrap_envelope({"code": 401, "message": "invalid_api_key", "data": {}})


def test_unwrap_envelope_missing_data():
    with pytest.raises(ValueError, match="unknown error"):
        unwrap_envelope({"code": 0, "message": None, "data": None})


def test_unwrap_envelope_bool_code():
    # False == 0 in Python; the contract uses numeric codes only.
    with pytest.raises(ValueError):
        unwrap_envelope({"code": False, "data": {}})


def test_result_from_dict_defaults():
    r = result_from_dict({"url": "https://x.test"})
    assert r.title == ""
    assert r.url == "https://x.test"
    assert r.snippet == ""
    assert r.content is None


def test_domain_enum_members():
    assert AnySearchDomain.FINANCE.value == "finance"
    assert AnySearchDomain.GENERAL.value == "general"
    assert len(AnySearchDomain) == 17


@pytest.mark.asyncio
async def test_client_search_posts_to_search_endpoint():
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = True
    resp.json.return_value = {
        "code": 0,
        "message": "success",
        "data": {"results": []},
    }
    client.requests.post = mock.AsyncMock(return_value=resp)

    out = await client.search({"query": "q", "max_results": 1})

    assert out["data"]["results"] == []
    call = client.requests.post.call_args
    assert call is not None
    args, kwargs = call
    assert args[0] == f"{ANYSEARCH_API_URL}/v1/search"
    assert kwargs["json"]["query"] == "q"
    assert client.requests.extra_headers == {
        "Authorization": "Bearer mock-anysearch-api-key"
    }


@pytest.mark.asyncio
async def test_client_extract_posts_to_extract_endpoint():
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = True
    resp.json.return_value = {
        "code": 0,
        "message": "success",
        "data": {"title": "t"},
    }
    client.requests.post = mock.AsyncMock(return_value=resp)

    out = await client.extract("https://agpt.co")

    assert out["data"]["title"] == "t"
    call = client.requests.post.call_args
    assert call is not None
    args, kwargs = call
    assert args[0] == f"{ANYSEARCH_API_URL}/v1/extract"
    assert kwargs["json"] == {"url": "https://agpt.co"}


@pytest.mark.asyncio
async def test_client_non_ok_status_raises():
    """raise_for_status only rejects >=400; a stranded 3xx (no Location)
    must not reach response.json()."""
    client = AnySearchClient(TEST_CREDENTIALS)
    resp = mock.Mock()
    resp.ok = False
    resp.status = 301
    client.requests.post = mock.AsyncMock(return_value=resp)

    with pytest.raises(ValueError, match="HTTP 301"):
        await client.search({"query": "q"})

    with pytest.raises(ValueError, match="HTTP 301"):
        await client.extract("https://agpt.co")


@pytest.mark.asyncio
async def test_search_happy_path():
    block = AnySearchBlock()
    _mock_block(block, {"_search": lambda *a, **k: _search_response([_row()])})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "query": "q", "max_results": 1},
    )

    assert len(outputs["results"]) == 1
    assert outputs["result"].url == "https://agpt.co"
    assert "AutoGPT" in outputs["context"]


@pytest.mark.asyncio
async def test_search_forwards_vertical_params():
    captured = {}

    def spy(creds, payload):
        captured.update(payload)
        return _search_response([])

    block = AnySearchBlock()
    _mock_block(block, {"_search": spy})

    outputs = await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "query": "NVDA earnings",
            "max_results": 3,
            "domain": "finance",
            "sub_domain": "finance.quote",
            "sub_domain_params": {"type": "stock", "symbol": "NVDA"},
        },
    )

    assert captured["tag"] == "finance.quote"
    assert captured["max_results"] == 3
    assert captured["params"] == {"type": "stock", "symbol": "NVDA"}
    assert "domain" not in captured
    assert "sub_domain" not in captured
    assert outputs["results"] == []
    assert "result" not in outputs


def test_domain_requires_matching_sub_domain():
    base = {
        "credentials": TEST_CREDENTIALS_INPUT,
        "query": "q",
        "domain": "finance",
    }

    with pytest.raises(ValidationError, match="sub_domain is required"):
        AnySearchBlock.Input.model_validate(base)

    with pytest.raises(ValidationError, match="must belong"):
        AnySearchBlock.Input.model_validate({**base, "sub_domain": "news.stock"})


@pytest.mark.asyncio
async def test_search_api_error_raises_block_error():
    block = AnySearchBlock()
    _mock_block(block, {"_search": _raise(RuntimeError("upstream boom"))})

    with pytest.raises(BlockExecutionError, match="Search failed"):
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"},
        )


@pytest.mark.asyncio
async def test_search_malformed_results_raise_block_error():
    block = AnySearchBlock()
    _mock_block(
        block,
        {
            "_search": lambda *a, **k: {
                "code": 0,
                "message": "success",
                "data": {"results": None},
            }
        },
    )

    with pytest.raises(BlockExecutionError, match="Search failed"):
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"},
        )


@pytest.mark.asyncio
async def test_parallel_partial_failure_returns_groups():
    def fn(creds, payload):
        if "fail" in payload["query"]:
            raise RuntimeError("upstream boom")
        return _search_response([_row()])

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": fn})

    outputs = await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "queries": ["fail me", "ok query"],
            "max_results": 1,
        },
    )

    groups = outputs["results"]
    assert groups[0].query == "fail me"
    assert groups[0].error
    assert groups[0].results == []
    assert groups[1].query == "ok query"
    assert groups[1].error == ""
    assert len(groups[1].results) == 1


@pytest.mark.asyncio
async def test_parallel_malformed_results_degrade_to_error_group():
    """null/non-object results must not escape the per-query boundary and
    abort ``asyncio.gather``."""

    def fn(creds, payload):
        if "null" in payload["query"]:
            return {
                "code": 0,
                "message": "success",
                "data": {"results": None},
            }
        if "rows" in payload["query"]:
            return _search_response(["not-a-dict"])
        return _search_response([_row()])

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": fn})

    outputs = await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "queries": ["null results", "bad rows", "healthy"],
        },
    )

    groups = outputs["results"]
    assert groups[0].error and groups[0].results == []
    assert groups[1].error and groups[1].results == []
    assert groups[2].error == "" and len(groups[2].results) == 1


@pytest.mark.asyncio
async def test_parallel_forwards_shared_params():
    payloads = []

    def spy(creds, payload):
        payloads.append(payload)
        return _search_response([])

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": spy})

    await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "queries": ["a", "b"],
            "max_results": 3,
            "domain": "finance",
            "sub_domain": "finance.quote",
            "sub_domain_params": {"type": "stock", "symbol": "NVDA"},
        },
    )

    assert sorted(p["query"] for p in payloads) == ["a", "b"]
    for p in payloads:
        assert p["tag"] == "finance.quote"
        assert p["params"] == {"type": "stock", "symbol": "NVDA"}
        assert p["max_results"] == 3


@pytest.mark.asyncio
async def test_extract_happy_path():
    block = AnySearchExtractBlock()
    _mock_block(
        block,
        {
            "_extract": lambda *a, **k: {
                "code": 0,
                "message": "success",
                "data": {
                    "url": "https://agpt.co",
                    "title": "AutoGPT",
                    "content": "AutoGPT is a platform for building AI agents.",
                },
            }
        },
    )

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"},
    )

    assert outputs["url"] == "https://agpt.co"
    assert outputs["title"] == "AutoGPT"
    assert "AI agents" in outputs["content"]


@pytest.mark.asyncio
async def test_extract_error_raises_block_error():
    block = AnySearchExtractBlock()
    _mock_block(block, {"_extract": _raise(RuntimeError("fetch failed"))})

    with pytest.raises(BlockExecutionError, match="Extract failed"):
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"},
        )


@pytest.mark.asyncio
async def test_extract_url_falls_back_to_input():
    block = AnySearchExtractBlock()
    _mock_block(
        block,
        {
            "_extract": lambda *a, **k: {
                "code": 0,
                "message": "success",
                "data": {},
            }
        },
    )

    outputs = await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "url": "https://example.test",
        },
    )

    assert outputs["url"] == "https://example.test"
    assert outputs["title"] == ""
    assert outputs["content"] == ""


@pytest.mark.asyncio
async def test_extract_malformed_fields_rejected_atomically():
    """A non-str field must fail inside the boundary before any output is
    yielded (no partial url/title emission)."""
    block = AnySearchExtractBlock()
    _mock_block(
        block,
        {
            "_extract": lambda *a, **k: {
                "code": 0,
                "message": "success",
                "data": {"url": "https://agpt.co", "title": "t", "content": ["x"]},
            }
        },
    )

    emitted = []
    with pytest.raises(BlockExecutionError, match="Extract failed"):
        async for name, _value in block.execute(
            {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"},
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        ):
            emitted.append(name)

    assert emitted == []
