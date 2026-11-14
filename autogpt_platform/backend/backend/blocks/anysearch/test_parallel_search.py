"""Unit tests for AnySearchParallelSearchBlock (client-side fan-out)."""

import asyncio

import aiohttp
import pytest
from pydantic import ValidationError

from backend.blocks.anysearch.parallel_search import AnySearchParallelSearchBlock
from backend.blocks.anysearch.test_helpers import (
    TEST_CREDENTIALS_INPUT,
    _collect,
    _mock_block,
    _raise,
    _row,
    _search_response,
)
from backend.util.exceptions import BlockUnknownError
from backend.util.request import HTTPClientError, HTTPServerError


@pytest.mark.asyncio
async def test_parallel_partial_failure_returns_groups():
    """An expected-type failure (envelope/HTTP/transport/timeout) stays
    inside the per-query boundary and becomes an error group."""

    def fn(creds, payload):
        if "fail" in payload["query"]:
            raise ValueError("AnySearch API error: boom")
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


_EXPECTED_EXC = [
    ValueError("AnySearch API error: quota"),
    HTTPClientError("HTTP 429 Error: Too Many Requests, Body: slow down", 429),
    HTTPServerError("HTTP 503 Error: Unavailable, Body: retry later", 503),
    aiohttp.ClientError("connection reset"),
    TimeoutError(),
    asyncio.TimeoutError(),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("exc", _EXPECTED_EXC, ids=type)
async def test_parallel_expected_exceptions_degrade_to_error_group(exc):
    """Every member of the narrowed catch tuple degrades to an error
    group - envelope errors, 4xx, 5xx, transport and timeouts alike."""
    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": _raise(exc)})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["q"]},
    )

    group = outputs["results"][0]
    assert group.results == []
    # bare exceptions with no message fall back to the type name.
    assert group.error == (str(exc) or type(exc).__name__)


@pytest.mark.asyncio
async def test_parallel_http_error_body_is_sanitized():
    """A 4xx body echoing credential material must not reach the
    group error output (HTTP-error text is upstream text too)."""
    exc = HTTPClientError(
        "HTTP 402 Error: Payment Required, Body: registered password p api_key k",
        402,
    )
    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": _raise(exc)})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["q"]},
    )

    error = outputs["results"][0].error
    assert "password" not in error
    assert "api_key" not in error
    assert "details withheld" in error


@pytest.mark.asyncio
async def test_parallel_non_dict_response_degrades_to_error_group():
    """A syntactically valid JSON body that isn't an object must not
    crash the fan-out."""
    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": lambda *a, **k: ["not", "a", "dict"]})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["q"]},
    )

    group = outputs["results"][0]
    assert group.results == []
    assert "malformed" in group.error


@pytest.mark.asyncio
async def test_parallel_unexpected_exception_propagates():
    """The narrowed except no longer swallows arbitrary exceptions -
    an unexpected failure aborts the block (surfaced as a block-level
    unknown error) instead of masquerading as a per-query error."""
    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": _raise(RuntimeError("programmer bug"))})

    with pytest.raises(BlockUnknownError, match="programmer bug"):
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["a"]},
        )


@pytest.mark.asyncio
async def test_parallel_unexpected_failure_drains_sibling_cleanup():
    """cancel() alone only schedules cancellation - the drain makes the
    cancelled sibling's cleanup finish before the block exits."""
    sibling_started = asyncio.Event()
    cleaned: list[str] = []

    async def fake_search_one(creds, payload):
        if payload["query"] == "ok":
            sibling_started.set()
            try:
                await asyncio.sleep(30)
            finally:
                cleaned.append("ok")
        await sibling_started.wait()
        raise RuntimeError("programmer bug")

    block = AnySearchParallelSearchBlock()
    # Async seam - patch directly; _mock_block only wraps sync functions.
    setattr(block, "_search_one", fake_search_one)

    with pytest.raises(BlockUnknownError, match="programmer bug"):
        await _collect(
            block,
            {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["ok", "boom"]},
        )

    # Reached only if the drain waited: without the gather-drain the
    # sibling's finally had not run yet when the block raised.
    assert cleaned == ["ok"]


@pytest.mark.asyncio
async def test_parallel_groups_carry_request_id():
    """Each AnySearchQueryResults group pins the request_id of its own
    HTTP call (per-query request ids, not one shared id)."""

    def fn(creds, payload):
        rid = "rid-a" if payload["query"] == "a" else "rid-b"
        return _search_response([_row()], request_id=rid)

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": fn})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["a", "b"]},
    )

    groups = outputs["results"]
    assert groups[0].request_id == "rid-a"
    assert groups[1].request_id == "rid-b"


@pytest.mark.asyncio
async def test_parallel_error_group_keeps_request_id():
    """A failed query still exposes its server request_id for debugging."""

    def fn(creds, payload):
        if "bad" in payload["query"]:
            return {
                "code": 400,
                "message": "bad request",
                "request_id": "rid-fail",
                "data": {},
            }
        return _search_response([_row()])

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": fn})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["bad", "ok"]},
    )

    groups = outputs["results"]
    assert groups[0].error
    assert "rid-fail" in groups[0].error
    assert groups[0].request_id == "rid-fail"


@pytest.mark.asyncio
async def test_parallel_missing_results_degrade_to_error_group():
    """code=0 without data.results is a response error on that query -
    it degrades to an error group instead of an empty result set."""

    def fn(creds, payload):
        if "missing" in payload["query"]:
            return {
                "code": 0,
                "message": "success",
                "request_id": "rid-1",
                "data": {},
            }
        return _search_response([_row()])

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": fn})

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["missing", "ok"]},
    )

    groups = outputs["results"]
    assert groups[0].error and "missing data.results" in groups[0].error
    assert groups[0].request_id == "rid-1"
    assert groups[1].error == "" and len(groups[1].results) == 1


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
            "zone": "intl",
            "language": "en",
            "format": "json",
        },
    )

    assert sorted(p["query"] for p in payloads) == ["a", "b"]
    for p in payloads:
        assert p["tag"] == "finance.quote"
        assert p["params"] == {"type": "stock", "symbol": "NVDA"}
        assert p["max_results"] == 3
        assert p["zone"] == "intl"
        assert p["language"] == "en"
        assert p["format"] == "json"


@pytest.mark.asyncio
async def test_parallel_omits_unset_optional_params():
    payloads = []

    def spy(creds, payload):
        payloads.append(payload)
        return _search_response([])

    block = AnySearchParallelSearchBlock()
    _mock_block(block, {"_search": spy})

    await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["a"]})

    for p in payloads:
        for key in ("tag", "params", "zone", "language", "format"):
            assert key not in p


@pytest.mark.parametrize(
    "queries",
    [[], ["a", "b", "c", "d", "e", "f"]],
    ids=["empty", "over_five"],
)
def test_parallel_queries_bounds(queries):
    with pytest.raises(ValidationError):
        AnySearchParallelSearchBlock.Input.model_validate(
            {"credentials": TEST_CREDENTIALS_INPUT, "queries": queries}
        )


def test_parallel_zone_and_format_literals():
    for bad in ({"zone": "us"}, {"format": "xml"}):
        with pytest.raises(ValidationError):
            AnySearchParallelSearchBlock.Input.model_validate(
                {"credentials": TEST_CREDENTIALS_INPUT, "queries": ["q"], **bad}
            )


@pytest.mark.asyncio
async def test_parallel_runs_anonymously_without_credentials():
    block = AnySearchParallelSearchBlock()
    captured = {}

    def spy(creds, payload):
        captured.setdefault("creds", []).append(creds)
        return _search_response([])

    _mock_block(block, {"_search": spy})
    outputs = await _collect(
        block,
        {"queries": ["a", "b"], "auth": "anonymous"},
        credentials=None,
    )
    assert captured["creds"] == [None, None]
    assert len(outputs["results"]) == 2
