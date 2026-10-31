"""Unit tests for AnySearchBlock (single-query search)."""

import pytest
from pydantic import ValidationError

from backend.blocks.anysearch.search import AnySearchBlock
from backend.blocks.anysearch.test_helpers import (
    TEST_CREDENTIALS_INPUT,
    _collect,
    _mock_block,
    _raise,
    _row,
    _search_response,
)
from backend.util.exceptions import BlockExecutionError


@pytest.mark.asyncio
async def test_search_happy_path():
    block = AnySearchBlock()
    _mock_block(
        block,
        {"_search": lambda *a, **k: _search_response([_row()], request_id="req-x")},
    )

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "query": "q", "max_results": 1},
    )

    assert len(outputs["results"]) == 1
    assert outputs["result"].url == "https://agpt.co"
    assert "AutoGPT" in outputs["context"]
    assert outputs["request_id"] == "req-x"


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


@pytest.mark.asyncio
async def test_search_forwards_zone_language_format():
    """zone/language/format pass straight into the /v1/search payload."""
    captured = {}

    def spy(creds, payload):
        captured.update(payload)
        return _search_response([])

    block = AnySearchBlock()
    _mock_block(block, {"_search": spy})

    await _collect(
        block,
        {
            "credentials": TEST_CREDENTIALS_INPUT,
            "query": "q",
            "zone": "cn",
            "language": "zh-CN",
            "format": "markdown",
        },
    )

    assert captured["zone"] == "cn"
    assert captured["language"] == "zh-CN"
    assert captured["format"] == "markdown"


@pytest.mark.asyncio
async def test_search_omits_unset_optional_params():
    captured = {}

    def spy(creds, payload):
        captured.update(payload)
        return _search_response([])

    block = AnySearchBlock()
    _mock_block(block, {"_search": spy})

    await _collect(block, {"credentials": TEST_CREDENTIALS_INPUT, "query": "q"})

    for key in ("tag", "params", "zone", "language", "format"):
        assert key not in captured


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
async def test_search_missing_results_key_raises_block_error():
    """code=0 with data present but no ``results`` key: the envelope
    contract rejects it (the old code silently returned an empty list)."""
    block = AnySearchBlock()
    _mock_block(
        block,
        {
            "_search": lambda *a, **k: {
                "code": 0,
                "message": "success",
                "request_id": "req-miss",
                "data": {},
            }
        },
    )

    with pytest.raises(BlockExecutionError, match="missing data.results.*req-miss"):
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
async def test_search_runs_anonymously_without_credentials():
    """Normal entry path: auth=anonymous and no credential - the block
    still executes and the request carries no Authorization header."""
    captured = {}

    def spy(creds, payload):
        captured["creds"] = creds
        return _search_response([_row()])

    block = AnySearchBlock()
    _mock_block(block, {"_search": spy})
    outputs = await _collect(
        block, {"query": "q", "auth": "anonymous"}, credentials=None
    )
    assert captured["creds"] is None
    assert len(outputs["results"]) == 1
