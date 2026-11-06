"""Unit tests for AnySearchExtractBlock (single-URL extraction)."""

import pytest

from backend.blocks.anysearch.extract import AnySearchExtractBlock
from backend.blocks.anysearch.test_helpers import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    _collect,
    _mock_block,
    _raise,
)
from backend.data.execution import ExecutionContext
from backend.util.exceptions import BlockExecutionError


def _extract_response(request_id: str = "req-e", **data) -> dict:
    return {
        "code": 0,
        "message": "success",
        "request_id": request_id,
        "data": data,
    }


@pytest.mark.asyncio
async def test_extract_happy_path():
    block = AnySearchExtractBlock()
    _mock_block(
        block,
        {
            "_extract": lambda *a, **k: _extract_response(
                url="https://agpt.co",
                title="AutoGPT",
                content="AutoGPT is a platform for building AI agents.",
            )
        },
    )

    outputs = await _collect(
        block,
        {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"},
    )

    assert outputs["url"] == "https://agpt.co"
    assert outputs["title"] == "AutoGPT"
    assert "AI agents" in outputs["content"]
    assert outputs["request_id"] == "req-e"


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
    _mock_block(block, {"_extract": lambda *a, **k: _extract_response()})

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
            "_extract": lambda *a, **k: _extract_response(
                url="https://agpt.co", title="t", content=["x"]
            )
        },
    )

    emitted = []
    with pytest.raises(BlockExecutionError, match="Extract failed"):
        async for name, _value in block.execute(
            {"credentials": TEST_CREDENTIALS_INPUT, "url": "https://agpt.co"},
            credentials=TEST_CREDENTIALS,
            execution_context=ExecutionContext(),
        ):
            emitted.append(name)  # pragma: no cover - unreached on raise

    assert emitted == []


@pytest.mark.asyncio
async def test_extract_runs_anonymously_without_credentials():
    block = AnySearchExtractBlock()
    captured = {}

    def spy(creds, url):
        captured["creds"] = creds
        return _extract_response(url=url, title="t", content="c")

    _mock_block(block, {"_extract": spy})
    outputs = await _collect(
        block,
        {"url": "https://x.test", "auth": "anonymous"},
        credentials=None,
    )
    assert captured["creds"] is None
    assert outputs["url"] == "https://x.test"
