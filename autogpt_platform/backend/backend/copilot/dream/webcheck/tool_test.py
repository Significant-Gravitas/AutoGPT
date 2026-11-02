"""Tests for the web-fact-check tool shape.

The actual search backend choice is deferred (see ``dream/p0-spec.md``
§P0.5 open item #3), so these tests only cover the protocol contract
the orchestrator will rely on:

* default (no backend) returns the documented null-backend error;
* the tool propagates a happy-path verified result;
* backend exceptions become an ``error=...`` field, not a raise.
"""

from __future__ import annotations

import pytest

from .tool import (
    SearchBackend,
    WebFactCheckResult,
    WebFactCheckTool,
    _NullSearchBackend,
)


class _StubVerifiedBackend:
    async def verify(self, fact_text: str) -> WebFactCheckResult:
        return WebFactCheckResult(
            fact_text=fact_text,
            verified=True,
            contradicted=False,
            sources=["https://example.com/source-a", "https://example.com/source-b"],
            confidence=0.92,
            error=None,
        )


class _StubRaisingBackend:
    async def verify(self, fact_text: str) -> WebFactCheckResult:
        raise RuntimeError("backend exploded")


@pytest.mark.asyncio
async def test_null_backend_returns_no_backend_configured_error():
    backend = _NullSearchBackend()
    result = await backend.verify("the sky is blue")

    assert isinstance(result, WebFactCheckResult)
    assert result.fact_text == "the sky is blue"
    assert result.verified is False
    assert result.contradicted is False
    assert result.sources == []
    assert result.confidence == 0.0
    assert result.error == "no_search_backend_configured"


@pytest.mark.asyncio
async def test_tool_with_default_backend_returns_result_shape():
    tool = WebFactCheckTool()

    result = await tool.verify("Python is a programming language")

    assert isinstance(result, WebFactCheckResult)
    assert result.fact_text == "Python is a programming language"
    assert result.error == "no_search_backend_configured"


@pytest.mark.asyncio
async def test_tool_with_verified_backend_propagates_sources_and_flag():
    backend: SearchBackend = _StubVerifiedBackend()
    tool = WebFactCheckTool(backend=backend)

    result = await tool.verify("Earth orbits the Sun")

    assert result.verified is True
    assert result.contradicted is False
    assert result.sources == [
        "https://example.com/source-a",
        "https://example.com/source-b",
    ]
    assert result.confidence == pytest.approx(0.92)
    assert result.error is None


@pytest.mark.asyncio
async def test_tool_with_raising_backend_returns_error_not_raise():
    backend: SearchBackend = _StubRaisingBackend()
    tool = WebFactCheckTool(backend=backend)

    result = await tool.verify("anything")

    assert result.verified is False
    assert result.contradicted is False
    assert result.sources == []
    assert result.confidence == 0.0
    assert result.error == "backend_error:RuntimeError"
