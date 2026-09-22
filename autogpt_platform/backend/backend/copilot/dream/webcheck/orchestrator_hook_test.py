"""Tests for the dream orchestrator's web-fact-check hook.

Guards the contract the dream orchestrator depends on:

* disabled or no-tool → cheap no-op, zero backend calls.
* a single raising backend lookup does not poison the rest of the
  batch.
* the returned map is keyed by ``fact.uuid`` so the orchestrator can
  join results back without re-matching on text.
"""

from __future__ import annotations

import pytest

from backend.copilot.dream.fetch import FactRow

from .orchestrator_hook import verify_stale_candidates
from .tool import WebFactCheckResult, WebFactCheckTool


def _fact(uuid: str, text: str = "some claim") -> FactRow:
    return FactRow(
        uuid=uuid,
        source=None,
        target=None,
        name=None,
        fact=text,
        scope="real:global",
        confidence=0.7,
        status="active",
        created_at="2026-05-01T00:00:00Z",
    )


class _RecordingBackend:
    def __init__(self):
        self.calls: list[str] = []

    async def verify(self, fact_text: str) -> WebFactCheckResult:
        self.calls.append(fact_text)
        return WebFactCheckResult(
            fact_text=fact_text,
            verified=True,
            contradicted=False,
            sources=["https://example.com/ok"],
            confidence=0.81,
        )


class _MixedBackend:
    """Raises on the candidate whose fact text starts with 'bad:'."""

    async def verify(self, fact_text: str) -> WebFactCheckResult:
        if fact_text.startswith("bad:"):
            raise RuntimeError("simulated lookup failure")
        return WebFactCheckResult(
            fact_text=fact_text,
            verified=True,
            contradicted=False,
            sources=["https://example.com/ok"],
            confidence=0.75,
        )


@pytest.mark.asyncio
async def test_disabled_flag_short_circuits_and_skips_tool():
    backend = _RecordingBackend()
    tool = WebFactCheckTool(backend=backend)

    result = await verify_stale_candidates(
        [_fact("u1"), _fact("u2")], enabled=False, tool=tool
    )

    assert result == {}
    assert backend.calls == []


@pytest.mark.asyncio
async def test_missing_tool_returns_empty_dict():
    result = await verify_stale_candidates([_fact("u1")], enabled=True, tool=None)
    assert result == {}


@pytest.mark.asyncio
async def test_mixed_batch_one_raise_other_results_land():
    tool = WebFactCheckTool(backend=_MixedBackend())
    candidates = [
        _fact("good-1", "good: alpha"),
        _fact("bad-1", "bad: beta"),
        _fact("good-2", "good: gamma"),
    ]

    result = await verify_stale_candidates(candidates, enabled=True, tool=tool)

    assert set(result.keys()) == {"good-1", "bad-1", "good-2"}
    assert result["good-1"].verified is True
    assert result["good-1"].error is None
    assert result["good-2"].verified is True
    assert result["good-2"].error is None
    # The tool wrapper catches the backend's RuntimeError and surfaces it.
    assert result["bad-1"].verified is False
    assert result["bad-1"].error == "backend_error:RuntimeError"


@pytest.mark.asyncio
async def test_result_map_is_keyed_by_fact_uuid():
    tool = WebFactCheckTool(backend=_RecordingBackend())
    candidates = [
        _fact("uuid-alpha", "alpha claim"),
        _fact("uuid-beta", "beta claim"),
    ]

    result = await verify_stale_candidates(candidates, enabled=True, tool=tool)

    assert set(result.keys()) == {"uuid-alpha", "uuid-beta"}
    assert result["uuid-alpha"].fact_text == "alpha claim"
    assert result["uuid-beta"].fact_text == "beta claim"
