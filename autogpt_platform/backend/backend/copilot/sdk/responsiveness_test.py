"""Unit tests for the latency-reduction helpers introduced in SECRT-1912.

Covers ``_fetch_graphiti_context`` (extracted to module level so all four
parallel I/O paths in ``stream_chat_completion_sdk`` are independently
testable) and the adaptive thinking ``effort`` resolution that depends on
the per-turn ``mode`` argument.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.model import ChatMessage, ChatSession

from .service import _fetch_graphiti_context


def _make_session(user_id: str, *, n_messages: int = 0) -> ChatSession:
    """Build a minimal ChatSession with a controllable message count."""
    session = ChatSession.new(user_id, dry_run=False)
    for i in range(n_messages):
        session.messages.append(ChatMessage(role="user", content=f"prior message {i}"))
    return session


# ---------------------------------------------------------------------------
# _fetch_graphiti_context
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_fetch_graphiti_context_returns_disabled_when_flag_off():
    """When LD flag is off, return (False, "") without fetching warm context."""
    session = _make_session("user-1")
    with patch(
        "backend.copilot.sdk.service.is_enabled_for_user",
        new=AsyncMock(return_value=False),
    ):
        enabled, ctx = await _fetch_graphiti_context("user-1", session, "hello")
    assert enabled is False
    assert ctx == ""


@pytest.mark.asyncio
async def test_fetch_graphiti_context_skips_warm_for_anonymous_users():
    """Anonymous turns (no user_id) skip warm context even when enabled."""
    session = _make_session("user-1")
    with patch(
        "backend.copilot.sdk.service.is_enabled_for_user",
        new=AsyncMock(return_value=True),
    ):
        enabled, ctx = await _fetch_graphiti_context(None, session, "hello")
    assert enabled is True
    assert ctx == ""


@pytest.mark.asyncio
async def test_fetch_graphiti_context_skips_warm_on_followup_turns():
    """Sessions with prior history skip warm context — it's a turn-1 only feature."""
    session = _make_session("user-1", n_messages=3)
    with patch(
        "backend.copilot.sdk.service.is_enabled_for_user",
        new=AsyncMock(return_value=True),
    ):
        enabled, ctx = await _fetch_graphiti_context("user-1", session, "follow-up")
    assert enabled is True
    assert ctx == ""


@pytest.mark.asyncio
async def test_fetch_graphiti_context_loads_warm_context_on_first_turn():
    """Turn 1 (≤1 prior message) with an authenticated user loads warm context."""
    session = _make_session("user-1")
    with (
        patch(
            "backend.copilot.sdk.service.is_enabled_for_user",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.graphiti.context.fetch_warm_context",
            new=AsyncMock(return_value="<warm_facts>important fact</warm_facts>"),
        ),
    ):
        enabled, ctx = await _fetch_graphiti_context("user-1", session, "first prompt")
    assert enabled is True
    assert ctx == "<warm_facts>important fact</warm_facts>"


@pytest.mark.asyncio
async def test_fetch_graphiti_context_handles_empty_warm_context():
    """``fetch_warm_context`` returning ``None`` collapses to empty string."""
    session = _make_session("user-1")
    with (
        patch(
            "backend.copilot.sdk.service.is_enabled_for_user",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.graphiti.context.fetch_warm_context",
            new=AsyncMock(return_value=None),
        ),
    ):
        enabled, ctx = await _fetch_graphiti_context("user-1", session, "first prompt")
    assert enabled is True
    assert ctx == ""


@pytest.mark.asyncio
async def test_fetch_graphiti_context_handles_none_message():
    """A None message (e.g. resume turn with no fresh user input) is normalised
    to an empty search string before hitting ``fetch_warm_context``."""
    session = _make_session("user-1")
    fetch_mock = AsyncMock(return_value="ctx")
    with (
        patch(
            "backend.copilot.sdk.service.is_enabled_for_user",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.graphiti.context.fetch_warm_context",
            new=fetch_mock,
        ),
    ):
        enabled, ctx = await _fetch_graphiti_context("user-1", session, None)
    assert enabled is True
    assert ctx == "ctx"
    fetch_mock.assert_awaited_once_with("user-1", "")


# ---------------------------------------------------------------------------
# Adaptive ``effort`` resolution (mirrors the inline expression in
# ``stream_chat_completion_sdk``).  A regression here means the SDK either
# loses the fast-mode override or stops honouring the operator's static
# ``CHAT_CLAUDE_AGENT_THINKING_EFFORT`` override.
# ---------------------------------------------------------------------------


def _resolve_effort(mode: str | None, configured: str | None) -> str:
    """Mirror the inline ``effort=`` expression so the branching is unit-tested
    without needing to construct ClaudeAgentOptions.  Keep this in sync with
    [service.py:stream_chat_completion_sdk] when the formula changes."""
    return "medium" if mode == "fast" else (configured or "high")


class TestAdaptiveEffortResolution:
    """Verify the ``effort`` value chosen for each mode/config combo.

    This tests the formula used at the ``ClaudeAgentOptions`` call site:
        ``"medium" if mode == "fast" else (config_value or "high")``
    """

    def test_fast_mode_pins_to_medium_regardless_of_config(self):
        """fast mode always uses 'medium' — the config override is ignored."""
        assert _resolve_effort("fast", configured=None) == "medium"
        assert _resolve_effort("fast", configured="low") == "medium"
        assert _resolve_effort("fast", configured="max") == "medium"

    def test_extended_thinking_mode_honours_config_override(self):
        """extended_thinking respects ``claude_agent_thinking_effort`` when set."""
        assert _resolve_effort("extended_thinking", configured="low") == "low"
        assert _resolve_effort("extended_thinking", configured="max") == "max"

    def test_extended_thinking_defaults_to_high_when_unconfigured(self):
        """No config override → 'high' (the responsive default for SDK turns)."""
        assert _resolve_effort("extended_thinking", configured=None) == "high"

    def test_default_mode_treats_none_as_extended_thinking(self):
        """``mode=None`` (server default) follows the same path as extended_thinking."""
        assert _resolve_effort(None, configured=None) == "high"
        assert _resolve_effort(None, configured="medium") == "medium"
