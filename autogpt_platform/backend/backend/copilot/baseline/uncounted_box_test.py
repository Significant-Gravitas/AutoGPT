"""A stream closed before the turn's try/finally is reached must not leave
the session's box running; an expert's shared box is left to its timeout."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.baseline import service


@pytest.mark.asyncio
async def test_session_box_is_paused_when_the_stream_is_closed_early():
    sandbox = MagicMock()
    with patch.object(service, "pause_sandbox_direct", AsyncMock()) as pause:
        task = service._pause_uncounted_box(sandbox, "sess-1", None)
        assert task is not None
        await task
    pause.assert_awaited_once_with(sandbox, "sess-1")


@pytest.mark.asyncio
async def test_expert_box_is_left_running_because_its_turn_was_never_counted():
    with patch.object(service, "pause_sandbox_direct", AsyncMock()) as pause:
        assert service._pause_uncounted_box(MagicMock(), "sess-1", "exp-1") is None
        await asyncio.sleep(0)
    pause.assert_not_awaited()
