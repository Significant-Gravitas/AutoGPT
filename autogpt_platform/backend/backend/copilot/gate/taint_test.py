"""Redis is a cache for taint, so an outage must read as tainted, never clean."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.gate import taint
from backend.copilot.model import ChatSession

_REDIS = "backend.copilot.gate.taint.get_redis_async"


@pytest.fixture
def redis_down():
    with patch(_REDIS, new=AsyncMock(side_effect=ConnectionError("redis down"))):
        yield


async def test_an_outage_reads_as_tainted(redis_down):
    assert await taint.is_tainted(ChatSession.new(user_id="u1", dry_run=False))


async def test_an_outage_reads_as_escalated(redis_down):
    assert await taint.is_escalated("s1", "bash_exec")


async def test_a_clean_session_reads_clean_when_redis_answers():
    """Control: the outage tests pass because of the outage, not the session."""
    redis = AsyncMock()
    redis.get = AsyncMock(return_value=None)
    with patch(_REDIS, new=AsyncMock(return_value=redis)):
        assert not await taint.is_tainted(ChatSession.new(user_id="u1", dry_run=False))
        assert not await taint.is_escalated("s1", "bash_exec")
