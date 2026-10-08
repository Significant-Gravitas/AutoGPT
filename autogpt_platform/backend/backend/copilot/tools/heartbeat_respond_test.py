from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.heartbeat.state import HEARTBEAT_SESSION_KIND
from backend.copilot.model import ChatSession, ChatSessionMetadata
from backend.copilot.tools import heartbeat_respond
from backend.copilot.tools.heartbeat_respond import HeartbeatRespondTool
from backend.copilot.tools.models import ErrorResponse, HeartbeatRespondResponse


def _session(kind: str) -> ChatSession:
    return ChatSession(
        session_id="hb",
        user_id="u",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="automation", kind=kind),
        messages=[],
    )


@pytest.fixture
def record():
    mock = AsyncMock(return_value=True)
    with patch.object(heartbeat_respond, "record_response", mock):
        yield mock


async def test_an_alert_is_recorded_for_the_runner(record):
    result = await HeartbeatRespondTool()._execute(
        "u", _session(HEARTBEAT_SESSION_KIND), notify=True, notification_text=" Hi "
    )
    assert isinstance(result, HeartbeatRespondResponse)
    assert result.notify
    record.assert_awaited_once_with("hb", True, "Hi")


async def test_it_refuses_outside_a_heartbeat(record):
    result = await HeartbeatRespondTool()._execute(
        "u", _session("normal"), notify=True, notification_text="Hi"
    )
    assert isinstance(result, ErrorResponse)
    assert result.error == "not_a_heartbeat"
    record.assert_not_awaited()


async def test_an_alert_needs_text(record):
    result = await HeartbeatRespondTool()._execute(
        "u", _session(HEARTBEAT_SESSION_KIND), notify=True
    )
    assert isinstance(result, ErrorResponse)
    record.assert_not_awaited()


async def test_quiet_is_recorded_too(record):
    result = await HeartbeatRespondTool()._execute(
        "u", _session(HEARTBEAT_SESSION_KIND), notify=False
    )
    assert isinstance(result, HeartbeatRespondResponse)
    record.assert_awaited_once_with("hb", False, "")
