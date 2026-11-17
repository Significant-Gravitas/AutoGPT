from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import ClientNotConnectedError

from backend.copilot import db as chat_module
from backend.copilot.model import ChatSession
from backend.copilot.tools.find_session import FindSessionTool
from backend.copilot.tools.message_session import MessageSessionTool
from backend.copilot.tools.models import ErrorResponse, SessionListResponse
from backend.data.db_manager import DatabaseManagerAsyncClient


@pytest.fixture
def rpc(mocker) -> MagicMock:
    """The copilot executor's view: no Prisma connection, only the RPC client."""
    client = MagicMock(spec_set=DatabaseManagerAsyncClient)
    mocker.patch("backend.data.db.is_connected", return_value=False)
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client", return_value=client
    )
    for name in (
        "list_recent_chat_sessions",
        "get_chat_session_metadata",
        "get_chat_session_status",
    ):
        mocker.patch.object(
            chat_module, name, new=AsyncMock(side_effect=ClientNotConnectedError())
        )
    return client


def _session() -> ChatSession:
    now = datetime.now(timezone.utc)
    return ChatSession(
        session_id="caller",
        user_id="owner",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[],
    )


async def test_find_session_reads_through_rpc_without_prisma(rpc: MagicMock):
    rpc.list_recent_chat_sessions = AsyncMock(return_value=[])
    result = await FindSessionTool()._execute("owner", _session())
    assert isinstance(result, SessionListResponse)
    rpc.list_recent_chat_sessions.assert_awaited_once()
    assert rpc.list_recent_chat_sessions.await_args.kwargs["user_id"] == "owner"


async def test_message_session_checks_ownership_through_rpc_without_prisma(
    rpc: MagicMock,
):
    rpc.get_chat_session_metadata = AsyncMock(return_value=None)
    result = await MessageSessionTool()._execute(
        "owner", _session(), session_id="another-session", message="hello"
    )
    assert isinstance(result, ErrorResponse)
    rpc.get_chat_session_metadata.assert_awaited_once_with("another-session")
