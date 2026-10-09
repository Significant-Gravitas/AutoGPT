"""Worker calls must use RPC while local Prisma is disconnected."""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.errors import ClientNotConnectedError

from backend.copilot.model import ChatSession
from backend.copilot.tools.find_session import FindSessionTool
from backend.copilot.tools.message_session import MessageSessionTool
from backend.copilot.tools.models import ErrorResponse, SessionListResponse
from backend.data.db_manager import DatabaseManager, DatabaseManagerAsyncClient


@pytest.fixture
def session() -> ChatSession:
    now = datetime.now(timezone.utc)
    return ChatSession(
        session_id="caller",
        user_id="owner",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[],
    )


async def test_find_sessions_routes_through_database_manager_without_prisma(
    session: ChatSession,
):
    client = MagicMock(spec=DatabaseManagerAsyncClient)
    client.list_recent_chat_sessions = AsyncMock(return_value=[])
    with (
        patch("backend.data.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ),
        patch(
            "backend.copilot.db.list_recent_chat_sessions",
            AsyncMock(side_effect=ClientNotConnectedError()),
        ),
    ):
        result = await FindSessionTool()._execute("owner", session)
    assert isinstance(result, SessionListResponse)
    client.list_recent_chat_sessions.assert_awaited_once_with(
        user_id="owner", expert_id=None, status=None, limit=50, skip=0
    )


async def test_message_session_ownership_check_routes_through_database_manager(
    session: ChatSession,
):
    client = MagicMock(spec=DatabaseManagerAsyncClient)
    client.get_chat_session_metadata = AsyncMock(return_value=None)
    with (
        patch("backend.data.db.is_connected", return_value=False),
        patch(
            "backend.util.clients.get_database_manager_async_client",
            return_value=client,
        ),
        patch(
            "backend.copilot.db.get_chat_session_metadata",
            AsyncMock(side_effect=ClientNotConnectedError()),
        ),
    ):
        result = await MessageSessionTool()._execute(
            "owner", session, session_id="another-session", message="hello"
        )
    assert isinstance(result, ErrorResponse)
    client.get_chat_session_metadata.assert_awaited_once_with("another-session")


@pytest.mark.parametrize(
    "method",
    [
        "list_recent_chat_sessions",
        "get_chat_session_metadata",
        "get_chat_session_status",
        "record_signup_consent",
        "record_marketing_opt_out_by_email",
        "users_share_active_org",
        "save_onboarding_role",
        "queue_onboarding_role",
    ],
)
def test_routed_rpc_request_schemas_are_constructible(method: str):
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(vars(DatabaseManager)[method])
    assert method in vars(DatabaseManagerAsyncClient)
