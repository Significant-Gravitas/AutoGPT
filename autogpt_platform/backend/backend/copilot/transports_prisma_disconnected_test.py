"""The default chat route, read from a process with no Prisma connection.

The scheduler and the copilot executor reach the database only through the
DatabaseManager RPC client. ``resolve_default_chat_route`` swallows any error
and falls back to ``platform``, so a lookup made in-process quietly ignores
the user's saved ChatGPT default on every scheduled turn.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_mock
from prisma.errors import ClientNotConnectedError

from backend.copilot import transports
from backend.copilot.transports import resolve_default_chat_route
from backend.copilot.transports_test import USER_ID, _codex_credentials
from backend.util.settings import BehaveAs


@pytest.fixture
def scheduler_db(mocker: pytest_mock.MockerFixture):
    """Prisma disconnected in-process, the DatabaseManager client answering."""
    mocker.patch.object(transports.settings.config, "behave_as", BehaveAs.CLOUD)
    mocker.patch.object(
        transports,
        "has_codex_access_for_discovery",
        new=AsyncMock(return_value=True),
    )
    mocker.patch.object(
        transports.credentials_manager.store,
        "get_creds_by_provider",
        new=AsyncMock(
            side_effect=lambda _user, provider: (
                [_codex_credentials("cred-codex")] if provider == "codex" else []
            )
        ),
    )
    mocker.patch("backend.data.db.is_connected", return_value=False)
    mocker.patch(
        "backend.data.user.get_user_by_id",
        new=AsyncMock(side_effect=ClientNotConnectedError()),
    )

    db_manager = MagicMock()
    db_manager.get_user_default_chat_route = AsyncMock(
        return_value=("codex", "cred-codex")
    )
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client",
        return_value=db_manager,
    )
    return db_manager


@pytest.mark.asyncio
async def test_scheduled_turn_uses_the_saved_default_without_prisma(scheduler_db):
    assert await resolve_default_chat_route(USER_ID) == ("codex", "cred-codex")
    scheduler_db.get_user_default_chat_route.assert_awaited_once_with(USER_ID)
