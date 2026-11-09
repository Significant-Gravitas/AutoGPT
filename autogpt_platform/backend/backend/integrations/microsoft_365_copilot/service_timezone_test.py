from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.errors import ClientNotConnectedError

from backend.data import user as user_module
from backend.data.db_manager import DatabaseManagerAsyncClient
from backend.integrations.microsoft_365_copilot import service


@pytest.mark.asyncio
async def test_timezone_uses_rpc_when_prisma_is_disconnected(mocker) -> None:
    client = MagicMock(spec=DatabaseManagerAsyncClient)
    client.get_user_by_id = AsyncMock(
        return_value=SimpleNamespace(timezone="America/Chicago")
    )
    mocker.patch("backend.data.db.is_connected", return_value=False)
    mocker.patch(
        "backend.util.clients.get_database_manager_async_client", return_value=client
    )
    redis = MagicMock()
    redis.get.return_value = None
    mocker.patch("backend.util.cache._get_redis", return_value=redis)
    local_query = mocker.patch.object(
        type(user_module.prisma.user),
        "find_unique",
        new=AsyncMock(side_effect=ClientNotConnectedError()),
    )

    assert (
        await service._get_timezone("worker-timezone-regression") == "America/Chicago"
    )
    client.get_user_by_id.assert_awaited_once_with("worker-timezone-regression")
    local_query.assert_not_awaited()


@pytest.mark.asyncio
async def test_timezone_without_user_needs_no_database(mocker) -> None:
    rpc = mocker.patch("backend.util.clients.get_database_manager_async_client")

    assert await service._get_timezone(None) == "UTC"
    rpc.assert_not_called()
