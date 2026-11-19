"""The SDK tests' database manager fake fails each call at once."""

import httpx
import pytest

from backend.blocks import persistence
from backend.copilot.database_manager_fixtures import DatabaseManagerFake
from backend.data import db, db_accessors
from backend.executor.scheduler import SchedulerClient
from backend.util import clients, service


@pytest.mark.asyncio
async def test_an_accessor_call_fails_at_once(
    database_manager: DatabaseManagerFake, monkeypatch: pytest.MonkeyPatch
):
    # Without a connected Prisma client the accessor picks the database manager.
    monkeypatch.setattr(db, "is_connected", lambda: False)
    with pytest.raises(httpx.ConnectError):
        await db_accessors.credit_db().spend_credits("user-1", 1)

    # Not the last entry: tasks an earlier test left running add calls too.
    assert "spend_credits" in database_manager.calls


@pytest.mark.asyncio
async def test_it_reaches_a_getter_imported_by_name(
    database_manager: DatabaseManagerFake,
):
    with pytest.raises(httpx.ConnectError):
        await persistence.get_database_manager_async_client().get_execution_kv_data(
            user_id="user-1", key="k"
        )

    # Not the last entry: tasks an earlier test left running add calls too.
    assert "get_execution_kv_data" in database_manager.calls


def test_the_sync_client_fails_at_once_too(database_manager: DatabaseManagerFake):
    with pytest.raises(httpx.ConnectError):
        clients.get_database_manager_client().get_graph_metadata("graph-1")

    # Not the last entry: tasks an earlier test left running add calls too.
    assert "get_graph_metadata" in database_manager.calls


def test_other_service_clients_are_left_alone():
    assert type(service.get_service_client(SchedulerClient)).__name__ == "DynamicClient"
