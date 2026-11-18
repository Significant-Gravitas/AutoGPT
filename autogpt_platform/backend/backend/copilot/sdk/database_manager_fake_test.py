"""The SDK tests' database manager fake fails each call at once."""

import time

import httpx
import pytest

from backend.blocks import persistence
from backend.data import db_accessors
from backend.executor.scheduler import SchedulerClient
from backend.util import clients, service

from .conftest import DatabaseManagerFake


@pytest.mark.asyncio
async def test_an_accessor_call_fails_at_once(database_manager: DatabaseManagerFake):
    started = time.monotonic()
    with pytest.raises(httpx.ConnectError):
        await db_accessors.credit_db().spend_credits("user-1", 1)

    assert time.monotonic() - started < 1
    assert database_manager.calls == ["spend_credits"]


@pytest.mark.asyncio
async def test_it_reaches_a_getter_imported_by_name(
    database_manager: DatabaseManagerFake,
):
    with pytest.raises(httpx.ConnectError):
        await persistence.get_database_manager_async_client().get_execution_kv_data(
            user_id="user-1", key="k"
        )

    assert database_manager.calls == ["get_execution_kv_data"]


def test_the_sync_client_fails_at_once_too(database_manager: DatabaseManagerFake):
    with pytest.raises(httpx.ConnectError):
        clients.get_database_manager_client().get_graph_metadata("graph-1")

    assert database_manager.calls == ["get_graph_metadata"]


def test_other_service_clients_are_left_alone():
    assert type(service.get_service_client(SchedulerClient)).__name__ == "DynamicClient"
