"""A credentials getter loads the user's store once, however many lookups it
serves and however many of them run at the same time."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.creds_manager import IntegrationCredentialsManager

STORED = APIKeyCredentials(
    id="user-tavily-key", provider="tavily", api_key=SecretStr("key"), title="Tavily"
)


@pytest.mark.asyncio
async def test_concurrent_lookups_share_one_load(mocker):
    manager = IntegrationCredentialsManager()
    release = asyncio.Event()

    async def slow_load(_user_id: str) -> list[APIKeyCredentials]:
        await release.wait()
        return [STORED]

    load = mocker.patch.object(
        manager.store, "get_all_creds", AsyncMock(side_effect=slow_load)
    )
    get_credentials = manager.cached_getter("user-a")
    ids = [STORED.id] + [f"unknown-{i}" for i in range(20)]

    lookups = asyncio.gather(*(get_credentials(i) for i in ids))
    await asyncio.sleep(0)
    release.set()
    found = await asyncio.wait_for(lookups, timeout=10)

    load.assert_awaited_once_with("user-a")
    assert found == [STORED] + [None] * 20


@pytest.mark.asyncio
async def test_an_empty_store_is_not_reloaded(mocker):
    manager = IntegrationCredentialsManager()
    load = mocker.patch.object(
        manager.store, "get_all_creds", AsyncMock(return_value=[])
    )
    get_credentials = manager.cached_getter("user-a")

    assert await get_credentials("a") is None
    assert await get_credentials("b") is None
    load.assert_awaited_once()


@pytest.mark.asyncio
async def test_a_duplicate_id_resolves_to_the_first_match(mocker):
    """The same choice `IntegrationCredentialsStore.get_creds_by_id` makes."""
    manager = IntegrationCredentialsManager()
    later = STORED.model_copy(update={"title": "Later"})
    mocker.patch.object(
        manager.store, "get_all_creds", AsyncMock(return_value=[STORED, later])
    )

    assert await manager.cached_getter("user-a")(STORED.id) is STORED
