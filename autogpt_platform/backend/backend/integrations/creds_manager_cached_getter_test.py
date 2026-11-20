"""A credentials getter loads the user's store once, however many lookups it
serves and however many of them run at the same time."""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.creds_manager import (
    _GETTER_REFRESH_CONCURRENCY,
    IntegrationCredentialsManager,
)
from backend.integrations.oauth.refresh_failure import (
    RECONNECT_REQUIRED_KEY,
    CredentialsNeedReconnectError,
)

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


@pytest.mark.asyncio
async def test_a_due_credential_marked_for_reconnect_is_refused_from_the_one_load(
    mocker,
):
    manager = IntegrationCredentialsManager()
    dead = [_due_oauth(f"oauth-{i}", reconnect_required=True) for i in range(5)]
    load = mocker.patch.object(
        manager.store, "get_all_creds", AsyncMock(return_value=dead)
    )
    _stub_refresh_machinery(mocker, manager)
    get_credentials = manager.cached_getter("user-a")

    results = await asyncio.gather(
        *(get_credentials(c.id) for c in dead), return_exceptions=True
    )

    assert all(isinstance(r, CredentialsNeedReconnectError) for r in results)
    load.assert_awaited_once_with("user-a")


@pytest.mark.asyncio
async def test_due_credentials_refresh_a_few_at_a_time(mocker):
    manager = IntegrationCredentialsManager()
    due = [_due_oauth(f"oauth-{i}") for i in range(10)]
    mocker.patch.object(manager.store, "get_all_creds", AsyncMock(return_value=due))
    _stub_refresh_machinery(mocker, manager)
    running = peak = 0
    release = asyncio.Event()

    async def refresh(_user_id, credential, lock=True):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        await release.wait()
        running -= 1
        return credential

    mocker.patch.object(manager, "refresh_if_needed", refresh)
    get_credentials = manager.cached_getter("user-a")
    lookups = asyncio.gather(*(get_credentials(c.id) for c in due))
    for _ in range(50):
        await asyncio.sleep(0)
    release.set()
    found = await asyncio.wait_for(lookups, timeout=10)

    assert peak == _GETTER_REFRESH_CONCURRENCY
    assert found == due


def _due_oauth(
    credential_id: str, *, reconnect_required: bool = False
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=credential_id,
        provider="github",
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        access_token_expires_at=1,
        scopes=[],
        metadata=(
            {RECONNECT_REQUIRED_KEY: {"error_code": "invalid_grant"}}
            if reconnect_required
            else {}
        ),
    )


def _stub_refresh_machinery(mocker, manager: IntegrationCredentialsManager) -> None:
    """A provider handler that calls every token due, and locks that need no Redis."""
    due = MagicMock(**{"needs_refresh.return_value": True})
    mocker.patch.object(manager, "_get_oauth_handler", AsyncMock(return_value=due))
    mocker.patch.object(manager, "_locked", lambda *_, **__: contextlib.nullcontext())
    mocker.patch.object(manager, "_acquire_lock", AsyncMock(return_value=AsyncMock()))
