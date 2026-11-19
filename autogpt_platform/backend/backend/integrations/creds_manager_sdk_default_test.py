"""A refused SDK default credential never reaches a block.

``<provider>-default`` holds the operator's own key and is appended to every
user's credentials. Run-time lookups refuse it unless its provider is listed in
``RUNNABLE_SDK_DEFAULT_PROVIDERS``.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials
from backend.integrations.creds_manager import IntegrationCredentialsManager

REFUSED = APIKeyCredentials(
    id="linear-default",
    provider="linear",
    api_key=SecretStr("platform-key"),
    title="Linear API Key",
)
RUNNABLE = APIKeyCredentials(
    id="tavily-default",
    provider="tavily",
    api_key=SecretStr("platform-key"),
    title="Tavily API Key",
)
REFUSAL = "platform's shared Linear key"


@asynccontextmanager
async def _noop_lock_context():
    yield


@pytest.fixture
def manager(mocker) -> IntegrationCredentialsManager:
    manager = IntegrationCredentialsManager()
    by_id = {c.id: c for c in (REFUSED, RUNNABLE)}
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(side_effect=lambda _user, cred_id: by_id.get(cred_id)),
    )
    mocker.patch.object(
        manager.store, "get_all_creds", AsyncMock(return_value=[REFUSED, RUNNABLE])
    )
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    mocker.patch.object(manager, "_acquire_lock", AsyncMock(return_value=MagicMock()))
    return manager


@pytest.mark.asyncio
async def test_get_refuses_sdk_default(manager: IntegrationCredentialsManager):
    with pytest.raises(ValueError, match=REFUSAL):
        await manager.get("user-a", REFUSED.id)


@pytest.mark.asyncio
async def test_acquire_refuses_sdk_default(manager: IntegrationCredentialsManager):
    with pytest.raises(ValueError, match=REFUSAL):
        await manager.acquire("user-a", REFUSED.id)


@pytest.mark.asyncio
async def test_cached_getter_refuses_sdk_default(
    manager: IntegrationCredentialsManager,
):
    get_credentials = manager.cached_getter("user-a")
    with pytest.raises(ValueError, match=REFUSAL):
        await get_credentials(REFUSED.id)


@pytest.mark.asyncio
async def test_opted_in_provider_still_resolves(manager: IntegrationCredentialsManager):
    assert await manager.get("user-a", RUNNABLE.id) == RUNNABLE
    credentials, _lock = await manager.acquire("user-a", RUNNABLE.id)
    assert credentials == RUNNABLE
    assert await manager.cached_getter("user-a")(RUNNABLE.id) == RUNNABLE


@pytest.mark.asyncio
async def test_webhook_housekeeping_can_still_load_the_key(
    manager: IntegrationCredentialsManager,
):
    """A webhook registered on the key earlier must still be deregistered."""
    assert await manager.get("user-a", REFUSED.id, allow_sdk_default=True) == REFUSED
    get_credentials = manager.cached_getter("user-a", allow_sdk_default=True)
    assert await get_credentials(REFUSED.id) == REFUSED
