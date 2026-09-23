from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.credentials_store import IntegrationCredentialsStore


def _codex_credentials(
    credential_id: str,
    *,
    title: str | None = None,
    managed: bool = False,
) -> OAuth2Credentials:
    return OAuth2Credentials(
        id=credential_id,
        provider="codex",
        title=title,
        is_managed=managed,
        access_token=SecretStr("access"),
        refresh_token=SecretStr("refresh"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr("state"),
        provider_state_version=1,
    )


@pytest.mark.asyncio
async def test_upsert_single_provider_preserves_id_and_removes_duplicates(mocker):
    store = IntegrationCredentialsStore()
    first = _codex_credentials("existing", title="My account")
    duplicate = _codex_credentials("duplicate", title="Duplicate")
    github = APIKeyCredentials(
        id="github",
        provider="github",
        api_key=SecretStr("key"),
    )
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        AsyncMock(return_value=[github, first, duplicate]),
    )
    persist = mocker.patch.object(
        store,
        "_set_user_integration_creds",
        new_callable=AsyncMock,
    )
    mocker.patch.object(
        store,
        "locked_user_integrations",
        AsyncMock(return_value=_NoopLockContext()),
    )

    updated = await store.upsert_single_provider_creds(
        "user-a",
        _codex_credentials("fresh", title="ChatGPT for Codex"),
    )

    assert updated.id == "existing"
    assert updated.title == "My account"
    persist.assert_awaited_once_with("user-a", [github, updated])


@pytest.mark.asyncio
async def test_upsert_single_provider_never_replaces_managed_credentials(mocker):
    store = IntegrationCredentialsStore()
    managed = _codex_credentials("managed", managed=True)
    mocker.patch.object(
        store,
        "_get_persisted_user_creds_unlocked",
        AsyncMock(return_value=[managed]),
    )
    persist = mocker.patch.object(
        store,
        "_set_user_integration_creds",
        new_callable=AsyncMock,
    )
    mocker.patch.object(
        store,
        "locked_user_integrations",
        AsyncMock(return_value=_NoopLockContext()),
    )

    with pytest.raises(ValueError, match="managed"):
        await store.upsert_single_provider_creds(
            "user-a",
            _codex_credentials("fresh"),
        )

    persist.assert_not_awaited()


class _NoopLockContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        return None
