from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.api.features.integrations import router as router_module
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.credential_refresh import (
    REFRESH_PENDING,
    REFRESH_REVISION,
    CredentialRefreshRequired,
)
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
    persist.assert_awaited_once_with(
        "user-a", [github, updated], reconnect_revisions={"existing": None}
    )


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


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [None, "provider", "username", "winner", "pending", "ordinary", "scopes"]
)
async def test_authenticated_reconnect_checks_fresh_identity_and_clears_pending(
    mocker, change
):
    store = IntegrationCredentialsStore()
    expected = _codex_credentials("existing").model_copy(
        update={
            "provider": "linear",
            "username": "alice",
            "scopes": ["read"],
            "metadata": {REFRESH_PENDING: "old", REFRESH_REVISION: "old:pending"},
            "access_token": SecretStr(""),
            "refresh_token": None,
            "provider_state": None,
        }
    )
    updated = expected.model_copy(
        update={
            "access_token": SecretStr("synthetic-new-access"),
            "refresh_token": SecretStr("synthetic-new-refresh"),
            "metadata": {REFRESH_REVISION: "new"},
        }
    )
    current = expected
    if change == "winner":
        current = updated.model_copy(update={"metadata": {REFRESH_REVISION: "winner"}})
    elif change in {"provider", "username"}:
        updated = updated.model_copy(update={change: "different"})
    elif change == "pending":
        updated = updated.model_copy(update={"metadata": expected.metadata})
    elif change == "scopes":
        updated = updated.model_copy(update={"scopes": []})
    mocker.patch.object(
        store, "_get_persisted_user_creds_unlocked", AsyncMock(return_value=[current])
    )
    mocker.patch.object(
        store, "locked_user_integrations", AsyncMock(return_value=_NoopLockContext())
    )
    persist = mocker.patch.object(store, "_set_user_integration_creds", AsyncMock())
    if change is None:
        await store.update_creds("user-a", updated, reconnect_expected=expected)
        persist.assert_awaited_once_with(
            "user-a", [updated], reconnect_revisions={"existing": "old:pending"}
        )
    else:
        with pytest.raises((ValueError, CredentialRefreshRequired)):
            await store.update_creds(
                "user-a",
                updated,
                reconnect_expected=None if change == "ordinary" else expected,
            )
        persist.assert_not_awaited()


@pytest.mark.asyncio
async def test_oauth_reconnect_route_preserves_user_metadata_and_original_response(
    mocker,
):
    existing = _codex_credentials("existing", title="My connection").model_copy(
        update={
            "provider": "linear",
            "username": "alice",
            "access_token": SecretStr(""),
            "refresh_token": None,
            "metadata": {
                REFRESH_PENDING: "pending",
                REFRESH_REVISION: "pending:old",
                "custom": "keep",
            },
        }
    )
    fresh = _codex_credentials("fresh").model_copy(
        update={
            "provider": "linear",
            "username": "alice",
            "metadata": {"fresh": "keep"},
        }
    )
    snapshot = fresh.model_copy(deep=True)
    manager = SimpleNamespace(
        store=SimpleNamespace(get_creds_by_id=AsyncMock(return_value=existing)),
        update=AsyncMock(),
    )
    mocker.patch.object(router_module, "creds_manager", manager)
    merged = await router_module._upgrade_existing_credential(
        "user-a", "existing", fresh
    )
    assert fresh == snapshot
    assert merged.id == "existing" and merged.title == "My connection"
    assert REFRESH_PENDING not in merged.metadata
    assert merged.metadata[REFRESH_REVISION] != "pending:old"
    assert merged.metadata["custom"] == merged.metadata["fresh"] == "keep"
    assert merged.refresh_token == fresh.refresh_token
    manager.update.assert_awaited_once_with(
        "user-a", merged, reconnect_expected=existing
    )
