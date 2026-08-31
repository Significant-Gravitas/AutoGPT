"""Tests for credential hooks and provider-runtime credential leases."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.integrations.creds_manager import (
    IntegrationCredentialsManager,
    _invoke_creds_changed_hook,
    register_creds_changed_hook,
    unregister_creds_changed_hook,
)


@pytest.fixture(autouse=True)
def _reset_hook():
    """Ensure global hook state is clean before and after every test."""
    unregister_creds_changed_hook()
    yield
    unregister_creds_changed_hook()


class TestRegisterCredsChangedHook:
    def test_register_and_invoke(self):
        calls: list[tuple[str, str]] = []
        register_creds_changed_hook(lambda u, p: calls.append((u, p)))

        _invoke_creds_changed_hook("user-1", "github")
        assert calls == [("user-1", "github")]

    def test_double_register_raises(self):
        register_creds_changed_hook(lambda u, p: None)
        with pytest.raises(RuntimeError, match="already registered"):
            register_creds_changed_hook(lambda u, p: None)

    def test_unregister_then_reregister(self):
        register_creds_changed_hook(lambda u, p: None)
        unregister_creds_changed_hook()
        # Should not raise after unregister.
        register_creds_changed_hook(lambda u, p: None)


class TestInvokeCredsChangedHook:
    def test_noop_when_no_hook_registered(self):
        # Must not raise even when no hook is registered.
        _invoke_creds_changed_hook("user-1", "github")

    def test_hook_exception_is_swallowed(self):
        def bad_hook(user_id: str, provider: str) -> None:
            raise ValueError("boom")

        register_creds_changed_hook(bad_hook)
        # Must not propagate the exception.
        _invoke_creds_changed_hook("user-1", "github")

    def test_hook_receives_correct_args(self):
        calls: list[tuple[str, str]] = []
        register_creds_changed_hook(lambda u, p: calls.append((u, p)))

        _invoke_creds_changed_hook("user-a", "github")
        _invoke_creds_changed_hook("user-b", "slack")

        assert calls == [("user-a", "github"), ("user-b", "slack")]


@pytest.mark.asyncio
async def test_provider_runtime_refresh_never_resolves_generic_handler(mocker):
    manager = IntegrationCredentialsManager()
    resolve_handler = mocker.patch.object(
        manager,
        "_get_oauth_handler",
        new_callable=AsyncMock,
    )
    credentials = _provider_runtime_credentials()

    locked = await manager.refresh_if_needed("user-a", credentials, lock=True)
    unlocked = await manager.refresh_if_needed("user-a", credentials, lock=False)

    assert locked is credentials
    assert unlocked is credentials
    resolve_handler.assert_not_awaited()


@pytest.mark.asyncio
async def test_acquire_refreshes_before_taking_the_long_lived_lock(mocker):
    manager = IntegrationCredentialsManager()
    credentials = _provider_runtime_credentials()
    lock = _owned_lock()
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    acquire_lock = mocker.patch.object(
        manager, "_acquire_lock", AsyncMock(return_value=lock)
    )
    get = AsyncMock(return_value=credentials)
    mocker.patch.object(manager, "get", get)
    stored = mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=credentials),
    )

    acquired, acquired_lock = await manager.acquire("user-a", "cred-id")

    assert acquired is credentials
    assert acquired_lock is lock
    get.assert_awaited_once_with("user-a", "cred-id")
    stored.assert_awaited_once_with("user-a", "cred-id")
    acquire_lock.assert_awaited_once()


@pytest.mark.asyncio
async def test_acquire_releases_lock_if_credential_disappears_after_refresh(mocker):
    manager = IntegrationCredentialsManager()
    lock = _owned_lock()
    mocker.patch.object(
        manager, "get", AsyncMock(return_value=_provider_runtime_credentials())
    )
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    mocker.patch.object(manager, "_acquire_lock", AsyncMock(return_value=lock))
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=None),
    )

    with pytest.raises(ValueError):
        await manager.acquire("user-a", "cred-id")

    lock.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_locked_refresh_reloads_after_waiting_for_rotating_provider(mocker):
    manager = IntegrationCredentialsManager()
    stale = _provider_runtime_credentials().model_copy(
        update={"refresh_strategy": "oauth_handler"}
    )
    fresh = stale.model_copy(update={"access_token_expires_at": 4_000_000_000})
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=fresh),
    )
    handler = MagicMock()
    handler.needs_refresh.return_value = False
    handler.refresh_tokens = AsyncMock()
    acquire_lock = mocker.patch.object(manager, "_acquire_lock", AsyncMock())

    result = await manager._refresh_locked("user-a", stale, handler=handler)

    assert result is fresh
    handler.needs_refresh.assert_called_once_with(fresh)
    handler.refresh_tokens.assert_not_awaited()
    acquire_lock.assert_not_awaited()


@pytest.mark.asyncio
async def test_locked_refresh_reports_a_credential_type_change(mocker):
    manager = IntegrationCredentialsManager()
    stale = _provider_runtime_credentials().model_copy(
        update={"refresh_strategy": "oauth_handler"}
    )
    changed = APIKeyCredentials(
        id=stale.id,
        provider=stale.provider,
        title=stale.title,
        api_key=SecretStr("replacement"),
    )
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=changed),
    )

    with pytest.raises(TypeError, match="changed type to 'api_key'"):
        await manager._refresh_locked("user-a", stale)


@pytest.mark.asyncio
async def test_update_acquired_requires_matching_owned_lock(mocker):
    manager = IntegrationCredentialsManager()
    update = mocker.patch.object(manager.store, "update_creds", new_callable=AsyncMock)
    current = _provider_runtime_credentials()
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=current),
    )
    lock = _owned_lock()

    await manager.update_acquired("user-a", current, lock)
    update.assert_awaited_once_with("user-a", current)

    lock.name = str(("user:user-a", "credentials:different"))
    with pytest.raises(RuntimeError, match="owned lock"):
        await manager.update_acquired("user-a", current, lock)

    lock.name = str(("user:user-a", "credentials:cred-id"))
    lock.owned.return_value = False
    with pytest.raises(RuntimeError, match="owned lock"):
        await manager.update_acquired("user-a", current, lock)


@pytest.mark.asyncio
async def test_update_acquired_rejects_provider_change(mocker):
    manager = IntegrationCredentialsManager()
    current = _provider_runtime_credentials()
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=current),
    )

    with pytest.raises(RuntimeError, match="identity"):
        await manager.update_acquired(
            "user-a",
            current.model_copy(update={"provider": "other"}),
            _owned_lock(),
        )


@pytest.mark.asyncio
async def test_update_acquired_allows_codex_to_move_refresh_to_http_handler(mocker):
    manager = IntegrationCredentialsManager()
    current = _provider_runtime_credentials()
    migrated = current.model_copy(update={"refresh_strategy": "oauth_handler"})
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=current),
    )
    update = mocker.patch.object(
        manager.store,
        "update_creds",
        new_callable=AsyncMock,
    )

    await manager.update_acquired("user-a", migrated, _owned_lock())

    update.assert_awaited_once_with("user-a", migrated)


@pytest.mark.asyncio
async def test_delete_acquired_deletes_while_lease_lock_is_owned(mocker):
    manager = IntegrationCredentialsManager()
    credentials = _provider_runtime_credentials()
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(return_value=credentials),
    )
    delete = mocker.patch.object(
        manager.store,
        "delete_creds_by_id",
        new_callable=AsyncMock,
    )
    lock = _owned_lock()

    await manager.delete_acquired("user-a", credentials, lock)

    delete.assert_awaited_once_with("user-a", credentials.id)
    lock.release.assert_not_awaited()


@pytest.mark.asyncio
async def test_upsert_single_provider_notifies_credential_listeners(mocker):
    manager = IntegrationCredentialsManager()
    credentials = _provider_runtime_credentials()
    mocker.patch.object(
        manager.store,
        "get_creds_by_provider",
        AsyncMock(return_value=[]),
    )
    upsert = mocker.patch.object(
        manager.store,
        "upsert_single_provider_creds",
        AsyncMock(return_value=credentials),
    )
    changes: list[tuple[str, str]] = []
    register_creds_changed_hook(lambda user, provider: changes.append((user, provider)))

    stored = await manager.upsert_single_provider("user-a", credentials)

    assert stored is credentials
    upsert.assert_awaited_once_with("user-a", credentials)
    assert changes == [("user-a", "codex")]


@pytest.mark.asyncio
async def test_upsert_single_provider_waits_for_existing_credential_lease(mocker):
    manager = IntegrationCredentialsManager()
    existing = _provider_runtime_credentials()
    replacement = existing.model_copy(update={"access_token": SecretStr("new-access")})
    mocker.patch.object(
        manager.store,
        "get_creds_by_provider",
        AsyncMock(return_value=[existing]),
    )
    lock = _owned_lock()
    acquire = mocker.patch.object(
        manager,
        "_acquire_lock",
        AsyncMock(return_value=lock),
    )

    async def upsert_while_locked(_user_id, credentials):
        lock.release.assert_not_awaited()
        return credentials

    upsert = mocker.patch.object(
        manager.store,
        "upsert_single_provider_creds",
        AsyncMock(side_effect=upsert_while_locked),
    )

    stored = await manager.upsert_single_provider("user-a", replacement)

    assert stored is replacement
    acquire.assert_awaited_once_with("user-a", existing.id)
    upsert.assert_awaited_once_with("user-a", replacement)
    lock.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_locked_provider_credentials_can_guard_state_recheck_and_upsert(mocker):
    manager = IntegrationCredentialsManager()
    existing = _provider_runtime_credentials()
    replacement = existing.model_copy(update={"title": "reconnected"})
    mocker.patch.object(
        manager.store,
        "get_creds_by_provider",
        AsyncMock(return_value=[existing]),
    )
    lock = _owned_lock()
    mocker.patch.object(manager, "_acquire_lock", AsyncMock(return_value=lock))
    upsert = mocker.patch.object(
        manager.store,
        "upsert_single_provider_creds",
        AsyncMock(return_value=replacement),
    )

    async with manager.locked_provider_credentials("user-a", "codex"):
        lock.release.assert_not_awaited()
        stored = await manager.upsert_single_provider_locked("user-a", replacement)
        lock.release.assert_not_awaited()

    assert stored is replacement
    upsert.assert_awaited_once_with("user-a", replacement)
    lock.release.assert_awaited_once()


class _NoopLockContext:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, _exc_type, _exc, _traceback) -> None:
        return None


def _noop_lock_context() -> _NoopLockContext:
    return _NoopLockContext()


def _owned_lock() -> AsyncMock:
    lock = AsyncMock()
    lock.name = str(("user:user-a", "credentials:cred-id"))
    lock.locked.return_value = True
    lock.owned.return_value = True
    lock.timeout = 60
    lock.extend.return_value = True
    return lock


def _provider_runtime_credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="cred-id",
        provider="codex",
        access_token=SecretStr("access"),
        access_token_expires_at=1,
        refresh_token=SecretStr("refresh"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr("state"),
        provider_state_version=1,
    )
