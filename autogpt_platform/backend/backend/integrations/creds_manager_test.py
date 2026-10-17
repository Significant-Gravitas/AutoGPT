"""Tests for credential hooks and provider-runtime credential leases."""

import time
from functools import partial
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.blocks.airtable._oauth import AirtableOAuthHandler
from backend.blocks.linear._oauth import LinearOAuthHandler
from backend.blocks.mcp.oauth import MCPOAuthHandler
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.data.user import get_near_expiry_oauth_credentials
from backend.integrations.creds_manager import (
    IntegrationCredentialsManager,
    _invoke_creds_changed_hook,
    register_creds_changed_hook,
    unregister_creds_changed_hook,
)
from backend.integrations.oauth.github import GitHubOAuthHandler
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.integrations.oauth.notion import NotionOAuthHandler
from backend.integrations.oauth.todoist import TodoistOAuthHandler
from backend.integrations.oauth.twitter import TwitterOAuthHandler
from backend.util.encryption import JSONCryptor


@pytest.fixture(autouse=True)
def _reset_hook():
    """Ensure global hook state is clean before and after every test."""
    unregister_creds_changed_hook()
    # The broadcast half is covered in creds_events_test; keep these tests off Redis.
    with patch(
        "backend.integrations.creds_manager.publish_creds_changed",
        new_callable=AsyncMock,
    ):
        yield
    unregister_creds_changed_hook()


class TestRegisterCredsChangedHook:
    @pytest.mark.asyncio
    async def test_register_and_invoke(self):
        calls: list[tuple[str, str]] = []
        register_creds_changed_hook(lambda u, p: calls.append((u, p)))

        await _invoke_creds_changed_hook("user-1", "github")
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
    @pytest.mark.asyncio
    async def test_noop_when_no_hook_registered(self):
        # Must not raise even when no hook is registered.
        await _invoke_creds_changed_hook("user-1", "github")

    @pytest.mark.asyncio
    async def test_hook_exception_is_swallowed(self):
        def bad_hook(user_id: str, provider: str) -> None:
            raise ValueError("boom")

        register_creds_changed_hook(bad_hook)
        # Must not propagate the exception.
        await _invoke_creds_changed_hook("user-1", "github")

    @pytest.mark.asyncio
    async def test_broadcast_happens_even_when_the_hook_raises(self):
        register_creds_changed_hook(MagicMock(side_effect=ValueError("boom")))

        with patch(
            "backend.integrations.creds_manager.publish_creds_changed",
            new_callable=AsyncMock,
        ) as publish:
            await _invoke_creds_changed_hook("user-1", "github")

        publish.assert_awaited_once_with("user-1", "github")

    @pytest.mark.asyncio
    async def test_hook_receives_correct_args(self):
        calls: list[tuple[str, str]] = []
        register_creds_changed_hook(lambda u, p: calls.append((u, p)))

        await _invoke_creds_changed_hook("user-a", "github")
        await _invoke_creds_changed_hook("user-b", "slack")

        assert calls == [("user-a", "github"), ("user-b", "slack")]


@pytest.mark.asyncio
async def test_refresh_idle_credentials_refreshes_each_candidate(mocker):
    manager = IntegrationCredentialsManager()
    first = _provider_runtime_credentials().model_copy(
        update={"id": "cred-1", "refresh_strategy": "oauth_handler"}
    )
    second = first.model_copy(update={"id": "cred-2"})
    get_candidates = mocker.patch.object(
        manager.store.db_manager,
        "get_near_expiry_oauth_credentials",
        AsyncMock(return_value=[("user-1", first), ("user-2", second)]),
    )
    refresh = mocker.patch.object(manager, "refresh_if_needed", AsyncMock())

    with patch("backend.integrations.creds_manager.time.time", return_value=1_000):
        await manager.refresh_idle_credentials()

    get_candidates.assert_awaited_once_with(1_000 - 14 * 86_400)
    assert refresh.await_args_list == [
        (("user-1", first), {}),
        (("user-2", second), {}),
    ]


@pytest.mark.asyncio
async def test_refresh_idle_credentials_continues_after_a_failure(mocker, caplog):
    manager = IntegrationCredentialsManager()
    first = _provider_runtime_credentials().model_copy(
        update={"id": "cred-1", "refresh_strategy": "oauth_handler"}
    )
    second = first.model_copy(update={"id": "cred-2"})
    mocker.patch.object(
        manager.store.db_manager,
        "get_near_expiry_oauth_credentials",
        AsyncMock(return_value=[("user-1", first), ("user-2", second)]),
    )
    refresh = mocker.patch.object(
        manager,
        "refresh_if_needed",
        AsyncMock(side_effect=[RuntimeError("provider rejected secret-token"), None]),
    )

    await manager.refresh_idle_credentials()

    assert refresh.await_args_list == [
        (("user-1", first), {}),
        (("user-2", second), {}),
    ]
    assert "secret-token" not in caplog.text


@pytest.mark.asyncio
@pytest.mark.parametrize("idle_days", [15, 30])
@pytest.mark.parametrize("fail_first", [False, True])
async def test_idle_sweep_maintains_unused_credentials_and_recovers_after_failure(
    mocker, idle_days, fail_first
):
    manager = IntegrationCredentialsManager()
    now = 1_750_000_000
    original = _provider_runtime_credentials().model_copy(
        update={
            "provider": "github",
            "refresh_strategy": "oauth_handler",
            "access_token_expires_at": now - idle_days * 86_400,
            "refresh_token_expires_at": now + 30 * 86_400,
        }
    )
    stored = {original.id: original}
    handler = GitHubOAuthHandler("client", "secret", "https://example.com/callback")
    clock = mocker.patch("time.time", return_value=now)
    database = mocker.patch("backend.data.user.prisma")
    database.integrationcredential.find_many = AsyncMock(
        side_effect=lambda **_: [
            MagicMock(
                id=c.id,
                ownerId="user-a",
                encryptedPayload=JSONCryptor().encrypt(c.model_dump()),
            )
            for c in stored.values()
        ]
    )
    mocker.patch.object(
        manager.store.db_manager,
        "get_near_expiry_oauth_credentials",
        AsyncMock(side_effect=get_near_expiry_oauth_credentials),
    )
    mocker.patch.object(manager, "_get_oauth_handler", AsyncMock(return_value=handler))
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    mocker.patch.object(manager, "_acquire_lock", AsyncMock(return_value=_owned_lock()))
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(side_effect=lambda _user, credential_id: stored[credential_id]),
    )

    async def persist(_user, credentials):
        assert _user == "user-a"
        stored[credentials.id] = credentials

    mocker.patch.object(manager.store, "update_creds", AsyncMock(side_effect=persist))
    response = MagicMock()
    response.json.return_value = {
        "access_token": "rotated-access",
        "refresh_token": "rotated-refresh",
        "expires_in": 3_600,
        "refresh_token_expires_in": 60 * 86_400,
        "scope": "repo",
    }
    post = AsyncMock(
        side_effect=([RuntimeError("temporary failure")] if fail_first else [])
        + [response, response]
    )
    mocker.patch(
        "backend.integrations.oauth.github.Requests", return_value=MagicMock(post=post)
    )
    mocker.patch.object(handler, "_request_username", AsyncMock(return_value="user"))

    await manager.refresh_idle_credentials()
    if fail_first:
        assert stored[original.id] == original
        clock.return_value += 86_400
        await manager.refresh_idle_credentials()

    assert post.await_count == 1 + fail_first
    assert post.await_args.kwargs["data"]["refresh_token"] == "refresh"
    assert stored[original.id].access_token.get_secret_value() == "rotated-access"
    assert stored[original.id].refresh_token == SecretStr("rotated-refresh")
    assert stored[original.id].access_token_expires_at == clock.return_value + 3_600

    for _ in range(14):
        clock.return_value += 86_400
        await manager.refresh_idle_credentials()
    assert post.await_count == 1 + fail_first

    clock.return_value += 86_400
    await manager.refresh_idle_credentials()
    assert post.await_count == 2 + fail_first
    assert post.await_args.kwargs["data"]["refresh_token"] == "rotated-refresh"


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
@pytest.mark.parametrize("replacement_exists", [True, False])
async def test_locked_refresh_reloads_after_waiting_for_a_lease(
    mocker, replacement_exists
):
    manager = IntegrationCredentialsManager()
    stale = _provider_runtime_credentials().model_copy(
        update={"refresh_strategy": "oauth_handler", "access_token_expires_at": 1_100}
    )
    fresh = stale.model_copy(update={"access_token_expires_at": 2_000})
    lock = _owned_lock()
    mocker.patch.object(manager, "_locked", return_value=_noop_lock_context())
    mocker.patch.object(manager, "_acquire_lock", AsyncMock(return_value=lock))
    mocker.patch.object(
        manager.store,
        "get_creds_by_id",
        AsyncMock(side_effect=[stale, fresh if replacement_exists else None]),
    )
    update = mocker.patch.object(manager.store, "update_creds", AsyncMock())
    handler = MagicMock()
    handler.needs_refresh.side_effect = lambda c: c.access_token_expires_at < 1_300
    handler.refresh_tokens = AsyncMock(return_value=fresh)

    if replacement_exists:
        assert await manager._refresh_locked("user-a", stale, handler) is fresh
    else:
        with pytest.raises(ValueError, match="not found"):
            await manager._refresh_locked("user-a", stale, handler)

    handler.refresh_tokens.assert_not_awaited()
    update.assert_not_awaited()
    lock.release.assert_awaited_once()


_HANDLER_ARGS = ("client", "secret", "redirect")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "make_handler,rotates",
    [
        pytest.param(partial(LinearOAuthHandler, *_HANDLER_ARGS), True, id="linear"),
        pytest.param(
            partial(AirtableOAuthHandler, *_HANDLER_ARGS), True, id="airtable"
        ),
        pytest.param(
            partial(
                MCPOAuthHandler,
                *_HANDLER_ARGS,
                authorize_url="https://auth.example.com/authorize",
                token_url="https://auth.example.com/token",
            ),
            True,
            id="mcp",
        ),
        pytest.param(partial(GitHubOAuthHandler, *_HANDLER_ARGS), True, id="github"),
        pytest.param(partial(TwitterOAuthHandler, *_HANDLER_ARGS), True, id="twitter"),
        pytest.param(partial(NotionOAuthHandler, *_HANDLER_ARGS), True, id="notion"),
        pytest.param(partial(TodoistOAuthHandler, *_HANDLER_ARGS), True, id="todoist"),
        pytest.param(partial(GoogleOAuthHandler, *_HANDLER_ARGS), False, id="google"),
    ],
)
async def test_copilot_refresh_takes_the_lock_only_for_rotating_providers(
    mocker, make_handler, rotates
):
    manager = IntegrationCredentialsManager()
    credentials = _provider_runtime_credentials().model_copy(
        update={"refresh_strategy": "oauth_handler"}
    )
    handler = make_handler()
    mocker.patch.object(manager, "_get_oauth_handler", AsyncMock(return_value=handler))
    locked = mocker.patch.object(
        manager, "_refresh_locked", AsyncMock(return_value=credentials)
    )
    unlocked = mocker.patch.object(
        manager, "_refresh_unlocked", AsyncMock(return_value=credentials)
    )

    result = await manager.refresh_if_needed("user-a", credentials, lock=False)

    assert result is credentials
    taken, skipped = (locked, unlocked) if rotates else (unlocked, locked)
    taken.assert_awaited_once_with("user-a", credentials, handler=handler)
    skipped.assert_not_awaited()


@pytest.mark.asyncio
async def test_copilot_read_of_an_unexpired_rotating_credential_takes_no_lock(
    mocker,
):
    manager = IntegrationCredentialsManager()
    credentials = _provider_runtime_credentials().model_copy(
        update={
            "refresh_strategy": "oauth_handler",
            "access_token_expires_at": int(time.time()) + 3600,
        }
    )
    handler = GitHubOAuthHandler(*_HANDLER_ARGS)
    mocker.patch.object(manager, "_get_oauth_handler", AsyncMock(return_value=handler))
    locked = mocker.patch.object(manager, "_refresh_locked", AsyncMock())

    result = await manager.refresh_if_needed("user-a", credentials, lock=False)

    assert result is credentials
    locked.assert_not_awaited()


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
