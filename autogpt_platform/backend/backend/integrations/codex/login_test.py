import asyncio
import base64
import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import SecretStr

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import CodexAuthBundleV1, CodexAuthTokensV1
from backend.integrations.codex.login import (
    CodexDeviceLoginState,
    CodexLoginCoordinator,
    CodexLoginPendingError,
    CodexLoginStatus,
    CodexLoginTransport,
    CodexSharedLoginState,
    RedisCodexLoginStateStore,
)
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexDeviceCodeDetails,
    CodexLoginCompletion,
    CodexRateLimitsSnapshot,
)


class FakeSession:
    def __init__(self) -> None:
        self.details = CodexDeviceCodeDetails(
            login_id="login-123",
            verification_url="https://auth.openai.com/codex/device",
            user_code="ABCD-EFGH",
        )
        self.result = asyncio.get_running_loop().create_future()
        self.cancel = AsyncMock()
        self.close = AsyncMock()

    async def wait(self) -> CodexLoginCompletion:
        return await self.result


class MemoryStateStore:
    def __init__(self) -> None:
        self.active: dict[str, str] = {}
        self.states: dict[tuple[str, str], CodexSharedLoginState] = {}
        self.lock = asyncio.Lock()
        self.refresh_count = 0

    @asynccontextmanager
    async def locked(self, _user_id: str) -> AsyncIterator[None]:
        async with self.lock:
            yield

    async def claim(self, state: CodexSharedLoginState, login_id: str) -> bool:
        if state.user_id in self.active:
            return False
        self.active[state.user_id] = login_id
        await self.write(state, login_id)
        return True

    async def get(self, user_id: str, login_id: str) -> CodexSharedLoginState | None:
        state = self.states.get((user_id, login_id))
        return state.model_copy(deep=True) if state else None

    async def write(self, state: CodexSharedLoginState, login_id: str) -> None:
        self.states[(state.user_id, login_id)] = state.model_copy(deep=True)

    async def release_active(self, user_id: str, login_id: str) -> None:
        if self.active.get(user_id) == login_id:
            del self.active[user_id]

    async def refresh_active(self, user_id: str, login_id: str) -> bool:
        if self.active.get(user_id) != login_id:
            return False
        self.refresh_count += 1
        return True

    async def delete(self, user_id: str, login_id: str) -> None:
        self.states.pop((user_id, login_id), None)


class FakeRedis:
    def __init__(self) -> None:
        self.values: dict[str, str] = {}
        self.expirations: dict[str, int] = {}

    async def set(
        self,
        key: str,
        value: str,
        *,
        nx: bool = False,
        ex: int | None = None,
    ) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        if ex is not None:
            self.expirations[key] = ex
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def delete(self, key: str) -> None:
        self.values.pop(key, None)

    async def expire(self, key: str, seconds: int) -> bool:
        if key not in self.values:
            return False
        self.expirations[key] = seconds
        return True

    async def eval(
        self,
        script: str,
        _numkeys: int,
        key: str,
        expected: str,
        *arguments: object,
    ) -> int:
        if self.values.get(key) != expected:
            return 0
        if "redis.call('del'" in script:
            self.values.pop(key, None)
            self.expirations.pop(key, None)
            return 1
        if "redis.call('expire'" in script:
            self.expirations[key] = int(arguments[0])
            return 1
        raise AssertionError("Unexpected Redis script")


@pytest.mark.asyncio
async def test_start_reserves_distributed_slot_before_requesting_device_code():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    transport_started = asyncio.Event()
    allow_transport = asyncio.Event()

    async def start_transport() -> FakeSession:
        transport_started.set()
        await allow_transport.wait()
        return session

    first_transport = MagicMock()
    first_transport.start_device_login = AsyncMock(side_effect=start_transport)
    losing_transport = MagicMock()
    losing_transport.start_device_login = AsyncMock()
    origin = CodexLoginCoordinator(
        transport_factory=lambda: first_transport,
        state_store=state_store,
        credentials_manager=MagicMock(),
        state_poll_interval_seconds=0.001,
    )
    other_replica = CodexLoginCoordinator(
        transport_factory=lambda: losing_transport,
        state_store=state_store,
        credentials_manager=MagicMock(),
    )

    start_task = asyncio.create_task(origin.start(user_id))
    await asyncio.wait_for(transport_started.wait(), timeout=1)

    with pytest.raises(CodexLoginPendingError):
        await other_replica.start(user_id)
    losing_transport.start_device_login.assert_not_awaited()

    allow_transport.set()
    details = await start_task
    assert details.login_id != session.details.login_id
    await origin.shutdown()


@pytest.mark.asyncio
async def test_failed_transport_start_rolls_back_provisional_claim():
    user_id = "user-123"
    state_store = MemoryStateStore()
    failing_transport = MagicMock()
    failing_transport.start_device_login = AsyncMock(
        side_effect=RuntimeError("startup failed")
    )
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: failing_transport,
        state_store=state_store,
        credentials_manager=MagicMock(),
        state_poll_interval_seconds=0.001,
    )

    with pytest.raises(RuntimeError, match="startup failed"):
        await coordinator.start(user_id)

    assert user_id not in state_store.active
    assert not state_store.states


@pytest.mark.asyncio
async def test_shutdown_cancels_in_progress_transport_start_and_rolls_back_claim():
    user_id = "user-123"
    state_store = MemoryStateStore()
    transport_started = asyncio.Event()
    transport_canceled = asyncio.Event()

    async def start_transport() -> FakeSession:
        transport_started.set()
        try:
            await asyncio.Future()
        finally:
            transport_canceled.set()
        raise AssertionError("unreachable")

    transport = MagicMock()
    transport.start_device_login = AsyncMock(side_effect=start_transport)
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: transport,
        state_store=state_store,
        credentials_manager=MagicMock(),
        state_poll_interval_seconds=0.001,
    )

    start_task = asyncio.create_task(coordinator.start(user_id))
    await asyncio.wait_for(transport_started.wait(), timeout=1)
    await coordinator.shutdown()

    assert start_task.cancelled()
    assert transport_canceled.is_set()
    assert user_id not in state_store.active
    assert not state_store.states


@pytest.mark.asyncio
async def test_completion_persists_before_cross_replica_callback():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    credentials = _credentials()
    manager = MagicMock()
    manager.locked_provider_credentials = MagicMock(
        side_effect=lambda *_args: _provider_locks()
    )
    manager.upsert_single_provider_locked = AsyncMock(return_value=credentials)
    manager.store.get_creds_by_id = AsyncMock(return_value=credentials)
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
    )

    details = await coordinator.start(user_id)
    session.result.set_result(_completion())
    await _wait_for_status(coordinator, user_id, details.login_id, "completed")
    await _wait_for_close(session)
    await _wait_for_active_release(state_store, user_id)

    manager.upsert_single_provider_locked.assert_awaited_once()
    second_replica = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
    )
    restored = await second_replica.complete(user_id, details.login_id)

    assert restored.id == credentials.id
    assert await state_store.get(user_id, details.login_id) is None
    assert session.close.await_count == 1


@pytest.mark.asyncio
async def test_cross_replica_cancel_prevents_origin_from_persisting():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    manager = MagicMock()
    manager.locked_provider_credentials = MagicMock(
        side_effect=lambda *_args: _provider_locks()
    )
    manager.upsert_single_provider_locked = AsyncMock(return_value=_credentials())
    origin = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
        state_poll_interval_seconds=0.001,
    )
    other_replica = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
    )

    details = await origin.start(user_id)
    assert await other_replica.cancel(user_id, details.login_id)
    assert state_store.active[user_id] == details.login_id
    await _wait_for_status(origin, user_id, details.login_id, "canceled")
    await _wait_for_close(session)
    await _wait_for_active_release(state_store, user_id)

    assert user_id not in state_store.active
    session.cancel.assert_awaited()
    manager.upsert_single_provider_locked.assert_not_awaited()


@pytest.mark.asyncio
async def test_same_replica_cancel_promptly_releases_slot_after_owner_cleanup():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=MagicMock(),
        state_poll_interval_seconds=0.001,
    )

    details = await coordinator.start(user_id)
    assert await coordinator.cancel(user_id, details.login_id)
    await _wait_for_close(session)
    await _wait_for_active_release(state_store, user_id)

    assert user_id not in state_store.active
    session.cancel.assert_awaited_once()


@pytest.mark.asyncio
async def test_shutdown_cancels_local_sessions_and_releases_active_slots():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=MagicMock(),
        state_poll_interval_seconds=0.001,
    )

    details = await coordinator.start(user_id)
    await coordinator.shutdown()

    state = await coordinator.get(user_id, details.login_id)
    assert state is not None
    assert state.status == "canceled"
    assert state.error == "ChatGPT sign-in was interrupted by service shutdown"
    assert user_id not in state_store.active
    session.cancel.assert_awaited_once()
    session.close.assert_awaited_once()


@pytest.mark.asyncio
async def test_cancel_during_provider_lock_wait_prevents_stale_persistence():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    provider_lock_entered = asyncio.Event()
    release_provider_lock = asyncio.Event()
    manager = MagicMock()

    @asynccontextmanager
    async def blocked_provider_locks() -> AsyncIterator[None]:
        provider_lock_entered.set()
        await release_provider_lock.wait()
        yield

    manager.locked_provider_credentials = MagicMock(
        side_effect=lambda *_args: blocked_provider_locks()
    )
    manager.upsert_single_provider_locked = AsyncMock(return_value=_credentials())
    origin = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
        state_poll_interval_seconds=0.001,
    )
    other_replica = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
    )

    details = await origin.start(user_id)
    session.result.set_result(_completion())
    await asyncio.wait_for(provider_lock_entered.wait(), timeout=1)

    assert await asyncio.wait_for(
        other_replica.cancel(user_id, details.login_id), timeout=1
    )
    await _wait_for_close(session)
    await _wait_for_active_release(state_store, user_id)
    release_provider_lock.set()

    manager.upsert_single_provider_locked.assert_not_awaited()
    assert user_id not in state_store.active


@pytest.mark.asyncio
async def test_expired_owner_cannot_persist_after_a_new_login_claims_slot():
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    manager = MagicMock()
    manager.locked_provider_credentials = MagicMock(
        side_effect=lambda *_args: _provider_locks()
    )
    manager.upsert_single_provider_locked = AsyncMock(return_value=_credentials())
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
        state_poll_interval_seconds=0.001,
    )

    details = await coordinator.start(user_id)
    state_store.active[user_id] = "new-owner-login"
    session.result.set_result(_completion())
    state = await _wait_for_status(coordinator, user_id, details.login_id, "failed")
    await _wait_for_close(session)
    await coordinator.shutdown()

    manager.upsert_single_provider_locked.assert_not_awaited()
    assert state_store.active[user_id] == "new-owner-login"
    assert state.error == "ChatGPT sign-in ownership was lost. Try again."


@pytest.mark.asyncio
async def test_redis_state_never_contains_device_code_or_verification_url():
    redis = FakeRedis()
    store = RedisCodexLoginStateStore(ttl_seconds=1200, owner_lease_seconds=30)
    state = CodexSharedLoginState(user_id="user-123", status="pending")
    with patch(
        "backend.integrations.codex.login.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        assert await store.claim(state, "login-123")

    serialized = json.dumps(redis.values)
    assert "ABCD-EFGH" not in serialized
    assert "auth.openai.com" not in serialized
    assert "access_token" not in serialized
    assert sorted(redis.expirations.values()) == [30, 1200]

    with patch(
        "backend.integrations.codex.login.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        assert await store.refresh_active("user-123", "login-123")
    assert sorted(redis.expirations.values()) == [30, 1200]


@pytest.mark.asyncio
async def test_stale_login_cannot_release_or_refresh_replacement_owner():
    user_id = "user-123"
    redis = FakeRedis()
    store = RedisCodexLoginStateStore(ttl_seconds=1200, owner_lease_seconds=30)

    with patch(
        "backend.integrations.codex.login.get_redis_async",
        new=AsyncMock(return_value=redis),
    ):
        original = CodexSharedLoginState(user_id=user_id, status="pending")
        assert await store.claim(original, "old-login")
        key = next(key for key in redis.values if key.endswith(":active"))
        redis.values[key] = "replacement-login"
        redis.expirations[key] = 17

        await store.release_active(user_id, "old-login")
        assert redis.values[key] == "replacement-login"
        assert not await store.refresh_active(user_id, "old-login")
        assert redis.expirations[key] == 17


@pytest.mark.asyncio
async def test_runtime_error_message_cannot_leak_into_state_or_logs(caplog):
    user_id = "user-123"
    session = FakeSession()
    state_store = MemoryStateStore()
    manager = MagicMock()
    coordinator = CodexLoginCoordinator(
        transport_factory=lambda: _transport(session),
        state_store=state_store,
        credentials_manager=manager,
    )

    details = await coordinator.start(user_id)
    session.result.set_exception(RuntimeError("access-secret refresh-secret"))
    state = await _wait_for_status(coordinator, user_id, details.login_id, "failed")
    await _wait_for_close(session)
    await _wait_for_active_release(state_store, user_id)

    assert state.error == "ChatGPT sign-in failed. Try again."
    assert "access-secret" not in caplog.text
    assert "refresh-secret" not in caplog.text


def _transport(session: FakeSession) -> CodexLoginTransport:
    transport = MagicMock()
    transport.start_device_login = AsyncMock(return_value=session)
    return cast(CodexLoginTransport, transport)


@asynccontextmanager
async def _provider_locks() -> AsyncIterator[None]:
    yield


async def _wait_for_status(
    coordinator: CodexLoginCoordinator,
    user_id: str,
    login_id: str,
    expected: CodexLoginStatus,
) -> CodexDeviceLoginState:
    for _ in range(20):
        state = await coordinator.get(user_id, login_id)
        if state and state.status == expected:
            return state
        await asyncio.sleep(0)
    raise AssertionError(f"Codex login did not reach {expected}")


async def _wait_for_close(session: FakeSession) -> None:
    for _ in range(20):
        if session.close.await_count:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Codex login session did not close")


async def _wait_for_active_release(state_store: MemoryStateStore, user_id: str) -> None:
    for _ in range(20):
        if user_id not in state_store.active:
            return
        await asyncio.sleep(0.001)
    raise AssertionError("Codex login slot was not released")


def _credentials() -> OAuth2Credentials:
    return OAuth2Credentials(
        id="codex-credential",
        provider="codex",
        title="ChatGPT for Codex",
        username="user@example.com",
        access_token=SecretStr("access-secret"),
        refresh_token=SecretStr("refresh-secret"),
        scopes=[],
        refresh_strategy="provider_runtime",
        provider_state=SecretStr("provider-secret"),
        provider_state_version=1,
    )


def _completion() -> CodexLoginCompletion:
    return CodexLoginCompletion(
        bundle=CodexAuthBundleV1(
            tokens=CodexAuthTokensV1(
                id_token=SecretStr(_jwt({"email": "user@example.com"})),
                access_token=SecretStr(_jwt({"exp": 4_000_000_000})),
                refresh_token=SecretStr("refresh-secret"),
            ),
            codex_runtime_version="0.144.4",
        ),
        account=CodexAccountSnapshot(
            connected=True,
            requires_openai_auth=False,
            account_type="chatgpt",
            email="user@example.com",
            plan_type="plus",
        ),
        rate_limits=CodexRateLimitsSnapshot(plan_type="plus"),
    )


def test_login_completion_allows_bundle_without_optional_enrichment():
    completion = CodexLoginCompletion(bundle=_completion().bundle)

    assert completion.account is None
    assert completion.rate_limits is None


def _jwt(payload: dict[str, object]) -> str:
    header = _base64url({"alg": "none"})
    body = _base64url(payload)
    return f"{header}.{body}.signature"


def _base64url(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(encoded).decode().rstrip("=")
