import asyncio
import hashlib
import logging
import secrets
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from typing import Literal, Protocol, TypeGuard, TypeVar

from autogpt_libs.utils.synchronize import AsyncRedisKeyedMutex
from pydantic import BaseModel, ConfigDict

from backend.data.model import OAuth2Credentials
from backend.data.redis_client import get_redis_async
from backend.integrations.codex.access import enforce_codex_access
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.models import (
    CodexDeviceCodeDetails,
    CodexLoginCompletion,
)
from backend.integrations.creds_manager import IntegrationCredentialsManager

logger = logging.getLogger(__name__)

CodexDeviceLogin = CodexDeviceCodeDetails
CodexLoginStatus = Literal["pending", "completed", "failed", "canceled"]

_COMPARE_DELETE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('del', KEYS[1])
end
return 0
"""
_COMPARE_EXPIRE_SCRIPT = """
if redis.call('get', KEYS[1]) == ARGV[1] then
  return redis.call('expire', KEYS[1], ARGV[2])
end
return 0
"""

_T = TypeVar("_T")


async def _await_redis_result(result: Awaitable[_T] | _T) -> _T:
    if isinstance(result, Awaitable):
        return await result
    return result


class CodexDeviceLoginSession(Protocol):
    details: CodexDeviceCodeDetails

    async def wait(self) -> CodexLoginCompletion: ...

    async def cancel(self) -> None: ...

    async def close(self) -> None: ...


class CodexLoginTransport(Protocol):
    async def start_device_login(self) -> CodexDeviceLoginSession: ...


class CodexLoginPendingError(RuntimeError):
    pass


class CodexLoginFailedError(RuntimeError):
    pass


class CodexDeviceLoginState(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: CodexLoginStatus
    error: str | None = None


class CodexSharedLoginState(CodexDeviceLoginState):
    user_id: str
    credential_id: str | None = None


class CodexLoginStateStore(Protocol):
    def locked(self, user_id: str) -> AbstractAsyncContextManager[None]: ...

    async def claim(self, state: CodexSharedLoginState, login_id: str) -> bool: ...

    async def get_active(self, user_id: str) -> str | None: ...

    async def get(
        self, user_id: str, login_id: str
    ) -> CodexSharedLoginState | None: ...

    async def write(self, state: CodexSharedLoginState, login_id: str) -> None: ...

    async def release_active(self, user_id: str, login_id: str) -> None: ...

    async def refresh_active(self, user_id: str, login_id: str) -> bool: ...

    async def delete(self, user_id: str, login_id: str) -> None: ...


class RedisCodexLoginStateStore:
    def __init__(self, ttl_seconds: int, owner_lease_seconds: int = 30) -> None:
        self._ttl_seconds = ttl_seconds
        self._owner_lease_seconds = owner_lease_seconds

    @asynccontextmanager
    async def locked(self, user_id: str) -> AsyncIterator[None]:
        mutex = AsyncRedisKeyedMutex(await get_redis_async())
        async with mutex.locked(_lock_key(user_id)):
            yield

    async def claim(self, state: CodexSharedLoginState, login_id: str) -> bool:
        redis = await get_redis_async()
        claimed = await redis.set(
            _active_key(state.user_id),
            login_id,
            nx=True,
            ex=self._owner_lease_seconds,
        )
        if not claimed:
            return False
        try:
            await self.write(state, login_id)
        except Exception:
            await _await_redis_result(
                redis.eval(
                    _COMPARE_DELETE_SCRIPT,
                    1,
                    _active_key(state.user_id),
                    login_id,
                )
            )
            raise
        return True

    async def get_active(self, user_id: str) -> str | None:
        redis = await get_redis_async()
        return await redis.get(_active_key(user_id))

    async def get(self, user_id: str, login_id: str) -> CodexSharedLoginState | None:
        redis = await get_redis_async()
        payload = await redis.get(_attempt_key(user_id, login_id))
        if payload is None:
            return None
        state = CodexSharedLoginState.model_validate_json(payload)
        return state if state.user_id == user_id else None

    async def write(self, state: CodexSharedLoginState, login_id: str) -> None:
        redis = await get_redis_async()
        await redis.set(
            _attempt_key(state.user_id, login_id),
            state.model_dump_json(),
            ex=self._ttl_seconds,
        )

    async def release_active(self, user_id: str, login_id: str) -> None:
        redis = await get_redis_async()
        await _await_redis_result(
            redis.eval(
                _COMPARE_DELETE_SCRIPT,
                1,
                _active_key(user_id),
                login_id,
            )
        )

    async def refresh_active(self, user_id: str, login_id: str) -> bool:
        redis = await get_redis_async()
        return bool(
            await _await_redis_result(
                redis.eval(
                    _COMPARE_EXPIRE_SCRIPT,
                    1,
                    _active_key(user_id),
                    login_id,
                    str(self._owner_lease_seconds),
                )
            )
        )

    async def delete(self, user_id: str, login_id: str) -> None:
        redis = await get_redis_async()
        await redis.delete(_attempt_key(user_id, login_id))


class CodexLoginCoordinator:
    def __init__(
        self,
        transport_factory: Callable[[], CodexLoginTransport] | None = None,
        timeout_seconds: float = 900,
        state_store: CodexLoginStateStore | None = None,
        credentials_manager: IntegrationCredentialsManager | None = None,
        state_poll_interval_seconds: float = 1,
    ) -> None:
        self._transport_factory = transport_factory or _get_default_transport
        self._timeout_seconds = timeout_seconds
        self._state_store = state_store or RedisCodexLoginStateStore(
            int(timeout_seconds) + 300
        )
        self._credentials_manager = (
            credentials_manager or IntegrationCredentialsManager()
        )
        self._state_poll_interval_seconds = state_poll_interval_seconds
        self._local_sessions: dict[str, CodexDeviceLoginSession] = {}
        self._local_tasks: dict[str, asyncio.Task[None]] = {}
        self._local_cancel_events: dict[str, asyncio.Event] = {}
        self._local_users: dict[str, str] = {}
        self._startup_tasks: dict[str, asyncio.Task[object]] = {}
        self._startup_users: dict[str, str] = {}

    async def start(self, user_id: str) -> CodexDeviceLogin:
        login_id = secrets.token_urlsafe(24)
        state = CodexSharedLoginState(user_id=user_id, status="pending")
        async with self._state_store.locked(user_id):
            active_login_id = await self._state_store.get_active(user_id)
            if active_login_id is not None:
                active_state = await self._state_store.get(user_id, active_login_id)
                if active_state is not None and active_state.status == "pending":
                    active_state.status = "canceled"
                    active_state.error = (
                        "ChatGPT sign-in was replaced by a newer attempt"
                    )
                    await self._state_store.write(active_state, active_login_id)
                await self._state_store.release_active(user_id, active_login_id)
            if not await self._state_store.claim(state, login_id):
                raise CodexLoginPendingError(
                    "A ChatGPT sign-in is already active for this user"
                )
        current_task = asyncio.current_task()
        if current_task is not None:
            self._startup_tasks[login_id] = current_task
            self._startup_users[login_id] = user_id
        startup_lease = asyncio.create_task(
            self._keep_owner_lease_alive(user_id, login_id)
        )
        session: CodexDeviceLoginSession | None = None
        try:
            session = await self._transport_factory().start_device_login()
            async with self._state_store.locked(user_id):
                current = await self._state_store.get(user_id, login_id)
                owns_login = (
                    current is not None
                    and current.status == "pending"
                    and await self._state_store.refresh_active(user_id, login_id)
                )
            if not owns_login:
                raise CodexLoginFailedError("Codex login ownership was lost")

            self._local_sessions[login_id] = session
            self._local_users[login_id] = user_id
            cancel_event = asyncio.Event()
            self._local_cancel_events[login_id] = cancel_event
            self._local_tasks[login_id] = asyncio.create_task(
                self._wait_for_login(user_id, login_id, session, cancel_event)
            )
            return session.details.model_copy(update={"login_id": login_id})
        except BaseException:
            try:
                if session is not None:
                    await session.close()
            finally:
                async with self._state_store.locked(user_id):
                    await self._state_store.release_active(user_id, login_id)
                    await self._state_store.delete(user_id, login_id)
                self._local_sessions.pop(login_id, None)
                self._local_tasks.pop(login_id, None)
                self._local_cancel_events.pop(login_id, None)
                self._local_users.pop(login_id, None)
            raise
        finally:
            startup_lease.cancel()
            with suppress(asyncio.CancelledError):
                await startup_lease
            self._startup_tasks.pop(login_id, None)
            self._startup_users.pop(login_id, None)

    async def get(self, user_id: str, login_id: str) -> CodexDeviceLoginState | None:
        state = await self._state_store.get(user_id, login_id)
        if state is None:
            return None
        return CodexDeviceLoginState(status=state.status, error=state.error)

    async def complete(self, user_id: str, login_id: str) -> OAuth2Credentials:
        async with self._state_store.locked(user_id):
            state = await self._state_store.get(user_id, login_id)
            if state is None:
                raise CodexLoginFailedError("Codex login not found")
            if state.status == "pending":
                raise CodexLoginPendingError("Codex login is still pending")
            if state.status != "completed" or state.credential_id is None:
                raise CodexLoginFailedError(state.error or "Codex login failed")
            credentials = await self._credentials_manager.store.get_creds_by_id(
                user_id, state.credential_id
            )
            if not _is_codex_credentials(credentials):
                raise CodexLoginFailedError("Stored Codex credential is unavailable")
            await self._state_store.delete(user_id, login_id)
            return credentials.model_copy(deep=True)

    async def cancel(self, user_id: str, login_id: str) -> bool:
        found, should_cancel = await self._mark_canceled(
            user_id, login_id, "ChatGPT sign-in was canceled"
        )
        if not found:
            return False
        session = self._local_sessions.get(login_id)
        if should_cancel and session is not None:
            cancel_event = self._local_cancel_events.get(login_id)
            if cancel_event is not None:
                cancel_event.set()
            await session.cancel()
        return True

    async def shutdown(self) -> None:
        starting_logins = [
            (login_id, user_id, task)
            for login_id, task in self._startup_tasks.items()
            if (user_id := self._startup_users.get(login_id)) is not None
        ]
        for login_id, user_id, task in starting_logins:
            await self._mark_canceled(
                user_id,
                login_id,
                "ChatGPT sign-in was interrupted by service shutdown",
            )
            if task is not asyncio.current_task():
                task.cancel()
        await asyncio.gather(
            *(
                task
                for _, _, task in starting_logins
                if task is not asyncio.current_task()
            ),
            return_exceptions=True,
        )

        local_logins = [
            (login_id, user_id, session)
            for login_id, session in self._local_sessions.items()
            if (user_id := self._local_users.get(login_id)) is not None
        ]
        tasks = [
            task
            for login_id, _, _ in local_logins
            if (task := self._local_tasks.get(login_id)) is not None
        ]

        for login_id, user_id, _ in local_logins:
            await self._mark_canceled(
                user_id,
                login_id,
                "ChatGPT sign-in was interrupted by service shutdown",
            )
            cancel_event = self._local_cancel_events.get(login_id)
            if cancel_event is not None:
                cancel_event.set()

        await asyncio.gather(
            *(session.cancel() for _, _, session in local_logins),
            return_exceptions=True,
        )
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _mark_canceled(
        self,
        user_id: str,
        login_id: str,
        message: str,
    ) -> tuple[bool, bool]:
        should_cancel = False
        async with self._state_store.locked(user_id):
            state = await self._state_store.get(user_id, login_id)
            if state is None:
                return False, False
            if state.status == "pending":
                state.status = "canceled"
                state.error = message
                await self._state_store.write(state, login_id)
                should_cancel = True
            elif state.status == "canceled":
                should_cancel = True
        return True, should_cancel

    async def _wait_for_login(
        self,
        user_id: str,
        login_id: str,
        session: CodexDeviceLoginSession,
        cancel_event: asyncio.Event,
    ) -> None:
        login_task = asyncio.create_task(session.wait())
        state_task = asyncio.create_task(
            self._wait_until_not_pending(user_id, login_id, cancel_event)
        )
        persist_task: asyncio.Task[None] | None = None
        try:
            done, _ = await asyncio.wait(
                {login_task, state_task},
                timeout=self._timeout_seconds,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not done:
                await session.cancel()
                login_task.cancel()
                await self._mark_failed(user_id, login_id, "ChatGPT sign-in timed out")
            elif login_task in done:
                completion = await login_task
                persist_task = asyncio.create_task(
                    self._persist_completed_login(user_id, login_id, completion)
                )
                persisted, _ = await asyncio.wait(
                    {persist_task, state_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                if persist_task in persisted:
                    await persist_task
                else:
                    persist_task.cancel()
            else:
                if not state_task.result():
                    await session.cancel()
                login_task.cancel()
        except asyncio.TimeoutError:
            await session.cancel()
            await self._mark_failed(user_id, login_id, "ChatGPT sign-in timed out")
        except asyncio.CancelledError:
            state = await self._state_store.get(user_id, login_id)
            if state is not None and state.status == "pending":
                with suppress(Exception):
                    await asyncio.shield(session.cancel())
            raise
        except Exception as error:
            logger.warning("Codex device login failed with %s", type(error).__name__)
            await self._mark_failed(
                user_id, login_id, "ChatGPT sign-in failed. Try again."
            )
        finally:
            child_tasks = [login_task, state_task]
            if persist_task is not None:
                child_tasks.append(persist_task)
            for child_task in child_tasks:
                if not child_task.done():
                    child_task.cancel()
            await asyncio.gather(*child_tasks, return_exceptions=True)
            try:
                await session.close()
            finally:
                try:
                    async with self._state_store.locked(user_id):
                        await self._state_store.release_active(user_id, login_id)
                finally:
                    self._local_sessions.pop(login_id, None)
                    self._local_tasks.pop(login_id, None)
                    self._local_cancel_events.pop(login_id, None)
                    self._local_users.pop(login_id, None)

    async def _wait_until_not_pending(
        self,
        user_id: str,
        login_id: str,
        cancel_event: asyncio.Event,
    ) -> bool:
        while True:
            if cancel_event.is_set():
                return True
            async with self._state_store.locked(user_id):
                state = await self._state_store.get(user_id, login_id)
                if state is None or state.status != "pending":
                    return False
                if not await self._state_store.refresh_active(user_id, login_id):
                    state.status = "failed"
                    state.error = "ChatGPT sign-in ownership was lost. Try again."
                    await self._state_store.write(state, login_id)
                    return False
            try:
                await asyncio.wait_for(
                    cancel_event.wait(),
                    timeout=self._state_poll_interval_seconds,
                )
            except asyncio.TimeoutError:
                pass

    async def _keep_owner_lease_alive(self, user_id: str, login_id: str) -> None:
        while True:
            await asyncio.sleep(self._state_poll_interval_seconds)
            async with self._state_store.locked(user_id):
                state = await self._state_store.get(user_id, login_id)
                if state is None or state.status != "pending":
                    return
                if not await self._state_store.refresh_active(user_id, login_id):
                    state.status = "failed"
                    state.error = "ChatGPT sign-in ownership was lost. Try again."
                    await self._state_store.write(state, login_id)
                    return

    async def _persist_completed_login(
        self,
        user_id: str,
        login_id: str,
        completion: CodexLoginCompletion,
    ) -> None:
        await enforce_codex_access(user_id)
        credentials = credentials_from_bundle(completion.bundle)
        async with self._credentials_manager.locked_provider_credentials(
            user_id, credentials.provider
        ):
            async with self._state_store.locked(user_id):
                state = await self._state_store.get(user_id, login_id)
                if state is None or state.status != "pending":
                    return
                if not await self._state_store.refresh_active(user_id, login_id):
                    state.status = "failed"
                    state.error = "ChatGPT sign-in ownership was lost. Try again."
                    await self._state_store.write(state, login_id)
                    return
                stored = await self._credentials_manager.upsert_single_provider_locked(
                    user_id, credentials
                )
                if not _is_codex_credentials(stored):
                    raise RuntimeError("Stored Codex credential type changed")
                state.status = "completed"
                state.credential_id = stored.id
                await self._state_store.write(state, login_id)

    async def _mark_failed(self, user_id: str, login_id: str, message: str) -> None:
        async with self._state_store.locked(user_id):
            state = await self._state_store.get(user_id, login_id)
            if state is None or state.status != "pending":
                return
            state.status = "failed"
            state.error = message
            await self._state_store.write(state, login_id)


def _is_codex_credentials(
    credentials: object,
) -> TypeGuard[OAuth2Credentials]:
    return (
        isinstance(credentials, OAuth2Credentials)
        and credentials.provider == "codex"
        and credentials.refresh_strategy == "provider_runtime"
    )


def _user_key(user_id: str) -> str:
    return hashlib.sha256(user_id.encode()).hexdigest()[:24]


def _active_key(user_id: str) -> str:
    slot = _user_key(user_id)
    return f"codex:login:{{{slot}}}:active"


def _lock_key(user_id: str) -> str:
    slot = _user_key(user_id)
    return f"codex:login:{{{slot}}}:lock"


def _attempt_key(user_id: str, login_id: str) -> str:
    slot = _user_key(user_id)
    attempt = hashlib.sha256(login_id.encode()).hexdigest()[:24]
    return f"codex:login:{{{slot}}}:attempt:{attempt}"


def _get_default_transport() -> CodexLoginTransport:
    from backend.integrations.codex.transport import get_codex_transport

    return get_codex_transport()
