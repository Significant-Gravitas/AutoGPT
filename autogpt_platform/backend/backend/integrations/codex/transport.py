import asyncio
import concurrent.futures
import inspect
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager, suppress
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast
from uuid import uuid4

from openai_codex.models import Notification
from pydantic import BaseModel, ConfigDict, Field

from backend.data.model import OAuth2Credentials
from backend.integrations.codex.auth_bundle import (
    CodexAuthBundleError,
    CodexAuthBundleV1,
    auth_bundle_fingerprint,
    materialize_auth_bundle,
    read_auth_bundle,
)
from backend.integrations.codex.credential_codec import (
    bundle_from_credentials,
    checkpoint_credentials_from_bundle,
)
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexDeviceCodeDetails,
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexLoginCompletion,
    CodexModelInfo,
    CodexRateLimitsSnapshot,
)
from backend.integrations.codex.runtime import CODEX_RUNTIME_VERSION, CodexRuntime
from backend.integrations.codex.temporary_home import TemporaryCodexHome
from backend.integrations.credential_lease import CredentialLease
from backend.util.settings import Settings

logger = logging.getLogger(__name__)


class CodexTransportError(RuntimeError):
    pass


class CodexInvocationTimeoutError(CodexTransportError):
    pass


class CodexTransportOverloadedError(CodexTransportError):
    pass


class CodexCredentialBusyError(CodexTransportError):
    pass


class CodexCredentialIntegrityError(CodexTransportError):
    pass


ResultT = TypeVar("ResultT")


class RuntimeLogin(Protocol):
    details: CodexDeviceCodeDetails

    async def wait(self) -> bool: ...

    async def cancel(self) -> None: ...


class RuntimeClient(Protocol):
    @property
    def closed(self) -> bool: ...

    async def start_device_code_login(self) -> RuntimeLogin: ...

    async def account(
        self,
        *,
        refresh_token: bool = True,
    ) -> CodexAccountSnapshot: ...

    async def rate_limits(self) -> CodexRateLimitsSnapshot: ...

    async def models(self) -> list[CodexModelInfo]: ...

    async def invoke(
        self,
        request: CodexInvocationRequest,
    ) -> CodexInvocationResult: ...

    async def invoke_agent(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None = None,
        *,
        tool_timeout_seconds: float,
    ) -> CodexInvocationResult: ...

    async def logout(self) -> None: ...

    async def close(self) -> None: ...


RuntimeFactory = Callable[[TemporaryCodexHome], Awaitable[RuntimeClient]]


class CodexAgentSession:
    def __init__(
        self,
        runtime: "_LeasedRuntime",
        *,
        turn_timeout_seconds: float,
        tool_timeout_seconds: float,
    ) -> None:
        self._runtime = runtime
        self._turn_timeout_seconds = turn_timeout_seconds
        self._tool_timeout_seconds = tool_timeout_seconds

    async def invoke(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None = None,
    ) -> CodexInvocationResult:
        try:
            result = await asyncio.wait_for(
                self._runtime.run(
                    self._runtime.runtime.invoke_agent(
                        request,
                        dynamic_tools,
                        tool_handler,
                        event_handler,
                        tool_timeout_seconds=self._tool_timeout_seconds,
                    )
                ),
                timeout=self._turn_timeout_seconds,
            )
            return result.model_copy(
                update={"resolved_model": result.resolved_model or request.model}
            )
        except asyncio.TimeoutError:
            raise CodexInvocationTimeoutError("codex_copilot_turn_timeout") from None

    async def models(self) -> list[CodexModelInfo]:
        return await self._runtime.run(self._runtime.runtime.models())

    @property
    def closed(self) -> bool:
        return self._runtime.runtime.closed

    @property
    def failure(self) -> BaseException | None:
        return self._runtime.failure


class CodexDeviceLoginSession:
    def __init__(
        self,
        home: TemporaryCodexHome,
        runtime: RuntimeClient,
        login: RuntimeLogin,
        timeout_seconds: float,
        cleanup_timeout_seconds: float,
        release_capacity: Callable[[], None],
    ) -> None:
        self.details = login.details
        self._home = home
        self._runtime = runtime
        self._login = login
        self._timeout_seconds = timeout_seconds
        self._cleanup_timeout_seconds = cleanup_timeout_seconds
        self._release_capacity = release_capacity
        self._closed = False
        self._close_lock = asyncio.Lock()

    async def wait(self) -> CodexLoginCompletion:
        try:
            completed = await asyncio.wait_for(
                self._login.wait(),
                timeout=self._timeout_seconds,
            )
            if not completed:
                raise CodexTransportError("Codex device login failed")
            bundle = read_auth_bundle(self._home.auth_path, CODEX_RUNTIME_VERSION)
            return CodexLoginCompletion(bundle=bundle)
        except asyncio.TimeoutError:
            raise CodexTransportError("Codex device login expired") from None
        finally:
            await self.close()

    async def cancel(self) -> None:
        try:
            await _bounded_phase(
                self._login.cancel(),
                self._cleanup_timeout_seconds,
                "Codex device login cancellation",
            )
        finally:
            await self.close()

    async def close(self) -> None:
        async with self._close_lock:
            if self._closed:
                return
            try:
                await _bounded_phase(
                    self._runtime.close(),
                    self._cleanup_timeout_seconds,
                    "Codex login runtime shutdown",
                )
            finally:
                try:
                    self._home.cleanup()
                finally:
                    self._release_capacity()
                    self._closed = True


class CodexTransport:
    def __init__(
        self,
        *,
        temp_root: Path | None = None,
        max_active_processes: int = 4,
        capacity_timeout_seconds: float = 10,
        startup_timeout_seconds: float = 30,
        control_timeout_seconds: float = 60,
        invocation_timeout_seconds: float = 180,
        copilot_turn_timeout_seconds: float = 21600,
        copilot_tool_timeout_seconds: float = 900,
        login_timeout_seconds: float = 900,
        checkpoint_interval_seconds: float = 0.25,
        runtime_factory: RuntimeFactory = CodexRuntime.start,
    ) -> None:
        self._temp_root = temp_root
        self._capacity = threading.BoundedSemaphore(max_active_processes)
        self._capacity_timeout_seconds = capacity_timeout_seconds
        self._startup_timeout_seconds = startup_timeout_seconds
        self._control_timeout_seconds = control_timeout_seconds
        self._invocation_timeout_seconds = invocation_timeout_seconds
        self._copilot_turn_timeout_seconds = copilot_turn_timeout_seconds
        self._copilot_tool_timeout_seconds = copilot_tool_timeout_seconds
        self._login_timeout_seconds = login_timeout_seconds
        self._checkpoint_interval_seconds = checkpoint_interval_seconds
        self._runtime_factory = runtime_factory
        self._runtime_pool = _CodexRuntimePool(self)

    async def start_device_login(self) -> CodexDeviceLoginSession:
        await self._acquire_capacity()
        home: TemporaryCodexHome | None = None
        runtime: RuntimeClient | None = None
        try:
            home = TemporaryCodexHome.create(self._temp_root)
            runtime = await _bounded_phase(
                self._runtime_factory(home),
                self._startup_timeout_seconds,
                "Codex runtime startup",
            )
            login = await _bounded_phase(
                runtime.start_device_code_login(),
                self._control_timeout_seconds,
                "Codex device login startup",
            )
            return CodexDeviceLoginSession(
                home,
                runtime,
                login,
                self._login_timeout_seconds,
                self._control_timeout_seconds,
                self._capacity.release,
            )
        except BaseException:
            try:
                if runtime is not None:
                    await _bounded_phase(
                        runtime.close(),
                        self._control_timeout_seconds,
                        "Codex runtime shutdown",
                    )
            finally:
                try:
                    if home is not None:
                        home.cleanup()
                finally:
                    self._capacity.release()
            raise

    async def acquire_runtime_lease(
        self,
        user_id: str,
        credential_id: str,
        *,
        lock_timeout_seconds: float,
    ) -> "PooledCodexRuntimeLease":
        return await self._runtime_pool.acquire(
            user_id,
            credential_id,
            lock_timeout_seconds=lock_timeout_seconds,
        )

    async def close_runtime_pool(self) -> None:
        await self._runtime_pool.close()

    async def _acquire_capacity(self) -> None:
        loop = asyncio.get_running_loop()
        deadline = loop.time() + self._capacity_timeout_seconds
        while not self._capacity.acquire(blocking=False):
            if loop.time() >= deadline:
                raise CodexTransportOverloadedError(
                    "Codex transport has no free process capacity"
                )
            await asyncio.sleep(0.01)

    async def account(self, lease: CredentialLease) -> CodexAccountSnapshot:
        return await self._run_control_operation(
            "account lookup",
            self._account(lease),
        )

    async def _account(self, lease: CredentialLease) -> CodexAccountSnapshot:
        async with self._credential_runtime(lease) as session:
            return await session.run(session.runtime.account(refresh_token=True))

    async def rate_limits(
        self,
        lease: CredentialLease,
    ) -> CodexRateLimitsSnapshot:
        return await self._run_control_operation(
            "rate-limit lookup",
            self._rate_limits(lease),
        )

    async def _rate_limits(
        self,
        lease: CredentialLease,
    ) -> CodexRateLimitsSnapshot:
        async with self._credential_runtime(lease) as session:
            return await session.run(session.runtime.rate_limits())

    async def models(self, lease: CredentialLease) -> list[CodexModelInfo]:
        return await self._run_control_operation(
            "model lookup",
            self._models(lease),
        )

    async def _models(self, lease: CredentialLease) -> list[CodexModelInfo]:
        async with self._credential_runtime(lease) as session:
            return await session.run(session.runtime.models())

    async def invoke(
        self,
        lease: CredentialLease,
        request: CodexInvocationRequest,
    ) -> CodexInvocationResult:
        timeout = request.timeout_seconds or self._invocation_timeout_seconds
        try:
            return await asyncio.wait_for(
                self._invoke_with_runtime(lease, request),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise CodexInvocationTimeoutError(
                f"Codex invocation exceeded {timeout:g} seconds"
            ) from None

    async def _invoke_with_runtime(
        self,
        lease: CredentialLease,
        request: CodexInvocationRequest,
    ) -> CodexInvocationResult:
        async with self._credential_runtime(lease) as session:
            await session.run(session.runtime.account(refresh_token=True))
            await session.checkpoint_now()
            models = await session.run(session.runtime.models())
            resolved_model = _resolve_invocation_model(request.model, models)
            result = await session.run(
                session.runtime.invoke(
                    request.model_copy(update={"model": resolved_model})
                )
            )
            return result.model_copy(update={"resolved_model": resolved_model})

    async def invoke_agent(
        self,
        lease: CredentialLease,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None = None,
    ) -> CodexInvocationResult:
        try:
            return await asyncio.wait_for(
                self._invoke_agent_with_runtime(
                    lease,
                    request,
                    dynamic_tools,
                    tool_handler,
                    event_handler,
                ),
                timeout=self._copilot_turn_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise CodexInvocationTimeoutError("codex_copilot_turn_timeout") from None

    @asynccontextmanager
    async def agent_session(
        self,
        lease: CredentialLease,
    ) -> AsyncIterator[CodexAgentSession]:
        async with self._credential_runtime(lease) as session:
            await session.run(session.runtime.account(refresh_token=True))
            await session.checkpoint_now()
            yield CodexAgentSession(
                session,
                turn_timeout_seconds=self._copilot_turn_timeout_seconds,
                tool_timeout_seconds=self._copilot_tool_timeout_seconds,
            )

    async def _invoke_agent_with_runtime(
        self,
        lease: CredentialLease,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None,
    ) -> CodexInvocationResult:
        async with self._credential_runtime(lease) as session:
            await session.run(session.runtime.account(refresh_token=True))
            await session.checkpoint_now()
            return await session.run(
                session.runtime.invoke_agent(
                    request,
                    dynamic_tools,
                    tool_handler,
                    event_handler,
                    tool_timeout_seconds=self._copilot_tool_timeout_seconds,
                )
            )

    async def logout(self, lease: CredentialLease) -> None:
        await self._run_control_operation("logout", self._logout(lease))

    async def _logout(self, lease: CredentialLease) -> None:
        async with self._credential_runtime(lease, checkpoint=False) as session:
            await session.run(session.runtime.logout())

    async def _run_control_operation(
        self,
        label: str,
        operation: Awaitable[ResultT],
    ) -> ResultT:
        return await _bounded_phase(
            operation,
            self._control_timeout_seconds,
            f"Codex {label}",
        )

    @asynccontextmanager
    async def _credential_runtime(
        self,
        lease: CredentialLease,
        *,
        checkpoint: bool = True,
    ) -> AsyncIterator["_LeasedRuntime"]:
        credentials = _codex_credentials(lease)
        bundle = bundle_from_credentials(credentials)
        await self._acquire_capacity()
        home: TemporaryCodexHome | None = None
        runtime: RuntimeClient | None = None
        session: _LeasedRuntime | None = None
        try:
            home = TemporaryCodexHome.create(self._temp_root)
            materialize_auth_bundle(bundle, home.auth_path)
            runtime = await _bounded_phase(
                self._runtime_factory(home),
                self._startup_timeout_seconds,
                "Codex runtime startup",
            )
            state = _AuthCheckpointState(credentials=credentials, bundle=bundle)
            session = _LeasedRuntime(
                runtime=runtime,
                lease=lease,
                state=state,
                home=home,
                checkpoint_timeout_seconds=self._control_timeout_seconds,
                checkpoint_interval_seconds=self._checkpoint_interval_seconds,
                monitor_auth=checkpoint,
            )
            session.start()
            yield session
        finally:
            try:
                if session is not None:
                    await session.stop()
            finally:
                try:
                    if runtime is not None:
                        await _bounded_phase(
                            runtime.close(),
                            self._control_timeout_seconds,
                            "Codex runtime shutdown",
                        )
                finally:
                    try:
                        if checkpoint and session is not None:
                            await session.checkpoint_now(validate_unchanged=True)
                    finally:
                        try:
                            if home is not None:
                                home.cleanup()
                        finally:
                            self._capacity.release()


_CODEX_TRANSPORT_SINGLETON_LOCK = threading.Lock()
_codex_transport_singleton: CodexTransport | None = None


def get_codex_transport() -> CodexTransport:
    global _codex_transport_singleton
    if _codex_transport_singleton is not None:
        return _codex_transport_singleton
    with _CODEX_TRANSPORT_SINGLETON_LOCK:
        if _codex_transport_singleton is None:
            config = Settings().config
            temp_root = Path(config.codex_temp_root) if config.codex_temp_root else None
            _codex_transport_singleton = CodexTransport(
                temp_root=temp_root,
                max_active_processes=config.codex_max_active_processes,
                capacity_timeout_seconds=config.codex_capacity_timeout_seconds,
                startup_timeout_seconds=config.codex_startup_timeout_seconds,
                control_timeout_seconds=config.codex_control_timeout_seconds,
                invocation_timeout_seconds=config.codex_invocation_timeout_seconds,
                copilot_turn_timeout_seconds=config.codex_copilot_turn_timeout_seconds,
                copilot_tool_timeout_seconds=config.codex_copilot_tool_timeout_seconds,
                login_timeout_seconds=config.codex_login_timeout_seconds,
                checkpoint_interval_seconds=(
                    config.codex_auth_checkpoint_interval_seconds
                ),
            )
        return _codex_transport_singleton


def _codex_credentials(lease: CredentialLease) -> OAuth2Credentials:
    credentials = lease.credentials
    if credentials.type != "oauth2":
        raise CodexTransportError("Codex transport requires OAuth credentials")
    oauth_credentials = cast(OAuth2Credentials, credentials)
    bundle_from_credentials(oauth_credentials)
    return oauth_credentials


def _resolve_invocation_model(
    requested_model: str | None,
    models: list[CodexModelInfo],
) -> str:
    available = {model.model: model for model in models}
    if requested_model is not None:
        if requested_model not in available:
            raise CodexTransportError(
                f"Codex model is not available for this account: {requested_model}"
            )
        return requested_model
    default = next((model for model in models if model.is_default), None)
    visible = next((model for model in models if not model.hidden), None)
    selected = default or visible or (models[0] if models else None)
    if selected is None:
        raise CodexTransportError("Codex account advertised no available models")
    return selected.model


class _AuthCheckpointState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    credentials: OAuth2Credentials
    bundle: CodexAuthBundleV1
    lock: asyncio.Lock = Field(default_factory=asyncio.Lock)


class _LeasedRuntime:
    def __init__(
        self,
        *,
        runtime: RuntimeClient,
        lease: CredentialLease,
        state: _AuthCheckpointState,
        home: TemporaryCodexHome,
        checkpoint_timeout_seconds: float,
        checkpoint_interval_seconds: float,
        monitor_auth: bool,
    ) -> None:
        self.runtime = runtime
        self._lease = lease
        self._state = state
        self._home = home
        self._checkpoint_timeout_seconds = checkpoint_timeout_seconds
        self._checkpoint_interval_seconds = checkpoint_interval_seconds
        self._monitor_auth = monitor_auth
        self._monitor_task: asyncio.Task[None] | None = None

    def start(self) -> None:
        if self._monitor_auth:
            self._monitor_task = asyncio.create_task(self._monitor_auth_file())

    async def run(self, operation: Awaitable[ResultT]) -> ResultT:
        return await _run_with_lease_guard(
            self._lease,
            operation,
            monitor_task=self._monitor_task,
        )

    @property
    def failure(self) -> BaseException | None:
        monitor = self._monitor_task
        if monitor is not None and monitor.done() and not monitor.cancelled():
            try:
                if error := monitor.exception():
                    return error
            except asyncio.CancelledError:
                pass
        return self._lease.failure

    async def checkpoint_now(self, *, validate_unchanged: bool = True) -> None:
        async with self._state.lock:
            try:
                after = await _read_runtime_auth(self._home.auth_path)
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                raise CodexCredentialIntegrityError(
                    "codex_credential_state_unavailable"
                ) from error
            if auth_bundle_fingerprint(after) == auth_bundle_fingerprint(
                self._state.bundle
            ):
                if validate_unchanged:
                    try:
                        await _bounded_phase(
                            self._lease.validate(),
                            self._checkpoint_timeout_seconds,
                            "Codex credential lease validation",
                        )
                    except asyncio.CancelledError:
                        raise
                    except BaseException as error:
                        raise CodexCredentialIntegrityError(
                            f"codex_credential_lease_lost: {error}"
                        ) from error
                return
            updated = checkpoint_credentials_from_bundle(
                self._state.credentials,
                after,
            )
            try:
                await _bounded_phase(
                    self._lease.checkpoint(updated),
                    self._checkpoint_timeout_seconds,
                    "Codex credential checkpoint",
                )
            except asyncio.CancelledError:
                raise
            except BaseException as error:
                raise CodexCredentialIntegrityError(
                    "codex_credential_checkpoint_failed"
                ) from error
            self._state.credentials = updated
            self._state.bundle = after

    async def stop(self) -> None:
        monitor = self._monitor_task
        if monitor is None:
            return
        self._monitor_task = None
        monitor.cancel()
        results = await asyncio.gather(monitor, return_exceptions=True)
        error = results[0]
        if isinstance(error, BaseException) and not isinstance(
            error, asyncio.CancelledError
        ):
            raise error

    async def _monitor_auth_file(self) -> None:
        while True:
            await asyncio.sleep(self._checkpoint_interval_seconds)
            await self.checkpoint_now(validate_unchanged=False)


class _CodexRuntimeActor:
    def __init__(
        self,
        *,
        lease: CredentialLease,
        context: AbstractAsyncContextManager[CodexAgentSession],
        session: CodexAgentSession,
    ) -> None:
        self.lease = lease
        self.context = context
        self.session = session
        self.handles: set[str] = set()
        self.calls: dict[str, set[asyncio.Task[object]]] = {}
        self.models_task: asyncio.Task[list[CodexModelInfo]] | None = None
        self.failed: BaseException | None = None


class PooledCodexRuntimeLease:
    def __init__(
        self,
        *,
        pool: "_CodexRuntimePool",
        key: tuple[str, str],
        handle_id: str,
        credentials: OAuth2Credentials,
    ) -> None:
        self.credentials = credentials
        self._pool = pool
        self._key = key
        self._handle_id = handle_id
        self._release_state = "active"
        self._release_completion: concurrent.futures.Future[None] | None = None
        self._release_lock = threading.Lock()

    async def models(self) -> list[CodexModelInfo]:
        self._ensure_active()
        return await self._pool.models(self._key, self._handle_id)

    async def invoke(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None = None,
    ) -> CodexInvocationResult:
        self._ensure_active()
        return await self._pool.invoke(
            self._key,
            self._handle_id,
            request,
            dynamic_tools,
            tool_handler,
            event_handler,
        )

    async def release(self) -> None:
        with self._release_lock:
            if self._release_state == "released":
                return
            if self._release_state == "releasing":
                completion = self._release_completion
                leader = False
            else:
                completion = concurrent.futures.Future()
                self._release_completion = completion
                self._release_state = "releasing"
                leader = True
        if completion is None:
            raise RuntimeError("Codex runtime release state is invalid")
        if not leader:
            await asyncio.shield(asyncio.wrap_future(completion))
            return
        try:
            await self._pool.release(self._key, self._handle_id)
        except BaseException as error:
            with self._release_lock:
                self._release_state = "active"
                self._release_completion = None
            completion.set_exception(error)
            raise
        else:
            with self._release_lock:
                self._release_state = "released"
                self._release_completion = None
            completion.set_result(None)

    def _ensure_active(self) -> None:
        with self._release_lock:
            if self._release_state != "active":
                raise RuntimeError("Codex runtime lease was released")


class _CodexRuntimePool:
    def __init__(self, transport: CodexTransport) -> None:
        self._transport = transport
        self._state_lock = threading.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._loop_ready: threading.Event | None = None
        self._loop_start_error: BaseException | None = None
        self._closing = False
        self._closed = False
        self._generation = 0
        self._close_complete = threading.Event()
        self._close_error: BaseException | None = None
        self._actors: dict[tuple[str, str], _CodexRuntimeActor] = {}
        self._starts: dict[tuple[str, str], asyncio.Task[_CodexRuntimeActor]] = {}
        self._retiring: dict[tuple[str, str], asyncio.Task[None]] = {}
        self._pending: dict[tuple[str, str], set[str]] = {}

    async def acquire(
        self,
        user_id: str,
        credential_id: str,
        *,
        lock_timeout_seconds: float,
    ) -> PooledCodexRuntimeLease:
        key = (user_id, credential_id)
        handle_id = uuid4().hex
        loop = self._ensure_loop()
        with self._state_lock:
            if self._closing or self._closed:
                raise CodexTransportError("Codex runtime pool is shutting down")
            generation = self._generation
            operation = self._borrow_on_actor_loop(
                key,
                handle_id,
                generation=generation,
                lock_timeout_seconds=lock_timeout_seconds,
            )
            try:
                future = asyncio.run_coroutine_threadsafe(operation, loop)
            except BaseException:
                operation.close()
                raise
        try:
            credentials = await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            with suppress(BaseException):
                cleanup = asyncio.run_coroutine_threadsafe(
                    self._release_on_actor_loop(key, handle_id),
                    loop,
                )
                await _await_cleanup_future(
                    cleanup,
                    timeout_seconds=self._cleanup_timeout_seconds,
                    label="Codex cancelled runtime borrow",
                )
            raise
        with self._state_lock:
            shutting_down = self._closing or self._closed
        if shutting_down:
            raise CodexTransportError("Codex runtime pool is shutting down")
        return PooledCodexRuntimeLease(
            pool=self,
            key=key,
            handle_id=handle_id,
            credentials=credentials,
        )

    async def models(
        self,
        key: tuple[str, str],
        handle_id: str,
    ) -> list[CodexModelInfo]:
        future = asyncio.run_coroutine_threadsafe(
            self._models_on_actor_loop(key, handle_id),
            self._ensure_loop(),
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def invoke(
        self,
        key: tuple[str, str],
        handle_id: str,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None,
    ) -> CodexInvocationResult:
        caller_loop = asyncio.get_running_loop()

        async def bridged_tool_handler(
            call: CodexDynamicToolCall,
        ) -> CodexDynamicToolResult:
            return await _await_threadsafe_callback(
                tool_handler(call),
                caller_loop,
            )

        async def bridged_event_handler(notification: Notification) -> None:
            if event_handler is not None:
                await _await_threadsafe_callback(
                    event_handler(notification),
                    caller_loop,
                )

        future = asyncio.run_coroutine_threadsafe(
            self._invoke_on_actor_loop(
                key,
                handle_id,
                request,
                dynamic_tools,
                bridged_tool_handler,
                bridged_event_handler if event_handler is not None else None,
            ),
            self._ensure_loop(),
        )
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    async def release(self, key: tuple[str, str], handle_id: str) -> None:
        loop = self._loop
        if loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(
            self._release_on_actor_loop(key, handle_id),
            loop,
        )
        await _await_cleanup_future(
            future,
            timeout_seconds=self._cleanup_timeout_seconds,
            label="Codex pooled runtime release",
        )

    async def close(self) -> None:
        close_complete = self._close_complete
        close_error: BaseException | None = None
        already_complete = False
        leader = False
        with self._state_lock:
            if self._closed:
                close_error = self._close_error
                already_complete = True
            elif self._closing:
                pass
            else:
                self._closing = True
                self._generation += 1
                leader = True
            loop = self._loop
            thread = self._thread
            ready = self._loop_ready
        if already_complete:
            if close_error is not None:
                raise close_error
            return
        if not leader:
            await self._wait_for_close(close_complete)
            return
        close_error, loop = await self._close_as_leader(loop, thread, ready)
        with self._state_lock:
            if self._loop is loop:
                self._loop = None
                self._thread = None
                self._loop_ready = None
            self._close_error = close_error
            self._closed = True
            close_complete.set()
        if close_error is not None:
            raise close_error

    async def _wait_for_close(self, close_complete: threading.Event) -> None:
        completed = await asyncio.to_thread(
            close_complete.wait,
            self._cleanup_timeout_seconds,
        )
        if not completed:
            raise CodexTransportError("Codex runtime pool shutdown timed out")
        with self._state_lock:
            close_error = self._close_error
        if close_error is not None:
            raise close_error

    async def _close_as_leader(
        self,
        loop: asyncio.AbstractEventLoop | None,
        thread: threading.Thread | None,
        ready: threading.Event | None,
    ) -> tuple[BaseException | None, asyncio.AbstractEventLoop | None]:
        close_error: BaseException | None = None
        try:
            loop = await self._resolve_shutdown_loop(loop, thread, ready)
            if loop is not None and not loop.is_closed():
                future = asyncio.run_coroutine_threadsafe(
                    self._close_all_on_actor_loop(), loop
                )
                await _await_cleanup_future(
                    future,
                    timeout_seconds=self._cleanup_timeout_seconds,
                    label="Codex runtime pool shutdown",
                )
        except BaseException as error:
            close_error = error
        stop_error = await self._stop_actor_loop(loop, thread)
        if close_error is not None and stop_error is not None:
            close_error.add_note(
                f"Actor-loop stop failure: "
                f"{type(stop_error).__name__}: {stop_error}"
            )
        return close_error or stop_error, loop

    async def _resolve_shutdown_loop(
        self,
        loop: asyncio.AbstractEventLoop | None,
        thread: threading.Thread | None,
        ready: threading.Event | None,
    ) -> asyncio.AbstractEventLoop | None:
        if thread is None or loop is not None:
            return loop
        if ready is None or not await asyncio.to_thread(
            ready.wait,
            self._cleanup_timeout_seconds,
        ):
            raise CodexTransportError(
                "Codex runtime pool loop startup timed out during shutdown"
            )
        with self._state_lock:
            loop = self._loop
            start_error = self._loop_start_error
        if start_error is not None:
            raise CodexTransportError(
                "Codex runtime pool loop failed to start"
            ) from start_error
        return loop

    async def _stop_actor_loop(
        self,
        loop: asyncio.AbstractEventLoop | None,
        thread: threading.Thread | None,
    ) -> BaseException | None:
        try:
            if loop is not None and not loop.is_closed():
                loop.call_soon_threadsafe(loop.stop)
            if thread is None:
                return None
            await asyncio.to_thread(thread.join, self._cleanup_timeout_seconds)
            if thread.is_alive():
                return CodexTransportError("Codex runtime pool loop did not stop")
        except BaseException as error:
            return error
        return None

    @property
    def _cleanup_timeout_seconds(self) -> float:
        return max(self._transport._control_timeout_seconds * 3, 5)

    def _ensure_loop(self) -> asyncio.AbstractEventLoop:
        with self._state_lock:
            if self._closing or self._closed:
                raise CodexTransportError("Codex runtime pool is shutting down")
            if self._loop is not None:
                return self._loop
            ready = self._loop_ready
            if ready is None:
                ready = threading.Event()
                self._loop_ready = ready
                self._loop_start_error = None

                def run() -> None:
                    loop: asyncio.AbstractEventLoop | None = None
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        with self._state_lock:
                            self._loop = loop
                        ready.set()
                        loop.run_forever()
                    except BaseException as error:
                        with self._state_lock:
                            self._loop_start_error = error
                        ready.set()
                    finally:
                        if loop is not None and not loop.is_closed():
                            loop.close()

                thread = threading.Thread(
                    target=run,
                    name="autogpt-codex-runtime-pool",
                    daemon=True,
                )
                self._thread = thread
                thread.start()
        if not ready.wait(self._cleanup_timeout_seconds):
            raise CodexTransportError("Codex runtime pool loop startup timed out")
        with self._state_lock:
            loop = self._loop
            start_error = self._loop_start_error
            shutting_down = self._closing or self._closed
        if start_error is not None:
            raise CodexTransportError(
                "Codex runtime pool loop failed to start"
            ) from start_error
        if shutting_down:
            raise CodexTransportError("Codex runtime pool is shutting down")
        if loop is None:
            raise RuntimeError("Codex runtime pool loop failed to start")
        return loop

    async def _borrow_on_actor_loop(
        self,
        key: tuple[str, str],
        handle_id: str,
        *,
        generation: int,
        lock_timeout_seconds: float,
    ) -> OAuth2Credentials:
        self._validate_generation(generation)
        pending = self._pending.setdefault(key, set())
        pending.add(handle_id)
        start: asyncio.Task[_CodexRuntimeActor] | None = None
        try:
            retiring = self._retiring.get(key)
            if retiring is not None:
                try:
                    await asyncio.shield(retiring)
                except BaseException:
                    raise
                else:
                    if self._retiring.get(key) is retiring:
                        self._retiring.pop(key, None)
            self._validate_generation(generation)
            actor = self._actors.get(key)
            if (
                actor is not None
                and (failure := self._actor_failure(actor)) is not None
            ):
                raise CodexTransportError(
                    "Codex credential runtime is unavailable"
                ) from failure
            if actor is None:
                start = self._starts.get(key)
                if start is None:
                    start = asyncio.create_task(
                        self._start_actor(
                            key,
                            lock_timeout_seconds=lock_timeout_seconds,
                        )
                    )
                    self._starts[key] = start
                actor = await asyncio.shield(start)
                with self._state_lock:
                    if self._closing or self._closed or generation != self._generation:
                        raise CodexTransportError("Codex runtime pool is shutting down")
                    actor = self._actors.setdefault(key, actor)
                if self._starts.get(key) is start:
                    self._starts.pop(key, None)
            else:
                self._validate_generation(generation)
            actor.handles.add(handle_id)
            return cast(OAuth2Credentials, actor.lease.credentials)
        except BaseException:
            pending.discard(handle_id)
            await self._cancel_unused_start(key, start)
            raise
        finally:
            pending.discard(handle_id)
            if not pending:
                self._pending.pop(key, None)

    async def _start_actor(
        self,
        key: tuple[str, str],
        *,
        lock_timeout_seconds: float,
    ) -> _CodexRuntimeActor:
        from backend.integrations.creds_manager import IntegrationCredentialsManager

        user_id, credential_id = key
        try:
            lease = await asyncio.wait_for(
                IntegrationCredentialsManager().acquire_lease(user_id, credential_id),
                timeout=lock_timeout_seconds,
            )
        except asyncio.TimeoutError:
            raise CodexCredentialBusyError("codex_credential_busy") from None
        context: AbstractAsyncContextManager[CodexAgentSession] | None = None
        try:
            _codex_credentials(lease)
            context = self._transport.agent_session(lease)
            session = await context.__aenter__()
            return _CodexRuntimeActor(
                lease=lease,
                context=context,
                session=session,
            )
        except BaseException:
            if context is not None:
                with suppress(BaseException):
                    await context.__aexit__(None, None, None)
            await lease.release()
            raise

    async def _cancel_unused_start(
        self,
        key: tuple[str, str],
        start: asyncio.Task[_CodexRuntimeActor] | None,
    ) -> None:
        if self._pending.get(key):
            return
        candidate = start or self._starts.get(key)
        if candidate is None:
            return
        if self._starts.get(key) is not candidate:
            return
        if not candidate.done():
            candidate.cancel()
            await asyncio.gather(candidate, return_exceptions=True)
        if self._starts.get(key) is candidate:
            self._starts.pop(key, None)
        if candidate.cancelled():
            return
        try:
            actor = candidate.result()
        except BaseException:
            return
        if not actor.handles:
            try:
                await self._close_actor_bounded(actor)
            except BaseException:
                logger.exception("Failed to clean up unused Codex runtime start")

    async def _models_on_actor_loop(
        self,
        key: tuple[str, str],
        handle_id: str,
    ) -> list[CodexModelInfo]:
        actor = self._active_actor(key, handle_id)
        models_task = actor.models_task
        if models_task is None:
            models_task = asyncio.create_task(actor.session.models())
            actor.models_task = models_task
        try:
            return list(await asyncio.shield(models_task))
        except BaseException as error:
            self._record_actor_failure(actor, error)
            if actor.models_task is models_task and models_task.done():
                actor.models_task = None
            raise

    async def _invoke_on_actor_loop(
        self,
        key: tuple[str, str],
        handle_id: str,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler: Callable[
            [CodexDynamicToolCall], Awaitable[CodexDynamicToolResult]
        ],
        event_handler: Callable[[Notification], Awaitable[None]] | None,
    ) -> CodexInvocationResult:
        actor = self._active_actor(key, handle_id)
        task = asyncio.current_task()
        if task is None:
            raise RuntimeError("Codex pooled invocation has no owning task")
        calls = actor.calls.setdefault(handle_id, set())
        calls.add(cast(asyncio.Task[object], task))
        try:
            return await actor.session.invoke(
                request,
                dynamic_tools,
                tool_handler,
                event_handler,
            )
        except BaseException as error:
            self._record_actor_failure(actor, error)
            raise
        finally:
            calls.discard(cast(asyncio.Task[object], task))
            if not calls and actor.calls.get(handle_id) is calls:
                actor.calls.pop(handle_id, None)

    async def _release_on_actor_loop(
        self,
        key: tuple[str, str],
        handle_id: str,
    ) -> None:
        actor = self._actors.get(key)
        if actor is None:
            retiring = self._retiring.get(key)
            if retiring is not None:
                await asyncio.shield(retiring)
            return
        if handle_id not in actor.handles:
            return
        actor.handles.discard(handle_id)
        calls = tuple(actor.calls.pop(handle_id, ()))
        for task in calls:
            task.cancel()
        if calls:
            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(
                    asyncio.gather(*calls, return_exceptions=True),
                    timeout=self._transport._control_timeout_seconds,
                )
        if actor.handles:
            return
        if self._actors.get(key) is actor:
            self._actors.pop(key, None)
        retiring = asyncio.create_task(self._close_actor_bounded(actor))
        self._retiring[key] = retiring
        try:
            await asyncio.shield(retiring)
        except BaseException:
            raise
        else:
            if self._retiring.get(key) is retiring:
                self._retiring.pop(key, None)

    async def _close_actor_bounded(self, actor: _CodexRuntimeActor) -> None:
        models_task, actor.models_task = actor.models_task, None
        if models_task is not None:
            if not models_task.done():
                models_task.cancel()
            await asyncio.gather(models_task, return_exceptions=True)

        async def cleanup() -> None:
            context_error: BaseException | None = None
            try:
                await actor.context.__aexit__(None, None, None)
            except BaseException as error:
                context_error = error
            try:
                await actor.lease.release()
            except BaseException as release_error:
                if context_error is None:
                    raise
                context_error.add_note(
                    f"Credential lease release also failed: {release_error}"
                )
            if context_error is not None:
                raise context_error

        task = asyncio.create_task(cleanup())
        try:
            await asyncio.wait_for(
                asyncio.shield(task),
                timeout=self._cleanup_timeout_seconds,
            )
        except BaseException:
            task.cancel()
            await _cancel_future_bounded(task)
            raise

    async def _close_all_on_actor_loop(self) -> None:
        errors: list[BaseException] = []
        starts = tuple(self._starts.values())
        self._starts.clear()
        for start in starts:
            start.cancel()
        started_actors: list[_CodexRuntimeActor] = []
        if starts:
            for result in await asyncio.gather(*starts, return_exceptions=True):
                if isinstance(result, _CodexRuntimeActor):
                    started_actors.append(result)
                elif isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    errors.append(result)
        actors_by_id = {
            id(actor): actor for actor in (*self._actors.values(), *started_actors)
        }
        actors = tuple(actors_by_id.values())
        self._actors.clear()
        self._pending.clear()
        for actor in actors:
            for calls in actor.calls.values():
                for task in calls:
                    task.cancel()
        calls = tuple(
            {
                id(task): task
                for actor in actors
                for group in actor.calls.values()
                for task in group
            }.values()
        )
        if calls:
            await asyncio.gather(*calls, return_exceptions=True)

        cleanup_tasks = [
            asyncio.create_task(self._close_actor_bounded(actor)) for actor in actors
        ]
        cleanup_tasks.extend(
            {id(task): task for task in self._retiring.values()}.values()
        )
        self._retiring.clear()
        if cleanup_tasks:
            for result in await asyncio.gather(
                *{id(task): task for task in cleanup_tasks}.values(),
                return_exceptions=True,
            ):
                if isinstance(result, BaseException) and not isinstance(
                    result, asyncio.CancelledError
                ):
                    errors.append(result)
        if errors:
            cleanup_error = CodexTransportError(
                f"Codex runtime pool cleanup failed ({len(errors)} error(s))"
            )
            for additional_error in errors[1:]:
                cleanup_error.add_note(
                    f"Additional cleanup failure: "
                    f"{type(additional_error).__name__}: {additional_error}"
                )
            raise cleanup_error from errors[0]

    def _active_actor(
        self,
        key: tuple[str, str],
        handle_id: str,
    ) -> _CodexRuntimeActor:
        actor = self._actors.get(key)
        if actor is None or handle_id not in actor.handles:
            raise RuntimeError("Codex runtime lease is no longer active")
        if failure := self._actor_failure(actor):
            raise CodexTransportError(
                "Codex credential runtime is unavailable"
            ) from failure
        return actor

    @staticmethod
    def _actor_failure(actor: _CodexRuntimeActor) -> BaseException | None:
        failure = actor.failed or actor.session.failure
        if failure is not None:
            actor.failed = failure
        return failure

    def _record_actor_failure(
        self,
        actor: _CodexRuntimeActor,
        error: BaseException,
    ) -> None:
        failure = actor.session.failure
        if (
            isinstance(error, CodexCredentialIntegrityError)
            or failure is not None
            or actor.session.closed
        ):
            actor.failed = failure or error

    def _validate_generation(self, generation: int) -> None:
        with self._state_lock:
            if self._closing or self._closed or generation != self._generation:
                raise CodexTransportError("Codex runtime pool is shutting down")


async def _await_threadsafe_callback(
    operation: Awaitable[ResultT],
    loop: asyncio.AbstractEventLoop,
) -> ResultT:
    async def await_operation() -> ResultT:
        return await operation

    future = asyncio.run_coroutine_threadsafe(await_operation(), loop)
    try:
        return await asyncio.wrap_future(future)
    except asyncio.CancelledError:
        future.cancel()
        raise


async def _await_cleanup_future(
    future: concurrent.futures.Future[ResultT],
    *,
    timeout_seconds: float,
    label: str,
) -> ResultT:
    wrapped = asyncio.wrap_future(future)
    try:
        return await asyncio.wait_for(
            asyncio.shield(wrapped),
            timeout=timeout_seconds,
        )
    except asyncio.CancelledError:
        with suppress(BaseException):
            await asyncio.wait_for(
                asyncio.shield(wrapped),
                timeout=timeout_seconds,
            )
        raise
    except asyncio.TimeoutError:
        future.cancel()
        raise CodexTransportError(f"{label} timed out") from None


async def _run_with_lease_guard(
    lease: CredentialLease,
    operation: Awaitable[ResultT],
    *,
    monitor_task: asyncio.Task[None] | None = None,
) -> ResultT:
    if failure := lease.failure:
        _close_unused_awaitable(operation)
        raise RuntimeError("Credential lease heartbeat failed") from failure
    if monitor_task is not None and monitor_task.done():
        _close_unused_awaitable(operation)
        if monitor_task.cancelled():
            raise CodexCredentialIntegrityError("codex_credential_monitor_stopped")
        if error := monitor_task.exception():
            raise error
        raise CodexCredentialIntegrityError("codex_credential_monitor_stopped")
    operation_task = asyncio.ensure_future(operation)
    heartbeat_task = asyncio.create_task(lease.wait_for_failure())
    try:
        watched_tasks = {operation_task, heartbeat_task}
        if monitor_task is not None:
            watched_tasks.add(monitor_task)
        done, _ = await asyncio.wait(
            watched_tasks,
            return_when=asyncio.FIRST_COMPLETED,
        )
        if heartbeat_task in done:
            await _cancel_future_bounded(operation_task)
            await heartbeat_task
        if monitor_task is not None and monitor_task in done:
            await _cancel_future_bounded(operation_task)
            await monitor_task
        return await operation_task
    finally:
        if not operation_task.done():
            await _cancel_future_bounded(operation_task)
        heartbeat_task.cancel()
        await asyncio.gather(heartbeat_task, return_exceptions=True)


def _close_unused_awaitable(operation: Awaitable[Any]) -> None:
    if inspect.iscoroutine(operation):
        operation.close()


async def _bounded_phase(
    operation: Awaitable[ResultT],
    timeout_seconds: float,
    label: str,
) -> ResultT:
    task = asyncio.ensure_future(operation)
    try:
        done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    except asyncio.CancelledError:
        await _cancel_future_bounded(task)
        raise
    if task in done:
        return await task
    await _cancel_future_bounded(task, timeout_seconds=0.1)
    raise CodexTransportError(f"{label} exceeded {timeout_seconds:g} seconds")


async def _cancel_future_bounded(
    task: asyncio.Future[Any],
    *,
    timeout_seconds: float = 1.0,
) -> None:
    task.cancel()
    done, _ = await asyncio.wait({task}, timeout=timeout_seconds)
    if task in done:
        try:
            task.result()
        except BaseException:
            pass
        return

    def consume_result(completed: asyncio.Future[Any]) -> None:
        try:
            completed.result()
        except BaseException:
            pass

    task.add_done_callback(consume_result)


async def _read_runtime_auth(auth_path: Path) -> CodexAuthBundleV1:
    for delay_seconds in (0.0, 0.02, 0.05, 0.1):
        if delay_seconds:
            await asyncio.sleep(delay_seconds)
        try:
            return read_auth_bundle(auth_path, CODEX_RUNTIME_VERSION)
        except CodexAuthBundleError:
            continue
    raise CodexTransportError("Codex runtime auth state is unavailable")
