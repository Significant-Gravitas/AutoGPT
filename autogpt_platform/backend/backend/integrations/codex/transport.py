import asyncio
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any, Protocol, TypeVar, cast

from openai_codex.models import Notification

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
    CodexRateLimitsSnapshot,
)
from backend.integrations.codex.runtime import CODEX_RUNTIME_VERSION, CodexRuntime
from backend.integrations.codex.temporary_home import TemporaryCodexHome
from backend.integrations.credential_lease import CredentialLease
from backend.util.settings import Settings


class CodexTransportError(RuntimeError):
    pass


class CodexInvocationTimeoutError(CodexTransportError):
    pass


class CodexTransportOverloadedError(CodexTransportError):
    pass


ResultT = TypeVar("ResultT")


class RuntimeLogin(Protocol):
    details: CodexDeviceCodeDetails

    async def wait(self) -> bool: ...

    async def cancel(self) -> None: ...


class RuntimeClient(Protocol):
    async def start_device_code_login(self) -> RuntimeLogin: ...

    async def account(
        self,
        *,
        refresh_token: bool = True,
    ) -> CodexAccountSnapshot: ...

    async def rate_limits(self) -> CodexRateLimitsSnapshot: ...

    async def models(self) -> list[str]: ...

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

    async def models(self, lease: CredentialLease) -> list[str]:
        return await self._run_control_operation(
            "model lookup",
            self._models(lease),
        )

    async def _models(self, lease: CredentialLease) -> list[str]:
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
            return await session.run(session.runtime.invoke(request))

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


@cache
def get_codex_transport() -> CodexTransport:
    config = Settings().config
    temp_root = Path(config.codex_temp_root) if config.codex_temp_root else None
    return CodexTransport(
        temp_root=temp_root,
        max_active_processes=config.codex_max_active_processes,
        capacity_timeout_seconds=config.codex_capacity_timeout_seconds,
        startup_timeout_seconds=config.codex_startup_timeout_seconds,
        control_timeout_seconds=config.codex_control_timeout_seconds,
        invocation_timeout_seconds=config.codex_invocation_timeout_seconds,
        copilot_turn_timeout_seconds=config.codex_copilot_turn_timeout_seconds,
        copilot_tool_timeout_seconds=config.codex_copilot_tool_timeout_seconds,
        login_timeout_seconds=config.codex_login_timeout_seconds,
        checkpoint_interval_seconds=config.codex_auth_checkpoint_interval_seconds,
    )


def _codex_credentials(lease: CredentialLease) -> OAuth2Credentials:
    credentials = lease.credentials
    if credentials.type != "oauth2":
        raise CodexTransportError("Codex transport requires OAuth credentials")
    oauth_credentials = cast(OAuth2Credentials, credentials)
    bundle_from_credentials(oauth_credentials)
    return oauth_credentials


@dataclass
class _AuthCheckpointState:
    credentials: OAuth2Credentials
    bundle: CodexAuthBundleV1
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)


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

    async def checkpoint_now(self, *, validate_unchanged: bool = True) -> None:
        async with self._state.lock:
            after = await _read_runtime_auth(self._home.auth_path)
            if auth_bundle_fingerprint(after) == auth_bundle_fingerprint(
                self._state.bundle
            ):
                if validate_unchanged:
                    await _bounded_phase(
                        self._lease.validate(),
                        self._checkpoint_timeout_seconds,
                        "Codex credential lease validation",
                    )
                return
            updated = checkpoint_credentials_from_bundle(
                self._state.credentials,
                after,
            )
            await _bounded_phase(
                self._lease.checkpoint(updated),
                self._checkpoint_timeout_seconds,
                "Codex credential checkpoint",
            )
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


async def _run_with_lease_guard(
    lease: CredentialLease,
    operation: Awaitable[ResultT],
    *,
    monitor_task: asyncio.Task[None] | None = None,
) -> ResultT:
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
