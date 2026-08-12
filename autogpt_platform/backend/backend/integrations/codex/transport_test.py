import asyncio
import base64
import json
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.integrations.codex.auth_bundle import (
    CodexAuthBundleV1,
    CodexAuthTokensV1,
    materialize_auth_bundle,
)
from backend.integrations.codex.credential_codec import credentials_from_bundle
from backend.integrations.codex.models import (
    CodexAccountSnapshot,
    CodexDeviceCodeDetails,
    CodexDynamicToolCall,
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexInvocationResult,
    CodexModelInfo,
    CodexRateLimitsSnapshot,
)
from backend.integrations.codex.transport import (
    CodexTransport,
    CodexTransportError,
    CodexTransportOverloadedError,
    _close_unused_awaitable,
    _run_with_lease_guard,
)
from backend.integrations.credential_lease import CredentialLease


def test_process_capacity_is_safe_across_event_loops():
    transport = CodexTransport(max_active_processes=1)
    first_acquired = threading.Event()
    release_first = threading.Event()
    second_acquired = threading.Event()
    errors: list[BaseException] = []

    async def hold_first_slot() -> None:
        await transport._acquire_capacity()
        first_acquired.set()
        while not release_first.is_set():
            await asyncio.sleep(0.01)
        transport._capacity.release()

    async def wait_for_second_slot() -> None:
        await transport._acquire_capacity()
        second_acquired.set()
        transport._capacity.release()

    def run(coroutine: Callable[[], Coroutine[object, object, None]]) -> None:
        try:
            asyncio.run(coroutine())
        except BaseException as error:
            errors.append(error)

    first = threading.Thread(target=run, args=(hold_first_slot,))
    second = threading.Thread(target=run, args=(wait_for_second_slot,))
    first.start()
    try:
        assert first_acquired.wait(timeout=2)
        second.start()
        assert not second_acquired.wait(timeout=0.05)
        release_first.set()
        assert second_acquired.wait(timeout=2)
    finally:
        release_first.set()
        first.join(timeout=2)
        if second.ident is not None:
            second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not errors


def test_close_unused_awaitable_only_closes_native_coroutines():
    class CustomAwaitable:
        def __init__(self) -> None:
            self.close_called = False

        def __await__(self):
            if False:
                yield None
            return None

        def close(self) -> None:
            self.close_called = True

    operation = CustomAwaitable()

    _close_unused_awaitable(operation)

    assert not operation.close_called


@pytest.mark.asyncio
async def test_invoke_checkpoints_rotated_auth_before_cleanup(tmp_path):
    lease, checkpoint = _lease()
    runtime = _FakeRuntime(rotate_auth=True)
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    result = await transport.invoke(
        lease,
        CodexInvocationRequest(prompt="Reply with ok"),
    )

    assert result.final_response == "ok"
    assert result.resolved_model == "gpt-test"
    assert runtime.last_request is not None
    assert runtime.last_request.model == "gpt-test"
    checkpoint.assert_awaited_once()
    updated = checkpoint.await_args.args[0]
    assert updated.refresh_token.get_secret_value() == "refresh-rotated"
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_invoke_rejects_model_not_advertised_by_account(tmp_path):
    lease, _ = _lease()
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    with pytest.raises(CodexTransportError, match="not available"):
        await transport.invoke(
            lease,
            CodexInvocationRequest(prompt="ok", model="gpt-not-on-account"),
        )

    assert runtime.last_request is None
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_runtime_failure_still_checkpoints_rotated_auth(tmp_path):
    lease, checkpoint = _lease()
    runtime = _FakeRuntime(rotate_auth=True, fail_invoke=True)
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    with pytest.raises(RuntimeError, match="runtime failed"):
        await transport.invoke(lease, CodexInvocationRequest(prompt="fail"))

    checkpoint.assert_awaited_once()
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_unchanged_auth_does_not_checkpoint(tmp_path):
    lease, checkpoint = _lease()
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    await transport.account(lease)

    checkpoint.assert_not_awaited()
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_unchanged_auth_still_validates_lease(tmp_path):
    lease, checkpoint = _lease()
    lease._lock.owned.return_value = False
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    with pytest.raises(RuntimeError, match="ownership was lost"):
        await transport.account(lease)

    checkpoint.assert_not_awaited()
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_materialization_failure_releases_capacity_and_cleans_home(
    tmp_path,
    monkeypatch,
):
    lease, _ = _lease()
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        max_active_processes=1,
        runtime_factory=_runtime_factory(runtime),
    )

    def fail_materialization(*_args):
        raise RuntimeError("materialization failed")

    monkeypatch.setattr(
        "backend.integrations.codex.transport.materialize_auth_bundle",
        fail_materialization,
    )
    with pytest.raises(RuntimeError, match="materialization failed"):
        await transport.account(lease)

    assert not tuple(tmp_path.iterdir())
    assert transport._capacity._value == 1


@pytest.mark.asyncio
async def test_device_login_returns_bundle_and_cleans_home(tmp_path):
    runtime = _FakeRuntime(complete_login=True)
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    session = await transport.start_device_login()
    completion = await session.wait()

    assert session.details.user_code == "CODE-1234"
    assert completion.bundle.tokens.account_id == "account-user"
    assert completion.account is None
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_device_login_home_creation_failure_releases_capacity(
    tmp_path,
    monkeypatch,
):
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        max_active_processes=1,
        runtime_factory=_runtime_factory(runtime),
    )

    def fail_home_creation(*_args):
        raise RuntimeError("home creation failed")

    monkeypatch.setattr(
        "backend.integrations.codex.transport.TemporaryCodexHome.create",
        fail_home_creation,
    )
    with pytest.raises(RuntimeError, match="home creation failed"):
        await transport.start_device_login()

    assert transport._capacity._value == 1


@pytest.mark.asyncio
async def test_capacity_wait_is_bounded():
    transport = CodexTransport(
        max_active_processes=1,
        capacity_timeout_seconds=0.02,
    )
    await transport._acquire_capacity()
    try:
        with pytest.raises(CodexTransportOverloadedError):
            await transport._acquire_capacity()
    finally:
        transport._capacity.release()


@pytest.mark.asyncio
async def test_runtime_startup_timeout_releases_capacity_and_cleans_home(tmp_path):
    release_startup = asyncio.Event()

    async def blocked_runtime_factory(_home):
        await release_startup.wait()
        return _FakeRuntime()

    transport = CodexTransport(
        temp_root=tmp_path,
        max_active_processes=1,
        startup_timeout_seconds=0.02,
        runtime_factory=blocked_runtime_factory,
    )

    with pytest.raises(CodexTransportError, match="runtime startup"):
        await transport.start_device_login()

    assert transport._capacity._value == 1
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_invoke_checkpoints_refresh_before_starting_turn(tmp_path):
    lease, checkpoint = _lease()
    runtime = _FakeRuntime(rotate_on_account=True)
    runtime.checkpoint_probe = lambda: checkpoint.await_count == 1
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )

    await transport.invoke(lease, CodexInvocationRequest(prompt="ok"))

    checkpoint.assert_awaited_once()
    assert runtime.refresh_was_checkpointed_before_invoke


@pytest.mark.asyncio
async def test_active_invocation_checkpoints_late_auth_rotation(tmp_path):
    lease, checkpoint = _lease()
    rotated = asyncio.Event()
    checkpointed = asyncio.Event()
    finish = asyncio.Event()

    async def record_checkpoint(*_args):
        checkpointed.set()

    checkpoint.side_effect = record_checkpoint

    class RotatingRuntime(_FakeRuntime):
        async def invoke(
            self,
            request: CodexInvocationRequest,
        ) -> CodexInvocationResult:
            payload = json.loads((self.home / "auth.json").read_text())
            payload["tokens"]["refresh_token"] = "refresh-during-turn"
            (self.home / "auth.json").write_text(json.dumps(payload))
            rotated.set()
            await finish.wait()
            return await super().invoke(request)

    runtime = RotatingRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        checkpoint_interval_seconds=0.01,
        runtime_factory=_runtime_factory(runtime),
    )

    invocation = asyncio.create_task(
        transport.invoke(lease, CodexInvocationRequest(prompt="ok"))
    )
    await asyncio.wait_for(rotated.wait(), timeout=1)
    await asyncio.wait_for(checkpointed.wait(), timeout=1)
    assert not invocation.done()
    finish.set()
    await invocation

    checkpoint.assert_awaited_once()


@pytest.mark.asyncio
async def test_invocation_aborts_when_credential_lease_heartbeat_fails(tmp_path):
    lease, _ = _lease()
    started = asyncio.Event()
    canceled = asyncio.Event()

    class BlockingRuntime(_FakeRuntime):
        async def invoke(
            self,
            request: CodexInvocationRequest,
        ) -> CodexInvocationResult:
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                canceled.set()
            raise AssertionError("unreachable")

    runtime = BlockingRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    invocation = asyncio.create_task(
        transport.invoke(lease, CodexInvocationRequest(prompt="ok"))
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    lease._heartbeat_error = RuntimeError("redis lease lost")
    lease._heartbeat_failed.set()

    with pytest.raises(RuntimeError, match="heartbeat failed"):
        await invocation

    assert canceled.is_set()
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_agent_invocation_uses_separate_tool_timeout_and_checkpoints(tmp_path):
    lease, checkpoint = _lease()
    runtime = _FakeRuntime(rotate_on_account=True)
    transport = CodexTransport(
        temp_root=tmp_path,
        copilot_tool_timeout_seconds=17,
        runtime_factory=_runtime_factory(runtime),
    )
    tool = CodexDynamicToolSpec(
        name="find_agent",
        description="Find an agent",
        input_schema={"type": "object"},
    )

    async def tool_handler(_call: CodexDynamicToolCall) -> CodexDynamicToolResult:
        return CodexDynamicToolResult(content="ok")

    result = await transport.invoke_agent(
        lease,
        CodexInvocationRequest(prompt="Find it"),
        [tool],
        tool_handler,
    )

    assert result.final_response == "ok"
    assert runtime.agent_tool_timeout == 17
    assert runtime.agent_dynamic_tools == [tool]
    checkpoint.assert_awaited_once()
    assert runtime.closed
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_runtime_pool_overlaps_borrowers_on_one_lease_and_runtime(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )

    both_started = threading.Event()
    allow_finish = threading.Event()

    class ConcurrentRuntime(_FakeRuntime):
        def __init__(self):
            super().__init__()
            self._active_lock = threading.Lock()
            self.active = 0
            self.max_active = 0
            self.requests: list[CodexInvocationRequest] = []

        async def invoke_agent(
            self,
            request,
            dynamic_tools,
            tool_handler,
            event_handler=None,
            *,
            tool_timeout_seconds,
        ):
            del dynamic_tools, tool_handler, event_handler, tool_timeout_seconds
            self.requests.append(request)
            with self._active_lock:
                self.active += 1
                self.max_active = max(self.max_active, self.active)
                if self.active == 2:
                    both_started.set()
            try:
                while not allow_finish.is_set():
                    await asyncio.sleep(0.01)
                return CodexInvocationResult(
                    response_id=f"turn-{len(self.requests)}",
                    final_response=request.prompt,
                    status="completed",
                )
            finally:
                with self._active_lock:
                    self.active -= 1

    runtime = ConcurrentRuntime()
    factory = AsyncMock(side_effect=_runtime_factory(runtime))
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=factory,
    )

    first, second = await asyncio.gather(
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        ),
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        ),
    )

    async def tool_handler(_call):
        return CodexDynamicToolResult(content="ok")

    first_call = asyncio.create_task(
        first.invoke(
            CodexInvocationRequest(prompt="first"),
            [],
            tool_handler,
        )
    )
    second_call = asyncio.create_task(
        second.invoke(
            CodexInvocationRequest(prompt="second"),
            [],
            tool_handler,
        )
    )
    assert await asyncio.to_thread(both_started.wait, 2)
    assert not first_call.done()
    assert not second_call.done()
    assert manager.acquire_lease.await_count == 1
    assert factory.await_count == 1
    assert runtime.max_active == 2

    allow_finish.set()
    results = await asyncio.gather(first_call, second_call)
    assert {result.final_response for result in results} == {"first", "second"}

    await first.release()
    assert not runtime.closed
    raw_lease._lock.release.assert_not_awaited()
    await second.release()
    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    assert not tuple(tmp_path.iterdir())
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_cancels_one_borrower_without_closing_sibling(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    cancel_started = threading.Event()
    cancel_observed = threading.Event()
    peer_started = threading.Event()
    finish_peer = threading.Event()

    class CancellationRuntime(_FakeRuntime):
        async def invoke_agent(
            self,
            request,
            dynamic_tools,
            tool_handler,
            event_handler=None,
            *,
            tool_timeout_seconds,
        ):
            del dynamic_tools, tool_handler, event_handler, tool_timeout_seconds
            if request.prompt == "cancel":
                cancel_started.set()
                try:
                    await asyncio.Event().wait()
                finally:
                    cancel_observed.set()
            peer_started.set()
            while not finish_peer.is_set():
                await asyncio.sleep(0.01)
            return CodexInvocationResult(
                response_id="peer-turn",
                final_response="peer-ok",
                status="completed",
            )

    runtime = CancellationRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    first, second = await asyncio.gather(
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        ),
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        ),
    )

    async def tool_handler(_call):
        return CodexDynamicToolResult(content="ok")

    canceled_call = asyncio.create_task(
        first.invoke(CodexInvocationRequest(prompt="cancel"), [], tool_handler)
    )
    peer_call = asyncio.create_task(
        second.invoke(CodexInvocationRequest(prompt="peer"), [], tool_handler)
    )
    assert await asyncio.to_thread(cancel_started.wait, 2)
    assert await asyncio.to_thread(peer_started.wait, 2)
    canceled_call.cancel()
    with pytest.raises(asyncio.CancelledError):
        await canceled_call
    assert await asyncio.to_thread(cancel_observed.wait, 2)
    assert not peer_call.done()
    assert not runtime.closed

    finish_peer.set()
    assert (await peer_call).final_response == "peer-ok"
    await first.release()
    assert not runtime.closed
    await second.release()
    assert runtime.closed
    await transport.close_runtime_pool()


def test_runtime_pool_loop_start_is_singleton_across_threads():
    transport = CodexTransport()
    barrier = threading.Barrier(8)
    loops: list[asyncio.AbstractEventLoop] = []
    errors: list[BaseException] = []

    def start_loop() -> None:
        try:
            barrier.wait(timeout=2)
            loops.append(transport._runtime_pool._ensure_loop())
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=start_loop) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=2)

    assert not errors
    assert len(loops) == 8
    assert len({id(loop) for loop in loops}) == 1
    asyncio.run(transport.close_runtime_pool())


@pytest.mark.asyncio
async def test_runtime_pool_lock_timeout_does_not_bound_healthy_startup(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()

    async def delayed_runtime_factory(home):
        await asyncio.sleep(0.05)
        runtime.home = home.path
        return runtime

    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=delayed_runtime_factory,
    )

    lease = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=0.01,
    )

    assert manager.acquire_lease.await_count == 1
    await lease.release()
    assert runtime.closed
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_canceled_acquire_releases_completed_borrow(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    pool = transport._runtime_pool
    original_borrow = pool._borrow_on_actor_loop
    borrowed = threading.Event()

    async def hold_completed_borrow(*args, **kwargs):
        credentials = await original_borrow(*args, **kwargs)
        borrowed.set()
        await asyncio.Event().wait()
        return credentials

    monkeypatch.setattr(pool, "_borrow_on_actor_loop", hold_completed_borrow)
    acquire = asyncio.create_task(
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        )
    )
    assert await asyncio.to_thread(borrowed.wait, 2)
    acquire.cancel()

    with pytest.raises(asyncio.CancelledError):
        await acquire

    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    assert not tuple(tmp_path.iterdir())
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_release_can_retry_after_submission_failure(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    lease = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=1,
    )
    pool = transport._runtime_pool
    original_release = pool.release
    attempts = 0

    async def fail_once(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("release submission failed")
        return await original_release(*args, **kwargs)

    monkeypatch.setattr(pool, "release", fail_once)

    with pytest.raises(RuntimeError, match="release submission failed"):
        await lease.release()

    assert not runtime.closed
    await lease.release()
    assert attempts == 2
    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_canceled_release_follower_does_not_cancel_leader(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    lease = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=1,
    )
    pool = transport._runtime_pool
    original_release = pool.release
    release_started = asyncio.Event()
    allow_release = asyncio.Event()

    async def delayed_release(*args, **kwargs):
        release_started.set()
        await allow_release.wait()
        return await original_release(*args, **kwargs)

    monkeypatch.setattr(pool, "release", delayed_release)
    leader = asyncio.create_task(lease.release())
    await asyncio.wait_for(release_started.wait(), timeout=1)
    follower = asyncio.create_task(lease.release())
    await asyncio.sleep(0)
    follower.cancel()

    with pytest.raises(asyncio.CancelledError):
        await follower

    allow_release.set()
    await leader
    await lease.release()

    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    await transport.close_runtime_pool()


def test_runtime_pool_close_serializes_with_acquire_submission(tmp_path, monkeypatch):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    real_submit = asyncio.run_coroutine_threadsafe
    submit_entered = threading.Event()
    allow_submit = threading.Event()
    close_complete = threading.Event()
    acquire_results: list[object] = []
    acquire_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def block_first_submit(operation, loop):
        monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", real_submit)
        submit_entered.set()
        if not allow_submit.wait(timeout=2):
            operation.close()
            raise RuntimeError("timed out waiting to submit")
        return real_submit(operation, loop)

    monkeypatch.setattr(asyncio, "run_coroutine_threadsafe", block_first_submit)

    def acquire() -> None:
        try:
            acquire_results.append(
                asyncio.run(
                    transport.acquire_runtime_lease(
                        "user-1",
                        "credential-1",
                        lock_timeout_seconds=1,
                    )
                )
            )
        except BaseException as error:
            acquire_errors.append(error)

    def close() -> None:
        try:
            asyncio.run(transport.close_runtime_pool())
        except BaseException as error:
            close_errors.append(error)
        finally:
            close_complete.set()

    acquire_thread = threading.Thread(target=acquire)
    close_thread = threading.Thread(target=close)
    acquire_thread.start()
    assert submit_entered.wait(timeout=2)
    close_thread.start()
    assert not close_complete.wait(timeout=0.05)
    allow_submit.set()
    acquire_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert not acquire_thread.is_alive()
    assert not close_thread.is_alive()
    assert not close_errors
    assert not acquire_results
    assert len(acquire_errors) == 1
    assert isinstance(acquire_errors[0], CodexTransportError)
    if manager.acquire_lease.await_count:
        assert runtime.closed
        raw_lease._lock.release.assert_awaited_once()
    else:
        assert not runtime.closed
        raw_lease._lock.release.assert_not_awaited()
    assert not tuple(tmp_path.iterdir())
    with pytest.raises(CodexTransportError, match="shutting down"):
        asyncio.run(
            transport.acquire_runtime_lease(
                "user-1",
                "credential-1",
                lock_timeout_seconds=1,
            )
        )


def test_runtime_pool_close_racing_completed_start_cleans_actor_once(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    pool = transport._runtime_pool
    original_start = pool._start_actor
    actor_ready = threading.Event()
    allow_start_return = threading.Event()

    async def gated_start(*args, **kwargs):
        actor = await original_start(*args, **kwargs)
        actor_ready.set()
        if not allow_start_return.wait(timeout=2):
            raise RuntimeError("timed out waiting to return started actor")
        return actor

    monkeypatch.setattr(pool, "_start_actor", gated_start)
    acquire_results: list[object] = []
    acquire_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def acquire() -> None:
        try:
            acquire_results.append(
                asyncio.run(
                    transport.acquire_runtime_lease(
                        "user-1",
                        "credential-1",
                        lock_timeout_seconds=1,
                    )
                )
            )
        except BaseException as error:
            acquire_errors.append(error)

    def close() -> None:
        try:
            asyncio.run(transport.close_runtime_pool())
        except BaseException as error:
            close_errors.append(error)

    acquire_thread = threading.Thread(target=acquire)
    close_thread = threading.Thread(target=close)
    acquire_thread.start()
    assert actor_ready.wait(timeout=2)
    close_thread.start()
    for _ in range(200):
        with pool._state_lock:
            if pool._closing:
                break
        threading.Event().wait(0.01)
    else:
        pytest.fail("runtime pool did not begin closing")
    allow_start_return.set()
    acquire_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert not acquire_thread.is_alive()
    assert not close_thread.is_alive()
    assert not close_errors
    assert not acquire_results
    assert len(acquire_errors) == 1
    assert isinstance(acquire_errors[0], CodexTransportError)
    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    assert not tuple(tmp_path.iterdir())


def test_runtime_pool_close_waits_for_loop_bootstrap(monkeypatch):
    transport = CodexTransport()
    pool = transport._runtime_pool
    real_new_event_loop = asyncio.new_event_loop
    bootstrap_entered = threading.Event()
    allow_bootstrap = threading.Event()
    acquire_errors: list[BaseException] = []
    close_errors: list[BaseException] = []

    def gated_new_event_loop():
        if threading.current_thread().name == "autogpt-codex-runtime-pool":
            bootstrap_entered.set()
            if not allow_bootstrap.wait(timeout=2):
                raise RuntimeError("timed out waiting to bootstrap actor loop")
        return real_new_event_loop()

    monkeypatch.setattr(asyncio, "new_event_loop", gated_new_event_loop)

    def acquire() -> None:
        try:
            asyncio.run(
                transport.acquire_runtime_lease(
                    "user-1",
                    "credential-1",
                    lock_timeout_seconds=1,
                )
            )
        except BaseException as error:
            acquire_errors.append(error)

    def close() -> None:
        try:
            asyncio.run(transport.close_runtime_pool())
        except BaseException as error:
            close_errors.append(error)

    acquire_thread = threading.Thread(target=acquire)
    close_thread = threading.Thread(target=close)
    acquire_thread.start()
    assert bootstrap_entered.wait(timeout=2)
    close_thread.start()
    for _ in range(200):
        with pool._state_lock:
            if pool._closing:
                break
        threading.Event().wait(0.01)
    else:
        pytest.fail("runtime pool did not begin closing")
    allow_bootstrap.set()
    acquire_thread.join(timeout=5)
    close_thread.join(timeout=5)

    assert not acquire_thread.is_alive()
    assert not close_thread.is_alive()
    assert not close_errors
    assert len(acquire_errors) == 1
    assert isinstance(acquire_errors[0], CodexTransportError)
    with pool._state_lock:
        assert pool._thread is None
        assert pool._loop is None
        assert pool._closed


@pytest.mark.asyncio
async def test_runtime_pool_release_rejects_late_invocation_while_draining(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    active_started = threading.Event()
    cancellation_seen = threading.Event()
    allow_cancellation = threading.Event()
    late_started = threading.Event()

    class ReleaseRaceRuntime(_FakeRuntime):
        async def invoke_agent(
            self,
            request,
            dynamic_tools,
            tool_handler,
            event_handler=None,
            *,
            tool_timeout_seconds,
        ):
            del dynamic_tools, tool_handler, event_handler, tool_timeout_seconds
            if request.prompt != "active":
                late_started.set()
                return CodexInvocationResult(
                    response_id="late",
                    final_response="late",
                    status="completed",
                )
            active_started.set()
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cancellation_seen.set()
                while not allow_cancellation.is_set():
                    await asyncio.sleep(0.01)
                raise

    runtime = ReleaseRaceRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    lease = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=1,
    )

    async def tool_handler(_call):
        return CodexDynamicToolResult(content="ok")

    active = asyncio.create_task(
        lease.invoke(
            CodexInvocationRequest(prompt="active"),
            [],
            tool_handler,
        )
    )
    assert await asyncio.to_thread(active_started.wait, 2)
    release = asyncio.create_task(lease.release())
    assert await asyncio.to_thread(cancellation_seen.wait, 2)

    with pytest.raises(RuntimeError, match="no longer active"):
        await lease._pool.invoke(
            lease._key,
            lease._handle_id,
            CodexInvocationRequest(prompt="late"),
            [],
            tool_handler,
            None,
        )

    assert not late_started.is_set()
    allow_cancellation.set()
    with pytest.raises(asyncio.CancelledError):
        await active
    await release

    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_release_cancels_and_drains_model_lookup(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    models_started = threading.Event()
    models_canceled = threading.Event()

    class BlockingModelsRuntime(_FakeRuntime):
        async def models(self):
            models_started.set()
            try:
                await asyncio.Event().wait()
            finally:
                models_canceled.set()

    runtime = BlockingModelsRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    lease = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=1,
    )
    models = asyncio.create_task(lease.models())
    assert await asyncio.to_thread(models_started.wait, 2)

    await lease.release()
    with pytest.raises(asyncio.CancelledError):
        await models

    assert models_canceled.is_set()
    assert runtime.closed
    raw_lease._lock.release.assert_awaited_once()
    assert not tuple(tmp_path.iterdir())
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_rejects_borrower_after_lease_heartbeat_failure(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    first = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=1,
    )
    raw_lease._heartbeat_error = RuntimeError("redis lease lost")

    with pytest.raises(CodexTransportError, match="runtime is unavailable"):
        await transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        )

    assert manager.acquire_lease.await_count == 1
    raw_lease._heartbeat_error = None
    await first.release()
    assert runtime.closed
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_runtime_pool_rejects_active_call_after_lease_heartbeat_failure(
    tmp_path,
    monkeypatch,
):
    raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(return_value=raw_lease)
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    runtime = _FakeRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=_runtime_factory(runtime),
    )
    lease = await transport.acquire_runtime_lease(
        "user-1",
        "credential-1",
        lock_timeout_seconds=1,
    )
    raw_lease._heartbeat_error = RuntimeError("redis lease lost")

    async def tool_handler(_call):
        return CodexDynamicToolResult(content="ok")

    with pytest.raises(CodexTransportError, match="runtime is unavailable"):
        await lease.invoke(
            CodexInvocationRequest(prompt="must not start"),
            [],
            tool_handler,
        )

    assert runtime.last_request is None
    raw_lease._heartbeat_error = None
    await lease.release()
    assert runtime.closed
    await transport.close_runtime_pool()


@pytest.mark.asyncio
async def test_lease_guard_closes_operation_before_known_heartbeat_failure():
    lease, _ = _lease()
    lease._heartbeat_error = RuntimeError("redis lease lost")

    async def operation() -> str:
        raise AssertionError("provider operation must not start")

    provider_operation = operation()
    with pytest.raises(RuntimeError, match="heartbeat failed"):
        await _run_with_lease_guard(lease, provider_operation)

    assert provider_operation.cr_frame is None


@pytest.mark.asyncio
async def test_lease_guard_closes_operation_before_failed_monitor():
    lease, _ = _lease()

    async def fail_monitor() -> None:
        raise RuntimeError("auth monitor failed")

    monitor = asyncio.create_task(fail_monitor())
    await asyncio.sleep(0)

    async def operation() -> str:
        raise AssertionError("provider operation must not start")

    provider_operation = operation()
    with pytest.raises(RuntimeError, match="auth monitor failed"):
        await _run_with_lease_guard(
            lease,
            provider_operation,
            monitor_task=monitor,
        )

    assert provider_operation.cr_frame is None


@pytest.mark.asyncio
async def test_runtime_pool_shutdown_cleans_other_actors_after_close_error(
    tmp_path,
    monkeypatch,
):
    first_raw_lease, _ = _lease()
    second_raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(side_effect=[first_raw_lease, second_raw_lease])
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )

    class FailingCloseRuntime(_FakeRuntime):
        async def close(self):
            self.closed = True
            raise RuntimeError("runtime close failed")

    first_runtime = FailingCloseRuntime()
    second_runtime = FailingCloseRuntime()
    runtimes = iter((first_runtime, second_runtime))

    async def runtime_factory(home):
        runtime = next(runtimes)
        runtime.home = home.path
        return runtime

    transport = CodexTransport(
        temp_root=tmp_path,
        control_timeout_seconds=0.1,
        runtime_factory=runtime_factory,
    )
    await asyncio.gather(
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        ),
        transport.acquire_runtime_lease(
            "user-2",
            "credential-2",
            lock_timeout_seconds=1,
        ),
    )

    results = await asyncio.gather(
        transport.close_runtime_pool(),
        transport.close_runtime_pool(),
        return_exceptions=True,
    )

    assert len(results) == 2
    assert all(isinstance(result, CodexTransportError) for result in results)
    assert first_runtime.closed
    assert second_runtime.closed
    assert any(
        "Additional cleanup failure: RuntimeError: runtime close failed" in note
        for result in results
        if isinstance(result, CodexTransportError)
        for note in result.__notes__
    )
    first_raw_lease._lock.release.assert_awaited_once()
    second_raw_lease._lock.release.assert_awaited_once()
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_runtime_pool_shutdown_closes_actors_concurrently(tmp_path, monkeypatch):
    first_raw_lease, _ = _lease()
    second_raw_lease, _ = _lease()
    manager = MagicMock()
    manager.acquire_lease = AsyncMock(side_effect=[first_raw_lease, second_raw_lease])
    monkeypatch.setattr(
        "backend.integrations.creds_manager.IntegrationCredentialsManager",
        lambda: manager,
    )
    first_runtime = _FakeRuntime()
    second_runtime = _FakeRuntime()
    runtimes = iter((first_runtime, second_runtime))

    async def runtime_factory(home):
        runtime = next(runtimes)
        runtime.home = home.path
        return runtime

    transport = CodexTransport(
        temp_root=tmp_path,
        runtime_factory=runtime_factory,
    )
    await asyncio.gather(
        transport.acquire_runtime_lease(
            "user-1",
            "credential-1",
            lock_timeout_seconds=1,
        ),
        transport.acquire_runtime_lease(
            "user-2",
            "credential-2",
            lock_timeout_seconds=1,
        ),
    )
    pool = transport._runtime_pool
    original_close = pool._close_actor_bounded
    close_count = 0
    close_count_lock = threading.Lock()
    both_closing = threading.Event()
    allow_close = threading.Event()

    async def delayed_close(actor):
        nonlocal close_count
        with close_count_lock:
            close_count += 1
            if close_count == 2:
                both_closing.set()
        while not allow_close.is_set():
            await asyncio.sleep(0.01)
        await original_close(actor)

    monkeypatch.setattr(pool, "_close_actor_bounded", delayed_close)
    shutdown = asyncio.create_task(transport.close_runtime_pool())
    reached_both = await asyncio.to_thread(both_closing.wait, 2)
    allow_close.set()
    await shutdown

    assert reached_both
    assert first_runtime.closed
    assert second_runtime.closed
    first_raw_lease._lock.release.assert_awaited_once()
    second_raw_lease._lock.release.assert_awaited_once()
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_never_returning_runtime_close_cannot_strand_home_or_capacity(tmp_path):
    lease, _ = _lease()

    class NonCancellingFuture(asyncio.Future[None]):
        def cancel(self, *_args, **_kwargs):
            return False

    class NeverClosingRuntime(_FakeRuntime):
        def close(self):
            return NonCancellingFuture()

    runtime = NeverClosingRuntime()
    transport = CodexTransport(
        temp_root=tmp_path,
        max_active_processes=1,
        control_timeout_seconds=0.01,
        runtime_factory=_runtime_factory(runtime),
    )

    with pytest.raises(CodexTransportError, match="runtime shutdown"):
        await asyncio.wait_for(transport._account(lease), timeout=0.2)

    assert transport._capacity._value == 1
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_cancel_during_runtime_startup_drains_factory_before_home_cleanup(
    tmp_path,
):
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked_runtime_factory(_home):
        started.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    transport = CodexTransport(
        temp_root=tmp_path,
        max_active_processes=1,
        runtime_factory=blocked_runtime_factory,
    )
    login = asyncio.create_task(transport.start_device_login())
    await asyncio.wait_for(started.wait(), timeout=1)
    login.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(login, timeout=2)

    assert cancelled.is_set()
    assert transport._capacity._value == 1
    assert not tuple(tmp_path.iterdir())


@pytest.mark.asyncio
async def test_caller_cancellation_stops_active_runtime_operation(tmp_path):
    lease, _ = _lease()
    started = asyncio.Event()
    cancelled = asyncio.Event()

    class BlockingRuntime(_FakeRuntime):
        async def invoke(self, request):
            started.set()
            try:
                await asyncio.Event().wait()
            finally:
                cancelled.set()

    runtime = BlockingRuntime()
    max_active_processes = 2
    transport = CodexTransport(
        temp_root=tmp_path,
        max_active_processes=max_active_processes,
        runtime_factory=_runtime_factory(runtime),
    )
    invocation = asyncio.create_task(
        transport.invoke(lease, CodexInvocationRequest(prompt="ok"))
    )
    await asyncio.wait_for(started.wait(), timeout=1)
    invocation.cancel()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(invocation, timeout=2)

    assert cancelled.is_set()
    assert runtime.closed
    assert transport._capacity._value == max_active_processes
    assert not tuple(tmp_path.iterdir())


def _runtime_factory(runtime: "_FakeRuntime"):
    async def create(home):
        runtime.home = home.path
        return runtime

    return create


def _lease() -> tuple[CredentialLease, AsyncMock]:
    credentials = credentials_from_bundle(_bundle())
    lock = AsyncMock()
    lock.locked.return_value = True
    lock.owned.return_value = True
    lock.timeout = 60
    checkpoint = AsyncMock()
    return CredentialLease(credentials, lock, checkpoint), checkpoint


class _FakeLogin:
    details = CodexDeviceCodeDetails(
        login_id="login-1",
        verification_url="https://example.com/device",
        user_code="CODE-1234",
    )

    async def wait(self) -> bool:
        return True

    async def cancel(self) -> None:
        return None


class _FakeRuntime:
    def __init__(
        self,
        *,
        rotate_auth: bool = False,
        fail_invoke: bool = False,
        complete_login: bool = False,
        rotate_on_account: bool = False,
    ) -> None:
        self.home = Path()
        self.rotate_auth = rotate_auth
        self.fail_invoke = fail_invoke
        self.complete_login = complete_login
        self.rotate_on_account = rotate_on_account
        self.closed = False
        self.refresh_was_checkpointed_before_invoke = False
        self.checkpoint_probe: Callable[[], bool] = lambda: False
        self.agent_tool_timeout: float | None = None
        self.agent_dynamic_tools: list[CodexDynamicToolSpec] | None = None
        self.last_request: CodexInvocationRequest | None = None

    async def start_device_code_login(self) -> _FakeLogin:
        if self.complete_login:
            materialize_auth_bundle(_bundle(), self.home / "auth.json")
        return _FakeLogin()

    async def account(self, *, refresh_token: bool = True) -> CodexAccountSnapshot:
        assert refresh_token
        if self.rotate_on_account:
            payload = json.loads((self.home / "auth.json").read_text())
            payload["tokens"]["refresh_token"] = "refresh-from-account"
            (self.home / "auth.json").write_text(json.dumps(payload))
        return CodexAccountSnapshot(
            connected=True,
            requires_openai_auth=True,
            account_type="chatgpt",
            email="user@example.com",
            plan_type="pro",
        )

    async def rate_limits(self) -> CodexRateLimitsSnapshot:
        return CodexRateLimitsSnapshot(plan_type="pro")

    async def models(self) -> list[CodexModelInfo]:
        return [
            CodexModelInfo(
                model="gpt-test",
                display_name="GPT Test",
                is_default=True,
                hidden=False,
                default_reasoning_effort="medium",
                supported_reasoning_efforts=["low", "medium", "high"],
                input_modalities=["text"],
            )
        ]

    async def invoke(self, request: CodexInvocationRequest) -> CodexInvocationResult:
        self.last_request = request
        if self.rotate_on_account:
            self.refresh_was_checkpointed_before_invoke = self.checkpoint_probe()
        if self.rotate_auth:
            payload = json.loads((self.home / "auth.json").read_text())
            payload["tokens"]["refresh_token"] = "refresh-rotated"
            (self.home / "auth.json").write_text(json.dumps(payload))
        if self.fail_invoke:
            raise RuntimeError("runtime failed")
        return CodexInvocationResult(
            response_id="turn-1",
            final_response="ok",
            status="completed",
        )

    async def invoke_agent(
        self,
        request: CodexInvocationRequest,
        dynamic_tools: list[CodexDynamicToolSpec],
        tool_handler,
        event_handler=None,
        *,
        tool_timeout_seconds: float,
    ) -> CodexInvocationResult:
        self.agent_tool_timeout = tool_timeout_seconds
        self.agent_dynamic_tools = dynamic_tools
        return await self.invoke(request)

    async def logout(self) -> None:
        return None

    async def close(self) -> None:
        self.closed = True


def _bundle() -> CodexAuthBundleV1:
    return CodexAuthBundleV1(
        tokens=CodexAuthTokensV1(
            id_token=_jwt({"email": "user@example.com"}),
            access_token=_jwt({"exp": 1_900_000_000}),
            refresh_token="refresh-original",
            account_id="account-user",
        ),
        codex_runtime_version="0.144.4",
    )


def _jwt(payload: dict[str, object]) -> str:
    encoded = (
        base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    )
    return f"header.{encoded}.signature"
