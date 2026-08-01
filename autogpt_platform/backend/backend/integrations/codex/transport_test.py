import asyncio
import base64
import json
import threading
from collections.abc import Callable, Coroutine
from pathlib import Path
from unittest.mock import AsyncMock

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
    transport = CodexTransport(
        temp_root=tmp_path,
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
    assert transport._capacity._value == 4
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
