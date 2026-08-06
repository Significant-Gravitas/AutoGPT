import asyncio
import concurrent.futures
import importlib.metadata
import os
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from codex_cli_bin import bundled_codex_path
from openai_codex import AsyncCodex

import backend.integrations.codex.runtime as runtime_module
from backend.integrations.codex.models import (
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
    CodexModelInfo,
)
from backend.integrations.codex.runtime import (
    CODEX_RUNTIME_VERSION,
    OPENAI_CODEX_VERSION,
    CodexRuntime,
    CodexRuntimeError,
    _deny_server_request,
    _install_concurrent_server_request_dispatcher,
    _install_fail_closed_approval_handler,
    assert_pinned_versions,
    build_runtime_config,
)
from backend.integrations.codex.temporary_home import TemporaryCodexHome


def test_sdk_and_runtime_versions_are_an_atomic_pin():
    assert importlib.metadata.version("openai-codex") == OPENAI_CODEX_VERSION
    assert importlib.metadata.version("openai-codex-cli-bin") == CODEX_RUNTIME_VERSION
    assert_pinned_versions()


def test_runtime_config_uses_sanitizing_launcher_and_isolated_home(tmp_path):
    with TemporaryCodexHome.create(tmp_path) as home:
        config = build_runtime_config(home)

    arguments = config.launch_args_override
    assert arguments is not None
    assert arguments[0] == sys.executable
    assert (
        Path(arguments[1]).resolve()
        == Path(__file__).with_name("launcher.py").resolve()
    )
    assert arguments[2] == str(bundled_codex_path())
    assert arguments[-3:] == ("app-server", "--listen", "stdio://")
    assert 'cli_auth_credentials_store="file"' in arguments
    assert 'forced_login_method="chatgpt"' in arguments
    assert config.experimental_api is True
    assert config.env is not None
    for variable in (
        "APPDATA",
        "CODEX_HOME",
        "CODEX_SQLITE_HOME",
        "HOME",
        "LOCALAPPDATA",
        "TEMP",
        "TMP",
        "TMPDIR",
        "USERPROFILE",
        "XDG_CACHE_HOME",
        "XDG_CONFIG_HOME",
        "XDG_DATA_HOME",
    ):
        assert config.env[variable].startswith(str(home.path))
    assert config.env["TEMP"] == str(home.temp_path)
    assert config.env["TMP"] == str(home.temp_path)
    assert config.env["TMPDIR"] == str(home.temp_path)
    assert config.env["RUST_LOG"] == "warn"
    assert "DATABASE_URL" not in config.env
    assert "ENCRYPTION_KEY" not in config.env
    if os.name == "nt":
        assert config.env["SYSTEMROOT"] == os.environ["SYSTEMROOT"]


def test_sanitizing_launcher_starts_from_isolated_working_directory(tmp_path):
    with TemporaryCodexHome.create(tmp_path) as home:
        config = build_runtime_config(home)
        arguments = config.launch_args_override
        assert arguments is not None
        result = subprocess.run(
            [*arguments[:-3], "--version"],
            cwd=home.workspace_path,
            env=config.env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )

    assert result.returncode == 0, result.stderr
    assert CODEX_RUNTIME_VERSION in result.stdout


def test_runtime_installs_protocol_exact_fail_closed_approval_handler(tmp_path):
    with TemporaryCodexHome.create(tmp_path) as home:
        client = AsyncCodex(build_runtime_config(home))
        _install_fail_closed_approval_handler(client)
        handler = client._client._sync._approval_handler

        assert handler("item/commandExecution/requestApproval", {}) == {
            "decision": "decline"
        }
        assert handler("item/fileChange/requestApproval", {}) == {"decision": "decline"}
        assert handler("applyPatchApproval", {}) == {"decision": "denied"}
        assert handler("execCommandApproval", {}) == {"decision": "denied"}
        assert handler("item/permissions/requestApproval", {}) == {
            "permissions": {},
            "scope": "turn",
        }


def test_approval_handler_fails_closed_for_unknown_server_requests():
    with pytest.raises(CodexRuntimeError, match="Unsupported Codex server request"):
        _deny_server_request("item/future/requestApproval", {})


async def test_model_discovery_preserves_account_catalog_metadata(tmp_path):
    client = SimpleNamespace(
        models=AsyncMock(
            return_value=SimpleNamespace(
                data=[
                    SimpleNamespace(
                        model="gpt-5.6-sol",
                        display_name="GPT-5.6 Sol",
                        is_default=True,
                        hidden=False,
                        default_reasoning_effort=SimpleNamespace(value="high"),
                        supported_reasoning_efforts=[
                            SimpleNamespace(
                                reasoning_effort=SimpleNamespace(value="low")
                            ),
                            SimpleNamespace(
                                reasoning_effort=SimpleNamespace(value="ultra")
                            ),
                        ],
                        input_modalities=[
                            SimpleNamespace(value="text"),
                            SimpleNamespace(value="image"),
                        ],
                    )
                ]
            )
        )
    )

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        models = await runtime.models()

    assert models == [
        CodexModelInfo(
            model="gpt-5.6-sol",
            display_name="GPT-5.6 Sol",
            is_default=True,
            hidden=False,
            default_reasoning_effort="high",
            supported_reasoning_efforts=["low", "ultra"],
            input_modalities=["text", "image"],
        )
    ]
    client.models.assert_awaited_once_with(include_hidden=True)


async def test_repeated_app_server_start_close_leaves_no_temporary_home(tmp_path):
    created_paths: list[Path] = []
    for _attempt in range(2):
        home = TemporaryCodexHome.create(tmp_path)
        created_paths.append(home.path)
        runtime: CodexRuntime | None = None
        try:
            runtime = await asyncio.wait_for(CodexRuntime.start(home), timeout=20)
            handler = runtime._client._client._sync._approval_handler
            assert handler("item/commandExecution/requestApproval", {}) == {
                "decision": "decline"
            }
        finally:
            if runtime is not None:
                await asyncio.wait_for(runtime.close(), timeout=10)
            home.cleanup()

    assert all(not path.exists() for path in created_paths)


async def test_thread_start_explicitly_removes_all_runtime_environments(tmp_path):
    request_call = AsyncMock(
        return_value=SimpleNamespace(thread=SimpleNamespace(id="thread-1"))
    )
    client = SimpleNamespace(_client=SimpleNamespace(request=request_call))

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        await runtime._start_toolless_thread(
            CodexInvocationRequest(
                prompt="Do not access the host",
                instructions="Return text only",
                model="gpt-test",
            )
        )

        method, params = request_call.await_args.args[:2]
        assert method == "thread/start"
        assert params["environments"] == []
        assert params["dynamicTools"] == []
        assert params["runtimeWorkspaceRoots"] == []
        assert params["selectedCapabilityRoots"] == []
        assert params["sandbox"] == "read-only"
        assert params["cwd"] == str(home.workspace_path)
        assert params["config"]["web_search"] == "disabled"
        assert all(
            enabled is False for enabled in params["config"]["features"].values()
        )


async def test_thread_start_exposes_only_supplied_dynamic_tools(tmp_path):
    request_call = AsyncMock(
        return_value=SimpleNamespace(thread=SimpleNamespace(id="thread-1"))
    )
    client = SimpleNamespace(_client=SimpleNamespace(request=request_call))

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        await runtime._start_thread(
            CodexInvocationRequest(prompt="List my agents"),
            [
                CodexDynamicToolSpec(
                    name="find_agent",
                    description="Find an AutoGPT agent",
                    input_schema={"type": "object", "properties": {}},
                )
            ],
        )

    params = request_call.await_args.args[1]
    assert params["dynamicTools"] == [
        {
            "type": "function",
            "name": "find_agent",
            "description": "Find an AutoGPT agent",
            "inputSchema": {"type": "object", "properties": {}},
        }
    ]


async def test_dynamic_tool_request_round_trips_on_copilot_loop(tmp_path):
    sync_client = SimpleNamespace(_approval_handler=None)
    client = SimpleNamespace(_client=SimpleNamespace(_sync=sync_client))
    observed = []

    async def execute(call):
        observed.append(call)
        return CodexDynamicToolResult(content='{"agents": []}', success=True)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        runtime._register_dynamic_tool_handler("thread-1", execute, timeout_seconds=1)
        response = await asyncio.to_thread(
            sync_client._approval_handler,
            "item/tool/call",
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "callId": "call-1",
                "namespace": "autogpt",
                "tool": "find_agent",
                "arguments": {"query": "demo"},
            },
        )

    assert observed[0].call_id == "call-1"
    assert observed[0].tool == "find_agent"
    assert observed[0].arguments == {"query": "demo"}
    assert response == {
        "contentItems": [{"type": "inputText", "text": '{"agents": []}'}],
        "success": True,
    }


async def test_dynamic_tool_timeout_cancels_callback_and_fails_closed(tmp_path):
    sync_client = SimpleNamespace(_approval_handler=None)
    client = SimpleNamespace(_client=SimpleNamespace(_sync=sync_client))
    callback_cancelled = asyncio.Event()

    async def execute(_call):
        try:
            await asyncio.Event().wait()
        finally:
            callback_cancelled.set()

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        runtime._register_dynamic_tool_handler(
            "thread-1", execute, timeout_seconds=0.01
        )
        response = await asyncio.to_thread(
            sync_client._approval_handler,
            "item/tool/call",
            {
                "threadId": "thread-1",
                "turnId": "turn-1",
                "callId": "call-1",
                "tool": "find_agent",
                "arguments": {},
            },
        )
        await asyncio.wait_for(callback_cancelled.wait(), timeout=1)

    assert response == {
        "contentItems": [{"type": "inputText", "text": "codex_tool_execution_timeout"}],
        "success": False,
    }


async def test_unregister_cancels_tool_reserved_during_dispatch(tmp_path, monkeypatch):
    class ObservedLock:
        def __init__(self) -> None:
            self._lock = threading.Lock()
            self.owner_ident: int | None = None
            self.unregister_attempted = threading.Event()
            self.unregister_acquired = threading.Event()

        def __enter__(self):
            is_unregister = threading.current_thread().name == "codex-unregister"
            if is_unregister:
                self.unregister_attempted.set()
            self._lock.acquire()
            self.owner_ident = threading.get_ident()
            if is_unregister:
                self.unregister_acquired.set()
            return self

        def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
            self.owner_ident = None
            self._lock.release()

    observed_lock = ObservedLock()
    sync_client = SimpleNamespace(_approval_handler=None)
    client = SimpleNamespace(_client=SimpleNamespace(_sync=sync_client))
    scheduler_entered = threading.Event()
    allow_scheduler_return = threading.Event()
    scheduler_held_lock: list[bool] = []
    future: concurrent.futures.Future[CodexDynamicToolResult] = (
        concurrent.futures.Future()
    )

    async def execute(_call):
        return CodexDynamicToolResult(content="unused", success=True)

    def schedule(coroutine, _loop):
        coroutine.close()
        scheduler_held_lock.append(observed_lock.owner_ident == threading.get_ident())
        scheduler_entered.set()
        assert allow_scheduler_return.wait(timeout=1)
        return future

    monkeypatch.setattr(runtime_module.asyncio, "run_coroutine_threadsafe", schedule)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        runtime._dynamic_tool_futures_lock = observed_lock  # type: ignore[assignment]
        runtime._register_dynamic_tool_handler("thread-1", execute, timeout_seconds=1)
        dispatch = asyncio.create_task(
            asyncio.to_thread(
                sync_client._approval_handler,
                "item/tool/call",
                {
                    "threadId": "thread-1",
                    "turnId": "turn-1",
                    "callId": "call-1",
                    "tool": "find_agent",
                    "arguments": {},
                },
            )
        )
        assert await asyncio.to_thread(scheduler_entered.wait, 1)

        unregister_done = threading.Event()

        def unregister() -> None:
            runtime._unregister_dynamic_tool_handler("thread-1")
            unregister_done.set()

        unregister_thread = threading.Thread(
            target=unregister,
            name="codex-unregister",
        )
        unregister_thread.start()
        assert await asyncio.to_thread(observed_lock.unregister_attempted.wait, 1)
        if scheduler_held_lock == [True]:
            assert not observed_lock.unregister_acquired.is_set()
        else:
            assert await asyncio.to_thread(unregister_done.wait, 1)

        allow_scheduler_return.set()
        assert await asyncio.to_thread(unregister_done.wait, 1)
        unregister_thread.join(timeout=1)
        assert not unregister_thread.is_alive()
        response = await dispatch

    assert scheduler_held_lock == [True]
    assert future.cancelled()
    assert "thread-1" not in runtime._dynamic_tool_handlers
    assert "thread-1" not in runtime._dynamic_tool_futures
    assert response == {
        "contentItems": [{"type": "inputText", "text": "codex_tool_execution_failed"}],
        "success": False,
    }


def test_concurrent_dispatcher_keeps_reading_while_tool_result_is_pending():
    handler_started = threading.Event()
    allow_handler = threading.Event()
    notification_routed = threading.Event()
    response_written = threading.Event()
    messages = [
        {"id": "request-1", "method": "item/tool/call", "params": {}},
        {"method": "item/agentMessage/delta", "params": {"delta": "still-live"}},
    ]

    class Router:
        def route_notification(self, _notification):
            notification_routed.set()

        def route_response(self, _message):
            raise AssertionError("unexpected response")

        def fail_all(self, _error):
            pass

    class SyncClient:
        _proc = object()
        _router = Router()

        def _read_message(self):
            if messages:
                return messages.pop(0)
            raise EOFError

        def _handle_server_request(self, _message):
            handler_started.set()
            assert allow_handler.wait(timeout=1)
            return {"success": True}

        def _write_message(self, message):
            assert message == {
                "id": "request-1",
                "result": {"success": True},
            }
            response_written.set()

        def _coerce_notification(self, method, params):
            return method, params

    sync_client = SyncClient()
    client = SimpleNamespace(_client=SimpleNamespace(_sync=sync_client))
    _install_concurrent_server_request_dispatcher(client)  # type: ignore[arg-type]

    reader = threading.Thread(target=sync_client._reader_loop, daemon=True)
    reader.start()
    assert handler_started.wait(timeout=1)
    assert notification_routed.wait(timeout=1)
    assert not response_written.is_set()
    allow_handler.set()
    assert response_written.wait(timeout=1)


def test_concurrent_dispatcher_rejects_requests_above_its_bound(monkeypatch):
    monkeypatch.setattr(runtime_module, "_SERVER_REQUEST_MAX_CONCURRENCY", 1)
    handler_started = threading.Event()
    allow_handler = threading.Event()
    notification_routed = threading.Event()
    first_response_written = threading.Event()
    overload_response_written = threading.Event()
    messages = [
        {"id": "request-1", "method": "item/tool/call", "params": {}},
        {"id": "request-2", "method": "item/tool/call", "params": {}},
        {"method": "item/agentMessage/delta", "params": {"delta": "still-live"}},
    ]

    class Router:
        def route_notification(self, _notification):
            notification_routed.set()

        def route_response(self, _message):
            raise AssertionError("unexpected response")

        def fail_all(self, _error):
            pass

    class SyncClient:
        _proc = object()
        _router = Router()

        def _read_message(self):
            if messages:
                return messages.pop(0)
            raise EOFError

        def _handle_server_request(self, message):
            assert message["id"] == "request-1"
            handler_started.set()
            assert allow_handler.wait(timeout=1)
            return {"success": True}

        def _write_message(self, message):
            if message.get("id") == "request-1":
                assert message == {
                    "id": "request-1",
                    "result": {"success": True},
                }
                first_response_written.set()
                return
            assert message == {
                "id": "request-2",
                "error": {
                    "code": -32001,
                    "message": "Codex tool bridge is busy",
                },
            }
            overload_response_written.set()

        def _coerce_notification(self, method, params):
            return method, params

    sync_client = SyncClient()
    client = SimpleNamespace(_client=SimpleNamespace(_sync=sync_client))
    _install_concurrent_server_request_dispatcher(client)  # type: ignore[arg-type]

    reader = threading.Thread(target=sync_client._reader_loop, daemon=True)
    reader.start()
    assert handler_started.wait(timeout=1)
    assert overload_response_written.wait(timeout=1)
    assert notification_routed.wait(timeout=1)
    allow_handler.set()
    assert first_response_written.wait(timeout=1)


async def test_runtime_close_drains_accepted_server_requests(tmp_path):
    handler_started = threading.Event()
    allow_handler = threading.Event()
    response_written = threading.Event()
    client_close_called = asyncio.Event()
    messages = [{"id": "request-1", "method": "item/tool/call", "params": {}}]

    class Router:
        def route_notification(self, _notification):
            raise AssertionError("unexpected notification")

        def route_response(self, _message):
            raise AssertionError("unexpected response")

        def fail_all(self, _error):
            pass

    class SyncClient:
        _proc = object()
        _router = Router()

        def _read_message(self):
            if messages:
                return messages.pop(0)
            raise EOFError

        def _handle_server_request(self, _message):
            handler_started.set()
            assert allow_handler.wait(timeout=1)
            return {"success": True}

        def _write_message(self, message):
            assert self._proc is not None
            assert message == {
                "id": "request-1",
                "result": {"success": True},
            }
            response_written.set()

        def _coerce_notification(self, method, params):
            return method, params

    sync_client = SyncClient()

    class Client:
        _client = SimpleNamespace(_sync=sync_client)

        async def close(self):
            client_close_called.set()
            sync_client._proc = None

    client = Client()
    _install_concurrent_server_request_dispatcher(client)  # type: ignore[arg-type]
    reader = threading.Thread(target=sync_client._reader_loop, daemon=True)
    reader.start()
    assert handler_started.wait(timeout=1)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            client,
            home,
            close_timeout_seconds=1,
        )
        closing = asyncio.create_task(runtime.close())
        await asyncio.sleep(0.01)
        assert not client_close_called.is_set()
        allow_handler.set()
        await asyncio.wait_for(closing, timeout=1)

    assert response_written.is_set()
    assert client_close_called.is_set()


async def test_runtime_close_abandons_stuck_server_request_after_deadline(tmp_path):
    handler_started = threading.Event()
    allow_handler = threading.Event()
    handler_finished = threading.Event()
    response_written = threading.Event()
    client_close_called = asyncio.Event()
    messages = [{"id": "request-1", "method": "item/tool/call", "params": {}}]

    class Router:
        def route_notification(self, _notification):
            raise AssertionError("unexpected notification")

        def route_response(self, _message):
            raise AssertionError("unexpected response")

        def fail_all(self, _error):
            pass

    class SyncClient:
        _proc = object()
        _router = Router()

        def _read_message(self):
            if messages:
                return messages.pop(0)
            raise EOFError

        def _handle_server_request(self, _message):
            handler_started.set()
            assert allow_handler.wait(timeout=1)
            handler_finished.set()
            return {"success": True}

        def _write_message(self, _message):
            if self._proc is None:
                raise RuntimeError("process closed")
            response_written.set()

        def _coerce_notification(self, method, params):
            return method, params

    sync_client = SyncClient()

    class Client:
        _client = SimpleNamespace(_sync=sync_client)

        async def close(self):
            client_close_called.set()
            sync_client._proc = None

    client = Client()
    _install_concurrent_server_request_dispatcher(client)  # type: ignore[arg-type]
    reader = threading.Thread(target=sync_client._reader_loop, daemon=True)
    reader.start()
    assert handler_started.wait(timeout=1)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            client,
            home,
            close_timeout_seconds=0.01,
        )
        await asyncio.wait_for(runtime.close(), timeout=0.2)

    assert client_close_called.is_set()
    assert not response_written.is_set()
    allow_handler.set()
    assert handler_finished.wait(timeout=1)


async def test_runtime_close_force_stops_child_when_client_close_never_returns(
    tmp_path,
):
    close_started = asyncio.Event()

    class ProcessIgnoringTerminate:
        stdin = None

        def __init__(self):
            self.terminated = False
            self.killed = False

        def terminate(self):
            self.terminated = True

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    process = ProcessIgnoringTerminate()

    class NeverClosingClient:
        _client = SimpleNamespace(_sync=SimpleNamespace(_proc=process))

        async def close(self):
            close_started.set()
            await asyncio.Event().wait()

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            NeverClosingClient(),
            home,
            close_timeout_seconds=0.01,
        )
        await asyncio.wait_for(runtime.close(), timeout=0.2)

    assert close_started.is_set()
    assert process.terminated
    assert process.killed


async def test_stream_close_timeout_force_closes_runtime(tmp_path):
    class NeverClosingStream:
        async def aclose(self):
            await asyncio.Event().wait()

    client = SimpleNamespace(
        close=AsyncMock(),
        _client=SimpleNamespace(_sync=SimpleNamespace(_proc=None)),
    )
    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            client,
            home,
            close_timeout_seconds=0.01,
        )
        with pytest.raises(CodexRuntimeError, match="codex_stream_shutdown_timeout"):
            await asyncio.wait_for(
                runtime._close_agent_stream(NeverClosingStream()),
                timeout=0.2,
            )

    client.close.assert_awaited_once()


async def test_cancelled_start_does_not_wait_forever_for_stuck_enter(
    tmp_path,
    monkeypatch,
):
    entered = asyncio.Event()
    release = asyncio.Event()

    class StuckStartupClient:
        def __init__(self, _config):
            self.closed = False
            self._client = SimpleNamespace(_sync=SimpleNamespace(_proc=None))

        async def __aenter__(self):
            entered.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return self

        async def close(self):
            self.closed = True

    fake_client = None

    def create_client(config):
        nonlocal fake_client
        fake_client = StuckStartupClient(config)
        return fake_client

    monkeypatch.setattr(runtime_module, "AsyncCodex", create_client)
    monkeypatch.setattr(runtime_module, "assert_pinned_versions", lambda: None)
    monkeypatch.setattr(
        runtime_module,
        "_install_concurrent_server_request_dispatcher",
        lambda _client: None,
    )
    monkeypatch.setattr(runtime_module, "_RUNTIME_CLOSE_TIMEOUT_SECONDS", 0.01)

    with TemporaryCodexHome.create(tmp_path) as home:
        startup = asyncio.create_task(CodexRuntime.start(home))
        await asyncio.wait_for(entered.wait(), timeout=1)
        startup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(startup, timeout=0.2)

    assert fake_client is not None
    assert fake_client.closed
    release.set()
    await asyncio.sleep(0)


async def test_cancelled_invoke_does_not_wait_for_stuck_turn_start(tmp_path):
    started = asyncio.Event()
    release = asyncio.Event()

    class StuckThread:
        async def turn(self, *_args, **_kwargs):
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return SimpleNamespace()

    client = SimpleNamespace(
        close=AsyncMock(),
        _client=SimpleNamespace(_sync=SimpleNamespace(_proc=None)),
    )
    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            client,
            home,
            close_timeout_seconds=0.01,
        )
        runtime._start_toolless_thread = AsyncMock(  # type: ignore[method-assign]
            return_value=StuckThread()
        )
        invocation = asyncio.create_task(
            runtime.invoke(CodexInvocationRequest(prompt="hello"))
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, timeout=0.2)

    client.close.assert_awaited_once()
    release.set()
    await asyncio.sleep(0)


async def test_cancelled_agent_turn_start_recovers_and_interrupts_without_closing(
    tmp_path,
):
    started = asyncio.Event()
    release = asyncio.Event()
    interrupt = AsyncMock()

    class RecoverableThread:
        id = "thread-1"

        async def turn(self, *_args, **_kwargs):
            started.set()
            await release.wait()
            return SimpleNamespace(id="turn-1", interrupt=interrupt)

    client = SimpleNamespace(
        close=AsyncMock(),
        _client=SimpleNamespace(
            _sync=SimpleNamespace(_approval_handler=None, _proc=None)
        ),
    )

    async def execute(_call):
        return CodexDynamicToolResult(content="ok", success=True)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            client,
            home,
            close_timeout_seconds=0.1,
        )
        runtime._start_thread = AsyncMock(  # type: ignore[method-assign]
            return_value=RecoverableThread()
        )
        invocation = asyncio.create_task(
            runtime.invoke_agent(
                CodexInvocationRequest(prompt="hello"),
                [],
                execute,
                tool_timeout_seconds=1,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        invocation.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, timeout=1)

        assert not runtime.closed

    interrupt.assert_awaited_once()
    client.close.assert_not_awaited()


async def test_cancelled_agent_turn_start_poisoned_when_handle_cannot_be_recovered(
    tmp_path,
):
    started = asyncio.Event()
    release = asyncio.Event()

    class StuckThread:
        id = "thread-1"

        async def turn(self, *_args, **_kwargs):
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    continue
            return SimpleNamespace(id="turn-1")

    client = SimpleNamespace(
        close=AsyncMock(),
        _client=SimpleNamespace(
            _sync=SimpleNamespace(_approval_handler=None, _proc=None)
        ),
    )

    async def execute(_call):
        return CodexDynamicToolResult(content="ok", success=True)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(  # type: ignore[arg-type]
            client,
            home,
            close_timeout_seconds=0.01,
        )
        runtime._start_thread = AsyncMock(  # type: ignore[method-assign]
            return_value=StuckThread()
        )
        invocation = asyncio.create_task(
            runtime.invoke_agent(
                CodexInvocationRequest(prompt="hello"),
                [],
                execute,
                tool_timeout_seconds=1,
            )
        )
        await asyncio.wait_for(started.wait(), timeout=1)
        invocation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(invocation, timeout=1)

        assert runtime.closed

    client.close.assert_awaited_once()
    release.set()
    await asyncio.sleep(0)


async def test_agent_stream_failure_interrupts_without_closing_sibling_runtime(
    tmp_path,
):
    interrupt = AsyncMock()

    async def failing_stream():
        if False:
            yield None
        raise RuntimeError("stream failed")

    turn = SimpleNamespace(
        id="turn-1",
        interrupt=interrupt,
        stream=failing_stream,
    )
    thread = SimpleNamespace(
        id="thread-1",
        turn=AsyncMock(return_value=turn),
    )
    client = SimpleNamespace(
        close=AsyncMock(),
        _client=SimpleNamespace(
            _sync=SimpleNamespace(_approval_handler=None, _proc=None)
        ),
    )

    async def execute(_call):
        return CodexDynamicToolResult(content="ok", success=True)

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        runtime._start_thread = AsyncMock(  # type: ignore[method-assign]
            return_value=thread
        )
        with pytest.raises(RuntimeError, match="stream failed"):
            await runtime.invoke_agent(
                CodexInvocationRequest(prompt="hello"),
                [],
                execute,
                tool_timeout_seconds=1,
            )

        assert not runtime.closed

    interrupt.assert_awaited_once()
    client.close.assert_not_awaited()


async def test_agent_event_failure_poisoned_when_interrupt_fails(tmp_path):
    interrupt = AsyncMock(side_effect=RuntimeError("interrupt failed"))

    async def one_event_stream():
        yield SimpleNamespace()

    turn = SimpleNamespace(
        id="turn-1",
        interrupt=interrupt,
        stream=one_event_stream,
    )
    thread = SimpleNamespace(
        id="thread-1",
        turn=AsyncMock(return_value=turn),
    )
    client = SimpleNamespace(
        close=AsyncMock(),
        _client=SimpleNamespace(
            _sync=SimpleNamespace(_approval_handler=None, _proc=None)
        ),
    )

    async def execute(_call):
        return CodexDynamicToolResult(content="ok", success=True)

    async def reject_event(_event):
        raise RuntimeError("event failed")

    with TemporaryCodexHome.create(tmp_path) as home:
        runtime = CodexRuntime(client, home)  # type: ignore[arg-type]
        runtime._start_thread = AsyncMock(  # type: ignore[method-assign]
            return_value=thread
        )
        with pytest.raises(RuntimeError, match="event failed"):
            await runtime.invoke_agent(
                CodexInvocationRequest(prompt="hello"),
                [],
                execute,
                reject_event,
                tool_timeout_seconds=1,
            )

        assert runtime.closed

    interrupt.assert_awaited_once()
    client.close.assert_awaited_once()
