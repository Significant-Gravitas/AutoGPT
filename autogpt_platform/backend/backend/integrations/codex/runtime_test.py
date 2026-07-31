import asyncio
import importlib.metadata
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from codex_cli_bin import bundled_codex_path
from openai_codex import AsyncCodex

import backend.integrations.codex.runtime as runtime_module
from backend.integrations.codex.runtime import (
    CODEX_RUNTIME_VERSION,
    OPENAI_CODEX_VERSION,
    CodexRuntime,
    CodexRuntimeError,
    _deny_server_request,
    _install_fail_closed_approval_handler,
    assert_pinned_versions,
    build_runtime_config,
)
from backend.integrations.codex.models import (
    CodexDynamicToolResult,
    CodexDynamicToolSpec,
    CodexInvocationRequest,
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
        runtime._install_dynamic_tool_handler(execute, timeout_seconds=1)
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
        runtime._install_dynamic_tool_handler(execute, timeout_seconds=0.01)
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
