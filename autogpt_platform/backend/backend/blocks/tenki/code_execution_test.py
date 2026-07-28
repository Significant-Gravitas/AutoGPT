import asyncio
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from backend.blocks.tenki import _config, code_execution
from backend.blocks.tenki._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT, tenki
from backend.blocks.tenki.code_execution import TenkiRunCodeBlock


class FakeResult:
    def __init__(
        self,
        *,
        exit_code: int = 0,
        stdout: str = "hello",
        stderr: str = "",
        duration_ms: int = 125,
    ):
        self.exit_code = exit_code
        self.stdout_text = stdout
        self.stderr_text = stderr
        self.duration_ms = duration_ms
        self.reason = None
        self.signal = None
        self.ok = exit_code == 0


class FakeSandbox:
    def __init__(self, result: FakeResult | None = None):
        self.id = "sandbox-123"
        self.state = "CREATING"
        self.result = result or FakeResult()
        self.close_calls = 0
        self.wait_error: Exception | None = None
        self.shell_error: Exception | None = None
        self.close_error: BaseException | None = None
        self.shell_started = asyncio.Event()
        self.shell_release: asyncio.Event | None = None
        self.close_started = asyncio.Event()
        self.close_release: asyncio.Event | None = None

    async def wait_ready(self, timeout: int):
        if self.wait_error:
            raise self.wait_error
        self.state = "RUNNING"

    async def shell(self, command: str, **kwargs):
        self.shell_started.set()
        if self.shell_release:
            await self.shell_release.wait()
        if self.shell_error:
            raise self.shell_error
        return self.result

    async def close_if_open(self):
        self.close_calls += 1
        self.close_started.set()
        if self.close_release:
            await self.close_release.wait()
        self.state = "TERMINATED"
        if self.close_error:
            raise self.close_error


class FakeClient:
    def __init__(self, sandbox: FakeSandbox, project_count: int = 1):
        self.sandbox = sandbox
        self.project_count = project_count
        self.create_kwargs: dict = {}
        self.closed = False
        self.close_error: BaseException | None = None

    async def who_am_i(self):
        projects = [
            SimpleNamespace(id=f"project-{index}")
            for index in range(self.project_count)
        ]
        return SimpleNamespace(
            workspaces=[SimpleNamespace(projects=projects)] if projects else []
        )

    async def create(self, **kwargs):
        self.create_kwargs = kwargs
        return self.sandbox

    async def close(self):
        self.closed = True
        if self.close_error:
            raise self.close_error


def _input(**overrides) -> TenkiRunCodeBlock.Input:
    return TenkiRunCodeBlock.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        command="printf hello",
        **overrides,
    )


async def _outputs(block: TenkiRunCodeBlock, input_data=None):
    return [
        item
        async for item in block.run(
            input_data or _input(), credentials=TEST_CREDENTIALS
        )
    ]


def test_tenki_provider_supports_api_keys():
    assert tenki.supported_auth_types == {"api_key"}


def test_client_uses_api_key(monkeypatch):
    async_client = Mock()
    monkeypatch.setattr(_config, "AsyncClient", async_client)

    client = _config._client(TEST_CREDENTIALS)

    assert client is async_client.return_value
    async_client.assert_called_once_with(auth_token="mock-tenki-api-key")


async def test_success_uses_ephemeral_lifecycle(monkeypatch):
    sandbox = FakeSandbox()
    client = FakeClient(sandbox)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs[0:3] == [("stdout", "hello"), ("stderr", ""), ("exit_code", 0)]
    assert client.create_kwargs["wait"] is False
    assert client.create_kwargs["project_id"] == "project-0"
    assert client.create_kwargs["allow_inbound"] is False
    assert client.create_kwargs["max_duration"] == 360
    assert sandbox.close_calls == 1
    assert client.closed


async def test_explicit_project_id_skips_discovery(monkeypatch):
    sandbox = FakeSandbox()
    client = FakeClient(sandbox, project_count=0)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(
        TenkiRunCodeBlock(), _input(project_id="  explicit-project  ")
    )

    assert outputs[0:3] == [("stdout", "hello"), ("stderr", ""), ("exit_code", 0)]
    assert client.create_kwargs["project_id"] == "explicit-project"
    assert sandbox.close_calls == 1
    assert client.closed


async def test_command_failure_surfaces_diagnostics_and_closes(monkeypatch):
    sandbox = FakeSandbox(FakeResult(exit_code=17, stderr="broken dependency"))
    client = FakeClient(sandbox)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs == [
        ("error", "Tenki command exited with code 17: broken dependency")
    ]
    assert sandbox.close_calls == 1
    assert client.closed


@pytest.mark.parametrize(
    "error",
    [TimeoutError("startup timed out"), RuntimeError("terminal state: TERMINATED")],
)
async def test_readiness_failure_closes(monkeypatch, error):
    sandbox = FakeSandbox()
    sandbox.wait_error = error
    client = FakeClient(sandbox)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs == [("error", f"Tenki sandbox execution failed: {error}")]
    assert sandbox.close_calls == 1
    assert client.closed


@pytest.mark.parametrize(
    "error", [RuntimeError("transport failed"), TimeoutError("command timed out")]
)
async def test_remote_command_exception_closes(monkeypatch, error):
    sandbox = FakeSandbox()
    sandbox.shell_error = error
    client = FakeClient(sandbox)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs == [("error", f"Tenki sandbox execution failed: {error}")]
    assert sandbox.close_calls == 1
    assert client.closed


async def test_cleanup_failure_does_not_mask_result(monkeypatch):
    sandbox = FakeSandbox()
    sandbox.close_error = RuntimeError("sandbox teardown transport failed")
    client = FakeClient(sandbox)
    client.close_error = RuntimeError("client teardown failed")
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs[0:3] == [("stdout", "hello"), ("stderr", ""), ("exit_code", 0)]
    assert sandbox.close_calls == 1
    assert client.closed


async def test_cleanup_failure_does_not_mask_command_exception(monkeypatch):
    sandbox = FakeSandbox()
    sandbox.shell_error = RuntimeError("transport failed")
    sandbox.close_error = RuntimeError("sandbox teardown transport failed")
    client = FakeClient(sandbox)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs == [("error", "Tenki sandbox execution failed: transport failed")]
    assert sandbox.close_calls == 1
    assert client.closed


@pytest.mark.parametrize("cancel_resource", ["sandbox", "client"])
async def test_resource_cleanup_cancellation_propagates(cancel_resource):
    sandbox = FakeSandbox()
    client = FakeClient(sandbox)
    if cancel_resource == "sandbox":
        sandbox.close_error = asyncio.CancelledError()
    else:
        client.close_error = asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await TenkiRunCodeBlock._cleanup(client, sandbox)

    assert sandbox.close_calls == 1
    assert client.closed


async def test_cancellation_waits_for_cleanup_before_propagating():
    sandbox = FakeSandbox()
    sandbox.close_release = asyncio.Event()
    client = FakeClient(sandbox)

    task = asyncio.create_task(TenkiRunCodeBlock._cleanup(client, sandbox))
    await sandbox.close_started.wait()
    task.cancel()
    await asyncio.sleep(0)

    assert not task.done()
    sandbox.close_release.set()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.close_calls == 1
    assert client.closed


async def test_cancellation_closes(monkeypatch):
    sandbox = FakeSandbox()
    sandbox.shell_release = asyncio.Event()
    client = FakeClient(sandbox)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)
    block = TenkiRunCodeBlock()

    task = asyncio.create_task(block.execute_in_sandbox(_input(), TEST_CREDENTIALS))
    await sandbox.shell_started.wait()
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task
    assert sandbox.close_calls == 1
    assert client.closed


async def test_multiple_projects_requires_explicit_project_id(monkeypatch):
    sandbox = FakeSandbox()
    client = FakeClient(sandbox, project_count=2)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs == [
        (
            "error",
            "Tenki sandbox execution failed: Multiple Tenki projects found; set the Tenki project ID",
        )
    ]
    assert not client.create_kwargs
    assert sandbox.close_calls == 0
    assert client.closed


async def test_no_projects_reports_error_and_closes_client(monkeypatch):
    sandbox = FakeSandbox()
    client = FakeClient(sandbox, project_count=0)
    monkeypatch.setattr(code_execution, "_client", lambda credentials: client)

    outputs = await _outputs(TenkiRunCodeBlock())

    assert outputs == [
        (
            "error",
            "Tenki sandbox execution failed: No Tenki project is available for this API key",
        )
    ]
    assert not client.create_kwargs
    assert sandbox.close_calls == 0
    assert client.closed
