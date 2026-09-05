import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import CommandExitException

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.code_executor_desktop import create_desktop_sandbox


async def test_live_view_returns_url_and_uses_same_sandbox_for_setup():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        enable_live_view=True,
        template_id="code-desktop",
        setup_code="print('ready')",
    )
    create = AsyncMock(return_value=("desktop-id", "https://preview.example/view"))
    execute = AsyncMock(return_value=([], "ready", "ready\n", "", "desktop-id", []))
    with (
        patch("backend.blocks.code_executor.create_desktop_sandbox", create),
        patch.object(block, "execute_code", execute),
    ):
        outputs = dict(
            [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
        )
    assert outputs["sandbox_id"] == "desktop-id"
    assert outputs["live_url"] == "https://preview.example/view"
    assert outputs["response"] == "ready"
    assert execute.await_args.kwargs["sandbox_id"] == "desktop-id"


async def test_default_does_not_start_desktop_or_emit_live_url():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(credentials=TEST_CREDENTIALS_INPUT)
    execute = AsyncMock(return_value=([], "", "", "", "code-id", []))
    with (
        patch("backend.blocks.code_executor.create_desktop_sandbox") as create,
        patch.object(block, "execute_code", execute),
    ):
        outputs = dict(
            [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
        )
    create.assert_not_called()
    assert outputs == {"sandbox_id": "code-id"}


async def test_live_url_available_before_setup_but_id_waits_for_setup():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, enable_live_view=True, template_id="custom"
    )
    execute = AsyncMock(return_value=([], "", "", "", "desktop-id", []))
    with (
        patch(
            "backend.blocks.code_executor.create_desktop_sandbox",
            AsyncMock(return_value=("desktop-id", "https://preview.example")),
        ),
        patch.object(block, "execute_code", execute),
    ):
        outputs = block.run(inputs, credentials=TEST_CREDENTIALS)
        assert await anext(outputs) == ("live_url", "https://preview.example")
        execute.assert_not_awaited()
        assert await anext(outputs) == ("sandbox_id", "desktop-id")
        execute.assert_awaited_once()
        await outputs.aclose()


async def test_failed_setup_cleans_up_desktop():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, enable_live_view=True, template_id="custom"
    )
    with (
        patch(
            "backend.blocks.code_executor.create_desktop_sandbox",
            AsyncMock(return_value=("desktop-id", "https://preview.example")),
        ),
        patch.object(
            block, "execute_code", AsyncMock(side_effect=RuntimeError("setup"))
        ),
        patch("backend.blocks.code_executor.kill_desktop_sandbox", AsyncMock()) as kill,
    ):
        outputs = dict(
            [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
        )
    assert outputs["error"] == "setup"
    assert "sandbox_id" not in outputs
    kill.assert_awaited_once_with("mock-e2b-api-key", "desktop-id")


async def test_abandoned_live_url_output_cleans_up_desktop():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, enable_live_view=True, template_id="custom"
    )
    with (
        patch(
            "backend.blocks.code_executor.create_desktop_sandbox",
            AsyncMock(return_value=("desktop-id", "https://preview.example")),
        ),
        patch("backend.blocks.code_executor.kill_desktop_sandbox", AsyncMock()) as kill,
    ):
        outputs = block.run(inputs, credentials=TEST_CREDENTIALS)
        await anext(outputs)
        await outputs.aclose()
    kill.assert_awaited_once_with("mock-e2b-api-key", "desktop-id")


@pytest.mark.parametrize("template_id", ["", " ", "base", "desktop"])
async def test_live_view_requires_combined_template(template_id):
    with (
        patch("backend.blocks.code_executor_desktop.DesktopSandbox.create") as create,
        pytest.raises(ValueError, match="both desktop dependencies"),
    ):
        await create_desktop_sandbox("key", template_id, 300)
    create.assert_not_called()


async def test_desktop_stream_requires_auth_and_allows_interaction():
    sandbox = MagicMock()
    sandbox.sandbox_id = "desktop-id"
    sandbox.stream.get_auth_key.return_value = "stream-password"
    sandbox.stream.get_url.return_value = (
        "https://preview.example?password=stream-password"
    )
    with patch(
        "backend.blocks.code_executor_desktop.DesktopSandbox.create",
        return_value=sandbox,
    ) as create:
        result = await create_desktop_sandbox("key", "custom", 600)
    create.assert_called_once_with(api_key="key", template="custom", timeout=600)
    sandbox.stream.start.assert_called_once_with(require_auth=True)
    sandbox.stream.get_url.assert_called_once_with(
        auth_key="stream-password", view_only=False
    )
    assert result == ("desktop-id", "https://preview.example?password=stream-password")
    sandbox.kill.assert_not_called()


async def test_stream_failure_kills_created_sandbox():
    sandbox = MagicMock()
    sandbox.stream.start.side_effect = RuntimeError("stream failed")
    with (
        patch(
            "backend.blocks.code_executor_desktop.DesktopSandbox.create",
            return_value=sandbox,
        ),
        pytest.raises(RuntimeError, match="stream failed"),
    ):
        await create_desktop_sandbox("key", "custom", 300)
    sandbox.kill.assert_called_once()


async def test_missing_interpreter_fails_clearly_and_cleans_up():
    sandbox = MagicMock()
    sandbox.commands.run.side_effect = CommandExitException(
        stderr="connection refused", stdout="", exit_code=7, error=None
    )
    with (
        patch(
            "backend.blocks.code_executor_desktop.DesktopSandbox.create",
            return_value=sandbox,
        ),
        pytest.raises(ValueError, match="code interpreter"),
    ):
        await create_desktop_sandbox("key", "desktop-without-interpreter", 300)
    sandbox.stream.start.assert_not_called()
    sandbox.kill.assert_called_once()


@pytest.mark.parametrize("worker_error", [False, True])
async def test_cancelled_creation_cleans_up_when_worker_finishes(worker_error, caplog):
    started, finish = asyncio.Event(), asyncio.Event()

    async def worker(*args):
        started.set()
        await finish.wait()
        if worker_error:
            raise RuntimeError("provisioning failed")
        return "desktop-id", "https://preview.example"

    with (
        patch("backend.blocks.code_executor_desktop.asyncio.to_thread", worker),
        patch(
            "backend.blocks.code_executor_desktop.kill_desktop_sandbox", AsyncMock()
        ) as kill,
    ):
        task = asyncio.create_task(create_desktop_sandbox("key", "custom", 300))
        await started.wait()
        task.cancel()
        try:
            done, _ = await asyncio.wait({task}, timeout=0.1)
            assert task in done
            with pytest.raises(asyncio.CancelledError):
                await task
            kill.assert_not_awaited()
        finally:
            finish.set()
            await asyncio.gather(task, return_exceptions=True)
            for _ in range(10):
                if kill.await_count or caplog.records:
                    break
                await asyncio.sleep(0)
    if worker_error:
        kill.assert_not_awaited()
        assert "Desktop provisioning failed after cancellation" in caplog.text
    else:
        kill.assert_awaited_once_with("key", "desktop-id")


async def test_setup_commands_execute_in_reconnected_desktop_before_code():
    block = InstantiateCodeSandboxBlock()
    sandbox = MagicMock()
    sandbox.sandbox_id = "desktop-id"
    sandbox.commands.run = AsyncMock()
    sandbox.run_code = AsyncMock(
        return_value=MagicMock(
            error=None, results=[], text="ready", logs=MagicMock(stdout=[], stderr=[])
        )
    )
    with patch(
        "backend.blocks.code_executor.AsyncSandbox.connect",
        AsyncMock(return_value=sandbox),
    ):
        await block.execute_code(
            api_key="key",
            sandbox_id="desktop-id",
            setup_commands=["git clone https://example.com/repo.git"],
            code="print('ready')",
            language=block.Input(credentials=TEST_CREDENTIALS_INPUT).language,
        )
    sandbox.commands.run.assert_awaited_once_with(
        "git clone https://example.com/repo.git"
    )
    sandbox.run_code.assert_awaited_once()
