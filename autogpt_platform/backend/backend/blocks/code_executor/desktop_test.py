import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import CommandExitException

from backend.blocks.code_executor.desktop import create_desktop_sandbox


@pytest.mark.parametrize(
    "template_id", ["", " ", "base", "desktop", "Desktop", " BASE "]
)
async def test_live_view_requires_combined_template(template_id):
    with (
        patch("backend.util.desktop_sdk.DesktopSandbox.create") as create,
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
        "backend.util.desktop_sdk.DesktopSandbox.create",
        return_value=sandbox,
    ) as create:
        result = await create_desktop_sandbox("key", "custom", 600)
    create.assert_called_once_with(
        api_key="key", template="custom", timeout=600, request_timeout=10
    )
    sandbox.commands.run.assert_called_once_with(
        "curl --fail --silent --retry 10 --retry-connrefused --retry-delay 1 "
        "--retry-max-time 10 --max-time 10 http://localhost:49999/health >/dev/null",
        timeout=25,
    )
    sandbox.stream.start.assert_called_once_with(require_auth=True)
    sandbox.stream.get_url.assert_called_once_with(
        auth_key="stream-password", view_only=False
    )
    assert result == ("desktop-id", "https://preview.example?password=stream-password")
    sandbox.kill.assert_not_called()


@pytest.mark.parametrize("cleanup_fails", [False, True])
async def test_stream_failure_kills_created_sandbox(cleanup_fails, caplog):
    sandbox = MagicMock()
    sandbox.stream.start.side_effect = RuntimeError("stream failed")
    if cleanup_fails:
        sandbox.kill.side_effect = RuntimeError("kill failed")
    with (
        patch(
            "backend.util.desktop_sdk.DesktopSandbox.create",
            return_value=sandbox,
        ),
        pytest.raises(RuntimeError, match="stream failed"),
    ):
        await create_desktop_sandbox("key", "custom", 300)
    sandbox.kill.assert_called_once()
    if cleanup_fails:
        assert caplog.records[-1].exc_info is not None


async def test_missing_interpreter_fails_clearly_and_cleans_up():
    sandbox = MagicMock()
    sandbox.commands.run.side_effect = CommandExitException(
        stderr="connection refused", stdout="", exit_code=7, error=None
    )
    with (
        patch(
            "backend.util.desktop_sdk.DesktopSandbox.create",
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
        patch("backend.blocks.code_executor.desktop.to_thread", worker),
        patch(
            "backend.blocks.code_executor.desktop.kill_desktop_sandbox", AsyncMock()
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
