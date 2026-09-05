"""Desktop teardown survives repeated cancellation and has a bounded wait."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    InstantiateCodeSandboxBlock,
)
from backend.blocks.code_executor_desktop import kill_desktop_sandbox


async def test_cancelled_cleanup_continues_until_kill_finishes():
    started, finish, killed = asyncio.Event(), asyncio.Event(), asyncio.Event()

    async def kill(**kwargs):
        started.set()
        await finish.wait()
        killed.set()

    with patch("backend.blocks.code_executor_desktop.AsyncSandbox.kill", kill):
        task = asyncio.create_task(kill_desktop_sandbox("key", "desktop-id"))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not killed.is_set()
        finish.set()
        await asyncio.wait_for(killed.wait(), timeout=1)


async def test_hung_cleanup_times_out_and_logs_reason(caplog):
    stopped = asyncio.Event()

    async def kill(**kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    with (
        patch("backend.blocks.code_executor_desktop.AsyncSandbox.kill", kill),
        patch("backend.blocks.code_executor_desktop.CLEANUP_TIMEOUT", 0.01),
    ):
        await asyncio.wait_for(kill_desktop_sandbox("key", "desktop-id"), timeout=1)
    assert stopped.is_set()
    assert "Could not clean up desktop sandbox" in caplog.text
    assert caplog.records[-1].exc_info is not None


async def test_missing_setup_id_never_hands_off_desktop():
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
            block,
            "execute_code",
            AsyncMock(return_value=([], "partial", "partial", "partial", "", [])),
        ),
        patch("backend.blocks.code_executor.kill_desktop_sandbox", AsyncMock()) as kill,
    ):
        outputs = dict(
            [
                item
                async for item in block.run(
                    inputs, credentials=TEST_CREDENTIALS, user_id="owner"
                )
            ]
        )
    assert outputs == {"error": "Sandbox ID not found"}
    kill.assert_awaited_once_with("mock-e2b-api-key", "desktop-id")
