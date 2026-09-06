from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest

from backend.blocks.code_executor._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.code_executor.instantiate import InstantiateCodeSandboxBlock


@pytest.fixture(autouse=True)
def preview_link():
    with patch(
        "backend.blocks.code_executor.instantiate.create_preview_link",
        side_effect=lambda user_id, url: "https://platform.example/preview",
    ) as create:
        yield create


async def test_live_view_returns_url_and_uses_same_sandbox_for_setup(preview_link):
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
        patch(
            "backend.blocks.code_executor.instantiate.create_desktop_sandbox", create
        ),
        patch.object(block, "execute_code", execute),
        patch(
            "backend.blocks.code_executor.instantiate.kill_desktop_sandbox", AsyncMock()
        ) as kill,
    ):
        outputs = dict(
            [
                item
                async for item in block.run(
                    inputs, credentials=TEST_CREDENTIALS, user_id="owner"
                )
            ]
        )
    kill.assert_not_awaited()
    assert outputs["sandbox_id"] == "desktop-id"
    assert outputs["live_url"] == "https://platform.example/preview"
    preview_link.assert_called_once_with("owner", "https://preview.example/view")
    assert outputs["response"] == "ready"
    assert execute.await_args.kwargs["sandbox_id"] == "desktop-id"


async def test_default_does_not_start_desktop_or_emit_live_url():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(credentials=TEST_CREDENTIALS_INPUT)
    execute = AsyncMock(return_value=([], "", "", "", "code-id", []))
    with (
        patch(
            "backend.blocks.code_executor.instantiate.create_desktop_sandbox"
        ) as create,
        patch.object(block, "execute_code", execute),
    ):
        outputs = dict(
            [
                item
                async for item in block.run(
                    inputs, credentials=TEST_CREDENTIALS, user_id="owner"
                )
            ]
        )
    create.assert_not_called()
    assert outputs == {"sandbox_id": "code-id"}


async def test_live_url_and_id_wait_for_setup():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, enable_live_view=True, template_id="custom"
    )
    execute = AsyncMock(return_value=([], "", "", "", "desktop-id", []))
    with (
        patch(
            "backend.blocks.code_executor.instantiate.create_desktop_sandbox",
            AsyncMock(return_value=("desktop-id", "https://preview.example")),
        ),
        patch.object(block, "execute_code", execute),
    ):
        outputs = block.run(inputs, credentials=TEST_CREDENTIALS, user_id="owner")
        assert await anext(outputs) == ("live_url", "https://platform.example/preview")
        execute.assert_awaited_once()
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
            "backend.blocks.code_executor.instantiate.create_desktop_sandbox",
            AsyncMock(return_value=("desktop-id", "https://preview.example")),
        ),
        patch.object(
            block, "execute_code", AsyncMock(side_effect=RuntimeError("setup"))
        ),
        patch(
            "backend.blocks.code_executor.instantiate.kill_desktop_sandbox", AsyncMock()
        ) as kill,
    ):
        outputs = dict(
            [
                item
                async for item in block.run(
                    inputs, credentials=TEST_CREDENTIALS, user_id="owner"
                )
            ]
        )
    assert outputs["error"] == "setup"
    assert "live_url" not in outputs
    assert "sandbox_id" not in outputs
    kill.assert_awaited_once_with("mock-e2b-api-key", "desktop-id")


@pytest.mark.parametrize("sandbox_id", ["desktop-id", ""])
async def test_abandoned_or_missing_sandbox_id_cleans_up_desktop(sandbox_id):
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, enable_live_view=True, template_id="custom"
    )
    with (
        patch(
            "backend.blocks.code_executor.instantiate.create_desktop_sandbox",
            AsyncMock(return_value=("desktop-id", "https://preview.example")),
        ),
        patch.object(
            block,
            "execute_code",
            AsyncMock(return_value=([], "", "", "", sandbox_id, [])),
        ),
        patch(
            "backend.blocks.code_executor.instantiate.kill_desktop_sandbox", AsyncMock()
        ) as kill,
    ):
        outputs = block.run(inputs, credentials=TEST_CREDENTIALS, user_id="owner")
        await anext(outputs)
        await outputs.aclose()
    kill.assert_awaited_once_with("mock-e2b-api-key", "desktop-id")


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
    calls = MagicMock()
    calls.attach_mock(sandbox.commands.run, "setup")
    calls.attach_mock(sandbox.run_code, "code")
    with patch(
        "backend.blocks.code_executor._base.AsyncSandbox.connect",
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

    assert calls.mock_calls[0] == call.setup("git clone https://example.com/repo.git")
    assert calls.mock_calls[1] == call.code(
        *sandbox.run_code.call_args.args, **sandbox.run_code.call_args.kwargs
    )


async def test_live_view_requires_owner_before_provisioning():
    block = InstantiateCodeSandboxBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT, enable_live_view=True, template_id="custom"
    )
    with patch(
        "backend.blocks.code_executor.instantiate.create_desktop_sandbox", AsyncMock()
    ) as create:
        outputs = dict(
            [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
        )
    assert outputs == {"error": "Live view requires an authenticated user"}
    create.assert_not_awaited()
