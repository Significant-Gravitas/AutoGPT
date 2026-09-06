from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.blocks.code_executor._test import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.code_executor.helpers import ProgrammingLanguage
from backend.blocks.code_executor.instantiate import InstantiateCodeSandboxBlock
from backend.blocks.code_executor.step import ExecuteCodeStepBlock


async def test_reconnect_preserves_explicit_sandbox_timeout():
    sandbox = MagicMock()
    sandbox.run_code = AsyncMock(
        return_value=MagicMock(
            error=None, results=[], text="", logs=MagicMock(stdout=[], stderr=[])
        )
    )
    with patch(
        "backend.blocks.code_executor._base.AsyncSandbox.connect",
        AsyncMock(return_value=sandbox),
    ) as connect:
        await InstantiateCodeSandboxBlock().execute_code(
            api_key="key",
            sandbox_id="desktop-id",
            timeout=1800,
            code="",
            language=ProgrammingLanguage.PYTHON,
        )
    connect.assert_awaited_once_with(
        sandbox_id="desktop-id", api_key="key", timeout=1800
    )


async def test_code_step_can_extend_testing_window():
    block = ExecuteCodeStepBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        sandbox_id="desktop-id",
        timeout=1800,
    )
    with patch.object(
        block,
        "execute_code",
        AsyncMock(return_value=([], "", "", "", "desktop-id", [])),
    ) as execute:
        outputs = dict(
            [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
        )
    assert outputs == {"results": []}
    assert execute.await_args.kwargs["timeout"] == 1800


@pytest.mark.parametrize("timeout", [0, -1])
def test_code_step_rejects_nonpositive_testing_window(timeout):
    with pytest.raises(ValidationError, match="greater than or equal to 1"):
        ExecuteCodeStepBlock.Input(
            credentials=TEST_CREDENTIALS_INPUT, sandbox_id="desktop-id", timeout=timeout
        )


async def test_code_step_default_preserves_existing_lifetime_behavior():
    block = ExecuteCodeStepBlock()
    inputs = block.Input(credentials=TEST_CREDENTIALS_INPUT, sandbox_id="code-id")
    with patch.object(
        block, "execute_code", AsyncMock(return_value=([], "", "", "", "code-id", []))
    ) as execute:
        _ = [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    assert execute.await_args.kwargs["timeout"] is None


async def test_code_step_can_dispose_desktop_after_testing():
    sandbox = MagicMock()
    sandbox.kill = AsyncMock()
    sandbox.run_code = AsyncMock(
        return_value=MagicMock(
            error=None, results=[], text="", logs=MagicMock(stdout=[], stderr=[])
        )
    )
    block = ExecuteCodeStepBlock()
    inputs = block.Input(
        credentials=TEST_CREDENTIALS_INPUT,
        sandbox_id="desktop-id",
        dispose_sandbox=True,
    )
    with patch(
        "backend.blocks.code_executor._base.AsyncSandbox.connect",
        AsyncMock(return_value=sandbox),
    ):
        _ = [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    sandbox.kill.assert_awaited_once()


@pytest.mark.parametrize(
    "block_type", [InstantiateCodeSandboxBlock, ExecuteCodeStepBlock]
)
@pytest.mark.parametrize("timeout", [0, -1, 3601])
def test_sandbox_lifetime_bounds(block_type, timeout):
    with pytest.raises(ValidationError):
        block_type.Input(
            credentials=TEST_CREDENTIALS_INPUT, sandbox_id="desktop-id", timeout=timeout
        )


@pytest.mark.parametrize(
    "block_type", [InstantiateCodeSandboxBlock, ExecuteCodeStepBlock]
)
@pytest.mark.parametrize("timeout", [1, 3600])
def test_sandbox_lifetime_accepts_boundaries(block_type, timeout):
    assert (
        block_type.Input(
            credentials=TEST_CREDENTIALS_INPUT, sandbox_id="desktop-id", timeout=timeout
        ).timeout
        == timeout
    )


def test_live_view_and_step_timeout_field_visibility():
    assert (
        InstantiateCodeSandboxBlock.Input.model_json_schema()["properties"][
            "enable_live_view"
        ]["advanced"]
        is True
    )
    assert (
        ExecuteCodeStepBlock.Input.model_json_schema()["properties"]["timeout"][
            "advanced"
        ]
        is False
    )
