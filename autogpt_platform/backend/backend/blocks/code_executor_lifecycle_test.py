from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from backend.blocks.code_executor import (
    TEST_CREDENTIALS,
    TEST_CREDENTIALS_INPUT,
    ExecuteCodeBlock,
    ExecuteCodeStepBlock,
    InstantiateCodeSandboxBlock,
    ProgrammingLanguage,
)
from backend.util.test import execute_block_test


async def test_reconnect_preserves_explicit_sandbox_timeout():
    sandbox = MagicMock()
    sandbox.run_code = AsyncMock(
        return_value=MagicMock(
            error=None, results=[], text="", logs=MagicMock(stdout=[], stderr=[])
        )
    )
    with patch(
        "backend.blocks.code_executor.AsyncSandbox.connect",
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


@pytest.mark.parametrize(
    "block_type", [ExecuteCodeBlock, InstantiateCodeSandboxBlock, ExecuteCodeStepBlock]
)
async def test_existing_block_contracts(block_type):
    await execute_block_test(block_type())


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
        "backend.blocks.code_executor.AsyncSandbox.connect",
        AsyncMock(return_value=sandbox),
    ):
        _ = [item async for item in block.run(inputs, credentials=TEST_CREDENTIALS)]
    sandbox.kill.assert_awaited_once()
