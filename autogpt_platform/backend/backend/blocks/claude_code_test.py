"""Claude Code only resumes a sandbox that was created for the same user, and
runs only on the user's own Anthropic key."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.claude_code import (
    OWN_ANTHROPIC_KEY_REQUIRED,
    TEST_ANTHROPIC_CREDENTIALS,
    TEST_ANTHROPIC_CREDENTIALS_INPUT,
    TEST_E2B_CREDENTIALS_INPUT,
    ClaudeCodeBlock,
    ClaudeCodeExecutionError,
)
from backend.data.model import APIKeyCredentials
from backend.executor.utils import ExecutionContext
from backend.integrations.credentials_store import (
    anthropic_credentials,
    e2b_credentials,
)


async def test_resuming_another_users_sandbox_is_refused():
    block = ClaudeCodeBlock()
    theirs = AsyncMock()
    theirs.get_info = AsyncMock(
        return_value=AsyncMock(
            metadata={
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "claude_code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
    )
    with patch("backend.blocks.claude_code.BaseAsyncSandbox") as cls:
        # The stamp is read through the static ``get_info`` before any connect.
        cls.get_info = AsyncMock(return_value=theirs.get_info.return_value)
        cls.connect = AsyncMock(return_value=theirs)
        # The block wraps every failure in its own error; the reason survives.
        with pytest.raises(ClaudeCodeExecutionError, match="does not belong"):
            await block.execute_claude_code(
                e2b_api_key="k",
                anthropic_api_key="a",
                prompt="hi",
                timeout=60,
                setup_commands=[],
                working_directory="/home/user",
                session_id="s1",
                existing_sandbox_id="sb-theirs",
                conversation_history="",
                dispose_sandbox=True,
                execution_context=ExecutionContext(user_id="user-b"),
            )
    cls.get_info.assert_awaited_once_with("sb-theirs", api_key="k")
    # Connecting would resume the other user's box on their bill.
    cls.connect.assert_not_awaited()
    theirs.commands.run.assert_not_awaited()
    theirs.kill.assert_not_awaited()


async def _run_with(anthropic: APIKeyCredentials) -> tuple[list, AsyncMock]:
    block = ClaudeCodeBlock()
    execute = AsyncMock(return_value=("done", [], "history", "s1", None))
    with patch.object(block, "execute_claude_code", execute):
        outputs = [
            output
            async for output in block.run(
                ClaudeCodeBlock.Input(
                    e2b_credentials=TEST_E2B_CREDENTIALS_INPUT,
                    anthropic_credentials=TEST_ANTHROPIC_CREDENTIALS_INPUT,
                    prompt="hi",
                    setup_commands=[],
                ),
                e2b_credentials=e2b_credentials,
                anthropic_credentials=anthropic,
                execution_context=ExecutionContext(user_id="user-a"),
            )
        ]
    return outputs, execute


async def test_the_platform_anthropic_key_is_refused():
    outputs, execute = await _run_with(anthropic_credentials)

    assert outputs == [("error", OWN_ANTHROPIC_KEY_REQUIRED)]
    execute.assert_not_awaited()


async def test_a_managed_anthropic_key_is_refused():
    managed = TEST_ANTHROPIC_CREDENTIALS.model_copy(update={"is_managed": True})

    outputs, execute = await _run_with(managed)

    assert outputs == [("error", OWN_ANTHROPIC_KEY_REQUIRED)]
    execute.assert_not_awaited()


async def test_the_users_own_anthropic_key_runs_claude_code():
    outputs, execute = await _run_with(TEST_ANTHROPIC_CREDENTIALS)

    execute.assert_awaited_once()
    assert execute.await_args.kwargs["anthropic_api_key"] == "mock-anthropic-api-key"
    assert ("response", "done") in outputs


async def test_a_resumed_sandbox_runs_claude_on_the_key_picked_for_this_run():
    block = ClaudeCodeBlock()
    box = AsyncMock()
    box.sandbox_id = "sb-mine"

    async def run(command: str, **kwargs):
        stdout = '{"result": "done"}' if "claude" in command else "2026-10-27T00:00:00"
        return AsyncMock(exit_code=0, stdout=stdout, stderr="")

    box.commands.run = AsyncMock(side_effect=run)
    info = AsyncMock(
        metadata={
            "service": "autogpt-platform",
            "autogpt_owner": "user:user-a",
            "autogpt_kind": "claude_code",
            "autogpt_source": "block",
            "autogpt_env": "dev",
            "autogpt_user": "user-a",
        }
    )
    with (
        patch("backend.blocks.claude_code.BaseAsyncSandbox") as cls,
        patch(
            "backend.blocks.claude_code.connect_sandbox",
            AsyncMock(return_value=box),
        ),
        patch(
            "backend.blocks.claude_code.extract_and_store_sandbox_files",
            AsyncMock(return_value=[]),
        ),
    ):
        cls.get_info = AsyncMock(return_value=info)
        response, *_ = await block.execute_claude_code(
            e2b_api_key="k",
            anthropic_api_key="the-users-own-key",
            prompt="hi",
            timeout=60,
            setup_commands=[],
            working_directory="/home/user",
            session_id="s1",
            existing_sandbox_id="sb-mine",
            conversation_history="",
            dispose_sandbox=False,
            execution_context=ExecutionContext(user_id="user-a"),
        )

    assert response == "done"
    claude_call = next(
        call for call in box.commands.run.await_args_list if "claude" in call.args[0]
    )
    assert claude_call.kwargs["envs"] == {"ANTHROPIC_API_KEY": "the-users-own-key"}
