"""Claude Code only resumes a sandbox that was created for the same user."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.blocks.claude_code import ClaudeCodeBlock, ClaudeCodeExecutionError
from backend.executor.utils import ExecutionContext


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
    theirs.commands.run.assert_not_awaited()
    theirs.kill.assert_not_awaited()
