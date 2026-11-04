"""Claude Code only resumes a Claude Code sandbox that was created for the same
user, runs only on the user's own Anthropic key, and on the platform's E2B key
pauses a sandbox it keeps."""

from unittest.mock import AsyncMock, call, patch

import pytest
from pydantic import SecretStr

from backend.blocks.claude_code import (
    OWN_ANTHROPIC_KEY_REQUIRED,
    TEST_ANTHROPIC_CREDENTIALS,
    TEST_ANTHROPIC_CREDENTIALS_INPUT,
    TEST_E2B_CREDENTIALS,
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
from backend.util.sandbox_output import SandboxOutputLimitError


@pytest.mark.parametrize(
    "dispose_sandbox, pause_when_kept",
    [(True, False), (False, True)],
    ids=["disposed", "kept-on-platform-key"],
)
async def test_resuming_another_users_sandbox_is_refused(
    dispose_sandbox, pause_when_kept
):
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
    hold = AsyncMock()
    with (
        patch("backend.blocks.claude_code.BaseAsyncSandbox") as cls,
        patch("backend.blocks.claude_code.hold_sandbox", hold),
        patch("backend.util.e2b_network.AsyncSandbox") as by_id,
    ):
        # The stamp is read through the static ``get_info`` before any connect.
        cls.get_info = AsyncMock(return_value=theirs.get_info.return_value)
        cls.connect = AsyncMock(return_value=theirs)
        by_id.pause = AsyncMock()
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
                dispose_sandbox=dispose_sandbox,
                execution_context=ExecutionContext(user_id="user-b"),
                pause_when_kept=pause_when_kept,
            )
    cls.get_info.assert_awaited_once_with("sb-theirs", api_key="k")
    # Connecting would resume the other user's box on their bill.
    cls.connect.assert_not_awaited()
    theirs.commands.run.assert_not_awaited()
    # Not ours to kill, count or pause, whatever the run would do with its own.
    theirs.kill.assert_not_awaited()
    hold.assert_not_awaited()
    by_id.pause.assert_not_awaited()


async def test_resuming_the_users_own_box_of_another_kind_is_refused():
    """A resume continues a Claude Code sandbox, not another box of the user's."""
    block = ClaudeCodeBlock()
    code_box = AsyncMock()
    code_box.get_info = AsyncMock(
        return_value=AsyncMock(
            metadata={
                "service": "autogpt-platform",
                "autogpt_owner": "user:user-a",
                "autogpt_kind": "code",
                "autogpt_source": "block",
                "autogpt_env": "dev",
                "autogpt_user": "user-a",
            }
        )
    )
    with (
        patch("backend.blocks.claude_code.BaseAsyncSandbox") as cls,
        patch("backend.blocks.claude_code.hold_sandbox", AsyncMock()) as hold,
        patch("backend.blocks.claude_code.release_sandbox", AsyncMock()) as release,
    ):
        cls.get_info = AsyncMock(return_value=code_box.get_info.return_value)
        cls.connect = AsyncMock(return_value=code_box)
        with pytest.raises(ClaudeCodeExecutionError, match="not a Claude Code sandbox"):
            await block.execute_claude_code(
                e2b_api_key="k",
                anthropic_api_key="a",
                prompt="hi",
                timeout=60,
                setup_commands=[],
                working_directory="/home/user",
                session_id="s1",
                existing_sandbox_id="sb-code",
                conversation_history="",
                dispose_sandbox=False,
                execution_context=ExecutionContext(user_id="user-a"),
                pause_when_kept=True,
            )
    # Refused before the box is counted as held, so nothing pauses it.
    hold.assert_not_awaited()
    release.assert_not_awaited()
    cls.connect.assert_not_awaited()
    code_box.commands.run.assert_not_awaited()


async def _run_with(
    anthropic: APIKeyCredentials, e2b: APIKeyCredentials = e2b_credentials
) -> tuple[list, AsyncMock]:
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
                e2b_credentials=e2b,
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


# A credential with no key of its own: E2B opens the box on the operator's,
# so the block must ask for the pause as it does for the platform's own.
EMPTY_KEY_E2B_CREDENTIALS = TEST_E2B_CREDENTIALS.model_copy(
    update={"api_key": SecretStr("")}
)


@pytest.mark.parametrize(
    "e2b, paused",
    [
        (e2b_credentials, True),
        (TEST_E2B_CREDENTIALS, False),
        (EMPTY_KEY_E2B_CREDENTIALS, True),
    ],
    ids=["platform-key", "own-key", "empty-key"],
)
async def test_a_kept_sandbox_is_paused_only_on_the_platform_e2b_key(e2b, paused):
    _, execute = await _run_with(TEST_ANTHROPIC_CREDENTIALS, e2b)

    assert execute.await_args.kwargs["pause_when_kept"] is paused


@pytest.mark.parametrize(
    "dispose_sandbox, pause_when_kept, outcome",
    [
        (True, True, "kill"),
        (True, False, "kill"),
        (False, True, "release"),
        (False, False, None),
    ],
)
async def test_the_sandbox_is_killed_released_or_left_running(
    dispose_sandbox, pause_when_kept, outcome
):
    box = _claude_box("sb-new")
    m = "backend.blocks.claude_code"
    with (
        patch(f"{m}.create_sandbox", AsyncMock(return_value=box)),
        patch(f"{m}.extract_and_store_sandbox_files", AsyncMock(return_value=[])),
        patch(f"{m}.kill_sandbox", AsyncMock()) as kill,
        patch(f"{m}.hold_sandbox", AsyncMock()) as hold,
        patch(f"{m}.release_sandbox", AsyncMock()) as release,
    ):
        await ClaudeCodeBlock().execute_claude_code(
            e2b_api_key="k",
            anthropic_api_key="a",
            prompt="hi",
            timeout=60,
            setup_commands=[],
            working_directory="/home/user",
            session_id="",
            existing_sandbox_id="",
            conversation_history="",
            dispose_sandbox=dispose_sandbox,
            execution_context=ExecutionContext(user_id="user-a"),
            pause_when_kept=pause_when_kept,
        )
    kept = outcome == "release"
    assert kill.await_args_list == ([call(box)] if outcome == "kill" else [])
    assert hold.await_args_list == ([call("sb-new")] if kept else [])
    assert release.await_args_list == (
        [call("sb-new", box, api_key="k")] if kept else []
    )


@pytest.mark.parametrize(
    "dispose_sandbox, outcome", [(True, "kill"), (False, "release")]
)
async def test_output_past_the_limit_still_kills_or_releases_the_sandbox(
    dispose_sandbox, outcome
):
    box = _claude_box("sb-new")
    m = "backend.blocks.claude_code"
    with (
        patch(f"{m}.create_sandbox", AsyncMock(return_value=box)),
        patch(
            f"{m}.run_sandbox_command",
            AsyncMock(side_effect=SandboxOutputLimitError("too much")),
        ),
        patch(f"{m}.kill_sandbox", AsyncMock()) as kill,
        patch(f"{m}.hold_sandbox", AsyncMock()),
        patch(f"{m}.release_sandbox", AsyncMock()) as release,
        pytest.raises(Exception, match="too much"),
    ):
        await ClaudeCodeBlock().execute_claude_code(
            e2b_api_key="k",
            anthropic_api_key="a",
            prompt="hi",
            timeout=60,
            setup_commands=[],
            working_directory="/home/user",
            session_id="",
            existing_sandbox_id="",
            conversation_history="",
            dispose_sandbox=dispose_sandbox,
            execution_context=ExecutionContext(user_id="user-a"),
            pause_when_kept=True,
        )
    assert kill.await_args_list == ([call(box)] if outcome == "kill" else [])
    assert release.await_args_list == (
        [call("sb-new", box, api_key="k")] if outcome == "release" else []
    )


async def test_a_resumed_sandbox_is_counted_first_and_runs_for_this_runs_timeout():
    """Another run returning during the connect must not pause the box under
    this one, and a box resumed from a pause must last as long as this run
    asks, not the SDK's default."""
    box = _claude_box("sb-mine")
    order: list[str] = []

    async def connect(*args, **kwargs):
        order.append("connect")
        return box

    m = "backend.blocks.claude_code"
    with (
        patch(f"{m}.BaseAsyncSandbox") as cls,
        patch(f"{m}.connect_sandbox", AsyncMock(side_effect=connect)) as connect_mock,
        patch(f"{m}.extract_and_store_sandbox_files", AsyncMock(return_value=[])),
        patch(
            f"{m}.hold_sandbox", AsyncMock(side_effect=lambda *_: order.append("hold"))
        ),
        patch(f"{m}.release_sandbox", AsyncMock()) as release,
    ):
        cls.get_info = AsyncMock(return_value=AsyncMock(metadata=_STAMP_A))
        await ClaudeCodeBlock().execute_claude_code(
            e2b_api_key="k",
            anthropic_api_key="a",
            prompt="hi",
            timeout=1800,
            setup_commands=[],
            working_directory="/home/user",
            session_id="s1",
            existing_sandbox_id="sb-mine",
            conversation_history="",
            dispose_sandbox=False,
            execution_context=ExecutionContext(user_id="user-a"),
            pause_when_kept=True,
        )
    assert order == ["hold", "connect"]
    assert connect_mock.await_args.kwargs["timeout"] == 1800
    assert release.await_args_list == [call("sb-mine", box, api_key="k")]


_STAMP_A = {
    "service": "autogpt-platform",
    "autogpt_owner": "user:user-a",
    "autogpt_kind": "claude_code",
    "autogpt_source": "block",
    "autogpt_env": "dev",
    "autogpt_user": "user-a",
}


def _claude_box(sandbox_id: str) -> AsyncMock:
    box = AsyncMock()
    box.sandbox_id = sandbox_id
    box.commands.run = AsyncMock(
        return_value=AsyncMock(exit_code=0, stdout='{"result": "done"}', stderr="")
    )
    return box
