import pytest

from backend.copilot.baseline import stream_chat_completion_baseline
from backend.copilot.model import ChatSession, LocalExecutionTargetMetadata
from backend.copilot.response_model import StreamError


@pytest.mark.asyncio
async def test_a_local_pc_chat_is_refused_before_any_tool_can_run() -> None:
    """Only the SDK engine is bound to the user's machine; this engine would
    run a Local PC chat's tools in a cloud sandbox, so it refuses the turn."""
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        session_id="session-1",
        execution_target=LocalExecutionTargetMetadata(
            machine_id="machine-1",
            directory_ref="directory-1",
            allowed_root="/workspace",
            root_fingerprint="a" * 64,
            root_grant="grant-1",
        ),
    )

    events = [
        event
        async for event in stream_chat_completion_baseline(
            "session-1", "list my files", user_id="user-1", session=session
        )
    ]

    assert len(events) == 1
    assert isinstance(events[0], StreamError)
    assert events[0].code == "local_pc_requires_sdk_engine"
