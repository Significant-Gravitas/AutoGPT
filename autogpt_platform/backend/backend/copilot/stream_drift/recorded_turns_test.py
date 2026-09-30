"""The recorded turns the frontend drift suite replays still match the pipeline.

Each test drives an engine through the real registry, Redis and Postgres,
then compares what it stored, served and persisted with the committed fixture.
A red here means the sender changed; re-record (see ``recording.py``) and rerun
the frontend suite.
"""

import uuid
from unittest.mock import AsyncMock

import pytest
from claude_agent_sdk import (
    AssistantMessage,
    ResultMessage,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from prisma.models import User

from backend.copilot.baseline import service as baseline
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk import dummy
from backend.data import db, redis_client
from backend.data.user import get_or_create_user
from backend.util.testing import is_tcp_port_reachable

from .recording import check_fixture, persisted_session, record_turn
from .scripted import baseline_turn, provider_round, sdk_turn

pytestmark = pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="no local Redis reachable; the stream registry needs one to run",
)


@pytest.fixture
async def user_id():
    """A user of its own, removed with its sessions after the test."""
    await db.connect()
    user_id = str(uuid.uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@drift.invalid"})
    yield user_id
    await User.prisma().delete_many(where={"id": user_id})


async def test_dummy_text_turn(user_id: str) -> None:
    session = await persisted_session(user_id, "Count to three")

    recorded = await record_turn(
        dummy.stream_chat_completion_dummy(
            session.session_id, message="Count to three", session=session
        ),
        session=session,
        turn_id=str(uuid.uuid4()),
    )

    check_fixture("dummy-text-turn", recorded)


async def test_baseline_tool_turn(
    monkeypatch: pytest.MonkeyPatch, baseline_offline: None, user_id: str
) -> None:
    session = await persisted_session(user_id, "What does example.com say?")
    rounds = [
        provider_round(
            ["Let me ", "fetch that ", "page."],
            tool_call={"id": "call-fetch", "name": "web_fetch", "url": "example.com"},
        ),
        provider_round(["It says ", "Example Domain."]),
    ]
    monkeypatch.setattr(baseline, "call_provider_stream", AsyncMock(side_effect=rounds))

    async def execute(*, tool_call_id: str, **_: object) -> StreamToolOutputAvailable:
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id,
            toolName="web_fetch",
            output="<h1>Example Domain</h1>",
        )

    monkeypatch.setattr(baseline, "execute_tool", AsyncMock(side_effect=execute))
    turn_id = str(uuid.uuid4())

    recorded = await record_turn(
        baseline_turn(session, turn_id), session=session, turn_id=turn_id
    )

    check_fixture("baseline-tool-turn", recorded)


async def test_sdk_late_tool_result_turn(user_id: str) -> None:
    """Dev session c68992f9's order: a tool result lands after the next text."""
    session = await persisted_session(user_id, "Check the task and list the files")
    messages = [
        SystemMessage(subtype="init", data={}),
        AssistantMessage(
            content=[ToolUseBlock(id="task-1", name="TaskOutput", input={})],
            model="drift",
        ),
        AssistantMessage(content=[TextBlock(text="The task finished.")], model="drift"),
        UserMessage(content=[ToolResultBlock(tool_use_id="task-1", content="late")]),
        AssistantMessage(
            content=[
                ToolUseBlock(id="bash-1", name="bash_exec", input={"command": "ls"})
            ],
            model="drift",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="bash-1", content="report.md")]
        ),
        AssistantMessage(
            content=[TextBlock(text="All done: report.md is ready.")], model="drift"
        ),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=3,
            session_id="drift",
        ),
    ]

    recorded = await record_turn(
        sdk_turn(session, messages), session=session, turn_id=str(uuid.uuid4())
    )

    check_fixture("sdk-late-tool-result", recorded)
