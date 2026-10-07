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
    ThinkingBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
)
from prisma.models import User

from backend.copilot.baseline import service as baseline
from backend.copilot.model import ChatMessage
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.sdk import dummy
from backend.copilot.sdk import service as sdk
from backend.data import db, redis_client
from backend.data.user import get_or_create_user
from backend.util.testing import is_tcp_port_reachable

from .recording import check_fixture, persisted_session, record_turn
from .scripted import baseline_turn, provider_round, sdk_service_turn, sdk_turn

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


async def test_baseline_drain_turn(
    monkeypatch: pytest.MonkeyPatch, baseline_offline: None, user_id: str
) -> None:
    """Follow-ups drained at turn start and between rounds, with reasoning."""
    session = await persisted_session(user_id, "Summarise example.com")
    rounds = [
        provider_round(
            ["Fetching it."],
            reasoning=["The user wants ", "a summary."],
            tool_call={"id": "call-fetch", "name": "web_fetch", "url": "example.com"},
        ),
        provider_round(["It is a placeholder page."], reasoning=["Short answer."]),
    ]
    monkeypatch.setattr(baseline, "call_provider_stream", AsyncMock(side_effect=rounds))

    async def execute(*, tool_call_id: str, **_: object) -> StreamToolOutputAvailable:
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id, toolName="web_fetch", output="Example Domain"
        )

    monkeypatch.setattr(baseline, "execute_tool", AsyncMock(side_effect=execute))
    monkeypatch.setattr(
        baseline,
        "drain_pending_safe",
        AsyncMock(return_value=[PendingMessage(content="Keep it short.")]),
    )
    monkeypatch.setattr(
        baseline,
        "drain_pending_messages",
        AsyncMock(side_effect=[[PendingMessage(content="In French, please.")], []]),
    )
    turn_id = str(uuid.uuid4())

    recorded = await record_turn(
        baseline_turn(session, turn_id), session=session, turn_id=turn_id
    )

    check_fixture("baseline-drain-turn", recorded)


async def test_sdk_reasoning_turn(
    monkeypatch: pytest.MonkeyPatch, user_id: str
) -> None:
    """Thinking before each reply, and a follow-up drained after a tool call."""
    session = await persisted_session(user_id, "What is in the repo?")
    follow_up = [PendingMessage(content="Only count markdown files.")]
    monkeypatch.setattr(
        sdk, "drain_pending_for_persist", AsyncMock(side_effect=[follow_up, []])
    )
    messages = [
        SystemMessage(subtype="init", data={}),
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="List the files first.", signature="s"),
                ToolUseBlock(id="bash-1", name="bash_exec", input={"command": "ls"}),
            ],
            model="drift",
        ),
        UserMessage(
            content=[ToolResultBlock(tool_use_id="bash-1", content="a.md b.py")]
        ),
        AssistantMessage(
            content=[
                ThinkingBlock(thinking="One markdown file.", signature="s"),
                TextBlock(text="There is one markdown file: a.md."),
            ],
            model="drift",
        ),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=2,
            session_id="drift",
        ),
    ]

    recorded = await record_turn(
        sdk_turn(session, messages), session=session, turn_id=str(uuid.uuid4())
    )

    check_fixture("sdk-reasoning-turn", recorded)


async def test_sdk_auto_continue_turn(user_id: str) -> None:
    """Two follow-ups queued after the last drain continue the same turn."""
    session = await persisted_session(
        user_id,
        "Draft the release note.",
        history=[
            ChatMessage(role="user", content="Hi"),
            ChatMessage(role="assistant", content="Hello! What can I do?"),
        ],
    )
    queued = [
        PendingMessage(content="Mention the new export."),
        PendingMessage(content="And keep it to one line."),
    ]

    turn_id = str(uuid.uuid4())

    recorded = await record_turn(
        sdk_service_turn(
            session,
            turn_id,
            [_reply("Release 1.2 adds sharing."), _reply("Release 1.2 adds export.")],
            queued_after_first=queued,
        ),
        session=session,
        turn_id=turn_id,
    )

    check_fixture("sdk-auto-continue-turn", recorded)


def _reply(text: str) -> list:
    return [
        SystemMessage(subtype="init", data={}),
        AssistantMessage(content=[TextBlock(text=text)], model="drift"),
        ResultMessage(
            subtype="success",
            result=text,
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=1,
            session_id="drift",
        ),
    ]


async def test_baseline_consecutive_tools_turn(
    monkeypatch: pytest.MonkeyPatch, baseline_offline: None, user_id: str
) -> None:
    """A tool call right after a tool result, with no text between them."""
    session = await persisted_session(user_id, "Compare example.com and example.org")
    rounds = [
        provider_round(
            [], tool_call={"id": "call-com", "name": "web_fetch", "url": "example.com"}
        ),
        provider_round(
            [], tool_call={"id": "call-org", "name": "web_fetch", "url": "example.org"}
        ),
        provider_round(["Both are placeholder pages."]),
    ]
    monkeypatch.setattr(baseline, "call_provider_stream", AsyncMock(side_effect=rounds))

    async def execute(*, tool_call_id: str, **_: object) -> StreamToolOutputAvailable:
        return StreamToolOutputAvailable(
            toolCallId=tool_call_id, toolName="web_fetch", output="Example Domain"
        )

    monkeypatch.setattr(baseline, "execute_tool", AsyncMock(side_effect=execute))
    turn_id = str(uuid.uuid4())

    recorded = await record_turn(
        baseline_turn(session, turn_id), session=session, turn_id=turn_id
    )

    check_fixture("baseline-consecutive-tools-turn", recorded)


async def test_sdk_consecutive_tools_turn(user_id: str) -> None:
    """A tool call right after a tool result, with no text between them."""
    session = await persisted_session(user_id, "Count the files, then the lines")
    messages = [
        SystemMessage(subtype="init", data={}),
        AssistantMessage(
            content=[
                ToolUseBlock(id="bash-1", name="bash_exec", input={"command": "ls"})
            ],
            model="drift",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="bash-1", content="a.md")]),
        AssistantMessage(
            content=[
                ToolUseBlock(
                    id="bash-2", name="bash_exec", input={"command": "wc a.md"}
                )
            ],
            model="drift",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="bash-2", content="3 a.md")]),
        AssistantMessage(
            content=[TextBlock(text="One file, three lines.")], model="drift"
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

    check_fixture("sdk-consecutive-tools-turn", recorded)
