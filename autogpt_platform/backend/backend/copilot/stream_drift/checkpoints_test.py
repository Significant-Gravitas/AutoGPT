"""A checkpoint follows a persist that landed, never with a block open.

Both engines run their real persist paths with the DB swapped for
``saving_into``; the timeline interleaves each save with the events around it,
so a checkpoint must equal the one the latest save before it describes.
"""

from collections.abc import AsyncGenerator
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

from backend.copilot.model import ChatSession
from backend.copilot.response_model import (
    ResponseType,
    StreamBaseResponse,
    StreamCheckpoint,
)
from backend.copilot.sdk import service as sdk
from backend.copilot.stream_checkpoint import turn_checkpoint

from .recording import (
    assert_fold_matches_rows,
    fixture_names,
    load_fixture,
    saving_into,
)
from .scripted import sdk_turn, session_with_prompt

# The prompt is row 0, so every turn's rows start at 1.
TURN_START = 1


async def test_the_sdk_engine_checkpoints_each_flush_that_landed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdk, "_FLUSH_MESSAGE_THRESHOLD", 1)
    timeline = Timeline()
    monkeypatch.setattr(sdk, "upsert_chat_session", timeline.save)

    await timeline.run(sdk_turn(session_with_prompt("List the files"), _sdk_script()))

    # The flush policy holds every flush but the one after the tool round
    # (assistant row plus tool row); the second checkpoint is the turn end's.
    assert [c.rows for c in timeline.checkpoints()] == [2, 3]


async def test_the_sdk_engine_publishes_no_checkpoint_for_a_failed_flush(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sdk, "_FLUSH_MESSAGE_THRESHOLD", 1)
    timeline = Timeline()
    monkeypatch.setattr(
        sdk, "upsert_chat_session", AsyncMock(side_effect=ConnectionError("db down"))
    )

    with pytest.raises(ConnectionError):  # the scripted turn-end persist
        await timeline.run(
            sdk_turn(session_with_prompt("List the files"), _sdk_script())
        )

    assert timeline.checkpoints() == []


@pytest.mark.parametrize("name", fixture_names())
def test_every_recorded_turn_folds_to_its_rows(name: str) -> None:
    assert_fold_matches_rows(load_fixture(name))


@pytest.mark.parametrize("name", fixture_names())
def test_no_recorded_checkpoint_sits_inside_a_block_or_a_tool_call(name: str) -> None:
    open_parts: set[str] = set()
    checkpoints = 0
    for entry in load_fixture(name).entries:
        chunk = entry["data"]
        _track_open_parts(open_parts, chunk["type"], chunk)
        if chunk["type"] == "data-checkpoint":
            assert not open_parts, f"{entry['id']} checkpoints inside {open_parts}"
            checkpoints += 1
    assert checkpoints


class Timeline:
    """Saves and published events in the order they happened."""

    def __init__(self) -> None:
        self.saves: list[ChatSession] = []
        self._save = saving_into(self.saves)
        self.log: list[ChatSession | StreamBaseResponse] = []

    async def save(self, session: ChatSession) -> ChatSession:
        saved = await self._save(session)
        self.log.append(self.saves[-1])
        return saved

    async def run(self, engine: AsyncGenerator[StreamBaseResponse, None]) -> None:
        async for event in engine:
            self.log.append(event)

    def checkpoints(self) -> list[StreamCheckpoint]:
        """Every checkpoint, each checked against the save before it and
        against the blocks and tool calls open at that point."""
        last_save: ChatSession | None = None
        open_parts: set[str] = set()
        checkpoints = []
        for item in self.log:
            if isinstance(item, ChatSession):
                last_save = item
                continue
            _track_open_parts(open_parts, item.type.value, item.model_dump())
            if isinstance(item, StreamCheckpoint):
                assert not open_parts, f"checkpoint inside {open_parts}"
                assert last_save is not None, "checkpoint before any save"
                assert item == turn_checkpoint(last_save.messages, TURN_START)
                checkpoints.append(item)
        return checkpoints


def _track_open_parts(open_parts: set[str], kind: str, chunk: dict) -> None:
    if kind in (ResponseType.TEXT_START, ResponseType.REASONING_START):
        open_parts.add(chunk["id"])
    elif kind in (ResponseType.TEXT_END, ResponseType.REASONING_END):
        open_parts.discard(chunk["id"])
    elif kind == ResponseType.TOOL_INPUT_AVAILABLE:
        open_parts.add(chunk["toolCallId"])
    elif kind == ResponseType.TOOL_OUTPUT_AVAILABLE:
        open_parts.discard(chunk["toolCallId"])


def _sdk_script() -> list:
    return [
        SystemMessage(subtype="init", data={}),
        AssistantMessage(content=[TextBlock(text="Let me look.")], model="drift"),
        AssistantMessage(
            content=[
                ToolUseBlock(id="bash-1", name="bash_exec", input={"command": "ls"})
            ],
            model="drift",
        ),
        UserMessage(content=[ToolResultBlock(tool_use_id="bash-1", content="a.md")]),
        AssistantMessage(content=[TextBlock(text="There is a.md.")], model="drift"),
        ResultMessage(
            subtype="success",
            duration_ms=100,
            duration_api_ms=50,
            is_error=False,
            num_turns=2,
            session_id="drift",
        ),
    ]
