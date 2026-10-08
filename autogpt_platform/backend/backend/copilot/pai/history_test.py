"""History storage, chat-row conversion, and compaction."""

from unittest.mock import AsyncMock, patch

import pytest
from pydantic_ai.messages import (
    BinaryContent,
    ModelMessage,
    ModelMessagesTypeAdapter,
    ModelRequest,
    ModelResponse,
    TextPart,
    ThinkingPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from backend.copilot.model import ChatMessage

from .compaction import compact
from .history import (
    HeldToolCall,
    chat_rows_to_messages,
    close_pending_calls,
    messages_to_chat_rows,
    pending_tool_calls,
)
from .history_store import (
    HISTORY_SUFFIX,
    decode_history,
    download_history,
    encode_history,
    history_path_parts,
    upload_history,
)


def _conversation() -> list[ModelMessage]:
    return [
        ModelRequest(
            parts=[UserPromptPart(content="find a cat")], instructions="BIG PROMPT"
        ),
        ModelResponse(
            parts=[
                ThinkingPart(content="hmm", signature="sig-1"),
                TextPart(content="Looking."),
                ToolCallPart("web_fetch", '{"url": "https://cat"}', tool_call_id="c1"),
            ]
        ),
        ModelRequest(
            parts=[ToolReturnPart("web_fetch", "a cat", tool_call_id="c1")],
            instructions="BIG PROMPT",
        ),
        ModelResponse(parts=[TextPart(content="Found a cat.")]),
    ]


def test_round_trip_keeps_messages_and_drops_instructions():
    held = [HeldToolCall(tool_call_id="c9", tool_name="t", review_id="r", output="o")]
    conversation = _conversation()
    loaded = decode_history(encode_history(conversation, held, watermark=7))
    assert loaded is not None
    assert loaded.watermark == 7
    assert loaded.held == held
    expected = ModelMessagesTypeAdapter.dump_python(conversation, mode="json")
    got = ModelMessagesTypeAdapter.dump_python(loaded.messages, mode="json")
    for message in expected:
        if message["kind"] == "request":
            message["instructions"] = None
    assert got == expected
    # The thinking signature survives (chat rows cannot carry it).
    response = loaded.messages[1]
    assert response.kind == "response"
    assert response.parts[0] == ThinkingPart(content="hmm", signature="sig-1")


def test_binary_content_is_not_stored():
    messages: list[ModelMessage] = [
        ModelRequest(
            parts=[
                UserPromptPart(
                    content=[
                        "see",
                        BinaryContent(data=b"\x89PNG", media_type="image/png"),
                    ]
                )
            ]
        )
    ]
    loaded = decode_history(encode_history(messages, [], watermark=1))
    assert loaded is not None
    request = loaded.messages[0]
    assert request.kind == "request"
    part = request.parts[0]
    assert isinstance(part, UserPromptPart)
    assert part.content == [
        "see",
        "[image/png attachment, 4 bytes, not kept in history]",
    ]


def test_corrupt_file_is_ignored():
    assert decode_history(b"not json") is None


def test_chat_rows_rebuild_a_valid_history():
    rows = [
        ChatMessage(role="user", content="hi"),
        ChatMessage(role="reasoning", content="thinking"),
        ChatMessage(
            role="assistant",
            content="Fetching",
            tool_calls=[
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{}"},
                },
                {
                    "id": "c2",
                    "type": "function",
                    "function": {"name": "web_fetch", "arguments": "{}"},
                },
            ],
        ),
        ChatMessage(role="tool", content="page", tool_call_id="c1"),
        ChatMessage(role="tool", content="orphan", tool_call_id="zz"),
        ChatMessage(role="assistant", content="Done"),
    ]
    messages = chat_rows_to_messages(rows)
    assert [m.kind for m in messages] == ["request", "response", "request", "response"]
    response = messages[1]
    assert response.kind == "response"
    # c2 never got a result, so it is dropped rather than left dangling.
    assert [c.tool_call_id for c in response.tool_calls] == ["c1"]
    tool_request = messages[2]
    assert tool_request.kind == "request"
    assert [p.part_kind for p in tool_request.parts] == ["tool-return"]
    assert pending_tool_calls(messages) == []
    back = messages_to_chat_rows(messages)
    assert [r.role for r in back] == ["user", "assistant", "tool", "assistant"]


def test_pending_calls_close_in_place():
    messages = _conversation()[:2]
    assert [c.tool_call_id for c in pending_tool_calls(messages)] == ["c1"]
    closed = close_pending_calls(messages, {"c1": "held result"})
    assert pending_tool_calls(closed) == []
    last = closed[-1]
    assert last.kind == "request"
    part = last.parts[0]
    assert isinstance(part, ToolReturnPart)
    assert (part.tool_name, part.content, part.tool_call_id) == (
        "web_fetch",
        "held result",
        "c1",
    )


class _MemoryStorage:
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}

    async def store(self, workspace_id, file_id, filename, content) -> str:
        path = f"local://{workspace_id}/{file_id}/{filename}"
        self.files[path] = content
        return path

    async def retrieve(self, path: str) -> bytes:
        if path not in self.files:
            raise FileNotFoundError(path)
        return self.files[path]


@pytest.mark.asyncio
async def test_upload_then_download_under_the_transcript_convention():
    storage = _MemoryStorage()
    user = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"
    session = "5e53486c-cf57-477e-ba2a-cb02dc828e1c"
    with patch(
        "backend.copilot.pai.history_store.get_workspace_storage",
        AsyncMock(return_value=storage),
    ):
        assert await download_history(user, session) == (True, None)
        await upload_history(user, session, _conversation(), [], watermark=4)
        upload_safe, loaded = await download_history(user, session)
    assert history_path_parts(user, session) == (
        "cli-sessions",
        user,
        f"{session}{HISTORY_SUFFIX}",
    )
    assert list(storage.files) == [f"local://cli-sessions/{user}/{session}.pai.json"]
    assert upload_safe is True
    assert loaded is not None and loaded.watermark == 4
    assert len(loaded.messages) == 4


@pytest.mark.asyncio
async def test_compaction_reuses_the_baseline_compressor():
    messages = [
        *_conversation(),
        ModelRequest(parts=[UserPromptPart("next")], instructions="I"),
    ]

    async def squash(rows, model):
        return [ChatMessage(role="user", content="[summary]"), rows[-1]]

    with patch("backend.copilot.pai.compaction._compress_session_messages", squash):
        compacted = await compact(messages, "anthropic/claude-test")
    assert [m.kind for m in compacted] == ["request"]
    last = compacted[-1]
    assert last.kind == "request"
    assert last.instructions == "I"
    assert [p.content for p in last.parts if isinstance(p, UserPromptPart)] == [
        "[summary]",
        "next",
    ]


@pytest.mark.asyncio
async def test_compaction_is_a_no_op_when_nothing_was_cut():
    messages = [*_conversation(), ModelRequest(parts=[UserPromptPart("next")])]

    async def untouched(rows, model):
        return rows

    with patch("backend.copilot.pai.compaction._compress_session_messages", untouched):
        assert await compact(messages, "m") is messages
