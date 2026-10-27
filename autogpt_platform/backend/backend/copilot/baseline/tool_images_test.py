import asyncio
import io
from collections.abc import Sequence
from functools import partial
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import orjson
import pytest
from openai.types.chat import ChatCompletionChunk
from PIL import Image

from backend.copilot.baseline import service
from backend.copilot.baseline.continuation_note_test import _sent_messages
from backend.copilot.baseline.service_cancellation_test import baseline_io
from backend.copilot.graphiti.context_marker import INJECTED_MEMORY_MARKER
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.tool_image import prepare_image
from backend.copilot.transcript_builder import TranscriptBuilder
from backend.util.tool_call_loop import LLMToolCall, ToolCallResult, tool_call_loop

__all__ = ["baseline_io"]


@pytest.mark.parametrize(("rounds", "batch_size"), [(12, 1), (1, 12)])
async def test_image_context_stays_bounded_through_tool_rounds(
    monkeypatch, baseline_io, rounds, batch_size
):
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(output, format="PNG")
    image = prepare_image(output.getvalue())
    sent: list[list[dict[str, Any]]] = []

    async def provider(**kwargs):
        sent.append(orjson.loads(orjson.dumps(kwargs["messages"])))
        round_number = len(sent)
        calls = (
            [
                {
                    "index": index,
                    "id": f"page-{round_number}-{index}",
                    "type": "function",
                    "function": {"name": "read_workspace_file", "arguments": "{}"},
                }
                for index in range(batch_size)
            ]
            if round_number <= rounds
            else []
        )
        chunk = ChatCompletionChunk.model_validate(
            {
                "id": "image-round",
                "created": 0,
                "model": "gpt-4o",
                "object": "chat.completion.chunk",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": calls,
                            "content": None if calls else "Done",
                        },
                        "finish_reason": "tool_calls" if calls else "stop",
                    }
                ],
            }
        )
        stream = MagicMock()
        stream.__aiter__.return_value = [chunk]
        stream.close = AsyncMock()
        return stream

    async def read_image(call: LLMToolCall, tools: Sequence[Any]) -> ToolCallResult:
        return ToolCallResult(
            tool_call_id=call.id,
            tool_name=call.name,
            content=f"workspace://{call.id}.png",
            image_blocks=[image.anthropic()],
        )

    monkeypatch.setattr(service, "call_provider_stream", provider)
    transcript = TranscriptBuilder()
    messages = [{"role": "user", "content": "Inspect the pages"}]
    async for _ in tool_call_loop(
        messages=messages,
        tools=[],
        llm_call=partial(
            service._baseline_llm_caller,
            state=service._BaselineStreamState(model="gpt-4o"),
        ),
        execute_tool=read_image,
        update_conversation=partial(
            service._baseline_conversation_updater, transcript_builder=transcript
        ),
        max_iterations=rounds + 1,
        last_iteration_message="Finish",
    ):
        pass

    assert len(sent) == rounds + 1
    for index, request in enumerate(sent):
        images = [
            part
            for message in request
            if isinstance(message.get("content"), list)
            for part in message["content"]
            if part["type"] == "image_url"
        ]
        assert len(images) == min(index * batch_size, 10)
        assert all(part == image.openai() for part in images)
    assert "omitted" in orjson.dumps(sent[-1]).decode()
    assert [
        message["content"] for message in sent[-1] if message["role"] == "tool"
    ] == [
        f"workspace://page-{round_number}-{index}.png"
        for round_number in range(1, rounds + 1)
        for index in range(batch_size)
    ]
    assert (
        orjson.dumps(transcript.entries_as_dicts()).count(image.data_base64.encode())
        == rounds * batch_size
    )


async def test_requeued_approved_image_survives_turn_start_budget(
    monkeypatch, baseline_io
):
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "blue").save(output, format="PNG")
    image = prepare_image(output.getvalue())
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Image inspection"
    session.messages = [
        ChatMessage(role="user", content="Inspect the page", sequence=0)
    ]
    pending = PendingMessage(content="Approved image", model_image=image)
    monkeypatch.setattr(
        service, "drain_pending_safe", AsyncMock(return_value=[pending])
    )
    monkeypatch.setattr(
        service, "persist_pending_as_user_rows", AsyncMock(return_value=True)
    )
    monkeypatch.setattr(
        service, "build_turn_budget_block", AsyncMock(return_value="Live budget: 10.\n")
    )
    messages = await _sent_messages(monkeypatch, session, message=None)
    assert isinstance(messages[-1]["content"], list)
    assert messages[-1]["content"][0] == {"type": "text", "text": "Live budget: 10.\n"}
    assert messages[-1]["content"][-1] == image.openai()
    assert image.data_base64 not in session.model_dump_json()


@pytest.mark.parametrize("pending_image", [False, True])
@pytest.mark.parametrize(
    "warm_context,has_memory",
    [
        ("<temporal_context>Relevant facts</temporal_context>", True),
        ("Unmarked memory must not reach the model", False),
    ],
)
async def test_context_and_attachments_survive_a_pending_image(
    monkeypatch, baseline_io, pending_image, warm_context, has_memory
):
    output = io.BytesIO()
    Image.new("RGB", (20, 20), "red").save(output, format="PNG")
    image = prepare_image(output.getvalue())
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Image inspection"
    session.metadata.builder_graph_id = "workflow-1"
    session.messages = [ChatMessage(role="user", content="Inspect the page")]
    pending = PendingMessage(
        content="Approved image", model_image=image if pending_image else None
    )
    hints = {
        "build_turn_budget_block": "Live budget: 10.\n",
        "build_builder_context_turn_prefix": "<builder_context>Live workflow</builder_context>",
        "build_skills_update_notice": "<skills_update>New skill</skills_update>",
    }
    for name, value in {
        **hints,
        "_fetch_graphiti_context": warm_context,
        "drain_pending_safe": [pending],
        "persist_pending_as_user_rows": True,
        "build_builder_system_prompt_suffix": "",
        "build_budget_ctx": "",
        "build_session_context": "",
        "build_skills_context": "",
        "inject_user_context": None,
        "clear_pending_question": None,
        "capture_chat_turn": None,
        "_prepare_baseline_attachments": ("Attached page", [image.anthropic()]),
    }.items():
        monkeypatch.setattr(service, name, AsyncMock(return_value=value))
    seen = "<seen_capabilities>Known tool</seen_capabilities>"
    monkeypatch.setattr(service, "build_seen_capabilities_notice", lambda _: seen)
    stream = MagicMock()
    stream.__aiter__.return_value = []
    stream.close = AsyncMock()
    provider = AsyncMock(return_value=stream)
    monkeypatch.setattr(service, "call_provider_stream", provider)

    async def consume() -> None:
        async for _ in service.stream_chat_completion_baseline(
            session.session_id,
            "Inspect the page",
            is_user_message=True,
            user_id=session.user_id,
            session=session,
            file_ids=["attached-image"],
            context={"url": "https://example.com/page", "content": "Shared context"},
        ):
            pass

    await asyncio.wait_for(consume(), timeout=5)

    parts = provider.await_args.kwargs["messages"][-1]["content"]
    assert isinstance(parts, list)
    text = "".join(part["text"] for part in parts if part["type"] == "text")
    expected_memory = [warm_context] if has_memory else []
    saved_text = "".join(message.content or "" for message in session.messages)
    for hint in [
        *hints.values(),
        *expected_memory,
        seen,
        "Attached page",
        "Shared context",
    ]:
        assert (
            hint.replace(
                "<temporal_context>", f"<temporal_context {INJECTED_MEMORY_MARKER}>"
            )
            in text
        )
        assert hint not in saved_text
    assert text.count(INJECTED_MEMORY_MARKER) == int(has_memory)
    if not has_memory:
        assert warm_context not in text
        assert "None" not in text
    assert warm_context not in session.model_dump_json()
    assert INJECTED_MEMORY_MARKER not in session.model_dump_json()
    assert text.index(seen) < text.index(hints["build_skills_update_notice"])
    assert text.index(hints["build_builder_context_turn_prefix"]) < text.index(
        hints["build_turn_budget_block"]
    )
    assert [part for part in parts if part["type"] == "image_url"] == [
        image.openai()
    ] * (2 if pending_image else 1)
    assert parts[-1] == image.openai()
