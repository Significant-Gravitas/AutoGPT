import asyncio
import io
from unittest.mock import AsyncMock, MagicMock

import pytest
from PIL import Image

from backend.copilot.baseline import service
from backend.copilot.baseline.continuation_note_test import _sent_messages
from backend.copilot.baseline.service_cancellation_test import baseline_io
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.tool_image import prepare_image

__all__ = ["baseline_io"]


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
async def test_context_and_attachments_survive_a_pending_image(
    monkeypatch, baseline_io, pending_image
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
        "_fetch_graphiti_context": "<temporal_context>Relevant facts</temporal_context>",
        "build_builder_context_turn_prefix": "<builder_context>Live workflow</builder_context>",
        "build_skills_update_notice": "<skills_update>New skill</skills_update>",
    }
    for name, value in {
        **hints,
        "drain_pending_safe": [pending],
        "persist_pending_as_user_rows": True,
        "is_enabled_for_user": True,
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
    for hint in [*hints.values(), seen, "Attached page", "Shared context"]:
        assert hint in text
        assert hint not in session.model_dump_json()
    assert text.index(seen) < text.index(hints["build_skills_update_notice"])
    assert text.index(hints["build_builder_context_turn_prefix"]) < text.index(
        hints["build_turn_budget_block"]
    )
    assert [part for part in parts if part["type"] == "image_url"] == [
        image.openai()
    ] * (2 if pending_image else 1)
    assert parts[-1] == image.openai()
