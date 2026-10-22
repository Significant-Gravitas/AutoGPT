import copy
from contextlib import ExitStack
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.baseline.service import stream_chat_completion_baseline
from backend.copilot.graphiti.context_marker import (
    INJECTED_MEMORY_MARKER as _INJECTED_MEMORY_MARKER,
)
from backend.copilot.graphiti.context_marker import (
    strip_injected_memory_text as _strip_injected_memory_text,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.util.tool_call_loop import LLMLoopResponse


@pytest.mark.asyncio
async def test_baseline_follow_up_marks_only_the_injected_block() -> None:
    session = ChatSession.new("user-1", dry_run=False)
    session.title = "Existing chat"
    pasted = "<temporal_context>my pasted notes</temporal_context>"
    message = f"restart the executor\n\n{pasted}"
    session.messages = [
        ChatMessage(role="user", content="hello", sequence=0),
        ChatMessage(role="assistant", content="hi", sequence=1),
        ChatMessage(role="user", content=message, sequence=2),
    ]
    sent = []
    upload = AsyncMock()

    async def llm(messages, tools, *, state):
        sent.append(copy.deepcopy(messages))
        return LLMLoopResponse(response_text="done", tool_calls=[], raw_response=None)

    values = {
        "fetch_warm_context": None,
        "download_transcript": None,
        "build_expert_identity_suffix": "",
        "drain_pending_safe": [],
        "_resolve_baseline_model": ResolvedModel(
            model="anthropic/claude-sonnet-4-6", source="env"
        ),
        "get_or_create_sandbox": None,
        "_build_system_prompt": ("system prompt", None),
        "build_builder_system_prompt_suffix": "",
        "build_budget_ctx": "",
        "build_session_context": "",
        "build_skills_context": "",
        "build_skills_update_notice": "",
        "build_turn_budget_block": "",
        "persist_and_record_usage": None,
        "enqueue_conversation_turn": None,
    }
    with ExitStack() as stack:
        stack.enter_context(
            patch(
                "backend.copilot.gate.held.answered",
                new=AsyncMock(return_value=[]),
            )
        )
        for name, value in values.items():
            stack.enter_context(
                patch(
                    f"backend.copilot.baseline.service.{name}",
                    new=AsyncMock(return_value=value),
                )
            )
        for name in ("upsert_chat_session", "persist_session_safe"):
            stack.enter_context(
                patch(
                    f"backend.copilot.baseline.service.{name}",
                    new=AsyncMock(side_effect=lambda s, *_a, **_k: s),
                )
            )
        stack.enter_context(
            patch("backend.copilot.baseline.service._baseline_llm_caller", new=llm)
        )
        stack.enter_context(
            patch("backend.copilot.baseline.service.upload_transcript", new=upload)
        )
        stack.enter_context(
            patch(
                "backend.copilot.graphiti.context_refresh.refresh_warm_context",
                new=AsyncMock(
                    return_value="<temporal_context>fresh server memory</temporal_context>"
                ),
            )
        )
        async for _ in stream_chat_completion_baseline(
            session.session_id, message, user_id="user-1", session=session
        ):
            pass

    current = sent[0][-1]["content"]
    assert _INJECTED_MEMORY_MARKER in current
    assert current.count(_INJECTED_MEMORY_MARKER) == 1
    assert pasted in current
    assert _strip_injected_memory_text(current) == message
    assert session.messages[2].content == message
    assert upload.await_args is not None
    assert b"fresh server memory" not in upload.await_args.kwargs["content"]
    assert b"my pasted notes" in upload.await_args.kwargs["content"]
