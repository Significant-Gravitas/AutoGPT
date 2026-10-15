"""Baseline-engine turns: ``stream_chat_completion_baseline`` over the OpenAI SDK.

A real ``AsyncOpenAI`` client streams from the fake provider, and the
baseline's own usage and cost parsing reads what it sends. Stubbed:
persistence, transcript storage, the system prompt, flag lookups and the
copilot's tools, which answer the provider's tool call with a canned result.
"""

import uuid
from typing import Any
from unittest.mock import AsyncMock

import pytest
from openai import AsyncOpenAI

from backend.copilot import pending_message_helpers
from backend.copilot.baseline import service as baseline
from backend.copilot.config import ChatConfig
from backend.copilot.model import ChatSession
from backend.copilot.model_router import ResolvedModel
from backend.copilot.response_model import StreamToolOutputAvailable
from backend.copilot.tree import root_envelope

from .wire import TOOL_NAME


class BaselineTurns:
    def __init__(
        self,
        monkeypatch: pytest.MonkeyPatch,
        config: ChatConfig,
        *,
        base_url: str,
        model: str,
    ) -> None:
        client = AsyncOpenAI(api_key="sk-billing-e2e", base_url=base_url)
        save = AsyncMock(side_effect=lambda session, *_, **__: session)
        stubs: dict[str, Any] = {
            "config": config,
            "_get_main_client": lambda: client,
            "resolve_model_route": AsyncMock(
                return_value=ResolvedModel(model=model, source="env")
            ),
            "execute_tool": AsyncMock(side_effect=_answer_tool),
            "upsert_chat_session": save,
            "download_transcript": AsyncMock(return_value=None),
            "upload_transcript": AsyncMock(),
            "drain_pending_safe": AsyncMock(return_value=[]),
            "drain_pending_messages": AsyncMock(return_value=[]),
            "_build_system_prompt": AsyncMock(
                return_value=("You are a test assistant. Reply briefly.", None)
            ),
            "build_budget_ctx": AsyncMock(return_value=None),
            "build_session_context": AsyncMock(return_value=""),
            "build_skills_context": AsyncMock(return_value=""),
            "build_skills_update_notice": AsyncMock(return_value=None),
            "inject_user_context": AsyncMock(return_value=None),
            "clear_pending_question": AsyncMock(),
            "resolve_answered": AsyncMock(return_value=[]),
            "build_turn_budget_block": AsyncMock(return_value=""),
            "is_enabled_for_user": AsyncMock(return_value=False),
            "is_feature_enabled": AsyncMock(return_value=False),
        }
        for name, value in stubs.items():
            monkeypatch.setattr(baseline, name, value)
        monkeypatch.setattr(pending_message_helpers, "upsert_chat_session", save)

    async def __call__(
        self, session: ChatSession, prompt: str, *, new_pod: bool
    ) -> None:
        # The baseline keeps nothing on disk between turns, so a new pod is
        # the same as the old one.
        async for _ in baseline.stream_chat_completion_baseline(
            session.session_id,
            message=prompt,
            is_user_message=True,
            user_id=session.user_id,
            session=session,
            envelope=root_envelope(str(uuid.uuid4()), session_id=session.session_id),
        ):
            pass


async def _answer_tool(*, tool_call_id: str, **_: Any) -> StreamToolOutputAvailable:
    return StreamToolOutputAvailable(
        toolCallId=tool_call_id, toolName=TOOL_NAME, output="The fact is 42."
    )
