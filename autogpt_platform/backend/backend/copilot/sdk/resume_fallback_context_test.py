"""A later turn that cannot ``--resume`` its CLI session keeps its first-turn context.

The skills index, memory, expert/team workflows and user context are injected
into the first user message only; later turns rely on ``--resume`` to carry
them. When the CLI session cannot be restored the turn falls back to a
``<conversation_history>`` block, and before SECRT-2801 that block was all the
model got: the first-turn blocks were either missing or buried inside the
history, where the system prompt tells the model to ignore them.

These tests drive the real ``stream_chat_completion_sdk`` generator with a
mocked ``ClaudeSDKClient`` and assert on the query the CLI would receive.
"""

from __future__ import annotations

import contextlib
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from claude_agent_sdk import ResultMessage

from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.sdk.service import (
    _format_conversation_context,
    stream_chat_completion_sdk,
)

from .conftest import build_test_transcript
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_SHARED = "backend.copilot.service"

_SESSION_ID = "test-session-id"
_FIRST_TURN_ROW = (
    "<available_skills>\n- name: retired-skill — gone since turn 1\n"
    "</available_skills>\n\n"
    "<memory_context>\nstale fact from turn 1\n</memory_context>\n\n"
    "Hi, I'm Sam. I run a bakery."
)
_CURRENT_MESSAGE = "Write this week's report."
_SKILLS_INDEX = (
    "- name: weekly-report — Drafts the weekly report — triggers: weekly report"
)
_MEMORY = "Sam's bakery is closed on Mondays."
_TEAM_BLOCK = (
    "<team_context>\nMax owns the weekly-report workflow.\n</team_context>\n\n"
)
_UNDERSTANDING = "Name: Sam\nBusiness: bakery"


def _session() -> ChatSession:
    now = datetime.now(UTC)
    return ChatSession(
        session_id=_SESSION_ID,
        user_id="test-user",
        title="Bakery",
        usage=[],
        started_at=now,
        updated_at=now,
        messages=[
            ChatMessage(role="user", content=_FIRST_TURN_ROW, sequence=0),
            ChatMessage(role="assistant", content="Nice to meet you, Sam.", sequence=1),
            ChatMessage(role="user", content=_CURRENT_MESSAGE, sequence=2),
        ],
    )


def _client(queries: list[str], *, prompt_too_long: bool = False):
    async def _receive():
        yield ResultMessage(
            subtype="success",
            result="done",
            duration_ms=10,
            duration_api_ms=5,
            is_error=False,
            num_turns=1,
            session_id=_SESSION_ID,
        )

    async def _query(prompt, session_id=None):
        queries.append(prompt)
        if prompt_too_long:
            raise Exception("prompt is too long (context_length_exceeded)")

    client = MagicMock()
    client.receive_response = _receive
    client.query = AsyncMock(side_effect=_query)
    client._transport = MagicMock()
    client._transport.write = AsyncMock()
    cm = AsyncMock()
    cm.__aenter__.return_value = client
    cm.__aexit__.return_value = None
    return cm


async def _passthrough_compress(messages, target_tokens=None):
    return messages, False, None


def _context_patches(*, resume_hit: bool) -> list[tuple[str, dict]]:
    understanding_db = MagicMock()
    understanding_db.get_business_understanding = AsyncMock(return_value=MagicMock())
    tier = MagicMock()
    tier.value = "PRO"
    transcript = build_test_transcript(
        [("user", _FIRST_TURN_ROW), ("assistant", "Nice to meet you, Sam.")]
    )
    patches = [
        (f"{_SVC}.build_skills_context", dict(return_value=_SKILLS_INDEX)),
        (
            f"{_SVC}.build_session_context",
            dict(return_value=f"session_id: {_SESSION_ID}"),
        ),
        (f"{_SVC}.is_enabled_for_user", dict(return_value=True)),
        (f"{_SVC}.fetch_warm_context", dict(return_value=_MEMORY)),
        # The ingest runs as a background task that outlives these patches.
        (f"{_SVC}._graphiti_ingest_allowed", dict(return_value=False)),
        (f"{_SVC}.build_skills_update_notice", dict(return_value="")),
        (f"{_SVC}._compress_messages", dict(side_effect=_passthrough_compress)),
        (f"{_SHARED}.understanding_db", dict(return_value=understanding_db)),
        (
            f"{_SHARED}.format_understanding_for_prompt",
            dict(return_value=_UNDERSTANDING),
        ),
        (f"{_SHARED}._fetch_langfuse_prompt", dict(return_value=None)),
        (f"{_SHARED}.build_expert_context", dict(return_value=_TEAM_BLOCK)),
        ("backend.copilot.rate_limit.get_user_tier", dict(return_value=tier)),
    ]
    if resume_hit:
        patches.append(
            (f"{_SVC}.process_cli_restore", dict(return_value=(transcript, True)))
        )
    else:
        patches.append((f"{_SVC}.download_transcript", dict(return_value=None)))
    return patches


async def _run_turn(session: ChatSession, client_factory, *, resume_hit: bool) -> None:
    transcript = build_test_transcript(
        [("user", _FIRST_TURN_ROW), ("assistant", "Nice to meet you, Sam.")]
    )
    # The real system-prompt builder runs, so a fallback turn can fetch the
    # user's business understanding through it like the first turn does.
    base = [
        (target, kwargs)
        for target, kwargs in _make_sdk_patches(
            session,
            original_transcript=transcript,
            compacted_transcript=None,
            client_side_effect=client_factory,
        )
        if target != f"{_SVC}._build_system_prompt"
    ]
    with contextlib.ExitStack() as stack:
        for target, kwargs in base + _context_patches(resume_hit=resume_hit):
            stack.enter_context(patch(target, **kwargs))
        async for _ in stream_chat_completion_sdk(
            session_id=_SESSION_ID,
            message=_CURRENT_MESSAGE,
            is_user_message=True,
            user_id="test-user",
            session=session,
        ):
            pass


def _assert_first_turn_context_leads(query: str) -> None:
    assert query.startswith(
        "<available_skills>\n"
    ), f"skills index must lead the fallback query, got: {query[:200]!r}"
    history_at = query.index("<conversation_history>")
    for block in (
        _SKILLS_INDEX,
        f"<memory_context>\n{_MEMORY}\n</memory_context>",
        "<team_context>\nMax owns the weekly-report workflow.",
        f"<user_context>\n{_UNDERSTANDING}\nPlan: PRO\n</user_context>",
    ):
        assert block in query, f"missing first-turn block: {block!r}"
        assert query.index(block) < history_at, f"{block!r} is not in the prefix"
    history = query[history_at:]
    assert "Hi, I'm Sam. I run a bakery." in history
    assert "Nice to meet you, Sam." in history
    assert query.rstrip().endswith(_CURRENT_MESSAGE)


class TestResumeMissKeepsFirstTurnContext:
    @pytest.mark.asyncio
    async def test_fallback_turn_leads_with_first_turn_context(self):
        session = _session()
        queries: list[str] = []

        await _run_turn(session, lambda *a, **kw: _client(queries), resume_hit=False)

        assert len(queries) == 1
        query = queries[0]
        _assert_first_turn_context_leads(query)
        # The stale turn-1 blocks persisted on the first row are not replayed
        # inside the history next to the fresh ones.
        assert "retired-skill" not in query
        assert "stale fact from turn 1" not in query
        # The rebuilt context rides the query only; the stored row keeps the
        # user's own words.
        assert session.messages[-1].content == _CURRENT_MESSAGE

    @pytest.mark.asyncio
    async def test_prompt_too_long_retry_keeps_first_turn_context(self):
        session = _session()
        queries: list[str] = []
        attempts = [0]

        def _factory(*a, **kw):
            attempts[0] += 1
            return _client(queries, prompt_too_long=attempts[0] == 1)

        await _run_turn(session, _factory, resume_hit=False)

        assert len(queries) == 2
        for query in queries:
            _assert_first_turn_context_leads(query)

    @pytest.mark.asyncio
    async def test_resumed_turn_does_not_reinject(self):
        session = _session()
        queries: list[str] = []

        await _run_turn(session, lambda *a, **kw: _client(queries), resume_hit=True)

        assert len(queries) == 1
        assert "<available_skills>" not in queries[0]
        assert "<memory_context>" not in queries[0]
        assert "<conversation_history>" not in queries[0]


class TestFormatConversationContext:
    def test_server_blocks_on_history_rows_are_dropped(self):
        context = _format_conversation_context(
            [
                ChatMessage(role="user", content=_FIRST_TURN_ROW),
                ChatMessage(role="assistant", content="Nice to meet you, Sam."),
            ]
        )

        assert context is not None
        assert "User: Hi, I'm Sam. I run a bakery." in context
        assert "<available_skills>" not in context
        assert "<memory_context>" not in context

    def test_user_typed_text_is_kept(self):
        context = _format_conversation_context(
            [
                ChatMessage(role="user", content="Explain <memory_context> tags"),
                ChatMessage(role="assistant", content="Sure."),
            ]
        )

        assert context is not None
        assert "User: Explain <memory_context> tags" in context
