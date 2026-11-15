"""SDK turns that carry a /command: auto-continue, retries and the first turn."""

import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.pending_messages import PendingMessage
from backend.copilot.response_model import StreamError
from backend.copilot.sdk.service import stream_chat_completion_sdk
from backend.copilot.skill_invocation import InvocableSkill
from backend.copilot.tools.skills import ParsedSkill
from backend.copilot.transcript_builder import TranscriptBuilder

from .conftest import build_test_transcript
from .retry_scenarios_test import TestFirstTurnWarmContextIsEphemeral as _FirstTurn
from .retry_scenarios_test import TestFollowUpWarmContextCallSite as _FollowUp
from .retry_scenarios_test import TestStreamChatCompletionRetryIntegration as _Retry
from .retry_scenarios_test import _make_sdk_patches

_SVC = "backend.copilot.sdk.service"
_REPORT = InvocableSkill(
    skill=ParsedSkill(name="report", description="Write a report", body="Write $0"),
    is_default=True,
)


@pytest.mark.asyncio
async def test_auto_continue_preserves_each_pending_user_row():
    helper = _Retry()
    session = helper._make_session()
    session.title = "Report"
    pending = [
        PendingMessage(content="/report weekly"),
        PendingMessage(content="Include totals"),
    ]
    combined = "\n\n".join(pm.content for pm in pending)
    client_cm = helper._make_client_mock(result_message=helper._make_result_message())
    patches = _make_sdk_patches(
        session,
        original_transcript=build_test_transcript(
            [("user", "hello"), ("assistant", "ready")]
        ),
        compacted_transcript=None,
        client_side_effect=lambda *a, **kw: client_cm,
    )
    db = MagicMock()
    db.update_message_content_by_sequence = AsyncMock(return_value=True)
    patches.extend(
        [
            (
                "backend.copilot.pending_message_helpers.persist_session_safe",
                dict(new=_persist_with_sequences),
            ),
            ("backend.copilot.service.chat_db", dict(return_value=db)),
            *_skill_patches(AsyncMock(return_value=_REPORT)),
        ]
    )
    events = []
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async for event in stream_chat_completion_sdk(
            session_id=session.session_id,
            message=combined,
            is_user_message=True,
            user_id=session.user_id,
            session=session,
            continued_pending=pending,
        ):
            events.append(event)
    assert not [event for event in events if isinstance(event, StreamError)]
    assert client_cm.__aenter__.return_value.query.await_count == 1
    assert [row.content for row in session.messages if row.role == "user"] == [
        "hello",
        "/report weekly",
        "Include totals",
    ]
    db.update_message_content_by_sequence.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("first_turn", [True, False])
@pytest.mark.parametrize("incoming", ["/report weekly", ""])
async def test_retry_keeps_persisted_skill_in_sdk_query_and_upload(
    first_turn, incoming
):
    expanded = "<skill_invocation>\nWrite the saved weekly report.\n</skill_invocation>\n\n/report weekly"
    helper = _FirstTurn()
    if first_turn:
        expanded = "<available_skills>report</available_skills>\n\n" + expanded
    session = (
        helper._first_turn(expanded)
        if first_turn
        else helper._second_turn("hello", expanded)
    )
    original_rows = len(session.messages)
    builders = []

    def build_transcript():
        builder = TranscriptBuilder()
        builders.append(builder)
        return builder

    queries = []
    upload = AsyncMock()
    resolve = AsyncMock(return_value=None)
    await helper._run(
        session,
        _FollowUp._clients(queries),
        message=incoming,
        first_block=None,
        refresh=helper._no_refresh,
        extra=[
            (f"{_SVC}.TranscriptBuilder", dict(side_effect=build_transcript)),
            (f"{_SVC}.persist_session_safe", dict(new=_persist_with_sequences)),
            (f"{_SVC}.download_transcript", dict(new=AsyncMock(return_value=None))),
            (f"{_SVC}.read_cli_session_from_disk", dict(new=helper._cli_file(queries))),
            (f"{_SVC}.upload_transcript", dict(new=upload)),
            *_skill_patches(resolve),
        ],
    )
    assert builders[0].to_jsonl().count("<skill_invocation>") == 1
    assert len(queries) == 1
    assert queries[0].count("<skill_invocation>") == 1
    assert upload.await_args.kwargs["content"].count(b"<skill_invocation>") == 1
    assert session.messages[original_rows - 1].content == expanded
    assert sum(m.role == "user" for m in session.messages) == (1 if first_turn else 2)
    resolve.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_first_turn_command_reaches_the_sdk_query():
    helper = _FirstTurn()
    queries = []
    await helper._run(
        helper._first_turn("/report weekly"),
        _FollowUp._clients(queries),
        first_block=None,
        refresh=helper._no_refresh,
        extra=_skill_patches(AsyncMock(return_value=_REPORT)),
    )
    assert len(queries) == 1
    assert queries[0].count("<skill_invocation>") == 1
    assert queries[0].count("</skill_invocation>") == 1
    block = (
        queries[0].split("<skill_invocation>", 1)[1].split("</skill_invocation>", 1)[0]
    )
    assert "Write weekly" in block


def _skill_patches(resolve: AsyncMock) -> list[tuple[str, dict]]:
    return [
        (
            "backend.copilot.skill_invocation.is_skills_feature_enabled",
            dict(new=AsyncMock(return_value=True)),
        ),
        ("backend.copilot.skill_invocation.resolve_invocable_skill", dict(new=resolve)),
    ]


async def _persist_with_sequences(session, *_args, **_kwargs):
    for sequence, row in enumerate(session.messages):
        row.sequence = sequence
    return session
