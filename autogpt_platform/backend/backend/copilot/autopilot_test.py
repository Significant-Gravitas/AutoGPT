import json
from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import AsyncMock

import prisma.enums
import pytest

from backend.copilot.autopilot import (
    AUTOPILOT_CALLBACK_EMAIL_TEMPLATE,
    AUTOPILOT_DISABLED_TOOLS,
    AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE,
    AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE,
    _create_autopilot_session,
    _crosses_local_midnight,
    _get_completion_email_template_name,
    _get_recent_manual_session_context,
    _resolve_timezone_name,
    consume_callback_token,
    dispatch_nightly_copilot,
    handle_non_manual_session_completion,
    send_nightly_copilot_emails,
    strip_internal_content,
    wrap_internal_message,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.session_types import ChatSessionStartType, StoredCompletionReport


def _build_autopilot_session() -> ChatSession:
    return ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
    )


def test_wrap_and_strip_internal_content() -> None:
    wrapped = wrap_internal_message("secret instruction")

    assert wrapped == "<internal>secret instruction</internal>"
    assert strip_internal_content(wrapped) is None
    assert (
        strip_internal_content("Visible<internal>secret</internal> text")
        == "Visible text"
    )


def test_resolve_timezone_name_falls_back_to_utc() -> None:
    assert _resolve_timezone_name(None) == "UTC"
    assert _resolve_timezone_name("not-set") == "UTC"
    assert _resolve_timezone_name("Definitely/Invalid") == "UTC"
    assert _resolve_timezone_name("Europe/Madrid") == "Europe/Madrid"


def test_crosses_local_midnight_supports_offset_timezones() -> None:
    assert _crosses_local_midnight(
        datetime(2026, 3, 13, 18, 0, tzinfo=UTC),
        datetime(2026, 3, 13, 18, 30, tzinfo=UTC),
        "Asia/Kathmandu",
    ) == date(2026, 3, 14)
    assert (
        _crosses_local_midnight(
            datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
            datetime(2026, 3, 13, 0, 30, tzinfo=UTC),
            "UTC",
        )
        is None
    )


@pytest.mark.asyncio
async def test_get_recent_manual_session_context_strips_internal_content(
    mocker,
) -> None:
    session = SimpleNamespace(
        id="sess-1",
        title="Manual work",
        updatedAt=datetime(2026, 3, 14, 9, 0, tzinfo=UTC),
    )
    session_prisma = SimpleNamespace(find_many=AsyncMock(return_value=[session]))
    message_prisma = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                SimpleNamespace(
                    role="user",
                    content="<internal>hidden</internal>",
                ),
                SimpleNamespace(
                    role="user",
                    content="Visible<internal>hidden</internal> text",
                ),
                SimpleNamespace(
                    role="assistant",
                    content="Completed a useful task for the user.",
                ),
            ]
        )
    )
    mocker.patch("prisma.models.ChatSession.prisma", return_value=session_prisma)
    mocker.patch("prisma.models.ChatMessage.prisma", return_value=message_prisma)

    context = await _get_recent_manual_session_context(
        "user-1",
        since_utc=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
    )

    assert "Manual work" in context
    assert "Visible text" in context
    assert "Completed a useful task for the user." in context
    assert "hidden" not in context


@pytest.mark.asyncio
async def test_handle_non_manual_session_completion_saves_report(mocker) -> None:
    session = _build_autopilot_session()
    session.messages = [
        ChatMessage(
            role="assistant",
            content="Finished the work.",
            tool_calls=[
                {
                    "id": "tool-call-1",
                    "type": "function",
                    "function": {
                        "name": "completion_report",
                        "arguments": json.dumps(
                            {
                                "thoughts": "I reviewed the recent context and prepared a useful next step.",
                                "should_notify_user": True,
                                "email_title": "Your nightly update",
                                "email_body": "I found something useful for you.",
                                "callback_session_message": "Open this chat and I will walk you through it.",
                                "approval_summary": None,
                            }
                        ),
                    },
                }
            ],
        ),
        ChatMessage(
            role="tool",
            tool_call_id="tool-call-1",
            content=json.dumps(
                {
                    "type": "completion_report_saved",
                    "message": "Completion report recorded successfully.",
                }
            ),
        ),
    ]

    mocker.patch(
        "backend.copilot.autopilot.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.autopilot._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot.upsert_chat_session",
        new_callable=AsyncMock,
    )

    await handle_non_manual_session_completion(session.session_id)

    assert session.completion_report is not None
    assert session.completion_report.email_title == "Your nightly update"
    assert session.completed_at is not None
    assert session.completion_report_repair_queued_at is None
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_handle_non_manual_session_completion_queues_repair(mocker) -> None:
    session = _build_autopilot_session()
    session.messages = [ChatMessage(role="assistant", content="Done.")]

    mocker.patch(
        "backend.copilot.autopilot.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.autopilot._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(1, "copilot-session-session-1"),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot.upsert_chat_session",
        new_callable=AsyncMock,
    )
    enqueue = mocker.patch(
        "backend.copilot.autopilot._enqueue_session_turn",
        new_callable=AsyncMock,
    )

    await handle_non_manual_session_completion(session.session_id)

    assert session.completion_report is None
    assert session.completion_report_repair_count == 1
    assert session.completion_report_repair_queued_at is not None
    assert session.completed_at is None
    assert session.messages[-1].role == "user"
    assert "completion_report" in (session.messages[-1].content or "")
    assert "approval" in (session.messages[-1].content or "").lower()
    upsert.assert_awaited_once_with(session)
    enqueue.assert_awaited_once()


@pytest.mark.parametrize(
    ("start_type", "expected_template"),
    [
        (
            ChatSessionStartType.AUTOPILOT_NIGHTLY,
            AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE,
        ),
        (
            ChatSessionStartType.AUTOPILOT_CALLBACK,
            AUTOPILOT_CALLBACK_EMAIL_TEMPLATE,
        ),
        (
            ChatSessionStartType.AUTOPILOT_INVITE_CTA,
            AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE,
        ),
    ],
)
def test_get_completion_email_template_name(
    start_type: ChatSessionStartType,
    expected_template: str,
) -> None:
    assert _get_completion_email_template_name(start_type) == expected_template


@pytest.mark.asyncio
async def test_create_autopilot_session_disables_configured_tools(mocker) -> None:
    created_session = ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
    )

    mocker.patch(
        "backend.copilot.autopilot._session_exists_for_execution_tag",
        new_callable=AsyncMock,
        return_value=False,
    )
    mocker.patch(
        "backend.copilot.autopilot._build_autopilot_system_prompt",
        new_callable=AsyncMock,
        return_value="system prompt",
    )
    create_chat_session = mocker.patch(
        "backend.copilot.autopilot.create_chat_session",
        new_callable=AsyncMock,
        return_value=created_session,
    )
    enqueue = mocker.patch(
        "backend.copilot.autopilot._enqueue_session_turn",
        new_callable=AsyncMock,
    )

    user = SimpleNamespace(id="user-1", name="User Name")

    session = await _create_autopilot_session(
        cast(Any, user),
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
        execution_tag="autopilot-nightly:2026-03-13",
        timezone_name="UTC",
        target_local_date=None,
    )

    assert session is created_session
    session_config = create_chat_session.await_args.kwargs["session_config"]
    assert session_config.extra_tools == ["completion_report"]
    assert session_config.disabled_tools == AUTOPILOT_DISABLED_TOOLS
    enqueue.assert_awaited_once()


@pytest.mark.asyncio
async def test_handle_non_manual_session_completion_stops_after_max_repairs(
    mocker,
) -> None:
    session = _build_autopilot_session()
    session.completion_report_repair_count = 2
    session.messages = [ChatMessage(role="assistant", content="Done.")]

    mocker.patch(
        "backend.copilot.autopilot.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.autopilot._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot.upsert_chat_session",
        new_callable=AsyncMock,
    )
    enqueue = mocker.patch(
        "backend.copilot.autopilot._enqueue_session_turn",
        new_callable=AsyncMock,
    )

    await handle_non_manual_session_completion(session.session_id)

    assert session.completed_at is not None
    assert session.completion_report_repair_queued_at is None
    upsert.assert_awaited_once_with(session)
    enqueue.assert_not_called()


@pytest.mark.asyncio
async def test_dispatch_nightly_copilot_respects_cohort_priority(mocker) -> None:
    fixed_now = datetime(2026, 3, 17, 0, 5, tzinfo=UTC)
    datetime_mock = mocker.patch(
        "backend.copilot.autopilot.datetime",
        wraps=datetime,
    )
    datetime_mock.now.return_value = fixed_now

    invite_user = SimpleNamespace(id="invite-user", timezone="UTC", name="Invite User")
    nightly_user = SimpleNamespace(
        id="nightly-user", timezone="UTC", name="Nightly User"
    )
    callback_user = SimpleNamespace(
        id="callback-user", timezone="UTC", name="Callback User"
    )
    disabled_user = SimpleNamespace(
        id="disabled-user", timezone="UTC", name="Disabled User"
    )

    invited = SimpleNamespace(
        authUserId="invite-user",
        status=prisma.enums.InvitedUserStatus.INVITED,
        createdAt=fixed_now - timedelta(hours=72),
    )
    user_prisma = SimpleNamespace(
        find_many=AsyncMock(
            return_value=[
                invite_user,
                nightly_user,
                callback_user,
                disabled_user,
            ]
        )
    )
    invite_prisma = SimpleNamespace(find_many=AsyncMock(return_value=[invited]))
    mocker.patch("prisma.models.User.prisma", return_value=user_prisma)
    mocker.patch("prisma.models.InvitedUser.prisma", return_value=invite_prisma)
    mocker.patch(
        "backend.copilot.autopilot.is_feature_enabled",
        new_callable=AsyncMock,
        side_effect=lambda _flag, user_id, default=False: user_id != "disabled-user",
    )
    mocker.patch(
        "backend.copilot.autopilot._crosses_local_midnight",
        side_effect=lambda *_args, **_kwargs: date(2026, 3, 17),
    )
    mocker.patch(
        "backend.copilot.autopilot._user_has_recent_manual_message",
        new_callable=AsyncMock,
        side_effect=lambda user_id, _since: user_id == "nightly-user",
    )
    mocker.patch(
        "backend.copilot.autopilot._user_has_session_since",
        new_callable=AsyncMock,
        side_effect=lambda user_id, _since: (
            user_id in {"nightly-user", "callback-user"}
        ),
    )
    mocker.patch(
        "backend.copilot.autopilot._session_exists_for_execution_tag",
        new_callable=AsyncMock,
        return_value=False,
    )
    create_autopilot_session = mocker.patch(
        "backend.copilot.autopilot._create_autopilot_session",
        new_callable=AsyncMock,
        side_effect=[
            object(),
            object(),
            object(),
        ],
    )

    created_count = await dispatch_nightly_copilot()

    assert created_count == 3
    create_calls = create_autopilot_session.await_args_list
    assert (
        create_calls[0].kwargs["start_type"]
        == ChatSessionStartType.AUTOPILOT_INVITE_CTA
    )
    assert (
        create_calls[1].kwargs["start_type"] == ChatSessionStartType.AUTOPILOT_NIGHTLY
    )
    assert (
        create_calls[2].kwargs["start_type"] == ChatSessionStartType.AUTOPILOT_CALLBACK
    )


@pytest.mark.asyncio
async def test_send_nightly_copilot_emails_queues_repair_for_missing_report(
    mocker,
) -> None:
    session = _build_autopilot_session()
    candidate = SimpleNamespace(id=session.session_id)

    chat_session_prisma = SimpleNamespace(find_many=AsyncMock(return_value=[candidate]))
    mocker.patch("prisma.models.ChatSession.prisma", return_value=chat_session_prisma)
    mocker.patch(
        "backend.copilot.autopilot.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    queue_repair = mocker.patch(
        "backend.copilot.autopilot._queue_completion_report_repair",
        new_callable=AsyncMock,
    )

    processed = await send_nightly_copilot_emails()

    assert processed == 0
    queue_repair.assert_awaited_once_with(session, pending_approval_count=0)


@pytest.mark.asyncio
async def test_send_nightly_copilot_emails_sends_and_marks_sent(mocker) -> None:
    session = _build_autopilot_session()
    session.completion_report = StoredCompletionReport(
        thoughts="Did useful work.",
        should_notify_user=True,
        email_title="Autopilot update",
        email_body="Useful update",
        callback_session_message="Open this session",
        approval_summary=None,
        has_pending_approvals=False,
        pending_approval_count=0,
        pending_approval_graph_exec_id=None,
        saved_at=datetime.now(UTC),
    )
    candidate = SimpleNamespace(id=session.session_id)

    chat_session_prisma = SimpleNamespace(find_many=AsyncMock(return_value=[candidate]))
    mocker.patch("prisma.models.ChatSession.prisma", return_value=chat_session_prisma)
    mocker.patch(
        "backend.copilot.autopilot.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(1, "copilot-session-session-1"),
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot._send_completion_email",
        new_callable=AsyncMock,
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot.upsert_chat_session",
        new_callable=AsyncMock,
    )

    processed = await send_nightly_copilot_emails()

    assert processed == 1
    assert session.notification_email_sent_at is not None
    assert session.completion_report.pending_approval_graph_exec_id == (
        "copilot-session-session-1"
    )
    send_email.assert_awaited_once_with(session)
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_send_nightly_copilot_emails_skips_when_should_not_notify(mocker) -> None:
    session = _build_autopilot_session()
    session.completion_report = StoredCompletionReport(
        thoughts="Did useful work.",
        should_notify_user=False,
        email_title=None,
        email_body=None,
        callback_session_message=None,
        approval_summary=None,
        has_pending_approvals=False,
        pending_approval_count=0,
        pending_approval_graph_exec_id=None,
        saved_at=datetime.now(UTC),
    )
    candidate = SimpleNamespace(id=session.session_id)

    chat_session_prisma = SimpleNamespace(find_many=AsyncMock(return_value=[candidate]))
    mocker.patch("prisma.models.ChatSession.prisma", return_value=chat_session_prisma)
    mocker.patch(
        "backend.copilot.autopilot.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot.upsert_chat_session",
        new_callable=AsyncMock,
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot._send_completion_email",
        new_callable=AsyncMock,
    )

    processed = await send_nightly_copilot_emails()

    assert processed == 1
    assert session.notification_email_skipped_at is not None
    send_email.assert_not_called()
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_consume_callback_token_reuses_existing_session(mocker) -> None:
    token = SimpleNamespace(
        id="token-1",
        userId="user-1",
        expiresAt=datetime.now(UTC) + timedelta(hours=1),
        consumedSessionId="sess-existing",
    )
    token_prisma = SimpleNamespace(
        find_unique=AsyncMock(return_value=token),
        update=AsyncMock(),
    )
    mocker.patch(
        "prisma.models.ChatSessionCallbackToken.prisma",
        return_value=token_prisma,
    )
    create_chat_session = mocker.patch(
        "backend.copilot.autopilot.create_chat_session",
        new_callable=AsyncMock,
    )

    result = await consume_callback_token("token-1", "user-1")

    assert result.session_id == "sess-existing"
    create_chat_session.assert_not_called()
    token_prisma.update.assert_not_called()


@pytest.mark.asyncio
async def test_consume_callback_token_creates_manual_session(mocker) -> None:
    token = SimpleNamespace(
        id="token-1",
        userId="user-1",
        expiresAt=datetime.now(UTC) + timedelta(hours=1),
        consumedSessionId=None,
        callbackSessionMessage="Open this chat",
    )
    token_prisma = SimpleNamespace(
        find_unique=AsyncMock(return_value=token),
        update=AsyncMock(),
    )
    created_session = ChatSession.new("user-1")
    mocker.patch(
        "prisma.models.ChatSessionCallbackToken.prisma",
        return_value=token_prisma,
    )
    create_chat_session = mocker.patch(
        "backend.copilot.autopilot.create_chat_session",
        new_callable=AsyncMock,
        return_value=created_session,
    )

    result = await consume_callback_token("token-1", "user-1")

    assert result.session_id == created_session.session_id
    create_chat_session.assert_awaited_once()
    create_kwargs = create_chat_session.await_args.kwargs
    assert create_kwargs["initial_messages"][0].role == "assistant"
    assert create_kwargs["initial_messages"][0].content == "Open this chat"
    token_prisma.update.assert_awaited_once()


@pytest.mark.asyncio
async def test_consume_callback_token_rejects_expired_token(mocker) -> None:
    token = SimpleNamespace(
        id="token-1",
        userId="user-1",
        expiresAt=datetime.now(UTC) - timedelta(minutes=1),
        consumedSessionId=None,
        callbackSessionMessage="Open this chat",
    )
    token_prisma = SimpleNamespace(
        find_unique=AsyncMock(return_value=token),
    )
    mocker.patch(
        "prisma.models.ChatSessionCallbackToken.prisma",
        return_value=token_prisma,
    )

    with pytest.raises(ValueError, match="expired"):
        await consume_callback_token("token-1", "user-1")
