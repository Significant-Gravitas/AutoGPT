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
    consume_callback_token,
    dispatch_nightly_copilot,
    handle_non_manual_session_completion,
    send_nightly_copilot_emails,
    send_pending_copilot_emails_for_user,
    strip_internal_content,
    trigger_autopilot_session_for_user,
    wrap_internal_message,
)
from backend.copilot.autopilot_dispatch import (
    _create_autopilot_session,
    _crosses_local_midnight,
    _resolve_timezone_name,
)
from backend.copilot.autopilot_email import (
    _get_completion_email_template_name,
    _send_completion_email,
)
from backend.copilot.autopilot_prompts import (
    _build_autopilot_system_prompt,
    _get_recent_manual_session_context,
    _get_recent_sent_email_context,
    _get_recent_session_summary_context,
)
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.session_types import ChatSessionStartType, StoredCompletionReport


def _build_autopilot_session() -> ChatSession:
    return ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
    )


def _build_completion_report(
    *,
    thoughts: str = "Did useful work.",
    email_title: str = "Autopilot update",
    email_body: str = "I found something useful for you.",
    callback_session_message: str = "Open this chat",
) -> StoredCompletionReport:
    return StoredCompletionReport(
        thoughts=thoughts,
        should_notify_user=True,
        email_title=email_title,
        email_body=email_body,
        callback_session_message=callback_session_message,
        approval_summary=None,
        has_pending_approvals=False,
        pending_approval_count=0,
        pending_approval_graph_exec_id=None,
        saved_at=datetime.now(UTC),
    )


def test_wrap_and_strip_internal_content() -> None:
    wrapped = wrap_internal_message("secret instruction")

    assert wrapped == "<internal>secret instruction</internal>"
    assert strip_internal_content(wrapped) is None
    assert (
        strip_internal_content("Visible<internal>secret</internal> text")
        == "Visible text"
    )


def test_autopilot_facade_exports_only_public_names() -> None:
    from backend.copilot import autopilot

    assert all(not name.startswith("_") for name in autopilot.__all__)


@pytest.mark.asyncio
async def test_send_completion_email_links_back_to_source_session(mocker) -> None:
    session = _build_autopilot_session()
    session.completion_report = _build_completion_report()

    sender = SimpleNamespace(send_template=mocker.Mock())
    mocker.patch("backend.copilot.autopilot_email.EmailSender", return_value=sender)
    mocker.patch(
        "backend.copilot.autopilot_email.user_db",
        return_value=SimpleNamespace(
            get_user_by_id=AsyncMock(
                return_value=SimpleNamespace(email="user@example.com")
            )
        ),
    )
    mocker.patch(
        "backend.copilot.autopilot_email.get_frontend_base_url",
        return_value="https://example.com",
    )

    await _send_completion_email(session)

    assert sender.send_template.call_args.kwargs["data"]["cta_url"] == (
        f"https://example.com/copilot?sessionId={session.session_id}&showAutopilot=1"
    )
    assert sender.send_template.call_args.kwargs["data"]["cta_label"] == "Open Copilot"


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


def test_crosses_local_midnight_only_triggers_once_across_dst_shift() -> None:
    assert _crosses_local_midnight(
        datetime(2026, 3, 8, 4, 30, tzinfo=UTC),
        datetime(2026, 3, 8, 5, 0, tzinfo=UTC),
        "America/New_York",
    ) == date(2026, 3, 8)
    assert (
        _crosses_local_midnight(
            datetime(2026, 3, 8, 5, 0, tzinfo=UTC),
            datetime(2026, 3, 8, 5, 30, tzinfo=UTC),
            "America/New_York",
        )
        is None
    )


@pytest.mark.asyncio
async def test_get_recent_manual_session_context_strips_internal_content(
    mocker,
) -> None:
    session = SimpleNamespace(
        session_id="sess-1",
        title="Manual work",
        updated_at=datetime(2026, 3, 14, 9, 0, tzinfo=UTC),
    )
    chat_store = SimpleNamespace(
        get_manual_chat_sessions_since=AsyncMock(return_value=[session]),
        get_chat_messages_since=AsyncMock(
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
        ),
    )
    mocker.patch("backend.copilot.autopilot_prompts.chat_db", return_value=chat_store)

    context = await _get_recent_manual_session_context(
        "user-1",
        since_utc=datetime(2026, 3, 13, 0, 0, tzinfo=UTC),
    )

    assert "Manual work" in context
    assert "Visible text" in context
    assert "Completed a useful task for the user." in context
    assert "hidden" not in context


@pytest.mark.asyncio
async def test_get_recent_sent_email_context_formats_sent_emails(mocker) -> None:
    sent_session = SimpleNamespace(
        start_type=ChatSessionStartType.AUTOPILOT_CALLBACK,
        notification_email_sent_at=datetime(2026, 3, 14, 10, 0, tzinfo=UTC),
        completion_report=_build_completion_report(
            email_title="Follow-up",
            email_body="Try this workflow next.",
            callback_session_message="Open the session",
        ),
    )
    chat_store = SimpleNamespace(
        get_recent_sent_email_chat_sessions=AsyncMock(return_value=[sent_session])
    )
    mocker.patch("backend.copilot.autopilot_prompts.chat_db", return_value=chat_store)

    context = await _get_recent_sent_email_context("user-1")

    assert "Follow-up" in context
    assert "Try this workflow next." in context
    assert "Open the session" in context
    assert "Callback" in context


@pytest.mark.asyncio
async def test_get_recent_session_summary_context_formats_completion_reports(
    mocker,
) -> None:
    summary_session = SimpleNamespace(
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
        updated_at=datetime(2026, 3, 14, 12, 0, tzinfo=UTC),
        title="Nightly run",
        completion_report=_build_completion_report(
            thoughts="Prepared a concrete plan for the user.",
            email_title="Nightly summary",
        ),
    )
    chat_store = SimpleNamespace(
        get_recent_completion_report_chat_sessions=AsyncMock(
            return_value=[summary_session]
        )
    )
    mocker.patch("backend.copilot.autopilot_prompts.chat_db", return_value=chat_store)

    context = await _get_recent_session_summary_context("user-1")

    assert "Nightly session updated" in context
    assert "Prepared a concrete plan for the user." in context
    assert "Nightly summary" in context
    assert "Nightly run" in context


@pytest.mark.parametrize(
    ("start_type", "expected_prompt_name"),
    [
        (ChatSessionStartType.AUTOPILOT_NIGHTLY, "CoPilot Nightly"),
        (ChatSessionStartType.AUTOPILOT_CALLBACK, "CoPilot Callback"),
        (ChatSessionStartType.AUTOPILOT_INVITE_CTA, "CoPilot Beta Invite CTA"),
    ],
)
@pytest.mark.asyncio
async def test_build_autopilot_system_prompt_selects_langfuse_prompt(
    mocker,
    start_type: ChatSessionStartType,
    expected_prompt_name: str,
) -> None:
    mocker.patch(
        "backend.copilot.autopilot_prompts.chat_config",
        new=SimpleNamespace(
            langfuse_autopilot_nightly_prompt_name="CoPilot Nightly",
            langfuse_autopilot_callback_prompt_name="CoPilot Callback",
            langfuse_autopilot_invite_cta_prompt_name="CoPilot Beta Invite CTA",
        ),
    )
    understanding_store = SimpleNamespace(
        get_business_understanding=AsyncMock(return_value=object())
    )
    mocker.patch(
        "backend.copilot.autopilot_prompts.understanding_db",
        return_value=understanding_store,
    )
    mocker.patch(
        "backend.copilot.autopilot_prompts.format_understanding_for_prompt",
        return_value="business understanding",
    )
    mocker.patch(
        "backend.copilot.autopilot_prompts._get_recent_sent_email_context",
        new_callable=AsyncMock,
        return_value="recent emails",
    )
    mocker.patch(
        "backend.copilot.autopilot_prompts._get_recent_session_summary_context",
        new_callable=AsyncMock,
        return_value="recent summaries",
    )
    mocker.patch(
        "backend.copilot.autopilot_prompts._get_recent_manual_session_context",
        new_callable=AsyncMock,
        return_value="recent manual sessions",
    )
    compile_prompt = mocker.patch(
        "backend.copilot.autopilot_prompts._get_system_prompt_template",
        new_callable=AsyncMock,
        return_value="compiled prompt",
    )

    invited_user = (
        SimpleNamespace(tally_understanding={"company": "Example"})
        if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA
        else None
    )

    prompt = await _build_autopilot_system_prompt(
        cast(Any, SimpleNamespace(id="user-1", name="User")),
        start_type=start_type,
        timezone_name="UTC",
        target_local_date=date(2026, 3, 16),
        invited_user=cast(Any, invited_user),
    )

    assert prompt == "compiled prompt"
    assert compile_prompt.await_args.kwargs["prompt_name"] == expected_prompt_name
    compiled_context = compile_prompt.await_args.args[0]
    template_vars = compile_prompt.await_args.kwargs["template_vars"]
    assert "business understanding" in compiled_context
    assert "## Recent Copilot Emails Sent To User\nrecent emails" in compiled_context
    assert "## Recent Copilot Session Summaries\nrecent summaries" in compiled_context
    assert template_vars["business_understanding"] == "business understanding"
    assert template_vars["recent_copilot_emails"] == "recent emails"
    assert template_vars["recent_session_summaries"] == "recent summaries"
    assert template_vars["users_information"] == compiled_context
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        assert template_vars["recent_manual_sessions"] == "recent manual sessions"
    else:
        assert template_vars["recent_manual_sessions"] == (
            "Not applicable for this prompt type."
        )
    if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA:
        assert template_vars["beta_application_context"] == '{"company": "Example"}'
    else:
        assert template_vars["beta_application_context"] == (
            "No beta application context available."
        )
    assert (
        "## Recent Manual Sessions Since Previous Nightly Run" not in compiled_context
    )
    assert "## Beta Application Context" not in compiled_context


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
        "backend.copilot.autopilot_completion.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.autopilot_completion._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_completion.upsert_chat_session",
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
        "backend.copilot.autopilot_completion.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.autopilot_completion._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(1, "copilot-session-session-1"),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_completion.upsert_chat_session",
        new_callable=AsyncMock,
    )
    enqueue = mocker.patch(
        "backend.copilot.autopilot_completion._enqueue_session_turn",
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
        "backend.copilot.autopilot_dispatch._session_exists_for_execution_tag",
        new_callable=AsyncMock,
        return_value=False,
    )
    mocker.patch(
        "backend.copilot.autopilot_dispatch._build_autopilot_system_prompt",
        new_callable=AsyncMock,
        return_value="system prompt",
    )
    create_chat_session = mocker.patch(
        "backend.copilot.autopilot_dispatch.create_chat_session",
        new_callable=AsyncMock,
        return_value=created_session,
    )
    enqueue = mocker.patch(
        "backend.copilot.autopilot_dispatch._enqueue_session_turn",
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
        "backend.copilot.autopilot_completion.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.autopilot_completion._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_completion.upsert_chat_session",
        new_callable=AsyncMock,
    )
    enqueue = mocker.patch(
        "backend.copilot.autopilot_completion._enqueue_session_turn",
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
        "backend.copilot.autopilot_dispatch.datetime",
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
        auth_user_id="invite-user",
        status=prisma.enums.InvitedUserStatus.INVITED,
        created_at=fixed_now - timedelta(hours=72),
    )
    user_store = SimpleNamespace(
        list_users=AsyncMock(
            return_value=[
                invite_user,
                nightly_user,
                callback_user,
                disabled_user,
            ]
        )
    )
    invited_user_store = SimpleNamespace(
        list_invited_users_for_auth_users=AsyncMock(return_value=[invited])
    )
    mocker.patch("backend.copilot.autopilot_dispatch.user_db", return_value=user_store)
    mocker.patch(
        "backend.copilot.autopilot_dispatch.invited_user_db",
        return_value=invited_user_store,
    )
    mocker.patch(
        "backend.copilot.autopilot_dispatch.is_feature_enabled",
        new_callable=AsyncMock,
        side_effect=lambda _flag, user_id, default=False: user_id != "disabled-user",
    )
    mocker.patch(
        "backend.copilot.autopilot_dispatch._crosses_local_midnight",
        side_effect=lambda *_args, **_kwargs: date(2026, 3, 17),
    )
    mocker.patch(
        "backend.copilot.autopilot_dispatch._user_has_recent_manual_message",
        new_callable=AsyncMock,
        side_effect=lambda user_id, _since: user_id == "nightly-user",
    )
    mocker.patch(
        "backend.copilot.autopilot_dispatch._user_has_session_since",
        new_callable=AsyncMock,
        side_effect=lambda user_id, _since: (
            user_id in {"nightly-user", "callback-user"}
        ),
    )
    mocker.patch(
        "backend.copilot.autopilot_dispatch._session_exists_for_execution_tag",
        new_callable=AsyncMock,
        return_value=False,
    )
    create_autopilot_session = mocker.patch(
        "backend.copilot.autopilot_dispatch._create_autopilot_session",
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
async def test_trigger_autopilot_session_for_user_uses_local_date_for_nightly(
    mocker,
) -> None:
    fixed_now = datetime(2026, 3, 16, 18, 30, tzinfo=UTC)
    datetime_mock = mocker.patch(
        "backend.copilot.autopilot_dispatch.datetime",
        wraps=datetime,
    )
    datetime_mock.now.return_value = fixed_now

    user = SimpleNamespace(
        id="user-1",
        timezone="Asia/Tokyo",
        name="Nightly User",
    )
    user_store = SimpleNamespace(get_user_by_id=AsyncMock(return_value=user))
    invited_user_store = SimpleNamespace(
        list_invited_users_for_auth_users=AsyncMock(return_value=[])
    )
    session = ChatSession.new(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
    )
    create_autopilot_session = mocker.patch(
        "backend.copilot.autopilot_dispatch._create_autopilot_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch("backend.copilot.autopilot_dispatch.user_db", return_value=user_store)
    mocker.patch(
        "backend.copilot.autopilot_dispatch.invited_user_db",
        return_value=invited_user_store,
    )

    created = await trigger_autopilot_session_for_user(
        "user-1",
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
    )

    assert created is session
    assert create_autopilot_session.await_args.kwargs["execution_tag"].startswith(
        "admin-autopilot:AUTOPILOT_NIGHTLY:"
    )
    assert create_autopilot_session.await_args.kwargs["timezone_name"] == "Asia/Tokyo"
    assert create_autopilot_session.await_args.kwargs["target_local_date"] == date(
        2026,
        3,
        17,
    )


@pytest.mark.asyncio
async def test_send_nightly_copilot_emails_queues_repair_for_missing_report(
    mocker,
) -> None:
    session = _build_autopilot_session()
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions=AsyncMock(return_value=[candidate])
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    queue_repair = mocker.patch(
        "backend.copilot.autopilot_email._queue_completion_report_repair",
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
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions=AsyncMock(return_value=[candidate])
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(1, "copilot-session-session-1"),
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_email.upsert_chat_session",
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
async def test_send_nightly_copilot_emails_uses_email_title_for_session_title(
    mocker,
) -> None:
    session = _build_autopilot_session()
    session.completion_report = _build_completion_report(
        email_title="Autopilot update",
        email_body="Useful update",
    )
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions=AsyncMock(return_value=[candidate])
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    update_title = mocker.patch(
        "backend.copilot.autopilot_email.update_session_title",
        new_callable=AsyncMock,
        return_value=True,
    )
    generate_title = mocker.patch(
        "backend.copilot.autopilot_email._generate_session_title",
        new_callable=AsyncMock,
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_email.upsert_chat_session",
        new_callable=AsyncMock,
    )

    processed = await send_nightly_copilot_emails()

    assert processed == 1
    assert session.title == "Autopilot update"
    update_title.assert_awaited_once_with(
        session.session_id,
        session.user_id,
        "Autopilot update",
        only_if_empty=True,
    )
    generate_title.assert_not_awaited()
    send_email.assert_awaited_once_with(session)
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_send_nightly_copilot_emails_generates_session_title_when_missing(
    mocker,
) -> None:
    session = _build_autopilot_session()
    session.completion_report = _build_completion_report(email_body="Useful update")
    session.completion_report = session.completion_report.model_copy(
        update={"email_title": None}
    )
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions=AsyncMock(return_value=[candidate])
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    update_title = mocker.patch(
        "backend.copilot.autopilot_email.update_session_title",
        new_callable=AsyncMock,
        return_value=True,
    )
    generate_title = mocker.patch(
        "backend.copilot.autopilot_email._generate_session_title",
        new_callable=AsyncMock,
        return_value="Generated title",
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_email.upsert_chat_session",
        new_callable=AsyncMock,
    )

    processed = await send_nightly_copilot_emails()

    assert processed == 1
    assert session.title == "Generated title"
    generate_title.assert_awaited_once_with(
        "Useful update",
        user_id=session.user_id,
        session_id=session.session_id,
    )
    update_title.assert_awaited_once_with(
        session.session_id,
        session.user_id,
        "Generated title",
        only_if_empty=True,
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
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions=AsyncMock(return_value=[candidate])
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_email.upsert_chat_session",
        new_callable=AsyncMock,
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
    )

    processed = await send_nightly_copilot_emails()

    assert processed == 1
    assert session.notification_email_skipped_at is not None
    send_email.assert_not_called()
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_send_pending_copilot_emails_for_user_returns_summary(mocker) -> None:
    session = _build_autopilot_session()
    session.completion_report = _build_completion_report()
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions_for_user=AsyncMock(
            return_value=[candidate]
        )
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    send_email = mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_email.upsert_chat_session",
        new_callable=AsyncMock,
    )

    result = await send_pending_copilot_emails_for_user("user-1")

    assert result.model_dump() == {
        "candidate_count": 1,
        "processed_count": 1,
        "sent_count": 1,
        "skipped_count": 0,
        "repair_queued_count": 0,
        "running_count": 0,
        "failed_count": 0,
    }
    send_email.assert_awaited_once_with(session)
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_send_pending_copilot_emails_for_user_counts_expected_send_failures(
    mocker,
) -> None:
    session = _build_autopilot_session()
    session.completion_report = _build_completion_report()
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions_for_user=AsyncMock(
            return_value=[candidate]
        )
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
        side_effect=ValueError("missing email"),
    )
    upsert = mocker.patch(
        "backend.copilot.autopilot_email.upsert_chat_session",
        new_callable=AsyncMock,
    )

    result = await send_pending_copilot_emails_for_user("user-1")

    assert result.model_dump() == {
        "candidate_count": 1,
        "processed_count": 1,
        "sent_count": 0,
        "skipped_count": 0,
        "repair_queued_count": 0,
        "running_count": 0,
        "failed_count": 1,
    }
    assert session.notification_email_skipped_at is not None
    upsert.assert_awaited_once_with(session)


@pytest.mark.asyncio
async def test_send_pending_copilot_emails_for_user_reraises_unexpected_send_failures(
    mocker,
) -> None:
    session = _build_autopilot_session()
    session.completion_report = _build_completion_report()
    candidate = SimpleNamespace(session_id=session.session_id)
    chat_store = SimpleNamespace(
        get_pending_notification_chat_sessions_for_user=AsyncMock(
            return_value=[candidate]
        )
    )
    mocker.patch("backend.copilot.autopilot_email.chat_db", return_value=chat_store)
    mocker.patch(
        "backend.copilot.autopilot_email.get_chat_session",
        new_callable=AsyncMock,
        return_value=session,
    )
    mocker.patch(
        "backend.copilot.stream_registry.get_session",
        new_callable=AsyncMock,
        return_value=None,
    )
    mocker.patch(
        "backend.copilot.autopilot_email._get_pending_approval_metadata",
        new_callable=AsyncMock,
        return_value=(0, None),
    )
    mocker.patch(
        "backend.copilot.autopilot_email._send_completion_email",
        new_callable=AsyncMock,
        side_effect=TypeError("unexpected bug"),
    )

    with pytest.raises(TypeError, match="unexpected bug"):
        await send_pending_copilot_emails_for_user("user-1")


@pytest.mark.asyncio
async def test_consume_callback_token_reuses_existing_session(mocker) -> None:
    token = SimpleNamespace(
        id="token-1",
        user_id="user-1",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        consumed_session_id="sess-existing",
    )
    chat_store = SimpleNamespace(
        get_chat_session_callback_token=AsyncMock(return_value=token),
        mark_chat_session_callback_token_consumed=AsyncMock(),
    )
    mocker.patch(
        "backend.copilot.autopilot.chat_db",
        return_value=chat_store,
    )
    create_chat_session = mocker.patch(
        "backend.copilot.autopilot.create_chat_session",
        new_callable=AsyncMock,
    )

    result = await consume_callback_token("token-1", "user-1")

    assert result.session_id == "sess-existing"
    create_chat_session.assert_not_called()
    chat_store.mark_chat_session_callback_token_consumed.assert_not_called()


@pytest.mark.asyncio
async def test_consume_callback_token_creates_manual_session(mocker) -> None:
    token = SimpleNamespace(
        id="token-1",
        user_id="user-1",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
        consumed_session_id=None,
        callback_session_message="Open this chat",
    )
    chat_store = SimpleNamespace(
        get_chat_session_callback_token=AsyncMock(return_value=token),
        mark_chat_session_callback_token_consumed=AsyncMock(),
    )
    created_session = ChatSession.new("user-1")
    mocker.patch(
        "backend.copilot.autopilot.chat_db",
        return_value=chat_store,
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
    chat_store.mark_chat_session_callback_token_consumed.assert_awaited_once_with(
        "token-1",
        created_session.session_id,
    )


@pytest.mark.asyncio
async def test_consume_callback_token_rejects_expired_token(mocker) -> None:
    token = SimpleNamespace(
        id="token-1",
        user_id="user-1",
        expires_at=datetime.now(UTC) - timedelta(minutes=1),
        consumed_session_id=None,
        callback_session_message="Open this chat",
    )
    chat_store = SimpleNamespace(
        get_chat_session_callback_token=AsyncMock(return_value=token),
    )
    mocker.patch(
        "backend.copilot.autopilot.chat_db",
        return_value=chat_store,
    )

    with pytest.raises(ValueError, match="expired"):
        await consume_callback_token("token-1", "user-1")
