from __future__ import annotations

import json
import logging
import re
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING, Any
from uuid import uuid4
from zoneinfo import ZoneInfo

import prisma.enums
from pydantic import BaseModel, Field, ValidationError

from backend.copilot import stream_registry
from backend.copilot.constants import COPILOT_SESSION_PREFIX
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    create_chat_session,
    get_chat_session,
    upsert_chat_session,
)
from backend.copilot.service import _get_system_prompt_template
from backend.copilot.service import config as chat_config
from backend.copilot.session_types import (
    ChatSessionConfig,
    ChatSessionStartType,
    CompletionReportInput,
    StoredCompletionReport,
)
from backend.data.db_accessors import (
    chat_db,
    invited_user_db,
    review_db,
    understanding_db,
    user_db,
)
from backend.data.model import User
from backend.data.understanding import format_understanding_for_prompt
from backend.notifications.email import EmailSender
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.settings import Settings
from backend.util.timezone_utils import get_user_timezone_or_utc
from backend.util.url import get_frontend_base_url

if TYPE_CHECKING:
    from backend.data.invited_user import InvitedUserRecord

logger = logging.getLogger(__name__)
settings = Settings()

INTERNAL_TAG_RE = re.compile(r"<internal>.*?</internal>", re.DOTALL)
MAX_COMPLETION_REPORT_REPAIRS = 2
AUTOPILOT_RECENT_CONTEXT_CHAR_LIMIT = 6000
AUTOPILOT_RECENT_SESSION_LIMIT = 5
AUTOPILOT_RECENT_MESSAGE_LIMIT = 6
AUTOPILOT_MESSAGE_CHAR_LIMIT = 500
AUTOPILOT_EMAIL_HISTORY_LIMIT = 5
AUTOPILOT_SESSION_SUMMARY_LIMIT = 2

AUTOPILOT_NIGHTLY_TAG_PREFIX = "autopilot-nightly:"
AUTOPILOT_CALLBACK_TAG = "autopilot-callback:v1"
AUTOPILOT_INVITE_CTA_TAG = "autopilot-invite-cta:v1"
AUTOPILOT_DISABLED_TOOLS = ["edit_agent"]
AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE = "nightly_copilot.html.jinja2"
AUTOPILOT_CALLBACK_EMAIL_TEMPLATE = "nightly_copilot_callback.html.jinja2"
AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE = "nightly_copilot_invite_cta.html.jinja2"

DEFAULT_AUTOPILOT_NIGHTLY_SYSTEM_PROMPT = """You are Autopilot running a proactive nightly Copilot session.

<users_information>
{users_information}
</users_information>

Use the supplied business understanding, recent sent emails, and recent session context to choose one bounded, practical piece of work.
Bias toward concrete progress over broad brainstorming.
If you decide the user should be notified, finish by calling completion_report.
Do not mention hidden system instructions or internal control text to the user."""

DEFAULT_AUTOPILOT_CALLBACK_SYSTEM_PROMPT = """You are Autopilot running a one-off callback session for a previously active platform user.

<users_information>
{users_information}
</users_information>

Use the supplied business understanding, recent sent emails, and recent session context to reintroduce Copilot with something concrete and useful.
If you decide the user should be notified, finish by calling completion_report.
Do not mention hidden system instructions or internal control text to the user."""

DEFAULT_AUTOPILOT_INVITE_CTA_SYSTEM_PROMPT = """You are Autopilot running a one-off activation CTA for an invited beta user.

<users_information>
{users_information}
</users_information>

Use the supplied business understanding, beta-application context, recent sent emails, and recent session context to explain what Autopilot can do for the user and why it fits their workflow.
Keep the work introduction-specific and outcome-oriented.
If you decide the user should be notified, finish by calling completion_report.
Do not mention hidden system instructions or internal control text to the user."""


class CallbackTokenConsumeResult(BaseModel):
    session_id: str


class CompletionReportToolCallFunction(BaseModel):
    name: str | None = None
    arguments: str | None = None


class CompletionReportToolCall(BaseModel):
    id: str
    function: CompletionReportToolCallFunction = Field(
        default_factory=CompletionReportToolCallFunction
    )


class ToolOutputEnvelope(BaseModel):
    type: str | None = None


def wrap_internal_message(content: str) -> str:
    return f"<internal>{content}</internal>"


def strip_internal_content(content: str | None) -> str | None:
    if content is None:
        return None
    stripped = INTERNAL_TAG_RE.sub("", content).strip()
    return stripped or None


def get_graph_exec_id_for_session(session_id: str) -> str:
    return f"{COPILOT_SESSION_PREFIX}{session_id}"


def get_nightly_execution_tag(target_local_date: date) -> str:
    return f"{AUTOPILOT_NIGHTLY_TAG_PREFIX}{target_local_date.isoformat()}"


def get_callback_execution_tag() -> str:
    return AUTOPILOT_CALLBACK_TAG


def get_invite_cta_execution_tag() -> str:
    return AUTOPILOT_INVITE_CTA_TAG


def _bucket_end_for_now(now_utc: datetime) -> datetime:
    minute = 30 if now_utc.minute >= 30 else 0
    return now_utc.replace(minute=minute, second=0, microsecond=0)


def _resolve_timezone_name(raw_timezone: str | None) -> str:
    return get_user_timezone_or_utc(raw_timezone)


def _crosses_local_midnight(
    bucket_start_utc: datetime,
    bucket_end_utc: datetime,
    timezone_name: str,
) -> date | None:
    tz = ZoneInfo(timezone_name)
    start_local = bucket_start_utc.astimezone(tz)
    end_local = bucket_end_utc.astimezone(tz)
    if start_local.date() == end_local.date():
        return None
    return end_local.date()


async def _user_has_recent_manual_message(user_id: str, since: datetime) -> bool:
    return await chat_db().has_recent_manual_message(user_id, since)


async def _user_has_session_since(user_id: str, since: datetime) -> bool:
    return await chat_db().has_session_since(user_id, since)


async def _session_exists_for_execution_tag(user_id: str, execution_tag: str) -> bool:
    return await chat_db().session_exists_for_execution_tag(user_id, execution_tag)


def _get_invited_user_tally_understanding(
    invited_user: InvitedUserRecord | None,
) -> dict[str, Any] | None:
    return invited_user.tally_understanding if invited_user is not None else None


def _render_initial_message(
    start_type: ChatSessionStartType,
    *,
    user_name: str | None,
    invited_user: InvitedUserRecord | None = None,
) -> str:
    display_name = user_name or "the user"
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        return wrap_internal_message(
            "This is a nightly proactive Copilot session. Review recent manual activity, "
            f"do one useful piece of work for {display_name}, and finish with completion_report."
        )
    if start_type == ChatSessionStartType.AUTOPILOT_CALLBACK:
        return wrap_internal_message(
            "This is a one-off callback session for a previously active user. "
            f"Reintroduce Copilot with something concrete and useful for {display_name}, "
            "then finish with completion_report."
        )

    invite_summary = ""
    tally_understanding = _get_invited_user_tally_understanding(invited_user)
    if tally_understanding is not None:
        invite_summary = "\nKnown context from the beta application:\n" + json.dumps(
            tally_understanding, ensure_ascii=False
        )
    return wrap_internal_message(
        "This is a one-off invite CTA session for an invited beta user who has not yet activated. "
        f"Create a tailored introduction for {display_name}, explain how Autopilot can help, "
        f"and finish with completion_report.{invite_summary}"
    )


def _truncate_prompt_text(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _get_autopilot_prompt_name(start_type: ChatSessionStartType) -> str:
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        return chat_config.langfuse_autopilot_nightly_prompt_name
    if start_type == ChatSessionStartType.AUTOPILOT_CALLBACK:
        return chat_config.langfuse_autopilot_callback_prompt_name
    if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA:
        return chat_config.langfuse_autopilot_invite_cta_prompt_name
    raise ValueError(f"Unsupported start type for autopilot prompt: {start_type}")


def _get_autopilot_fallback_prompt(start_type: ChatSessionStartType) -> str:
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        return DEFAULT_AUTOPILOT_NIGHTLY_SYSTEM_PROMPT
    if start_type == ChatSessionStartType.AUTOPILOT_CALLBACK:
        return DEFAULT_AUTOPILOT_CALLBACK_SYSTEM_PROMPT
    if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA:
        return DEFAULT_AUTOPILOT_INVITE_CTA_SYSTEM_PROMPT
    raise ValueError(f"Unsupported start type for autopilot prompt: {start_type}")


def _format_start_type_label(start_type: ChatSessionStartType) -> str:
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        return "Nightly"
    if start_type == ChatSessionStartType.AUTOPILOT_CALLBACK:
        return "Callback"
    if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA:
        return "Beta Invite CTA"
    return start_type.value


def _get_previous_local_midnight_utc(
    target_local_date: date,
    timezone_name: str,
) -> datetime:
    tz = ZoneInfo(timezone_name)
    previous_midnight_local = datetime.combine(
        target_local_date - timedelta(days=1),
        time.min,
        tzinfo=tz,
    )
    return previous_midnight_local.astimezone(UTC)


async def _get_recent_manual_session_context(
    user_id: str,
    *,
    since_utc: datetime,
) -> str:
    sessions = await chat_db().get_manual_chat_sessions_since(
        user_id,
        since_utc,
        AUTOPILOT_RECENT_SESSION_LIMIT,
    )

    if not sessions:
        return "No recent manual sessions since the previous nightly run."

    blocks: list[str] = []
    used_chars = 0

    for session in sessions:
        messages = await chat_db().get_chat_messages_since(
            session.session_id, since_utc
        )

        visible_messages: list[str] = []
        for message in messages[-AUTOPILOT_RECENT_MESSAGE_LIMIT:]:
            content = message.content or ""
            if message.role == "user":
                visible = strip_internal_content(content)
            else:
                visible = content.strip() or None
            if not visible:
                continue

            role_label = {
                "user": "User",
                "assistant": "Assistant",
                "tool": "Tool",
            }.get(message.role, message.role.title())
            visible_messages.append(
                f"{role_label}: {_truncate_prompt_text(visible, AUTOPILOT_MESSAGE_CHAR_LIMIT)}"
            )

        if not visible_messages:
            continue

        title_suffix = f" ({session.title})" if session.title else ""
        block = (
            f"### Session updated {session.updated_at.isoformat()}{title_suffix}\n"
            + "\n".join(visible_messages)
        )
        if used_chars + len(block) > AUTOPILOT_RECENT_CONTEXT_CHAR_LIMIT:
            break

        blocks.append(block)
        used_chars += len(block)

    return (
        "\n\n".join(blocks)
        if blocks
        else "No recent manual sessions since the previous nightly run."
    )


async def _get_recent_sent_email_context(user_id: str) -> str:
    sessions = await chat_db().get_recent_sent_email_chat_sessions(
        user_id,
        AUTOPILOT_EMAIL_HISTORY_LIMIT,
    )
    if not sessions:
        return "No recent Copilot or Autopilot emails have been sent to this user."

    blocks: list[str] = []
    for session in sessions:
        report = session.completion_report
        sent_at = session.notification_email_sent_at
        if report is None or sent_at is None:
            continue

        lines = [
            f"### Sent {sent_at.isoformat()} ({_format_start_type_label(session.start_type)})",
        ]
        if report.email_title:
            lines.append(
                f"Subject: {_truncate_prompt_text(report.email_title, AUTOPILOT_MESSAGE_CHAR_LIMIT)}"
            )
        if report.email_body:
            lines.append(
                f"Body: {_truncate_prompt_text(report.email_body, AUTOPILOT_MESSAGE_CHAR_LIMIT)}"
            )
        if report.callback_session_message:
            lines.append(
                "CTA Message: "
                + _truncate_prompt_text(
                    report.callback_session_message,
                    AUTOPILOT_MESSAGE_CHAR_LIMIT,
                )
            )
        blocks.append("\n".join(lines))

    return (
        "\n\n".join(blocks)
        if blocks
        else "No recent Copilot or Autopilot emails have been sent to this user."
    )


async def _get_recent_session_summary_context(user_id: str) -> str:
    sessions = await chat_db().get_recent_completion_report_chat_sessions(
        user_id,
        AUTOPILOT_SESSION_SUMMARY_LIMIT,
    )
    if not sessions:
        return "No recent Copilot session summaries are available."

    blocks: list[str] = []
    for session in sessions:
        report = session.completion_report
        if report is None:
            continue

        title_suffix = f" ({session.title})" if session.title else ""
        lines = [
            f"### {_format_start_type_label(session.start_type)} session updated {session.updated_at.isoformat()}{title_suffix}",
            f"Summary: {_truncate_prompt_text(report.thoughts, AUTOPILOT_MESSAGE_CHAR_LIMIT)}",
        ]
        if report.email_title:
            lines.append(
                "Email Title: "
                + _truncate_prompt_text(
                    report.email_title, AUTOPILOT_MESSAGE_CHAR_LIMIT
                )
            )
        blocks.append("\n".join(lines))

    return (
        "\n\n".join(blocks)
        if blocks
        else "No recent Copilot session summaries are available."
    )


async def _build_autopilot_system_prompt(
    user: User,
    *,
    start_type: ChatSessionStartType,
    timezone_name: str,
    target_local_date: date | None = None,
    invited_user: InvitedUserRecord | None = None,
) -> str:
    understanding = await understanding_db().get_business_understanding(user.id)
    context_sections = [
        (
            format_understanding_for_prompt(understanding)
            if understanding
            else "No saved business understanding yet."
        )
    ]
    context_sections.append(
        "## Recent Copilot Emails Sent To User\n"
        + await _get_recent_sent_email_context(user.id)
    )
    context_sections.append(
        "## Recent Copilot Session Summaries\n"
        + await _get_recent_session_summary_context(user.id)
    )

    if (
        start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY
        and target_local_date is not None
    ):
        recent_context = await _get_recent_manual_session_context(
            user.id,
            since_utc=_get_previous_local_midnight_utc(
                target_local_date,
                timezone_name,
            ),
        )
        context_sections.append(
            "## Recent Manual Sessions Since Previous Nightly Run\n" + recent_context
        )

    tally_understanding = _get_invited_user_tally_understanding(invited_user)
    if tally_understanding is not None:
        invite_context = json.dumps(tally_understanding, ensure_ascii=False)
        context_sections.append("## Beta Application Context\n" + invite_context)

    return await _get_system_prompt_template(
        "\n\n".join(context_sections),
        prompt_name=_get_autopilot_prompt_name(start_type),
        fallback_prompt=_get_autopilot_fallback_prompt(start_type),
    )


async def _enqueue_session_turn(
    session: ChatSession,
    *,
    message: str,
    tool_name: str,
) -> None:
    turn_id = str(uuid4())
    await stream_registry.create_session(
        session_id=session.session_id,
        user_id=session.user_id,
        tool_call_id=tool_name,
        tool_name=tool_name,
        turn_id=turn_id,
        blocking=False,
    )
    await enqueue_copilot_turn(
        session_id=session.session_id,
        user_id=session.user_id,
        message=message,
        turn_id=turn_id,
        is_user_message=True,
    )


async def _create_autopilot_session(
    user: User,
    *,
    start_type: ChatSessionStartType,
    execution_tag: str,
    timezone_name: str,
    target_local_date: date | None = None,
    invited_user: InvitedUserRecord | None = None,
) -> ChatSession | None:
    if await _session_exists_for_execution_tag(user.id, execution_tag):
        return None

    system_prompt = await _build_autopilot_system_prompt(
        user,
        start_type=start_type,
        timezone_name=timezone_name,
        target_local_date=target_local_date,
        invited_user=invited_user,
    )
    initial_message = _render_initial_message(
        start_type,
        user_name=user.name,
        invited_user=invited_user,
    )
    session_config = ChatSessionConfig(
        system_prompt_override=system_prompt,
        initial_user_message=initial_message,
        extra_tools=["completion_report"],
        disabled_tools=AUTOPILOT_DISABLED_TOOLS,
    )

    session = await create_chat_session(
        user.id,
        start_type=start_type,
        execution_tag=execution_tag,
        session_config=session_config,
        initial_messages=[ChatMessage(role="user", content=initial_message)],
    )
    await _enqueue_session_turn(
        session,
        message=initial_message,
        tool_name="autopilot_dispatch",
    )
    return session


async def _try_create_invite_cta_session(
    user: User,
    *,
    invited_user: InvitedUserRecord | None,
    now_utc: datetime,
    timezone_name: str,
    invite_cta_start: date,
    invite_cta_delay: timedelta,
) -> bool:
    if invited_user is None:
        return False
    if invited_user.status != prisma.enums.InvitedUserStatus.INVITED:
        return False
    if invited_user.created_at.date() < invite_cta_start:
        return False
    if invited_user.created_at > now_utc - invite_cta_delay:
        return False
    if await _session_exists_for_execution_tag(user.id, get_invite_cta_execution_tag()):
        return False

    created = await _create_autopilot_session(
        user,
        start_type=ChatSessionStartType.AUTOPILOT_INVITE_CTA,
        execution_tag=get_invite_cta_execution_tag(),
        timezone_name=timezone_name,
        invited_user=invited_user,
    )
    return created is not None


async def _try_create_nightly_session(
    user: User,
    *,
    now_utc: datetime,
    timezone_name: str,
    target_local_date: date,
) -> bool:
    if not await _user_has_recent_manual_message(
        user.id,
        now_utc - timedelta(hours=24),
    ):
        return False

    created = await _create_autopilot_session(
        user,
        start_type=ChatSessionStartType.AUTOPILOT_NIGHTLY,
        execution_tag=get_nightly_execution_tag(target_local_date),
        timezone_name=timezone_name,
        target_local_date=target_local_date,
    )
    return created is not None


async def _try_create_callback_session(
    user: User,
    *,
    callback_start: datetime,
    timezone_name: str,
) -> bool:
    if not await _user_has_session_since(user.id, callback_start):
        return False
    if await _session_exists_for_execution_tag(user.id, get_callback_execution_tag()):
        return False

    created = await _create_autopilot_session(
        user,
        start_type=ChatSessionStartType.AUTOPILOT_CALLBACK,
        execution_tag=get_callback_execution_tag(),
        timezone_name=timezone_name,
    )
    return created is not None


async def _dispatch_nightly_copilot() -> int:
    now_utc = datetime.now(UTC)
    bucket_end = _bucket_end_for_now(now_utc)
    bucket_start = bucket_end - timedelta(minutes=30)
    callback_start = datetime.combine(
        settings.config.nightly_copilot_callback_start_date,
        time.min,
        tzinfo=UTC,
    )
    invite_cta_start = settings.config.nightly_copilot_invite_cta_start_date
    invite_cta_delay = timedelta(
        hours=settings.config.nightly_copilot_invite_cta_delay_hours
    )

    users = await user_db().list_users()
    invites = await invited_user_db().list_invited_users_for_auth_users(
        [user.id for user in users]
    )
    invites_by_user_id = {
        invite.auth_user_id: invite for invite in invites if invite.auth_user_id
    }

    created_count = 0
    for user in users:
        if not await is_feature_enabled(Flag.NIGHTLY_COPILOT, user.id, default=False):
            continue

        timezone_name = _resolve_timezone_name(user.timezone)
        target_local_date = _crosses_local_midnight(
            bucket_start,
            bucket_end,
            timezone_name,
        )
        if target_local_date is None:
            continue

        invited_user = invites_by_user_id.get(user.id)
        if await _try_create_invite_cta_session(
            user,
            invited_user=invited_user,
            now_utc=now_utc,
            timezone_name=timezone_name,
            invite_cta_start=invite_cta_start,
            invite_cta_delay=invite_cta_delay,
        ):
            created_count += 1
            continue

        if await _try_create_nightly_session(
            user,
            now_utc=now_utc,
            timezone_name=timezone_name,
            target_local_date=target_local_date,
        ):
            created_count += 1
            continue

        if await _try_create_callback_session(
            user,
            callback_start=callback_start,
            timezone_name=timezone_name,
        ):
            created_count += 1

    return created_count


async def dispatch_nightly_copilot() -> int:
    return await _dispatch_nightly_copilot()


async def _get_pending_approval_metadata(
    session: ChatSession,
) -> tuple[int, str | None]:
    graph_exec_id = get_graph_exec_id_for_session(session.session_id)
    pending_count = await review_db().count_pending_reviews_for_graph_exec(
        graph_exec_id,
        session.user_id,
    )
    return pending_count, graph_exec_id if pending_count > 0 else None


def _extract_completion_report_from_session(
    session: ChatSession,
    *,
    pending_approval_count: int,
) -> CompletionReportInput | None:
    tool_outputs = {
        message.tool_call_id: message.content
        for message in session.messages
        if message.role == "tool" and message.tool_call_id
    }

    latest_report: CompletionReportInput | None = None
    for message in session.messages:
        if message.role != "assistant" or not message.tool_calls:
            continue

        for tool_call in message.tool_calls:
            try:
                parsed_tool_call = CompletionReportToolCall.model_validate(tool_call)
            except ValidationError:
                continue

            if parsed_tool_call.function.name != "completion_report":
                continue

            output = tool_outputs.get(parsed_tool_call.id)
            if not output:
                continue

            try:
                output_payload = ToolOutputEnvelope.model_validate_json(output)
            except ValidationError:
                output_payload = None

            if output_payload is not None and output_payload.type == "error":
                continue

            try:
                raw_arguments = parsed_tool_call.function.arguments or "{}"
                report = CompletionReportInput.model_validate_json(raw_arguments)
            except ValidationError:
                continue

            if pending_approval_count > 0 and not report.approval_summary:
                continue

            latest_report = report

    return latest_report


def _build_completion_report_repair_message(
    *,
    attempt: int,
    pending_approval_count: int,
) -> str:
    approval_instruction = ""
    if pending_approval_count > 0:
        approval_instruction = (
            f" There are currently {pending_approval_count} pending approval item(s). "
            "If they still exist, include approval_summary."
        )

    return wrap_internal_message(
        "The session completed without a valid completion_report tool call. "
        f"This is repair attempt {attempt}. Call completion_report now and do not do any additional user-facing work."
        + approval_instruction
    )


async def _queue_completion_report_repair(
    session: ChatSession,
    *,
    pending_approval_count: int,
) -> None:
    attempt = session.completion_report_repair_count + 1
    repair_message = _build_completion_report_repair_message(
        attempt=attempt,
        pending_approval_count=pending_approval_count,
    )
    session.messages.append(ChatMessage(role="user", content=repair_message))
    session.completion_report_repair_count = attempt
    session.completion_report_repair_queued_at = datetime.now(UTC)
    session.completed_at = None
    session.completion_report = None
    await upsert_chat_session(session)
    await _enqueue_session_turn(
        session,
        message=repair_message,
        tool_name="completion_report_repair",
    )


async def handle_non_manual_session_completion(session_id: str) -> None:
    session = await get_chat_session(session_id)
    if session is None or session.is_manual:
        return

    pending_approval_count, graph_exec_id = await _get_pending_approval_metadata(
        session
    )
    report = _extract_completion_report_from_session(
        session,
        pending_approval_count=pending_approval_count,
    )

    if report is not None:
        session.completion_report = StoredCompletionReport(
            **report.model_dump(),
            has_pending_approvals=pending_approval_count > 0,
            pending_approval_count=pending_approval_count,
            pending_approval_graph_exec_id=graph_exec_id,
            saved_at=datetime.now(UTC),
        )
        session.completion_report_repair_queued_at = None
        session.completed_at = datetime.now(UTC)
        await upsert_chat_session(session)
        return

    if session.completion_report_repair_count >= MAX_COMPLETION_REPORT_REPAIRS:
        session.completion_report_repair_queued_at = None
        session.completed_at = datetime.now(UTC)
        await upsert_chat_session(session)
        return

    await _queue_completion_report_repair(
        session,
        pending_approval_count=pending_approval_count,
    )


def _split_email_paragraphs(text: str | None) -> list[str]:
    return [segment.strip() for segment in (text or "").splitlines() if segment.strip()]


async def _create_callback_token(
    session: ChatSession,
) -> str:
    if session.completion_report is None:
        raise ValueError("Missing completion report")
    callback_session_message = session.completion_report.callback_session_message
    if callback_session_message is None:
        raise ValueError("Missing callback session message")

    token = await chat_db().create_chat_session_callback_token(
        user_id=session.user_id,
        source_session_id=session.session_id,
        callback_session_message=callback_session_message,
        expires_at=datetime.now(UTC)
        + timedelta(hours=settings.config.nightly_copilot_callback_token_ttl_hours),
    )
    return token.id


def _build_session_link(session_id: str, *, show_autopilot: bool) -> str:
    base_url = get_frontend_base_url()
    suffix = "&showAutopilot=1" if show_autopilot else ""
    return f"{base_url}/copilot?sessionId={session_id}{suffix}"


def _build_callback_link(token_id: str) -> str:
    return f"{get_frontend_base_url()}/copilot?callbackToken={token_id}"


def _get_completion_email_template_name(start_type: ChatSessionStartType) -> str:
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        return AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE
    if start_type == ChatSessionStartType.AUTOPILOT_CALLBACK:
        return AUTOPILOT_CALLBACK_EMAIL_TEMPLATE
    if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA:
        return AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE
    raise ValueError(f"Unsupported start type for completion email: {start_type}")


async def _send_completion_email(session: ChatSession) -> None:
    report = session.completion_report
    if report is None:
        raise ValueError("Missing completion report")
    try:
        user = await user_db().get_user_by_id(session.user_id)
    except ValueError as exc:
        raise ValueError(f"User {session.user_id} not found") from exc

    if not user.email:
        raise ValueError(f"User {session.user_id} not found")

    approval_cta = report.has_pending_approvals
    template_name = _get_completion_email_template_name(session.start_type)
    if approval_cta:
        cta_url = _build_session_link(session.session_id, show_autopilot=True)
        cta_label = "Review in Copilot"
    else:
        token_id = await _create_callback_token(session)
        cta_url = _build_callback_link(token_id)
        cta_label = (
            "Try Copilot"
            if session.start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA
            else "Open Copilot"
        )

    EmailSender().send_template(
        user_email=user.email,
        subject=report.email_title or "Autopilot update",
        template_name=template_name,
        data={
            "email_body_paragraphs": _split_email_paragraphs(report.email_body),
            "approval_summary_paragraphs": _split_email_paragraphs(
                report.approval_summary
            ),
            "cta_url": cta_url,
            "cta_label": cta_label,
        },
    )


async def _send_nightly_copilot_emails() -> int:
    candidates = await chat_db().get_pending_notification_chat_sessions(limit=200)

    processed_count = 0
    for candidate in candidates:
        session = await get_chat_session(candidate.session_id)
        if session is None or session.is_manual:
            continue

        active = await stream_registry.get_session(session.session_id)
        is_running = active is not None and active.status == "running"
        if is_running:
            continue

        pending_approval_count, graph_exec_id = await _get_pending_approval_metadata(
            session
        )

        if session.completion_report is None:
            if session.completion_report_repair_count < MAX_COMPLETION_REPORT_REPAIRS:
                await _queue_completion_report_repair(
                    session,
                    pending_approval_count=pending_approval_count,
                )
                continue

            session.completed_at = session.completed_at or datetime.now(UTC)
            session.completion_report_repair_queued_at = None
            session.notification_email_skipped_at = datetime.now(UTC)
            await upsert_chat_session(session)
            processed_count += 1
            continue

        session.completed_at = session.completed_at or datetime.now(UTC)
        if (
            session.completion_report.pending_approval_graph_exec_id is None
            and graph_exec_id
        ):
            session.completion_report = session.completion_report.model_copy(
                update={
                    "has_pending_approvals": pending_approval_count > 0,
                    "pending_approval_count": pending_approval_count,
                    "pending_approval_graph_exec_id": graph_exec_id,
                }
            )

        if not session.completion_report.should_notify_user:
            session.notification_email_skipped_at = datetime.now(UTC)
            await upsert_chat_session(session)
            processed_count += 1
            continue

        try:
            await _send_completion_email(session)
        except Exception:
            logger.exception(
                "Failed to send nightly copilot email for session %s",
                session.session_id,
            )
            continue

        session.notification_email_sent_at = datetime.now(UTC)
        await upsert_chat_session(session)
        processed_count += 1

    return processed_count


async def send_nightly_copilot_emails() -> int:
    return await _send_nightly_copilot_emails()


async def consume_callback_token(
    token_id: str,
    user_id: str,
) -> CallbackTokenConsumeResult:
    token = await chat_db().get_chat_session_callback_token(token_id)
    if token is None or token.user_id != user_id:
        raise ValueError("Callback token not found")
    if token.expires_at <= datetime.now(UTC):
        raise ValueError("Callback token has expired")

    if token.consumed_session_id:
        return CallbackTokenConsumeResult(session_id=token.consumed_session_id)

    session = await create_chat_session(
        user_id,
        initial_messages=[
            ChatMessage(role="assistant", content=token.callback_session_message)
        ],
    )
    await chat_db().mark_chat_session_callback_token_consumed(
        token_id,
        session.session_id,
    )
    return CallbackTokenConsumeResult(session_id=session.session_id)
