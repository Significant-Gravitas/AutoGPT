from __future__ import annotations

import logging
from datetime import UTC, datetime
from urllib.parse import urlencode

import bleach
from markdown_it import MarkdownIt
from postmarker.exceptions import PostmarkerException
from pydantic import BaseModel

from backend.copilot import stream_registry
from backend.copilot.autopilot_completion import (
    _get_pending_approval_metadata,
    _queue_completion_report_repair,
)
from backend.copilot.autopilot_prompts import (
    AUTOPILOT_CALLBACK_EMAIL_TEMPLATE,
    AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE,
    AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE,
    MAX_COMPLETION_REPORT_REPAIRS,
)
from backend.copilot.model import (
    ChatSession,
    ChatSessionInfo,
    get_chat_session,
    update_session_title,
    upsert_chat_session,
)
from backend.copilot.service import _generate_session_title
from backend.copilot.session_types import ChatSessionStartType
from backend.data.db_accessors import chat_db, user_db
from backend.notifications.email import EmailSender
from backend.util.url import get_frontend_base_url

logger = logging.getLogger(__name__)
PENDING_NOTIFICATION_SWEEP_LIMIT = 200
_NON_FATAL_EMAIL_SWEEP_ERRORS = (
    ConnectionError,
    OSError,
    PostmarkerException,
    TimeoutError,
    ValueError,
)

_md = MarkdownIt("commonmark", {"html": False})

_EMAIL_INLINE_STYLES: list[tuple[str, str]] = [
    (
        "<p>",
        '<p style="font-size: 15px; line-height: 170%;'
        " margin-top: 0; margin-bottom: 16px;"
        ' color: #1F1F20;">',
    ),
    (
        "<li>",
        '<li style="font-size: 15px; line-height: 170%;'
        " margin-top: 0; margin-bottom: 8px;"
        ' color: #1F1F20;">',
    ),
    (
        "<ul>",
        '<ul style="padding: 0 0 0 24px;' ' margin-top: 0; margin-bottom: 16px;">',
    ),
    (
        "<ol>",
        '<ol style="padding: 0 0 0 24px;' ' margin-top: 0; margin-bottom: 16px;">',
    ),
    (
        "<a ",
        '<a style="color: #7733F5;'
        " text-decoration: underline;"
        ' font-weight: 500;" ',
    ),
    (
        "<h2>",
        '<h2 style="font-size: 20px; font-weight: 600;'
        ' margin-top: 0; margin-bottom: 12px; color: #1F1F20;">',
    ),
    (
        "<h3>",
        '<h3 style="font-size: 18px; font-weight: 600;'
        ' margin-top: 0; margin-bottom: 12px; color: #1F1F20;">',
    ),
]


_BLEACH_ALLOWED_TAGS = [
    "p",
    "br",
    "strong",
    "b",
    "em",
    "i",
    "ul",
    "ol",
    "li",
    "a",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "code",
    "pre",
    "blockquote",
]
_BLEACH_ALLOWED_ATTRS = {"a": ["href", "title"]}
_BLEACH_ALLOWED_PROTOCOLS = ["http", "https", "mailto"]


def _markdown_to_email_html(text: str | None) -> str:
    """Convert markdown text to email-safe HTML with inline styles.

    The output is sanitized via bleach to strip any raw HTML that the LLM may
    have injected (e.g. ``<img onerror=...>``, phishing links, CSS exfil).
    """
    if not text or not text.strip():
        return ""
    html = _md.render(text.strip())
    html = bleach.clean(
        html,
        tags=_BLEACH_ALLOWED_TAGS,
        attributes=_BLEACH_ALLOWED_ATTRS,
        protocols=_BLEACH_ALLOWED_PROTOCOLS,
        strip=True,
    )
    for tag, styled_tag in _EMAIL_INLINE_STYLES:
        html = html.replace(tag, styled_tag)
    return html.strip()


# --------------- link builders --------------- #


def _build_session_link(session_id: str, *, show_autopilot: bool) -> str:
    base_url = get_frontend_base_url()
    params = {"sessionId": session_id}
    if show_autopilot:
        params["showAutopilot"] = "1"
    return f"{base_url}/copilot?{urlencode(params)}"


def _get_completion_email_template_name(start_type: ChatSessionStartType) -> str:
    if start_type == ChatSessionStartType.AUTOPILOT_NIGHTLY:
        return AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE
    if start_type == ChatSessionStartType.AUTOPILOT_CALLBACK:
        return AUTOPILOT_CALLBACK_EMAIL_TEMPLATE
    if start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA:
        return AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE
    raise ValueError(f"Unsupported start type for completion email: {start_type}")


class PendingCopilotEmailSweepResult(BaseModel):
    candidate_count: int = 0
    processed_count: int = 0
    sent_count: int = 0
    skipped_count: int = 0
    repair_queued_count: int = 0
    running_count: int = 0
    failed_count: int = 0


async def _ensure_session_title_for_completed_session(session: ChatSession) -> None:
    if session.title or not session.user_id:
        return

    report = session.completion_report
    if report is None:
        return

    title = report.email_title.strip() if report.email_title else ""
    if not title:
        title_seed = report.email_body or report.thoughts
        if title_seed:
            generated_title = await _generate_session_title(
                title_seed,
                user_id=session.user_id,
                session_id=session.session_id,
            )
            title = generated_title.strip() if generated_title else ""

    if not title:
        return

    updated = await update_session_title(
        session.session_id,
        session.user_id,
        title,
        only_if_empty=True,
    )
    if updated:
        session.title = title


# --------------- send email --------------- #


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
    cta_url = _build_session_link(session.session_id, show_autopilot=True)
    if approval_cta:
        cta_label = "Review in Copilot"
    else:
        cta_label = (
            "Try Copilot"
            if session.start_type == ChatSessionStartType.AUTOPILOT_INVITE_CTA
            else "Open Copilot"
        )

    sender = EmailSender()
    await sender.send_template(
        user_email=user.email,
        subject=report.email_title or "Autopilot update",
        template_name=template_name,
        data={
            "email_body_html": _markdown_to_email_html(report.email_body),
            "approval_summary_html": _markdown_to_email_html(report.approval_summary),
            "cta_url": cta_url,
            "cta_label": cta_label,
        },
    )


# --------------- email sweep --------------- #


async def _process_pending_copilot_email_candidates(
    candidates: list[ChatSessionInfo],
) -> PendingCopilotEmailSweepResult:
    result = PendingCopilotEmailSweepResult(candidate_count=len(candidates))

    for candidate in candidates:
        session = await get_chat_session(candidate.session_id)
        if session is None or session.is_manual:
            continue

        active = await stream_registry.get_session(session.session_id)
        is_running = active is not None and active.status == "running"
        if is_running:
            result.running_count += 1
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
                result.repair_queued_count += 1
                continue

            session.completed_at = session.completed_at or datetime.now(UTC)
            session.completion_report_repair_queued_at = None
            session.notification_email_skipped_at = datetime.now(UTC)
            await upsert_chat_session(session)
            result.skipped_count += 1
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

        await _ensure_session_title_for_completed_session(session)

        if not session.completion_report.should_notify_user:
            session.notification_email_skipped_at = datetime.now(UTC)
            await upsert_chat_session(session)
            result.skipped_count += 1
            continue

        try:
            await _send_completion_email(session)
        except _NON_FATAL_EMAIL_SWEEP_ERRORS:
            logger.exception(
                "Failed to send nightly copilot email for session %s",
                session.session_id,
            )
            # Without a persisted retry state, leave the session in a terminal
            # skipped state so the sweep does not pick it up forever.
            session.notification_email_skipped_at = datetime.now(UTC)
            await upsert_chat_session(session)
            result.failed_count += 1
            continue

        session.notification_email_sent_at = datetime.now(UTC)
        await upsert_chat_session(session)
        result.sent_count += 1

    result.processed_count = (
        result.sent_count + result.skipped_count + result.failed_count
    )
    return result


async def _send_nightly_copilot_emails() -> int:
    candidates = await chat_db().get_pending_notification_chat_sessions(
        limit=PENDING_NOTIFICATION_SWEEP_LIMIT
    )
    result = await _process_pending_copilot_email_candidates(candidates)
    return result.processed_count


async def send_nightly_copilot_emails() -> int:
    return await _send_nightly_copilot_emails()


async def send_pending_copilot_emails_for_user(
    user_id: str,
) -> PendingCopilotEmailSweepResult:
    candidates = await chat_db().get_pending_notification_chat_sessions_for_user(
        user_id,
        limit=PENDING_NOTIFICATION_SWEEP_LIMIT,
    )
    return await _process_pending_copilot_email_candidates(candidates)
