from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

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
from backend.copilot.model import ChatSession, get_chat_session, upsert_chat_session
from backend.copilot.session_types import ChatSessionStartType
from backend.data.db_accessors import chat_db, user_db
from backend.notifications.email import EmailSender
from backend.util.settings import Settings
from backend.util.url import get_frontend_base_url

logger = logging.getLogger(__name__)
settings = Settings()


def _split_email_paragraphs(text: str | None) -> list[str]:
    return [segment.strip() for segment in (text or "").splitlines() if segment.strip()]


# --------------- link builders --------------- #


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


# --------------- callback token --------------- #


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

    # EmailSender.send_template is synchronous (blocking HTTP call to Postmark).
    # Run it in a thread to avoid blocking the async event loop.
    sender = EmailSender()
    await asyncio.to_thread(
        sender.send_template,
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


# --------------- email sweep --------------- #


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
