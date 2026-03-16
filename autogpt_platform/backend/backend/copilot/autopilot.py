"""Autopilot public API — thin facade re-exporting from sub-modules.

Implementation is split by responsibility:
- autopilot_prompts: constants, prompt templates, context builders
- autopilot_dispatch: timezone helpers, session creation, dispatch/scheduling
- autopilot_completion: completion report extraction, repair, handler
- autopilot_email: email sending, link building, notification sweep
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime

from pydantic import BaseModel

from backend.copilot.autopilot_completion import (  # noqa: F401
    CompletionReportToolCall,
    CompletionReportToolCallFunction,
    ToolOutputEnvelope,
    _build_completion_report_repair_message,
    _extract_completion_report_from_session,
    _get_pending_approval_metadata,
    _queue_completion_report_repair,
    handle_non_manual_session_completion,
)
from backend.copilot.autopilot_dispatch import (  # noqa: F401
    _bucket_end_for_now,
    _create_autopilot_session,
    _crosses_local_midnight,
    _enqueue_session_turn,
    _resolve_timezone_name,
    _session_exists_for_execution_tag,
    _try_create_callback_session,
    _try_create_invite_cta_session,
    _try_create_nightly_session,
    _user_has_recent_manual_message,
    _user_has_session_since,
    dispatch_nightly_copilot,
    get_callback_execution_tag,
    get_graph_exec_id_for_session,
    get_invite_cta_execution_tag,
    get_nightly_execution_tag,
    trigger_autopilot_session_for_user,
)
from backend.copilot.autopilot_email import (  # noqa: F401
    _build_callback_link,
    _build_session_link,
    _create_callback_token,
    _get_completion_email_template_name,
    _send_completion_email,
    _send_nightly_copilot_emails,
    _split_email_paragraphs,
    send_nightly_copilot_emails,
)
from backend.copilot.autopilot_prompts import (  # noqa: F401
    AUTOPILOT_CALLBACK_EMAIL_TEMPLATE,
    AUTOPILOT_CALLBACK_TAG,
    AUTOPILOT_DISABLED_TOOLS,
    AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE,
    AUTOPILOT_INVITE_CTA_TAG,
    AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE,
    AUTOPILOT_NIGHTLY_TAG_PREFIX,
    DEFAULT_AUTOPILOT_CALLBACK_SYSTEM_PROMPT,
    DEFAULT_AUTOPILOT_INVITE_CTA_SYSTEM_PROMPT,
    DEFAULT_AUTOPILOT_NIGHTLY_SYSTEM_PROMPT,
    INTERNAL_TAG_RE,
    MAX_COMPLETION_REPORT_REPAIRS,
    _build_autopilot_system_prompt,
    _format_start_type_label,
    _get_recent_manual_session_context,
    _get_recent_sent_email_context,
    _get_recent_session_summary_context,
    strip_internal_content,
    wrap_internal_message,
)
from backend.copilot.model import ChatMessage, create_chat_session
from backend.data.db_accessors import chat_db

# -- re-exports from sub-modules (preserves existing import paths) ---------- #


logger = logging.getLogger(__name__)


class CallbackTokenConsumeResult(BaseModel):
    session_id: str


async def consume_callback_token(
    token_id: str,
    user_id: str,
) -> CallbackTokenConsumeResult:
    """Consume a callback token and return the resulting session.

    Uses an atomic consume-and-update to prevent the TOCTOU race where two
    concurrent requests could each see the token as unconsumed and create
    duplicate sessions.
    """
    db = chat_db()
    token = await db.get_chat_session_callback_token(token_id)
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

    # Atomically mark consumed — if another request already consumed the token
    # concurrently, the DB will have a non-null consumed_session_id; we re-read
    # and return the winner's session instead.
    await db.mark_chat_session_callback_token_consumed(
        token_id,
        session.session_id,
    )

    # Re-read to see if we won the race
    refreshed = await db.get_chat_session_callback_token(token_id)
    if (
        refreshed
        and refreshed.consumed_session_id
        and refreshed.consumed_session_id != session.session_id
    ):
        return CallbackTokenConsumeResult(session_id=refreshed.consumed_session_id)

    return CallbackTokenConsumeResult(session_id=session.session_id)
