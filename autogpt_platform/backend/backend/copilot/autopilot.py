"""Autopilot public API — thin facade re-exporting from sub-modules.

Implementation is split by responsibility:
- autopilot_prompts: constants, prompt templates, context builders
- autopilot_dispatch: timezone helpers, session creation, dispatch/scheduling
- autopilot_completion: completion report extraction, repair, handler
- autopilot_email: email sending, link building, notification sweep
"""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import BaseModel

from backend.copilot.autopilot_completion import (
    CompletionReportToolCall,
    CompletionReportToolCallFunction,
    ToolOutputEnvelope,
    handle_non_manual_session_completion,
)
from backend.copilot.autopilot_dispatch import (
    dispatch_nightly_copilot,
    get_callback_execution_tag,
    get_graph_exec_id_for_session,
    get_invite_cta_execution_tag,
    get_nightly_execution_tag,
    trigger_autopilot_session_for_user,
)
from backend.copilot.autopilot_email import (
    PendingCopilotEmailSweepResult,
    send_nightly_copilot_emails,
    send_pending_copilot_emails_for_user,
)
from backend.copilot.autopilot_prompts import (
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
    strip_internal_content,
    unwrap_internal_content,
    wrap_internal_message,
)
from backend.copilot.model import ChatMessage, create_chat_session, delete_chat_session
from backend.data.db_accessors import chat_db

__all__ = [
    # autopilot_completion
    "CompletionReportToolCall",
    "CompletionReportToolCallFunction",
    "ToolOutputEnvelope",
    "handle_non_manual_session_completion",
    # autopilot_dispatch
    "dispatch_nightly_copilot",
    "get_callback_execution_tag",
    "get_graph_exec_id_for_session",
    "get_invite_cta_execution_tag",
    "get_nightly_execution_tag",
    "trigger_autopilot_session_for_user",
    # autopilot_email
    "PendingCopilotEmailSweepResult",
    "send_nightly_copilot_emails",
    "send_pending_copilot_emails_for_user",
    # autopilot_prompts
    "AUTOPILOT_CALLBACK_EMAIL_TEMPLATE",
    "AUTOPILOT_CALLBACK_TAG",
    "AUTOPILOT_DISABLED_TOOLS",
    "AUTOPILOT_INVITE_CTA_EMAIL_TEMPLATE",
    "AUTOPILOT_INVITE_CTA_TAG",
    "AUTOPILOT_NIGHTLY_EMAIL_TEMPLATE",
    "AUTOPILOT_NIGHTLY_TAG_PREFIX",
    "DEFAULT_AUTOPILOT_CALLBACK_SYSTEM_PROMPT",
    "DEFAULT_AUTOPILOT_INVITE_CTA_SYSTEM_PROMPT",
    "DEFAULT_AUTOPILOT_NIGHTLY_SYSTEM_PROMPT",
    "INTERNAL_TAG_RE",
    "MAX_COMPLETION_REPORT_REPAIRS",
    "strip_internal_content",
    "unwrap_internal_content",
    "wrap_internal_message",
    # local
    "CallbackTokenConsumeResult",
    "consume_callback_token",
]


class CallbackTokenConsumeResult(BaseModel):
    session_id: str


async def consume_callback_token(
    token_id: str,
    user_id: str,
) -> CallbackTokenConsumeResult:
    """Consume a callback token and return the resulting session.

    Uses an atomic conditional UPDATE (WHERE consumedSessionId IS NULL) to
    prevent TOCTOU races. Only the request that wins the conditional update
    keeps its session; the loser cleans up its orphaned session.
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

    consumed = await db.mark_chat_session_callback_token_consumed(
        token_id,
        session.session_id,
    )

    if consumed:
        return CallbackTokenConsumeResult(session_id=session.session_id)

    # Lost the race — another request consumed the token first.
    # Clean up the orphaned session and return the winner's session.
    await delete_chat_session(session.session_id, user_id)

    refreshed = await db.get_chat_session_callback_token(token_id)
    if refreshed and refreshed.consumed_session_id:
        return CallbackTokenConsumeResult(session_id=refreshed.consumed_session_id)

    raise ValueError("Callback token was consumed but session ID is missing")
