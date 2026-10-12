"""Rich content for copilot completion notifications.

The backend composes the display strings so every OS surface (desktop
``Notification`` fallback, PWA service worker) renders the same copy without
duplicating the formatting rules: ``title`` names who did it (the expert, or
AutoPilot), ``body`` says what happened in a few words, ``icon`` carries the
expert's avatar, and ``url`` is the click-through path.

Enrichment is best-effort and must never break the completion path: every
lookup failure degrades to the generic copy the clients already fall back to.
"""

import logging
from typing import Literal
from urllib.parse import urlparse

from backend.api.model import CopilotCompletionPayload
from backend.data.db_accessors import chat_db, experts_db
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)

logger = logging.getLogger(__name__)

AUTOPILOT_DISPLAY_NAME = "AutoPilot"

FALLBACK_BODY_COMPLETED = "Task completed"
FALLBACK_BODY_FAILED = "Task failed"

MAX_TITLE_LEN = 80
MAX_BODY_LEN = 160
MAX_QUESTION_LEN = 120


def _truncate(text: str, max_len: int) -> str:
    """Shorten to ``max_len`` characters, marking the cut with an ellipsis."""
    cleaned = " ".join(text.split())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1].rstrip() + "…"


def build_completion_content(
    *,
    status: Literal["completed", "failed"],
    session_title: str | None,
    expert_name: str | None,
    pending_question: str | None,
) -> tuple[str, str]:
    """Compose the notification title and body for a finished turn.

    Pure function over already-fetched values so the copy rules are unit
    tested without touching the database.
    """
    title = (
        _truncate(expert_name, MAX_TITLE_LEN) if expert_name else AUTOPILOT_DISPLAY_NAME
    )
    if status == "failed":
        body = (
            f"Couldn't finish: {session_title}"
            if session_title
            else FALLBACK_BODY_FAILED
        )
    elif pending_question:
        body = f"Needs your answer: {_truncate(pending_question, MAX_QUESTION_LEN)}"
    elif session_title:
        body = f"Finished: {session_title}"
    else:
        body = FALLBACK_BODY_COMPLETED
    return title, _truncate(body, MAX_BODY_LEN)


def _safe_avatar_url(avatar_url: str | None) -> str | None:
    """Keep only avatar values a notification icon may safely load.

    Same rule as the expert API's own validator: an absolute https URL or a
    site-relative path. Anything else (notably ``data:``/``javascript:`` and
    protocol-relative URLs) is dropped so a legacy row can never smuggle an
    unexpected fetch into the notification surface.
    """
    if not avatar_url:
        return None
    stripped = avatar_url.strip()
    if not stripped or any(char in stripped for char in ("\\", "\t", "\r", "\n")):
        return None
    if stripped.startswith("//"):
        return None
    if stripped.startswith("/"):
        return stripped
    parsed = urlparse(stripped)
    if parsed.scheme != "https" or not parsed.netloc:
        return None
    return stripped


async def build_copilot_completion_payload(
    session_id: str, user_id: str, status: Literal["completed", "failed"]
) -> CopilotCompletionPayload:
    """Build the completion payload, enriching it with session/expert details.

    Each lookup is individually guarded: a missing session still notifies
    (with generic copy), and a missing expert still names the session's work
    under the AutoPilot title.
    """
    session_title: str | None = None
    pending_question: str | None = None
    expert_id: str | None = None
    try:
        session = await chat_db().get_chat_session_metadata(session_id)
    except Exception as e:
        logger.warning(
            "Completion notification for session %s: session lookup failed: %s",
            session_id,
            e,
        )
        session = None
    if session is not None:
        if session.user_id != user_id:
            logger.warning(
                "Completion notification for session %s: owner mismatch, "
                "skipping enrichment",
                session_id,
            )
            session = None
        else:
            session_title = session.title
            expert_id = session.expert_id
            if session.metadata.pending_question is not None:
                pending_question = session.metadata.pending_question.text

    expert_name: str | None = None
    avatar_url: str | None = None
    if expert_id:
        try:
            expert = await experts_db().get_expert(
                user_id, expert_id, include_workflows=False
            )
        except Exception as e:
            logger.warning(
                "Completion notification for session %s: expert lookup failed: %s",
                session_id,
                e,
            )
            expert = None
        if expert is not None:
            expert_name = expert.name
            avatar_url = _safe_avatar_url(expert.avatar_url)

    title, body = build_completion_content(
        status=status,
        session_title=session_title,
        expert_name=expert_name,
        pending_question=pending_question if status == "completed" else None,
    )
    return CopilotCompletionPayload(
        type="copilot_completion",
        event="session_completed",
        session_id=session_id,
        status=status,
        title=title,
        body=body,
        icon=avatar_url,
        url=f"/copilot?sessionId={session_id}",
        expert_name=expert_name,
        session_title=session_title,
    )


async def publish_copilot_completion(
    session_id: str,
    user_id: str,
    status: Literal["completed", "failed"],
    *,
    cancelled: bool = False,
) -> None:
    """Publish the completion notification, never raising.

    A user-initiated Stop notifies nothing: the user took the action
    themselves, so an OS popup can only ever be noise. (It also skips the
    enrichment reads below.)

    Even payload construction is guarded: if enrichment itself blows up in
    an unexpected way, the bare payload (today's behavior) still goes out.
    """
    if cancelled:
        return
    try:
        payload = await build_copilot_completion_payload(session_id, user_id, status)
    except Exception as e:
        logger.warning(
            "Completion notification for session %s: enrichment failed, "
            "publishing bare payload: %s",
            session_id,
            e,
        )
        payload = CopilotCompletionPayload(
            type="copilot_completion",
            event="session_completed",
            session_id=session_id,
            status=status,
        )
    try:
        await AsyncRedisNotificationEventBus().publish(
            NotificationEvent(user_id=user_id, payload=payload)
        )
    except Exception as e:
        logger.warning(
            "Failed to publish copilot completion notification for session %s: %s",
            session_id,
            e,
        )
