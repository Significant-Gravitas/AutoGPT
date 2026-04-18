"""Poll / wait on / cancel a sub-AutoPilot started by ``run_sub_session``.

Companion to :mod:`run_sub_session`. Both tools operate on the sub's
``ChatSession`` directly — there is no separate registry. Ownership is
re-verified on every call by loading the ChatSession and comparing its
``user_id`` against the authenticated caller.

* **Wait** — subscribe to ``stream_registry`` for the session and drain
  until ``StreamFinish`` / ``StreamError`` (terminal) or the per-call
  cap fires. On cap fire, return ``status="running"`` with the sub's
  session_id so the agent can re-poll.
* **Just check** — ``wait_if_running=0`` skips the subscription and
  returns whatever the session currently shows.
* **Cancel** — fan out a ``CancelCoPilotEvent`` on the shared cancel
  exchange. Whichever worker is running the sub breaks out of its
  stream and finalises the session as ``failed``.
"""

import json
import logging
import time
from typing import Any

from backend.copilot.constants import MAX_TOOL_WAIT_SECONDS
from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import ChatSession, get_chat_session
from backend.copilot.sdk.session_waiter import (
    SessionOutcome,
    wait_for_session_completion,
)

from .base import BaseTool
from .models import (
    ErrorResponse,
    SubSessionProgressSnapshot,
    SubSessionStatusResponse,
    ToolResponseBase,
)
from .run_sub_session import (
    MAX_SUB_SESSION_WAIT_SECONDS,
    _response_from_outcome,
    _sub_session_link,
)

logger = logging.getLogger(__name__)

# Cap on how many recent messages we echo back in a progress snapshot.
_PROGRESS_MESSAGE_LIMIT = 5
_PROGRESS_CONTENT_PREVIEW_CHARS = 400


class GetSubSessionResultTool(BaseTool):
    """Wait for, inspect, or cancel a sub-AutoPilot."""

    @property
    def name(self) -> str:
        return "get_sub_session_result"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Poll / wait / cancel a sub-AutoPilot from run_sub_session. "
            f"Waits up to wait_if_running sec (max {MAX_SUB_SESSION_WAIT_SECONDS}); "
            "cancel=true aborts; include_progress=true returns recent messages "
            "from the still-running sub. Works across turns."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "sub_session_id": {
                    "type": "string",
                    "description": (
                        "The sub's session_id returned by run_sub_session "
                        "(also accepted: sub_autopilot_session_id — same value)."
                    ),
                },
                "wait_if_running": {
                    "type": "integer",
                    "description": (
                        f"Seconds to wait. 0 = just check. Clamped to "
                        f"{MAX_SUB_SESSION_WAIT_SECONDS}."
                    ),
                    "default": 60,
                },
                "cancel": {
                    "type": "boolean",
                    "description": (
                        "Cancel the sub; takes precedence over wait_if_running."
                    ),
                    "default": False,
                },
                "include_progress": {
                    "type": "boolean",
                    "description": (
                        "Populate progress.last_messages when status=running."
                    ),
                    "default": False,
                },
            },
            "required": ["sub_session_id"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        sub_session_id: str = "",
        wait_if_running: int = 60,
        cancel: bool = False,
        include_progress: bool = False,
        **kwargs,
    ) -> ToolResponseBase:
        inner_session_id = sub_session_id.strip()
        if not inner_session_id:
            return ErrorResponse(
                message="sub_session_id is required",
                session_id=session.session_id,
            )
        if user_id is None:
            return ErrorResponse(
                message="Authentication required",
                session_id=session.session_id,
            )

        # Ownership check on every call — loads the ChatSession and
        # confirms the caller owns it. Returning the same "not found"
        # shape for "doesn't exist" and "belongs to someone else" avoids
        # leaking session existence.
        sub = await get_chat_session(inner_session_id)
        if sub is None or sub.user_id != user_id:
            return ErrorResponse(
                message=(
                    f"No sub-session with id {inner_session_id}. It may have "
                    "never existed or belongs to another user."
                ),
                session_id=session.session_id,
            )

        link = _sub_session_link(inner_session_id)
        started_at = time.monotonic()

        if cancel:
            # Fan out the cancel event. Whichever worker is running the
            # sub will break out of its stream and finalise the session
            # as failed. We return "cancelled" immediately; the sub may
            # still emit a little more output before the worker notices,
            # but the agent doesn't need to wait for that.
            await enqueue_cancel_task(inner_session_id)
            return SubSessionStatusResponse(
                message="Sub-AutoPilot cancel requested.",
                session_id=session.session_id,
                status="cancelled",
                sub_session_id=inner_session_id,
                sub_autopilot_session_id=inner_session_id,
                sub_autopilot_session_link=link,
                elapsed_seconds=0.0,
            )

        # If the sub's last turn is already terminal (assistant message
        # present with no pending user message), skip the wait entirely.
        # Otherwise, subscribe to stream_registry and wait for the
        # terminal event or the cap.
        effective_wait = max(0, min(wait_if_running, MAX_SUB_SESSION_WAIT_SECONDS))
        outcome: SessionOutcome
        if _looks_terminal(sub):
            outcome = "completed"
        elif effective_wait > 0:
            outcome = await wait_for_session_completion(
                session_id=inner_session_id,
                user_id=user_id,
                timeout=effective_wait,
            )
        else:
            outcome = "running"

        elapsed = time.monotonic() - started_at

        if outcome == "running" and include_progress:
            # Running + caller wants progress — hand-assemble the response
            # with progress attached. (`_response_from_outcome` doesn't
            # carry the progress snapshot.)
            progress = await _build_progress_snapshot(inner_session_id)
            return SubSessionStatusResponse(
                message=(
                    f"Sub-AutoPilot still running after {elapsed:.0f}s."
                    f"{f' Watch live at {link}.' if link else ''} "
                    "Call again to keep waiting, or cancel=true to abort."
                ),
                session_id=session.session_id,
                status="running",
                sub_session_id=inner_session_id,
                sub_autopilot_session_id=inner_session_id,
                sub_autopilot_session_link=link,
                elapsed_seconds=round(elapsed, 2),
                progress=progress,
            )

        response = await _response_from_outcome(
            outcome=outcome,
            inner_session_id=inner_session_id,
            parent_session=session,
            elapsed=elapsed,
        )
        return response


def _looks_terminal(sub: ChatSession) -> bool:
    """Cheap check for "has the sub's latest turn already finished?".

    Returns True when the last assistant message carries content. This
    avoids subscribing to stream_registry for a session whose terminal
    event fired before we arrived (common when the agent polls long
    after the sub actually completed).
    """
    if not sub.messages:
        return False
    last = sub.messages[-1]
    if last.role != "assistant":
        return False
    return bool(last.content) or bool(last.tool_calls)


async def _build_progress_snapshot(
    inner_session_id: str | None,
) -> SubSessionProgressSnapshot | None:
    """Read the sub's ChatSession and return a preview of recent messages.

    Returns ``None`` silently on lookup failure — progress is best-effort;
    missing progress shouldn't abort the normal ``still running`` response.
    """
    if not inner_session_id:
        return None
    try:
        sub = await get_chat_session(inner_session_id)
        if sub is None:
            return None
        messages = list(sub.messages)
    except Exception as exc:  # best-effort peek
        logger.debug(
            "Progress snapshot unavailable for sub %s: %s",
            inner_session_id,
            exc,
        )
        return None

    tail = messages[-_PROGRESS_MESSAGE_LIMIT:]
    previews: list[dict[str, Any]] = []
    for msg in tail:
        content = getattr(msg, "content", "") or ""
        if not isinstance(content, str):
            try:
                content = json.dumps(content, default=str)
            except (TypeError, ValueError):
                content = str(content)
        if len(content) > _PROGRESS_CONTENT_PREVIEW_CHARS:
            content = content[:_PROGRESS_CONTENT_PREVIEW_CHARS] + "…"
        previews.append(
            {
                "role": getattr(msg, "role", "unknown"),
                "content": content,
            }
        )
    return SubSessionProgressSnapshot(
        message_count=len(messages),
        last_messages=previews,
    )
