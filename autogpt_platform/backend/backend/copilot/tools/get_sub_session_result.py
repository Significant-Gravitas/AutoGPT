"""Poll / wait on / cancel a sub-AutoPilot started by ``run_sub_session``.

Companion to :mod:`run_sub_session`. Operates on the sub's
``ChatSession`` directly — there is no separate registry. Ownership is
re-verified on every call by loading the ChatSession and comparing its
``user_id`` against the authenticated caller.

* **Wait** — subscribe to ``stream_registry`` for the session and drain
  until ``StreamFinish`` / ``StreamError`` (terminal) or the per-call
  cap fires. On terminal, the aggregated :class:`SessionResult` comes
  back in memory — no DB round-trip for the response content.
* **Just check** — ``wait_if_running=0`` skips the subscription. If the
  sub's last assistant message already looks terminal, returns
  ``completed`` with that content.
* **Cancel** — fan out a ``CancelCoPilotEvent`` on the shared cancel
  exchange. Whichever worker is running the sub breaks out of its
  stream and finalises the session as ``failed``.
"""

import json
import logging
import time
from typing import Any

from backend.copilot import stream_registry
from backend.copilot.executor.utils import enqueue_cancel_task
from backend.copilot.model import ChatSession, get_chat_session
from backend.copilot.sdk.session_waiter import (
    SessionOutcome,
    SessionResult,
    wait_for_session_result,
)
from backend.copilot.sdk.stream_accumulator import ToolCallEntry

from .base import BaseTool
from .models import (
    ErrorResponse,
    SubSessionProgressSnapshot,
    SubSessionStatusResponse,
    ToolResponseBase,
)
from .run_sub_session import (
    MAX_SUB_SESSION_WAIT_SECONDS,
    _sub_session_link,
    response_from_outcome,
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

        started_at = time.monotonic()

        if cancel:
            # Fan out the cancel event. Whichever worker is running the
            # sub will break out of its stream and finalise the session
            # as failed. Return "cancelled" immediately; the sub may
            # still emit a little more output before the worker notices,
            # but the agent doesn't need to wait for that.
            await enqueue_cancel_task(inner_session_id)
            return SubSessionStatusResponse(
                message="Sub-AutoPilot cancel requested.",
                session_id=session.session_id,
                status="cancelled",
                sub_session_id=inner_session_id,
                sub_autopilot_session_id=inner_session_id,
                sub_autopilot_session_link=_sub_session_link(inner_session_id),
                elapsed_seconds=0.0,
            )

        # If a turn is currently running for this session (stream registry
        # meta shows status=running), we can NOT short-circuit on the
        # persisted last assistant message — that message belongs to a
        # PRIOR turn, and surfacing it here would hand the caller stale
        # data while the new turn is mid-flight (sentry r3105409601).
        # Only short-circuit when there's no active turn AND the last
        # persisted message already looks terminal.
        effective_wait = max(0, min(wait_if_running, MAX_SUB_SESSION_WAIT_SECONDS))
        registry_session = await stream_registry.get_session(inner_session_id)
        turn_in_flight = registry_session is not None and (
            getattr(registry_session, "status", "") == "running"
        )
        terminal_result = None if turn_in_flight else _already_terminal_result(sub)
        outcome: SessionOutcome
        result: SessionResult
        if terminal_result is not None:
            outcome, result = "completed", terminal_result
        elif effective_wait > 0:
            outcome, result = await wait_for_session_result(
                session_id=inner_session_id,
                user_id=user_id,
                timeout=effective_wait,
            )
        else:
            outcome, result = "running", SessionResult()

        elapsed = time.monotonic() - started_at

        if outcome == "running" and include_progress:
            # Running + caller wants progress — hand-assemble the response
            # with the progress snapshot attached. response_from_outcome
            # doesn't carry progress, so we build the response here.
            progress = await _build_progress_snapshot(inner_session_id)
            link = _sub_session_link(inner_session_id)
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

        return response_from_outcome(
            outcome=outcome,
            result=result,
            inner_session_id=inner_session_id,
            parent_session_id=session.session_id,
            elapsed=elapsed,
        )


def _already_terminal_result(sub: ChatSession) -> SessionResult | None:
    """Rebuild the aggregated result from the sub's persisted last turn,
    when the last message is a terminal assistant message.

    Lets ``get_sub_session_result`` short-circuit the subscribe+wait
    when the agent polls well after the sub actually finished (a common
    case when the user pauses and later asks "what's the result?").
    Returns ``None`` if the last message isn't terminal.
    """
    if not sub.messages:
        return None
    last = sub.messages[-1]
    if last.role != "assistant":
        return None
    if not last.content and not last.tool_calls:
        return None
    result = SessionResult()
    result.response_text = last.content or ""
    # Persisted tool calls are OpenAI-shape dicts; translate to
    # ToolCallEntry so the downstream ``response_from_outcome`` can
    # ``.model_dump()`` them uniformly with the live-drain path.
    for tc in last.tool_calls or []:
        fn = tc.get("function") or {}
        result.tool_calls.append(
            ToolCallEntry(
                tool_call_id=tc.get("id", ""),
                tool_name=fn.get("name") or tc.get("name") or "",
                input=fn.get("arguments") or tc.get("arguments") or tc.get("input"),
                output=tc.get("output"),
                success=tc.get("success"),
            )
        )
    return result


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
