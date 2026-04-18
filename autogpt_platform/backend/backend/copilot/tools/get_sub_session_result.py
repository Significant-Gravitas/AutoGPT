"""Poll / wait on / cancel a sub-AutoPilot started by ``run_sub_session``.

The companion tool to :mod:`run_sub_session`. All three operations (wait,
just-check, cancel) go through the same registry entry — ownership is scoped
to the authenticated user so cross-user access is impossible.
"""

import asyncio
import logging
import time
from typing import Any

from backend.copilot.model import ChatSession
from backend.copilot.sdk.sub_session_registry import (
    MAX_SUB_SESSION_WAIT_SECONDS,
    cancel_sub_session,
    get_sub_session,
    prune_finished,
    unregister_sub_session,
)

from .base import BaseTool
from .models import (
    ErrorResponse,
    SubSessionProgressSnapshot,
    SubSessionStatusResponse,
    ToolResponseBase,
)
from .run_sub_session import _response_from_task, _sub_session_link  # reuse the mapping

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
                    "description": "id from run_sub_session (e.g. 'sub-abc123').",
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
        prune_finished()

        if not sub_session_id.strip():
            return ErrorResponse(
                message="sub_session_id is required",
                session_id=session.session_id,
            )
        if user_id is None:
            return ErrorResponse(
                message="Authentication required",
                session_id=session.session_id,
            )

        entry = get_sub_session(sub_session_id, user_id)
        if entry is None:
            return ErrorResponse(
                message=(
                    f"No sub-session with id {sub_session_id}. It may have "
                    "finished and been purged, never existed, or belongs to "
                    "another user."
                ),
                session_id=session.session_id,
            )

        task: asyncio.Task = entry["task"]

        inner_session_id = entry.get("inner_session_id")

        if cancel:
            # Race guard: if the task finished before the cancel was
            # requested, prefer returning the real result over "cancelled".
            if task.done():
                return _finalize(session, sub_session_id, entry, task)
            cancel_sub_session(sub_session_id, user_id)
            return SubSessionStatusResponse(
                message="Sub-AutoPilot cancelled.",
                session_id=session.session_id,
                status="cancelled",
                sub_session_id=sub_session_id,
                sub_autopilot_session_id=inner_session_id,
                sub_autopilot_session_link=_sub_session_link(inner_session_id),
                elapsed_seconds=round(time.monotonic() - entry["started_at"], 2),
            )

        if task.done():
            return _finalize(session, sub_session_id, entry, task)

        effective_wait = max(0, min(wait_if_running, MAX_SUB_SESSION_WAIT_SECONDS))
        if effective_wait > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=effective_wait)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                raise
            if task.done():
                return _finalize(session, sub_session_id, entry, task)

        elapsed = time.monotonic() - entry["started_at"]
        progress = None
        if include_progress:
            progress = await _build_progress_snapshot(inner_session_id)
        link = _sub_session_link(inner_session_id)
        link_hint = f" Watch live at {link}." if link else ""
        return SubSessionStatusResponse(
            message=(
                f"Sub-AutoPilot still running after {elapsed:.0f}s total.{link_hint} "
                "Call again to keep waiting, or cancel=true to abort."
            ),
            session_id=session.session_id,
            status="running",
            sub_session_id=sub_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
            progress=progress,
        )


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
        from backend.copilot.model import get_chat_session  # noqa: PLC0415

        sub = await get_chat_session(inner_session_id)
        if sub is None:
            return None
        messages = list(sub.messages)
    except Exception as exc:  # pragma: no cover — best-effort peek
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
                import json as _json  # noqa: PLC0415

                content = _json.dumps(content, default=str)
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


def _finalize(
    session: ChatSession,
    sub_session_id: str,
    entry: dict[str, Any],
    task: asyncio.Task,
) -> ToolResponseBase:
    """Map a finished task to a response and unregister its entry."""
    elapsed = time.monotonic() - entry["started_at"]
    response = _response_from_task(
        task=task,
        sub_session_id=sub_session_id,
        session=session,
        elapsed=elapsed,
        inner_session_id_when_running=entry.get("inner_session_id"),
    )
    # Terminal state consumed — drop from registry (prune_finished would
    # catch it eventually anyway, but explicit removal keeps it tight).
    if isinstance(response, SubSessionStatusResponse) and response.status in (
        "completed",
        "cancelled",
        "error",
    ):
        unregister_sub_session(sub_session_id)
    return response
