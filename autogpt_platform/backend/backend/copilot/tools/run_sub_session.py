"""Start a sub-AutoPilot conversation via the copilot_executor queue.

Mirror-image of ``run_agent`` + ``view_agent_output`` for copilot turns:

1. The tool creates (or validates ownership of) an inner ``ChatSession``
   and calls :func:`run_copilot_turn_via_queue` — the shared primitive
   that creates the stream-registry session meta, enqueues a
   ``CoPilotExecutionEntry``, and waits on the Redis stream until the
   terminal event arrives or the cap fires.
2. Any available ``copilot_executor`` worker claims the job, runs
   the SDK stream to completion, and publishes the final
   ``StreamFinish`` event on the session's Redis stream.
3. If the terminal event arrives in the wait window, the aggregated
   :class:`SessionResult` (response text, tool calls, usage) comes back
   in memory — no DB round-trip. Otherwise the tool returns
   ``status="running"`` + the sub's ``session_id`` and the agent polls
   via :mod:`get_sub_session_result`.

Compared to the prior in-process ``asyncio.Task`` implementation this
gives us deploy/crash resilience, natural load balancing across
workers, and a uniform conversation model — a sub is just another
copilot turn routed through the same queue and event bus as every
other turn.
"""

import logging
import time
from typing import Any

from backend.copilot.active_turns import concurrent_turn_limit_message
from backend.copilot.constants import MAX_TOOL_WAIT_SECONDS
from backend.copilot.context import get_current_permissions
from backend.copilot.model import ChatSession, create_chat_session, get_chat_session
from backend.copilot.sdk.session_waiter import (
    SessionOutcome,
    SessionResult,
    run_copilot_turn_via_queue,
)

from .base import BaseTool
from .models import ErrorResponse, SubSessionStatusResponse, ToolResponseBase

logger = logging.getLogger(__name__)


# Max wait for a single run_sub_session / get_sub_session_result call.
# Shared with every other long-running tool so the stream idle timeout's
# 2x headroom holds uniformly.
MAX_SUB_SESSION_WAIT_SECONDS = MAX_TOOL_WAIT_SECONDS


class RunSubSessionTool(BaseTool):
    """Delegate a task to a fresh sub-AutoPilot via the copilot_executor queue."""

    @property
    def name(self) -> str:
        return "run_sub_session"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Delegate a task to a fresh sub-AutoPilot. Runs on the copilot "
            "executor queue — survives tab-close AND worker restarts. Waits "
            f"up to wait_for_result sec (max {MAX_SUB_SESSION_WAIT_SECONDS}). "
            "If not done, returns status=running + sub_session_id — poll via "
            "get_sub_session_result."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "The task for the sub-AutoPilot to execute.",
                },
                "system_context": {
                    "type": "string",
                    "description": "Optional context prepended to the prompt.",
                    "default": "",
                },
                "sub_autopilot_session_id": {
                    "type": "string",
                    "description": ("Continue/queue-into a prior sub; empty = new."),
                    "default": "",
                },
                "wait_for_result": {
                    "type": "integer",
                    "description": (
                        "Seconds to wait inline. 0 = return immediately. "
                        f"Clamped to {MAX_SUB_SESSION_WAIT_SECONDS}."
                    ),
                    "default": 60,
                },
            },
            "required": ["prompt"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        *,
        prompt: str = "",
        system_context: str = "",
        sub_autopilot_session_id: str = "",
        wait_for_result: int = 60,
        **kwargs,
    ) -> ToolResponseBase:
        if not prompt.strip():
            return ErrorResponse(
                message="prompt is required",
                session_id=session.session_id,
            )
        if user_id is None:
            return ErrorResponse(
                message="Authentication required",
                session_id=session.session_id,
            )

        # Resolve the sub's ChatSession id — either resume an owned one or
        # create a fresh session that inherits the parent's dry_run so a
        # sub spawned inside a dry-run conversation doesn't silently
        # escalate to a live run.
        sub_session_param = sub_autopilot_session_id.strip()
        if sub_session_param:
            owned = await get_chat_session(sub_session_param)
            if owned is None or owned.user_id != user_id:
                return ErrorResponse(
                    message=(
                        f"sub_autopilot_session_id {sub_session_param} is not "
                        "a session you own. Leave empty to start a fresh sub, "
                        "or pass a session_id returned by a previous "
                        "run_sub_session call of yours."
                    ),
                    session_id=session.session_id,
                )
            inner_session_id = sub_session_param
        else:
            new_session = await create_chat_session(user_id, dry_run=session.dry_run)
            inner_session_id = new_session.session_id

        effective_prompt = prompt
        if system_context.strip():
            effective_prompt = f"[System Context: {system_context.strip()}]\n\n{prompt}"

        cap = max(0, min(wait_for_result, MAX_SUB_SESSION_WAIT_SECONDS))
        started_at = time.monotonic()
        outcome, result = await run_copilot_turn_via_queue(
            session_id=inner_session_id,
            user_id=user_id,
            message=effective_prompt,
            timeout=cap,
            permissions=get_current_permissions(),
            tool_call_id=(f"sub:{session.session_id}" if session.session_id else "sub"),
            tool_name="run_sub_session",
        )
        elapsed = time.monotonic() - started_at
        return response_from_outcome(
            outcome=outcome,
            result=result,
            inner_session_id=inner_session_id,
            parent_session_id=session.session_id,
            elapsed=elapsed,
        )


def _sub_session_link(inner_session_id: str | None) -> str | None:
    """Build the CoPilot UI URL for a sub-AutoPilot session.

    Kept in one place so the format stays consistent across the
    running/completed/error paths, and so the frontend only has one
    contract to honour.
    """
    if not inner_session_id:
        return None
    return f"/copilot?sessionId={inner_session_id}"


def response_from_outcome(
    *,
    outcome: SessionOutcome,
    result: SessionResult,
    inner_session_id: str,
    parent_session_id: str | None,
    elapsed: float,
) -> SubSessionStatusResponse:
    """Translate a ``(SessionOutcome, SessionResult)`` tuple into the
    ``SubSessionStatusResponse`` contract the LLM sees.

    ``completed`` surfaces the aggregated response text + tool calls.
    ``failed`` returns the error marker with the same handles.
    ``running`` returns just the polling handles so the agent can resume.
    ``queued`` means the target session already had a turn in flight; the
    message was appended to its pending buffer and will be processed by
    the existing turn on its next drain.
    """
    link = _sub_session_link(inner_session_id)
    if outcome == "queued":
        return SubSessionStatusResponse(
            message=(
                f"Target session already had a turn in flight; the message "
                f"was queued ({result.pending_buffer_length} now pending) and "
                "will be processed by the existing turn on its next drain. "
                f"Call get_sub_session_result to poll progress"
                f"{f' or watch live at {link}' if link else ''}."
            ),
            session_id=parent_session_id,
            status="queued",
            sub_session_id=inner_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
        )

    if outcome == "running":
        return SubSessionStatusResponse(
            message=(
                f"Sub-AutoPilot is still running after {elapsed:.0f}s."
                f"{f' Watch live at {link}.' if link else ''} "
                "Call get_sub_session_result (optionally with "
                "include_progress=true) to wait, poll, or inspect progress."
            ),
            session_id=parent_session_id,
            status="running",
            sub_session_id=inner_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
        )

    if outcome == "rejected_concurrent_turn_cap":
        # No sub-session record / transcript exists yet — the per-user
        # concurrent-turn cap rejected before ``create_session`` ran.
        # Render the actionable message instead of a "see transcript"
        # pointer to nothing.
        return SubSessionStatusResponse(
            message=concurrent_turn_limit_message(),
            session_id=parent_session_id,
            status="error",
            sub_session_id=inner_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
        )

    if outcome == "failed":
        return SubSessionStatusResponse(
            message="Sub-AutoPilot failed. See the sub's transcript for details.",
            session_id=parent_session_id,
            status="error",
            sub_session_id=inner_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
        )

    # completed
    return SubSessionStatusResponse(
        message=f"Sub-AutoPilot completed.{f' View at {link}.' if link else ''}",
        session_id=parent_session_id,
        status="completed",
        sub_session_id=inner_session_id,
        sub_autopilot_session_id=inner_session_id,
        sub_autopilot_session_link=link,
        response=result.response_text,
        tool_calls=[tc.model_dump() for tc in result.tool_calls],
        elapsed_seconds=round(elapsed, 2),
    )
