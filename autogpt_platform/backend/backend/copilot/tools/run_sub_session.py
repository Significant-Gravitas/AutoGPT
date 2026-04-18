"""Start a sub-AutoPilot conversation via the copilot_executor queue.

Mirror-image of ``run_agent`` + ``view_agent_output`` for copilot turns:

1. The tool creates (or validates ownership of) an inner ``ChatSession``,
   creates a stream-registry session meta record, and enqueues a
   ``CoPilotExecutionEntry`` on the copilot execution queue.
2. Any available ``copilot_executor`` worker claims the job, runs
   ``collect_copilot_response`` to completion, and publishes the final
   ``StreamFinish`` event on the session's Redis stream.
3. The tool optionally waits up to ``wait_for_result`` seconds (capped at
   :data:`MAX_SUB_SESSION_WAIT_SECONDS`, 5 min) by subscribing to that
   stream. If the terminal event arrives in that window, the tool reads
   the sub's last assistant message and returns ``status="completed"``
   inline. Otherwise it returns ``status="running"`` + the sub's
   ``session_id``; the agent polls via :mod:`get_sub_session_result`.

Compared to the previous in-process ``asyncio.Task`` implementation,
queue-backed execution gives us:

* **Deploy/crash resilience** — RabbitMQ redelivers the job if a worker
  dies mid-run, so a 30-minute sub survives a rolling deploy.
* **Natural load balancing** — parallel subs from one user fan out across
  workers instead of pinning one event loop.
* **Uniform conversation model** — a sub is just another copilot turn;
  no bespoke task registry, shield dance, or abandoned-task cap.
* **No cross-process cancellation gymnastics** — cancel is a fan-out
  event on the existing ``copilot_cancel`` exchange.
"""

import logging
import time
import uuid
from typing import Any

from backend.copilot import stream_registry
from backend.copilot.constants import MAX_TOOL_WAIT_SECONDS
from backend.copilot.context import get_current_permissions
from backend.copilot.executor.utils import enqueue_copilot_turn
from backend.copilot.model import ChatSession, create_chat_session, get_chat_session
from backend.copilot.sdk.session_waiter import (
    SessionOutcome,
    wait_for_session_completion,
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
                    "description": (
                        "Continue a prior sub via its session_id; empty = new."
                    ),
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

        # Propagate the parent's capability filter so the sub can't escalate
        # past whatever the parent was restricted to.
        inherited_permissions = get_current_permissions()

        # Create a stream_registry meta record for the sub's turn, then
        # enqueue the job. Any worker picks it up; we wait on the shared
        # Redis stream for the terminal event.
        turn_id = str(uuid.uuid4())
        await stream_registry.create_session(
            session_id=inner_session_id,
            user_id=user_id,
            tool_call_id=f"sub:{session.session_id}" if session.session_id else "sub",
            tool_name="run_sub_session",
            turn_id=turn_id,
        )
        await enqueue_copilot_turn(
            session_id=inner_session_id,
            user_id=user_id,
            message=effective_prompt,
            turn_id=turn_id,
            permissions=inherited_permissions,
        )

        cap = max(0, min(wait_for_result, MAX_SUB_SESSION_WAIT_SECONDS))
        started_at = time.monotonic()
        outcome: SessionOutcome = "running"
        if cap > 0:
            outcome = await wait_for_session_completion(
                session_id=inner_session_id,
                user_id=user_id,
                timeout=cap,
            )
        elapsed = time.monotonic() - started_at

        return await _response_from_outcome(
            outcome=outcome,
            inner_session_id=inner_session_id,
            parent_session=session,
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


async def _response_from_outcome(
    *,
    outcome: SessionOutcome,
    inner_session_id: str,
    parent_session: ChatSession,
    elapsed: float,
) -> SubSessionStatusResponse:
    """Translate a ``SessionOutcome`` + the sub's ChatSession state into the
    ``SubSessionStatusResponse`` contract the LLM sees.

    ``completed`` reads the sub's last assistant message for the response
    text + tool_calls. ``failed`` returns the error with the same handles.
    ``running`` returns the polling handles so the agent can resume.
    """
    link = _sub_session_link(inner_session_id)
    if outcome == "running":
        return SubSessionStatusResponse(
            message=(
                f"Sub-AutoPilot is still running after {elapsed:.0f}s."
                f"{f' Watch live at {link}.' if link else ''} "
                "Call get_sub_session_result (optionally with "
                "include_progress=true) to wait, poll, or inspect progress."
            ),
            session_id=parent_session.session_id,
            status="running",
            sub_session_id=inner_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
        )

    if outcome == "failed":
        return SubSessionStatusResponse(
            message="Sub-AutoPilot failed. See the sub's transcript for details.",
            session_id=parent_session.session_id,
            status="error",
            sub_session_id=inner_session_id,
            sub_autopilot_session_id=inner_session_id,
            sub_autopilot_session_link=link,
            elapsed_seconds=round(elapsed, 2),
        )

    # completed — load the sub's last assistant turn for the content.
    response_text = ""
    tool_calls: list[dict[str, Any]] = []
    sub = await get_chat_session(inner_session_id)
    if sub is not None:
        last_assistant = next(
            (m for m in reversed(sub.messages) if m.role == "assistant"), None
        )
        if last_assistant is not None:
            response_text = last_assistant.content or ""
            tool_calls = list(last_assistant.tool_calls or [])

    return SubSessionStatusResponse(
        message=f"Sub-AutoPilot completed.{f' View at {link}.' if link else ''}",
        session_id=parent_session.session_id,
        status="completed",
        sub_session_id=inner_session_id,
        sub_autopilot_session_id=inner_session_id,
        sub_autopilot_session_link=link,
        response=response_text,
        tool_calls=tool_calls,
        elapsed_seconds=round(elapsed, 2),
    )
