"""Start a sub-AutoPilot conversation as an async task.

Long-running work is decoupled from the caller's stream lifecycle:

1. The tool spawns ``collect_copilot_response`` as an ``asyncio.Task`` and
   registers it in the process-wide :mod:`sub_session_registry` — so it
   survives the HTTP request that created it (tab close, stream teardown).
2. The tool waits up to ``wait_for_result`` seconds (capped at
   :data:`MAX_SUB_SESSION_WAIT_SECONDS`, 5 min). If the sub completes in
   that window the result is returned inline.
3. Otherwise the tool returns immediately with ``status="running"`` and the
   ``sub_session_id``. The agent calls :mod:`get_sub_session_result` — which
   can wait another 5 min — repeatedly until the sub finishes.

This mirrors the ``run_agent`` / ``view_agent_output`` pattern that already
handles long-running graph executions. No single MCP tool call ever blocks
the copilot stream for more than ~5 min, so the stream-level idle timeout
(10 min) stays simple and effective.
"""

import asyncio
import dataclasses
import logging
import time
from typing import TYPE_CHECKING, Any

from backend.copilot.context import get_current_permissions
from backend.copilot.model import ChatSession

if TYPE_CHECKING:
    from backend.copilot.permissions import CopilotPermissions
from backend.copilot.sdk.sub_session_registry import (
    MAX_SUB_SESSION_WAIT_SECONDS,
    prune_finished,
    register_sub_session,
)

from .base import BaseTool
from .models import ErrorResponse, SubSessionStatusResponse, ToolResponseBase

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class _SubAutopilotResult:
    """Result of a completed sub-AutoPilot run, keyed back to its session_id.

    ``CopilotResult`` itself doesn't carry the session_id (it's passed in at
    the call site), so we wrap it here so ``_response_from_task`` can echo
    the sub's session_id back to the agent for continuation.
    """

    session_id: str
    response_text: str
    tool_calls: list[dict[str, Any]]


class RunSubSessionTool(BaseTool):
    """Delegate a task to a fresh AutoPilot instance.

    Replaces the ``run_block`` path that used to invoke ``AutoPilotBlock``
    inline. Gives the agent an explicit async start + poll model so long
    sub-AutoPilot runs survive tab-close and don't block the parent stream.
    """

    @property
    def name(self) -> str:
        return "run_sub_session"

    @property
    def requires_auth(self) -> bool:
        return True

    @property
    def description(self) -> str:
        return (
            "Delegate a task to a fresh sub-AutoPilot. Runs as a background "
            "task that survives the current turn. Waits up to wait_for_result "
            f"sec (max {MAX_SUB_SESSION_WAIT_SECONDS}). If not done, returns "
            "status=running + sub_session_id — poll via get_sub_session_result."
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
        prune_finished()

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

        # Resolve the sub's ChatSession id BEFORE spawning the task so we
        # can surface it immediately in the response and also pass it to
        # get_sub_session_result for progress peeks while the sub runs.
        # Resume-path MUST verify ownership to prevent one user reusing
        # another user's session_id as the "sub" session.
        from backend.copilot.model import (  # noqa: PLC0415
            create_chat_session,
            get_chat_session,
        )

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
            # Inherit parent's dry_run so a sub spawned inside a dry-run
            # conversation doesn't silently escalate to a live run.
            new_session = await create_chat_session(user_id, dry_run=session.dry_run)
            inner_session_id = new_session.session_id

        effective_prompt = prompt
        if system_context.strip():
            effective_prompt = f"[System Context: {system_context.strip()}]\n\n{prompt}"

        # Propagate the parent's capability filter so the sub can't escalate
        # (e.g. call blocks or tools the parent was restricted from).
        inherited_permissions = get_current_permissions()

        # Spawn the sub-AutoPilot work as an asyncio.Task so it can outlive
        # the current request. Wrapping the await in asyncio.shield below
        # prevents the caller's cancellation (tab close) from propagating
        # into this task.
        task = asyncio.create_task(
            _run_sub_autopilot(
                user_id=user_id,
                inner_session_id=inner_session_id,
                effective_prompt=effective_prompt,
                permissions=inherited_permissions,
            ),
            name=f"sub-session:{session.session_id}",
        )

        # Register immediately so the task has a strong ref and survives
        # request teardown. Cancel the orphaned task if registry insertion
        # fails (e.g. session-cap hit, exotic errors) so we don't leak.
        try:
            sub_session_id = register_sub_session(
                task=task,
                user_id=user_id,
                parent_session_id=session.session_id or "",
                prompt=prompt,
                inner_session_id=inner_session_id,
            )
        except Exception:
            task.cancel()
            raise

        cap = max(0, min(wait_for_result, MAX_SUB_SESSION_WAIT_SECONDS))
        started_at = time.monotonic()
        if cap > 0:
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=cap)
            except asyncio.TimeoutError:
                pass
            except asyncio.CancelledError:
                # Caller was cancelled (e.g. tab close). The shielded task
                # keeps running in the background; re-raise so the handler
                # unwinds cleanly.
                raise
        elapsed = time.monotonic() - started_at

        return _response_from_task(
            task=task,
            sub_session_id=sub_session_id,
            session=session,
            elapsed=elapsed,
            inner_session_id_when_running=inner_session_id,
        )


async def _run_sub_autopilot(
    *,
    user_id: str,
    inner_session_id: str,
    effective_prompt: str,
    permissions: "CopilotPermissions | None" = None,
):
    """Entry point for the spawned sub-AutoPilot task.

    Isolated from the handler so it can be unit-tested and so the task's
    coroutine is a stable target — we don't want inner closure state that
    the handler closure might try to clean up on its own cancellation.

    The caller has already created (or chosen) ``inner_session_id`` so
    progress polling via ``get_sub_session_result`` can find the session
    even while the task is still in flight. ``permissions`` is the parent
    stream's capability filter; forwarding it prevents the sub from
    escalating past whatever the parent was restricted to.
    """
    from backend.copilot.sdk.collect import (  # noqa: PLC0415; avoid circular import at module load
        collect_copilot_response,
    )

    result = await collect_copilot_response(
        session_id=inner_session_id,
        message=effective_prompt,
        user_id=user_id,
        permissions=permissions,
    )
    return _SubAutopilotResult(
        session_id=inner_session_id,
        response_text=result.response_text,
        tool_calls=list(result.tool_calls),
    )


def _response_from_task(
    *,
    task: asyncio.Task,
    sub_session_id: str,
    session: ChatSession,
    elapsed: float,
    inner_session_id_when_running: str | None = None,
) -> ToolResponseBase:
    """Map a sub-session asyncio.Task's state to SubSessionStatusResponse.

    When the task is still running, the caller can supply
    ``inner_session_id_when_running`` so the response echoes the sub's
    ChatSession id back — the agent uses it to check progress via
    get_sub_session_result(include_progress=true) or to continue the
    conversation once the sub completes.
    """
    if not task.done():
        return SubSessionStatusResponse(
            message=(
                f"Sub-AutoPilot is still running after {elapsed:.0f}s. "
                "Call get_sub_session_result (optionally with "
                "include_progress=true) to wait, poll, or inspect "
                "progress."
            ),
            session_id=session.session_id,
            status="running",
            sub_session_id=sub_session_id,
            sub_autopilot_session_id=inner_session_id_when_running,
            elapsed_seconds=round(elapsed, 2),
        )

    if task.cancelled():
        return SubSessionStatusResponse(
            message="Sub-AutoPilot was cancelled.",
            session_id=session.session_id,
            status="cancelled",
            sub_session_id=sub_session_id,
            elapsed_seconds=round(elapsed, 2),
        )

    exc = task.exception()
    if exc is not None:
        logger.warning(
            "Sub-session %s failed: %s",
            sub_session_id,
            exc,
            exc_info=exc,
        )
        return SubSessionStatusResponse(
            message=f"Sub-AutoPilot failed: {type(exc).__name__}",
            session_id=session.session_id,
            status="error",
            sub_session_id=sub_session_id,
            error=str(exc),
            elapsed_seconds=round(elapsed, 2),
        )

    result = task.result()
    return SubSessionStatusResponse(
        message="Sub-AutoPilot completed.",
        session_id=session.session_id,
        status="completed",
        sub_session_id=sub_session_id,
        sub_autopilot_session_id=result.session_id,
        response=result.response_text,
        tool_calls=result.tool_calls,
        elapsed_seconds=round(elapsed, 2),
    )
