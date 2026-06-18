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

import json
import logging
import time
from typing import Any

from backend.copilot.active_turns import running_turn_limit_message
from backend.copilot.constants import MAX_TOOL_WAIT_SECONDS
from backend.copilot.context import get_current_permissions, get_workspace_manager
from backend.copilot.model import ChatSession, create_chat_session, get_chat_session
from backend.copilot.sdk.session_waiter import (
    SessionOutcome,
    SessionResult,
    run_copilot_turn_via_queue,
)
from backend.copilot.sdk.stream_accumulator import ToolCallEntry

from .base import BaseTool
from .models import (
    ErrorResponse,
    SubSessionStatusResponse,
    ToolResponseBase,
    WorkspaceFileInfoData,
)

logger = logging.getLogger(__name__)


# Max wait for a single run_sub_session / get_sub_session_result call.
# Shared with every other long-running tool so the stream idle timeout's
# 2x headroom holds uniformly.
MAX_SUB_SESSION_WAIT_SECONDS = MAX_TOOL_WAIT_SECONDS

# Cap on how many sub-written files we enumerate in a completed response's
# manifest. Bounds context size; a sub producing more than this is pathological.
_WORKSPACE_FILE_MANIFEST_LIMIT = 50


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
        workspace_files = (
            await list_sub_workspace_files(user_id, inner_session_id)
            if outcome == "completed"
            else None
        )
        return response_from_outcome(
            outcome=outcome,
            result=result,
            inner_session_id=inner_session_id,
            parent_session_id=session.session_id,
            elapsed=elapsed,
            workspace_files=workspace_files,
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


async def list_sub_workspace_files(
    user_id: str,
    inner_session_id: str,
) -> list[WorkspaceFileInfoData] | None:
    """Authoritative manifest of the persistent files a sub wrote, read from
    the sub's session workspace.

    A sub may deliver its real output by writing workspace files and only
    summarising in its final message (SECRT-2377). This queries the sub's
    session for ``agent-created`` files directly, so it captures writes from
    *any* turn — including ones absent from the current turn's tool-call log
    (e.g. the cold-poll / already-terminal path in ``get_sub_session_result``).

    The ``origin=agent-created`` filter means only files the sub persisted via
    ``write_workspace_file`` are listed — transient working-directory artefacts
    (e.g. a ``git clone`` the sub inspects but never persists) are not workspace
    files and never appear here. When the sub persists more than
    ``_WORKSPACE_FILE_MANIFEST_LIMIT`` files, the most recently written ones win
    (the listing is ordered ``createdAt`` descending).

    Returns ``None`` on lookup failure so callers can fall back to mining the
    tool-call log; an empty list means the sub genuinely wrote nothing.
    """
    try:
        manager = await get_workspace_manager(user_id, inner_session_id)
        files = await manager.list_files(
            limit=_WORKSPACE_FILE_MANIFEST_LIMIT,
            metadata_equals={"origin": "agent-created"},
        )
    except Exception:
        logger.warning(
            f"Failed to list workspace files for sub {inner_session_id[:12]}",
            exc_info=True,
        )
        return None
    return [
        WorkspaceFileInfoData(
            file_id=f.id,
            name=f.name,
            path=f.path,
            mime_type=f.mime_type,
            size_bytes=f.size_bytes,
        )
        for f in files
    ]


def _workspace_files_from_tool_calls(
    tool_calls: list[ToolCallEntry],
) -> list[WorkspaceFileInfoData]:
    """Mine the files a sub wrote from its tool-call log — the cheap fallback
    when the authoritative workspace listing is unavailable.

    ``write_workspace_file`` outputs already carry the fully session-qualified
    ``path`` (the workspace manager resolves it on write), so it is directly
    usable with ``read_workspace_file``. The output is an opaque
    ``WorkspaceWriteResponse`` payload — a JSON string on the live-drain path,
    a dict on the persisted-replay path — so we parse defensively and skip
    anything that doesn't carry the fields we need.
    """
    files: list[WorkspaceFileInfoData] = []
    seen_ids: set[str] = set()
    for tc in tool_calls:
        if tc.tool_name != "write_workspace_file" or tc.success is False:
            continue
        payload = _as_payload(tc.output)
        if payload is None:
            continue
        file_id = payload.get("file_id")
        path = payload.get("path")
        if not file_id or not path or file_id in seen_ids:
            continue
        seen_ids.add(file_id)
        files.append(
            WorkspaceFileInfoData(
                file_id=file_id,
                name=payload.get("name") or path.rsplit("/", 1)[-1],
                path=path,
                mime_type=payload.get("mime_type") or "",
                size_bytes=_coerce_size_bytes(payload.get("size_bytes")),
            )
        )
    return files


def _coerce_size_bytes(raw: Any) -> int:
    """Coerce a mined ``size_bytes`` to a non-negative int — the payload is
    untrusted (JSON parsed from the tool log), so a missing, malformed, or
    negative value must not surface as invalid size data."""
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return 0


def _as_payload(output: Any) -> dict[str, Any] | None:
    """Coerce a tool-call ``output`` (JSON string or dict) into a dict, or
    ``None`` when it isn't a usable object."""
    if isinstance(output, dict):
        return output
    if isinstance(output, str):
        try:
            parsed = json.loads(output)
        except (json.JSONDecodeError, ValueError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def response_from_outcome(
    *,
    outcome: SessionOutcome,
    result: SessionResult,
    inner_session_id: str,
    parent_session_id: str | None,
    elapsed: float,
    workspace_files: list[WorkspaceFileInfoData] | None = None,
) -> SubSessionStatusResponse:
    """Translate a ``(SessionOutcome, SessionResult)`` tuple into the
    ``SubSessionStatusResponse`` contract the LLM sees.

    ``completed`` surfaces the aggregated response text + tool calls, plus a
    manifest of any workspace files the sub wrote (SECRT-2377). Pass
    ``workspace_files`` to supply the authoritative listing from the sub's
    session; when omitted, the files are mined from ``result.tool_calls`` as a
    fallback.
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
            message=running_turn_limit_message(),
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

    # completed — prefer the authoritative listing supplied by the caller;
    # fall back to mining the tool-call log when it's unavailable.
    if workspace_files is None:
        workspace_files = _workspace_files_from_tool_calls(result.tool_calls)
    message = f"Sub-AutoPilot completed.{f' View at {link}.' if link else ''}"
    if workspace_files:
        # The sub may have put its real output in files and only summarised in
        # `response`. Flag the files explicitly so the parent reads them rather
        # than treating the run as empty (SECRT-2377).
        message += (
            f" It wrote {len(workspace_files)} workspace file(s); read them via "
            "read_workspace_file(path=<read_path>) — see sub_workspace_files."
        )
    return SubSessionStatusResponse(
        message=message,
        session_id=parent_session_id,
        status="completed",
        sub_session_id=inner_session_id,
        sub_autopilot_session_id=inner_session_id,
        sub_autopilot_session_link=link,
        response=result.response_text,
        tool_calls=[tc.model_dump() for tc in result.tool_calls],
        sub_workspace_files=workspace_files or None,
        elapsed_seconds=round(elapsed, 2),
    )
