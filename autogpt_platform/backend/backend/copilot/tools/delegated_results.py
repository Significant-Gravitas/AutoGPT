"""Bounded result packets for work delegated to a hired expert."""

import re
from typing import Literal

from backend.copilot.active_turns import running_turn_limit_message
from backend.copilot.sdk.session_waiter import SessionOutcome, SessionResult

from .models import (
    DelegatedArtifact,
    DelegatedExpertInfo,
    DelegatedExpertStatusResponse,
    WorkspaceFileInfoData,
)
from .run_sub_session import _sub_session_link, _workspace_files_from_tool_calls

DeliverableMode = Literal["message", "workspace_files"]

MAX_DELEGATED_PACKET_BYTES = 16 * 1024
MAX_DELEGATED_SUMMARY_CHARS = 12_000
_MAX_ARTIFACT_PREVIEW = 25
_MAX_ARTIFACT_NAME_CHARS = 160
_MAX_ARTIFACT_PATH_CHARS = 400
_MAX_MIME_TYPE_CHARS = 100
_INTERNAL_CONTEXT_BLOCK = re.compile(
    r"<(?:user_context|memory_context|env_context|budget_context|"
    r"session_context|available_skills|expert_identity|expert_workflows|"
    r"team_context)\b[^>]*>[\s\S]*?</(?:user_context|memory_context|"
    r"env_context|budget_context|session_context|available_skills|"
    r"expert_identity|expert_workflows|team_context)>",
    re.IGNORECASE,
)
_ABSOLUTE_PATH = re.compile(
    r"(?:[a-z]:\\(?:users|temp|windows|workspace)\\|"
    r"/(?:Users|home|tmp|private|var|opt|workspace|mnt)/)[^\s,;)]+",
    re.IGNORECASE,
)
_UUID = re.compile(
    r"\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-" r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b",
    re.IGNORECASE,
)
_INTERNAL_ID = re.compile(
    r"\b(?:tool(?:_call)?|graph|block|session|execution|node(?:_exec)?)"
    r"[_-]?id\s*[:=]\s*[\"']?[a-z0-9._:-]+[\"']?",
    re.IGNORECASE,
)
_SHELL_OUTPUT_LINE = re.compile(
    r"^\s*(?:\$\s|stdout\s*:|stderr\s*:|exit\s+code\s*:)", re.IGNORECASE
)


def delegated_response_from_outcome(
    *,
    outcome: SessionOutcome,
    result: SessionResult,
    inner_session_id: str,
    parent_session_id: str | None,
    elapsed: float,
    expert: DelegatedExpertInfo,
    deliverable_mode: DeliverableMode,
    workspace_files: list[WorkspaceFileInfoData] | None = None,
) -> DelegatedExpertStatusResponse:
    """Translate a delegated turn without returning its tool transcript."""
    link = _sub_session_link(inner_session_id)
    common = {
        "session_id": parent_session_id,
        "sub_session_id": inner_session_id,
        "sub_autopilot_session_id": inner_session_id,
        "sub_autopilot_session_link": link,
        "elapsed_seconds": round(max(0, elapsed), 2),
        "expert": expert,
        "deliverable_mode": deliverable_mode,
        "tool_call_count": len(result.tool_calls),
    }

    if outcome == "queued":
        return _fit_packet(
            DelegatedExpertStatusResponse(
                message=f"{expert.name}'s next task is queued.",
                status="queued",
                **common,
            )
        )
    if outcome == "running":
        return _fit_packet(
            DelegatedExpertStatusResponse(
                message=f"{expert.name} is working.",
                status="running",
                **common,
            )
        )
    if outcome == "rejected_concurrent_turn_cap":
        return _fit_packet(
            DelegatedExpertStatusResponse(
                message=running_turn_limit_message(),
                status="error",
                blockers=["The concurrent expert-work limit was reached."],
                **common,
            )
        )
    if outcome == "failed":
        return _fit_packet(
            DelegatedExpertStatusResponse(
                message=f"{expert.name} could not complete the delegated task.",
                status="error",
                blockers=["The delegated expert turn failed."],
                **common,
            )
        )

    files = workspace_files
    if files is None:
        files = _workspace_files_from_tool_calls(result.tool_calls)
    artifacts = [_artifact(file) for file in files[:_MAX_ARTIFACT_PREVIEW]]
    artifact_count = len(files)
    missing_files = deliverable_mode == "workspace_files" and not artifacts
    status = "incomplete" if missing_files else "completed"
    blockers = (
        [
            "The task required persistent files, but the expert did not promote "
            "any output with write_workspace_file(source_path=...)."
        ]
        if missing_files
        else []
    )
    message = (
        f"{expert.name} finished, but the required files are not accessible."
        if missing_files
        else f"{expert.name} completed the delegated task."
    )
    response = DelegatedExpertStatusResponse(
        message=message,
        status=status,
        summary=_summary_preview(result.response_text),
        blockers=blockers,
        artifacts=artifacts,
        artifact_count=artifact_count,
        **common,
    )
    return _fit_packet(response)


def _artifact(file: WorkspaceFileInfoData) -> DelegatedArtifact:
    return DelegatedArtifact(
        name=_clip(file.name, _MAX_ARTIFACT_NAME_CHARS),
        read_path=_clip(file.path, _MAX_ARTIFACT_PATH_CHARS),
        mime_type=_clip(file.mime_type, _MAX_MIME_TYPE_CHARS),
        size_bytes=max(0, file.size_bytes),
    )


def _summary_preview(text: str) -> str:
    return _clip(_safe_summary(text), MAX_DELEGATED_SUMMARY_CHARS)


def _safe_summary(text: str) -> str:
    """Remove developer-only material before a manager can relay the result."""
    without_context = _INTERNAL_CONTEXT_BLOCK.sub("", text)
    safe_lines: list[str] = []
    for line in without_context.splitlines():
        stripped = line.strip()
        if _SHELL_OUTPUT_LINE.match(line):
            continue
        if (stripped.startswith("{") and stripped.endswith("}")) or (
            stripped.startswith("[") and stripped.endswith("]")
        ):
            continue
        safe_lines.append(
            _UUID.sub(
                "reference",
                _INTERNAL_ID.sub(
                    "internal reference", _ABSOLUTE_PATH.sub("a workspace file", line)
                ),
            ).rstrip()
        )
    return "\n".join(safe_lines).strip()


def _clip(value: str, limit: int) -> str:
    if limit <= 0:
        return ""
    if len(value) <= limit:
        return value
    return value[: limit - 1] + "…"


def _fit_packet(
    response: DelegatedExpertStatusResponse,
) -> DelegatedExpertStatusResponse:
    """Fit the serialized model-facing packet below the hard byte ceiling."""
    while _packet_size(response) >= MAX_DELEGATED_PACKET_BYTES and response.artifacts:
        response = response.model_copy(update={"artifacts": response.artifacts[:-1]})
    if _packet_size(response) < MAX_DELEGATED_PACKET_BYTES or not response.summary:
        return response

    low = 0
    high = len(response.summary)
    while low < high:
        midpoint = (low + high + 1) // 2
        candidate = response.model_copy(
            update={"summary": _clip(response.summary, midpoint)}
        )
        if _packet_size(candidate) < MAX_DELEGATED_PACKET_BYTES:
            low = midpoint
        else:
            high = midpoint - 1
    return response.model_copy(update={"summary": _clip(response.summary, low)})


def _packet_size(response: DelegatedExpertStatusResponse) -> int:
    return len(response.model_dump_json(exclude_none=True).encode("utf-8"))
