from __future__ import annotations

import hashlib
import json
import time
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Literal

FLAGGED_AUTONOMY_MAX_TOOL_CALLS = 36
FLAGGED_AUTONOMY_MAX_ELAPSED_SECONDS = 8 * 60
FLAGGED_AUTONOMY_MAX_AGENT_ROUNDS = 36
FLAGGED_AUTONOMY_MAX_UNCHANGED_RESULTS = 2
FLAGGED_AUTONOMY_FINAL_RESPONSE_RESERVE_SECONDS = 30

_NON_PROGRESS_TYPES = {
    "error",
    "input_validation_error",
    "no_results",
    "review_required",
    "setup_requirements",
}
_NON_PROGRESS_STATUSES = {
    "blocked",
    "blocked_manager",
    "cancelled",
    "error",
    "failed",
    "incomplete",
    "needs_setup",
    "partial",
    "rejected",
}


@dataclass
class AutonomyDecision:
    allowed: bool
    reason: Literal["elapsed", "tool_calls", "unchanged"] | None = None
    message: str | None = None


@dataclass
class _AutonomyState:
    started_at: float
    tool_calls: int = 0
    last_non_progress: dict[str, str] = field(default_factory=dict)
    unchanged_counts: dict[str, int] = field(default_factory=dict)


_state: ContextVar[_AutonomyState | None] = ContextVar(
    "expert_autonomy_budget", default=None
)


def start_autonomy_budget(*, enabled: bool) -> None:
    _state.set(_AutonomyState(started_at=time.monotonic()) if enabled else None)


def bounded_agent_rounds(configured: int, *, enabled: bool) -> int:
    if not enabled:
        return configured
    return min(configured, FLAGGED_AUTONOMY_MAX_AGENT_ROUNDS)


def remaining_tool_seconds() -> float | None:
    """Return the flagged turn's remaining tool time.

    A short reserve is kept for the model to turn a timed-out tool into a
    truthful Delivered/Blocked/Fallback/Next response.
    """
    state = _state.get()
    if state is None:
        return None
    elapsed = time.monotonic() - state.started_at
    return max(
        0.0,
        FLAGGED_AUTONOMY_MAX_ELAPSED_SECONDS
        - FLAGGED_AUTONOMY_FINAL_RESPONSE_RESERVE_SECONDS
        - elapsed,
    )


def elapsed_stop_message() -> str:
    return _stopped("elapsed").message or "This turn reached its time limit."


def before_tool(
    tool_name: str, arguments: dict[str, Any] | None = None
) -> AutonomyDecision:
    state = _state.get()
    if state is None:
        return AutonomyDecision(allowed=True)

    elapsed = time.monotonic() - state.started_at
    if elapsed >= FLAGGED_AUTONOMY_MAX_ELAPSED_SECONDS:
        return _stopped("elapsed")
    if state.tool_calls >= FLAGGED_AUTONOMY_MAX_TOOL_CALLS:
        return _stopped("tool_calls")
    path_key = _path_key(tool_name, arguments)
    if (
        state.unchanged_counts.get(path_key, 0)
        >= FLAGGED_AUTONOMY_MAX_UNCHANGED_RESULTS
    ):
        return _stopped("unchanged")

    state.tool_calls += 1
    return AutonomyDecision(allowed=True)


def after_tool(
    tool_name: str, result: Any, arguments: dict[str, Any] | None = None
) -> None:
    state = _state.get()
    if state is None:
        return
    path_key = _path_key(tool_name, arguments)
    fingerprint = _non_progress_fingerprint(result)
    if fingerprint is None:
        state.last_non_progress.pop(path_key, None)
        state.unchanged_counts.pop(path_key, None)
        return
    if state.last_non_progress.get(path_key) == fingerprint:
        state.unchanged_counts[path_key] = state.unchanged_counts.get(path_key, 1) + 1
    else:
        state.last_non_progress[path_key] = fingerprint
        state.unchanged_counts[path_key] = 1


def _path_key(tool_name: str, arguments: dict[str, Any] | None) -> str:
    try:
        encoded = json.dumps(arguments or {}, sort_keys=True, default=str)
    except (TypeError, ValueError):
        encoded = repr(arguments)
    digest = hashlib.sha256(encoded.encode()).hexdigest()[:16]
    return f"{tool_name}:{digest}"


def _non_progress_fingerprint(result: Any) -> str | None:
    payload = _payload(result)
    response_type = str(payload.get("type", "")).lower()
    status = str(payload.get("status", "")).lower()
    is_error = bool(payload.get("isError")) or payload.get("success") is False
    if (
        not is_error
        and response_type not in _NON_PROGRESS_TYPES
        and status not in _NON_PROGRESS_STATUSES
    ):
        return None
    stable = {
        "type": response_type,
        "status": status,
        "error": str(payload.get("error", ""))[:300],
        "blocker": str(payload.get("blocker", ""))[:300],
        "message": str(payload.get("message", ""))[:300],
    }
    return json.dumps(stable, sort_keys=True)


def _payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "model_dump"):
        dumped = result.model_dump(mode="json", exclude_none=True)
        if not isinstance(dumped, dict):
            return {}
        output = dumped.get("output")
        if isinstance(output, str):
            try:
                nested = json.loads(output)
                if isinstance(nested, dict):
                    return {**nested, "success": dumped.get("success", True)}
            except json.JSONDecodeError:
                pass
        return dumped
    if isinstance(result, dict):
        content = result.get("content")
        if isinstance(content, list) and content:
            first = content[0]
            if isinstance(first, dict) and isinstance(first.get("text"), str):
                try:
                    nested = json.loads(first["text"])
                    if isinstance(nested, dict):
                        return {**nested, "isError": result.get("isError", False)}
                except json.JSONDecodeError:
                    pass
        return result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _stopped(
    reason: Literal["elapsed", "tool_calls", "unchanged"],
) -> AutonomyDecision:
    reason_text = {
        "elapsed": "this turn reached its elapsed-time limit",
        "tool_calls": "this turn reached its tool-call limit",
        "unchanged": "this tool returned the same blocked or failed state twice",
    }[reason]
    return AutonomyDecision(
        allowed=False,
        reason=reason,
        message=(
            f"STOP: {reason_text}. Do not retry this path in the current turn. "
            "Finish now with a concise manager handoff containing: Delivered, "
            "Blocked, the safest useful degraded fallback, and Next. Keep any "
            "independent work moving in a later event-driven continuation."
        ),
    )
