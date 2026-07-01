"""MCP tools that expose the LocalPC shim's workflow-recording surface.

Registered alongside the file/exec/computer-use MCP tools when (a) the
active executor is a :class:`LocalPCShim` and (b) the shim's HELLO
advertised the ``recording`` capability — same gating pattern as the
``local_pc_*`` computer-use tools.

Tools:
  * ``record_workflow`` — start / stop a recording from the agent side.
  * ``list_recordings`` — the recordings the agent started this session.
  * ``generate_skill_from_recording`` — fetch + generalize into a skill
    (returns ``needs_confirmation`` per §8 when params aren't confirmed).
  * ``dry_run_skill`` — drive replay over multiple rows (§8/§9).

Each handler is thin over ``recording_skill`` / the shim's ``_RecordingProxy``;
structured errors go through the existing :mod:`local_pc_errors` translator
via :class:`ShimRecordingError`. See
``experimental/local-pc-executor/docs/WORKFLOW_RECORDING.md``.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable

from backend.copilot.context import get_current_sandbox
from backend.copilot.tools.local_pc_shim import LocalPCShim, ShimRecordingError
from backend.copilot.tools.recording_skill import (
    GeneratedSkill,
    dry_run,
    generate_skill_from_recording,
    propose_clarifications,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Per-session state — keyed by shim.sandbox_id so it survives shim
# reconnects (the LocalPCShim object is recreated on reconnect, but the
# session id is stable). Holds (a) the recordings the agent started and
# (b) the most recent generated skill per recording so dry_run_skill can
# reference it without re-generating.
# ---------------------------------------------------------------------------


class _SessionRecordingState:
    def __init__(self) -> None:
        # recording_id -> metadata about the start/stop lifecycle.
        self.recordings: dict[str, dict[str, Any]] = {}
        # recording_id -> last GeneratedSkill (final or draft).
        self.skills: dict[str, GeneratedSkill] = {}


_SESSION_STATE: dict[str, _SessionRecordingState] = {}


def _state_for(shim: LocalPCShim) -> _SessionRecordingState:
    sid = shim.sandbox_id
    state = _SESSION_STATE.get(sid)
    if state is None:
        state = _SessionRecordingState()
        _SESSION_STATE[sid] = state
    return state


# ---------------------------------------------------------------------------
# MCP result helpers (mirror computer_use_tools._ok / _err)
# ---------------------------------------------------------------------------


def _ok(payload: dict | list | str | int | None) -> dict[str, Any]:
    text = payload if isinstance(payload, str) else json.dumps(payload, default=str)
    return {"content": [{"type": "text", "text": text}], "isError": False}


def _err(code: str, message: str, details: dict | None = None) -> dict[str, Any]:
    body: dict[str, Any] = {"code": code, "error": message}
    if details:
        body["details"] = details
    return {
        "content": [{"type": "text", "text": json.dumps(body, default=str)}],
        "isError": True,
    }


def _get_local_pc_shim() -> LocalPCShim | None:
    sb = get_current_sandbox()
    return sb if isinstance(sb, LocalPCShim) else None


def _require_recording() -> tuple[LocalPCShim | None, dict[str, Any] | None]:
    """Resolve the active LocalPCShim, refusing if the capability isn't granted."""
    shim = _get_local_pc_shim()
    if shim is None:
        return None, _err(
            "NO_LOCAL_PC_EXECUTOR",
            "No LocalPC shim is connected for this session. Workflow "
            "recording requires the autogpt-local-executor daemon to be "
            "running on the user's machine.",
        )
    if "recording" not in (shim.capabilities or []):
        return None, _err(
            "CAPABILITY_NOT_GRANTED",
            "The connected shim did not advertise the `recording` "
            "capability. The user would need to re-run `autogpt-shim auth` "
            "with the recording scope to enable workflow recording.",
            details={"capabilities": list(shim.capabilities or [])},
        )
    return shim, None


def _handle_shim_error(exc: ShimRecordingError) -> dict[str, Any]:
    return _err(exc.code, str(exc), exc.details)


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


async def _h_record_workflow(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_recording()
    if shim is None:
        return gate  # type: ignore[return-value]
    action = str(args.get("action", "")).lower()
    if action == "start":
        consent_token = args.get("consent_token")
        if not isinstance(consent_token, str) or not consent_token:
            # The platform cannot self-assert consent (§9). Surface the
            # same CONSENT_REQUIRED contract the shim would, so the LLM
            # knows it must obtain a shim-issued token first.
            return _err(
                "CONSENT_REQUIRED",
                "Recording can't start without a consent token the shim "
                "issues after the user confirms in its OS-native dialog. "
                "Ask the user to confirm recording; the shim then supplies "
                "the token.",
            )
        mode = str(args.get("mode", "copilot"))
        route = str(args.get("interpretation_route", "extract_then_cloud"))
        channels = args.get("channels") or ["floor"]
        if not isinstance(channels, list):
            return _err("INVALID_ARGUMENT", "channels must be a list of strings")
        try:
            recording_id = await shim.recording.start(
                mode=mode,
                interpretation_route=route,
                channels=[str(c) for c in channels],
                consent_token=consent_token,
            )
        except ShimRecordingError as exc:
            return _handle_shim_error(exc)
        state = _state_for(shim)
        state.recordings[recording_id] = {
            "recording_id": recording_id,
            "mode": mode,
            "interpretation_route": route,
            "channels": [str(c) for c in channels],
            "status": "recording",
            "started_at": time.time(),
        }
        return _ok({"recording_id": recording_id, "status": "recording", "mode": mode})

    if action == "stop":
        recording_id = args.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            return _err("INVALID_ARGUMENT", "recording_id is required to stop")
        try:
            summary = await shim.recording.stop(recording_id)
        except ShimRecordingError as exc:
            return _handle_shim_error(exc)
        # Close the live step stream if one was open (co-pilot mode).
        shim.close_recording(recording_id)
        state = _state_for(shim)
        meta = state.recordings.setdefault(recording_id, {"recording_id": recording_id})
        meta["status"] = "stopped"
        meta["stopped_at"] = time.time()
        meta["summary"] = summary.to_dict()
        return _ok({"recording_id": recording_id, "summary": summary.to_dict()})

    return _err(
        "INVALID_ARGUMENT",
        "action must be 'start' or 'stop'",
        details={"action": action},
    )


async def _h_list_recordings(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_recording()
    if shim is None:
        return gate  # type: ignore[return-value]
    state = _state_for(shim)
    recordings = sorted(
        state.recordings.values(),
        key=lambda r: r.get("started_at", 0),
    )
    return _ok({"recordings": recordings})


async def _h_generate_skill(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_recording()
    if shim is None:
        return gate  # type: ignore[return-value]
    recording_id = args.get("recording_id")
    if not isinstance(recording_id, str) or not recording_id:
        return _err("INVALID_ARGUMENT", "recording_id is required")
    try:
        recording = await shim.recording.fetch(recording_id)
    except ShimRecordingError as exc:
        return _handle_shim_error(exc)

    clarifications = args.get("clarifications")
    if clarifications is not None and not isinstance(clarifications, dict):
        return _err("INVALID_ARGUMENT", "clarifications must be an object")
    data_source_hint = args.get("data_source_hint")
    if data_source_hint is not None and not isinstance(data_source_hint, str):
        return _err("INVALID_ARGUMENT", "data_source_hint must be a string")

    skill = generate_skill_from_recording(
        recording,
        data_source_hint=data_source_hint,
        clarifications=clarifications,
    )
    state = _state_for(shim)
    state.skills[recording_id] = skill

    result = skill.to_dict()
    # When confirmation is still needed, also surface the §3.3 proposed
    # clarifications so the co-pilot can drive the trust loop in one step.
    if skill.needs_confirmation:
        result["proposed_clarifications"] = [
            q.to_dict() for q in propose_clarifications(recording)
        ]
    return _ok(result)


async def _h_dry_run_skill(args: dict[str, Any]) -> dict[str, Any]:
    shim, gate = _require_recording()
    if shim is None:
        return gate  # type: ignore[return-value]
    recording_id = args.get("recording_id")
    if not isinstance(recording_id, str) or not recording_id:
        return _err("INVALID_ARGUMENT", "recording_id is required")
    state = _state_for(shim)
    skill = state.skills.get(recording_id)
    if skill is None:
        return _err(
            "SKILL_NOT_GENERATED",
            "No generated skill for this recording yet. Call "
            "`generate_skill_from_recording` first.",
            details={"recording_id": recording_id},
        )
    if skill.needs_confirmation:
        # §8 — don't dry-run an unconfirmed skill; the parameter set isn't
        # settled, so the run would be meaningless.
        return _err(
            "SKILL_NEEDS_CONFIRMATION",
            "This skill still has unconfirmed parameters — confirm them "
            "(answer the clarifying questions or record a second row) "
            "before a dry run.",
            details={
                "recording_id": recording_id,
                "questions": [q.to_dict() for q in skill.questions],
            },
        )

    data_rows = args.get("data_rows")
    if not isinstance(data_rows, list) or not data_rows:
        return _err(
            "INVALID_ARGUMENT",
            "data_rows must be a non-empty list of row objects. A multi-row "
            "dry-run is required — a single row can't catch a "
            "parameter-as-constant mistake (§8).",
        )
    rows: list[dict[str, Any]] = [r for r in data_rows if isinstance(r, dict)]
    if not rows:
        return _err("INVALID_ARGUMENT", "data_rows must contain row objects")
    if len(rows) < 2:
        logger.info(
            "[Recording] dry_run_skill called with a single row for %s — "
            "multi-row recommended to catch parameter-as-constant",
            recording_id,
        )

    allow_destructive = bool(args.get("allow_destructive", False))
    try:
        result = await dry_run(
            skill, rows, shim=shim, allow_destructive=allow_destructive
        )
    except Exception as exc:  # defensive — replay drives the live machine
        logger.exception("[Recording] dry_run_skill failed for %s", recording_id)
        return _err("DRY_RUN_FAILED", f"Dry run failed: {exc}")
    return _ok(result.to_dict())


# ---------------------------------------------------------------------------
# Tool descriptors
# ---------------------------------------------------------------------------


RECORDING_TOOLS: list[
    tuple[str, str, dict[str, Any], Callable[[dict[str, Any]], Any]]
] = [
    (
        "record_workflow",
        "Start or stop recording a workflow on the user's machine via the "
        "LocalPC shim. `action: start` begins a recording — it REQUIRES a "
        "`consent_token` the shim issues only after the user confirms in an "
        "OS-native dialog (the platform cannot self-assert this). "
        "`action: stop` ends it and returns a summary (step count, "
        "enrichment coverage, duration). Use `mode: copilot` (default) to "
        "narrate live and ask clarifying questions, or `mode: demonstration` "
        "to let the user perform the whole task uninterrupted.",
        {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["start", "stop"],
                    "description": "Start or stop a recording.",
                },
                "mode": {
                    "type": "string",
                    "enum": ["copilot", "demonstration"],
                    "description": "Interaction mode (start only). Default: copilot.",
                },
                "interpretation_route": {
                    "type": "string",
                    "enum": [
                        "extract_then_cloud",
                        "local_vlm",
                        "screenshots_to_cloud",
                    ],
                    "description": "How the recording is turned into a skill "
                    "(start only). Default: extract_then_cloud (pixels stay "
                    "local). screenshots_to_cloud needs the shim's consent gate.",
                },
                "channels": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Capture channels (start only): 'floor' "
                    "(always), 'browser', 'desktop_ax'. Default: ['floor'].",
                },
                "consent_token": {
                    "type": "string",
                    "description": "Shim-issued consent token (start only, "
                    "REQUIRED). Obtained after the user confirms in the "
                    "shim's OS-native dialog.",
                },
                "recording_id": {
                    "type": "string",
                    "description": "The recording to stop (stop only).",
                },
            },
            "required": ["action"],
        },
        _h_record_workflow,
    ),
    (
        "list_recordings",
        "List the workflow recordings started in this session via the "
        "LocalPC shim, with their status (recording / stopped), mode, "
        "interpretation route, and (if stopped) summary.",
        {"type": "object", "properties": {}},
        _h_list_recordings,
    ),
    (
        "generate_skill_from_recording",
        "Fetch a stopped recording and generalize it into a reusable, "
        "parameterized skill (a SKILL.md plus a replay manifest). "
        "IMPORTANT: parameter inference is confirmed, not guessed — if the "
        "recording is a single row with unconfirmed parameters, this "
        "returns `needs_confirmation: true` with the questions to ask the "
        "user (or `needs_second_row: true` for demonstration mode). Pass "
        "`clarifications` to confirm: "
        "`{confirmed_parameters: {<field label>: true|false}}` from a "
        "co-pilot answer, or `{data_source_columns: {<field label>: "
        "<column name>}}` from a data-source join. The skill only becomes "
        "final when every parameter is confirmed.",
        {
            "type": "object",
            "properties": {
                "recording_id": {
                    "type": "string",
                    "description": "The stopped recording to generalize.",
                },
                "data_source_hint": {
                    "type": "string",
                    "description": "Where run data comes from, e.g. "
                    "'customers.csv'. Seeds the data-source contract.",
                },
                "clarifications": {
                    "type": "object",
                    "description": "Confirmation evidence: "
                    "confirmed_parameters / data_source_columns / name / "
                    "description / trigger / success_criterion.",
                },
            },
            "required": ["recording_id"],
        },
        _h_generate_skill,
    ),
    (
        "dry_run_skill",
        "Dry-run a generated skill over MULTIPLE data rows via the LocalPC "
        "shim, with per-step read-back asserts, before any unattended run. "
        "Multiple rows are required — a single-row dry-run can't catch a "
        "parameter that was mistakenly treated as a constant. Destructive "
        "steps (submit / upload / run / file ops) are skipped by default; "
        "pass `allow_destructive: true` only when the user explicitly wants "
        "the dry-run to actually perform them. Returns per-row outcomes the "
        "user can watch before trusting the skill.",
        {
            "type": "object",
            "properties": {
                "recording_id": {
                    "type": "string",
                    "description": "The recording whose generated skill to "
                    "dry-run (must be generated + confirmed first).",
                },
                "data_rows": {
                    "type": "array",
                    "items": {"type": "object"},
                    "description": "Rows of {parameter: value}. Use >= 2 rows "
                    "with different values to catch parameter-as-constant.",
                },
                "allow_destructive": {
                    "type": "boolean",
                    "description": "Actually perform submit/irreversible "
                    "steps during the dry-run. Default: false (skipped).",
                },
            },
            "required": ["recording_id", "data_rows"],
        },
        _h_dry_run_skill,
    ),
]


RECORDING_TOOL_NAMES: list[str] = [name for name, *_ in RECORDING_TOOLS]
