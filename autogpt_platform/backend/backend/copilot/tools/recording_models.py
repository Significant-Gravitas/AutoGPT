"""Wire-schema dataclasses for workflow recordings.

Mirrors the spec-locked schema in
``experimental/local-pc-executor/docs/WORKFLOW_RECORDING.md`` §1
(``WorkflowRecording`` / ``TrajectoryStep``). These are the platform-side
parsed forms of the ``RECORDING_DATA`` / ``RECORDING_STEP`` wire payloads.

Parsing is deliberately tolerant: the shim is a separate repo on its own
release cadence, so unknown fields are ignored and missing optional fields
default rather than raising. The floor fields (``action`` / ``screenshot_ref``
/ ``cursor`` / ``active_app``) are always expected per §1 but we still
default them defensively so a malformed step never crashes the recv loop.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# §1.1 semantic verbs + the two replay-control verbs the recorder
# synthesizes / the user adds. Kept as a frozenset so the generalizer can
# cheaply classify a step as state-mutating without re-deriving the list.
SEMANTIC_VERBS: frozenset[str] = frozenset(
    {
        "navigate",
        "fill",
        "select",
        "click",
        "submit",
        "upload",
        "launch_app",
        "focus_window",
        "copy",
        "paste",
        "run",
        "file_op",
        "wait",
        "assert",
    }
)

# Verbs that mutate state on the user's machine — the §9 "every replayed
# step that mutates state gets a read-back assert" set, and the basis for
# the `destructive: true` flag (submit/upload/run/file_op are irreversible
# enough to gate behind explicit confirmation).
MUTATING_VERBS: frozenset[str] = frozenset(
    {"fill", "select", "submit", "upload", "paste", "run", "file_op"}
)

# Verbs whose presence makes a generated skill destructive (§2/§7: submit
# + irreversible steps). Narrower than MUTATING_VERBS — filling a field is
# mutating but reversible; submitting / running / file ops are not.
DESTRUCTIVE_VERBS: frozenset[str] = frozenset({"submit", "upload", "run", "file_op"})

# §1 value.type enum.
VALUE_TYPES: frozenset[str] = frozenset(
    {"text", "email", "number", "date", "secret", "file", "enum"}
)

# §3 interpretation routes.
INTERPRETATION_ROUTES: frozenset[str] = frozenset(
    {"extract_then_cloud", "local_vlm", "screenshots_to_cloud"}
)

# §6 START_RECORDING modes.
RECORDING_MODES: frozenset[str] = frozenset({"demonstration", "copilot"})


@dataclass
class StepEnrichment:
    """The additive enrichment block on a TrajectoryStep (§1).

    ``kind`` is ``dom`` | ``ax`` | ``none``; the floor is always present
    regardless. ``selectors`` are browser-DOM only, most-stable-first (§7).
    """

    kind: str = "none"
    selectors: list[dict[str, str]] = field(default_factory=list)
    ax_path: str | None = None
    role: str | None = None
    label: str | None = None
    url: str | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "StepEnrichment":
        if not isinstance(payload, dict):
            return cls()
        raw_selectors = payload.get("selectors") or []
        selectors: list[dict[str, str]] = []
        if isinstance(raw_selectors, list):
            for sel in raw_selectors:
                if isinstance(sel, dict) and "strategy" in sel and "value" in sel:
                    selectors.append(
                        {
                            "strategy": str(sel["strategy"]),
                            "value": str(sel["value"]),
                        }
                    )
        kind = payload.get("kind") or "none"
        return cls(
            kind=str(kind),
            selectors=selectors,
            ax_path=payload.get("ax_path"),
            role=payload.get("role"),
            label=payload.get("label"),
            url=payload.get("url"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "selectors": [dict(s) for s in self.selectors],
            "ax_path": self.ax_path,
            "role": self.role,
            "label": self.label,
            "url": self.url,
        }


@dataclass
class StepValue:
    """The value block on a TrajectoryStep (§1).

    ``raw`` is the demonstration value — handled per interpretation route.
    ``is_parameter`` is ``None`` until inferred AND confirmed during
    generalization (§8); a None here means "not yet decided", which is
    why a single-row recording can't auto-save (§8).
    """

    raw: Any = None
    type: str = "text"
    is_parameter: bool | None = None

    @classmethod
    def from_payload(cls, payload: Any) -> "StepValue | None":
        if not isinstance(payload, dict):
            return None
        return cls(
            raw=payload.get("raw"),
            type=str(payload.get("type") or "text"),
            is_parameter=payload.get("is_parameter"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "type": self.type,
            "is_parameter": self.is_parameter,
        }


@dataclass
class TrajectoryStep:
    """One ordered step in a recording (§1 TrajectoryStep)."""

    seq: int = 0
    ts: float = 0.0
    actor: str = "human"
    action: str = ""
    screenshot_ref: str | None = None
    cursor: list[int] | None = None
    active_app: str | None = None
    active_window: str | None = None
    enrichment: StepEnrichment = field(default_factory=StepEnrichment)
    value: StepValue | None = None
    narration: str | None = None
    outcome: str = "unknown"
    redacted: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "TrajectoryStep":
        if not isinstance(payload, dict):
            return cls()
        cursor = payload.get("cursor")
        parsed_cursor: list[int] | None = None
        if isinstance(cursor, (list, tuple)) and len(cursor) == 2:
            try:
                parsed_cursor = [int(cursor[0]), int(cursor[1])]
            except (TypeError, ValueError):
                parsed_cursor = None
        return cls(
            seq=int(payload.get("seq") or 0),
            ts=float(payload.get("ts") or 0.0),
            actor=str(payload.get("actor") or "human"),
            action=str(payload.get("action") or ""),
            screenshot_ref=payload.get("screenshot_ref"),
            cursor=parsed_cursor,
            active_app=payload.get("active_app"),
            active_window=payload.get("active_window"),
            enrichment=StepEnrichment.from_payload(payload.get("enrichment")),
            value=StepValue.from_payload(payload.get("value")),
            narration=payload.get("narration"),
            outcome=str(payload.get("outcome") or "unknown"),
            redacted=bool(payload.get("redacted", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "ts": self.ts,
            "actor": self.actor,
            "action": self.action,
            "screenshot_ref": self.screenshot_ref,
            "cursor": list(self.cursor) if self.cursor else None,
            "active_app": self.active_app,
            "active_window": self.active_window,
            "enrichment": self.enrichment.to_dict(),
            "value": self.value.to_dict() if self.value else None,
            "narration": self.narration,
            "outcome": self.outcome,
            "redacted": self.redacted,
        }

    @property
    def is_mutating(self) -> bool:
        return self.action in MUTATING_VERBS

    @property
    def is_destructive(self) -> bool:
        return self.action in DESTRUCTIVE_VERBS


@dataclass
class WorkflowRecording:
    """A full recording as returned by RECORDING_DATA / RECORDING_FETCH (§1)."""

    recording_id: str = ""
    version: str = "1.0"
    created_at: float = 0.0
    machine_id: str = ""
    interpretation_route: str = "extract_then_cloud"
    steps: list[TrajectoryStep] = field(default_factory=list)
    redaction_applied: bool = False

    @classmethod
    def from_payload(cls, payload: Any) -> "WorkflowRecording":
        if not isinstance(payload, dict):
            return cls()
        raw_steps = payload.get("steps") or []
        steps = [
            TrajectoryStep.from_payload(s) for s in raw_steps if isinstance(s, dict)
        ]
        # Keep wire order, but defend against a shim that ships steps
        # out-of-order by sorting on seq when every step carries one.
        if steps and all(s.seq for s in steps):
            steps.sort(key=lambda s: s.seq)
        return cls(
            recording_id=str(payload.get("recording_id") or ""),
            version=str(payload.get("version") or "1.0"),
            created_at=float(payload.get("created_at") or 0.0),
            machine_id=str(payload.get("machine_id") or ""),
            interpretation_route=str(
                payload.get("interpretation_route") or "extract_then_cloud"
            ),
            steps=steps,
            redaction_applied=bool(payload.get("redaction_applied", False)),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "version": self.version,
            "created_at": self.created_at,
            "machine_id": self.machine_id,
            "interpretation_route": self.interpretation_route,
            "steps": [s.to_dict() for s in self.steps],
            "redaction_applied": self.redaction_applied,
        }


@dataclass
class RecordingSummary:
    """RECORDING_SUMMARY payload after STOP_RECORDING (§6)."""

    recording_id: str = ""
    step_count: int = 0
    enrichment_coverage: dict[str, int] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @classmethod
    def from_payload(cls, payload: Any) -> "RecordingSummary":
        if not isinstance(payload, dict):
            return cls()
        coverage = payload.get("enrichment_coverage")
        parsed_coverage: dict[str, int] = {}
        if isinstance(coverage, dict):
            for k, v in coverage.items():
                try:
                    parsed_coverage[str(k)] = int(v)
                except (TypeError, ValueError):
                    continue
        return cls(
            recording_id=str(payload.get("recording_id") or ""),
            step_count=int(payload.get("step_count") or 0),
            enrichment_coverage=parsed_coverage,
            duration_seconds=float(payload.get("duration_seconds") or 0.0),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "step_count": self.step_count,
            "enrichment_coverage": dict(self.enrichment_coverage),
            "duration_seconds": self.duration_seconds,
        }
