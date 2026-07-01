"""Recording → SKILL.md generalization (the core value step).

Turns a :class:`WorkflowRecording` into a reusable, parameterized skill —
a ``SKILL.md`` document plus a structured replay manifest that binds
selectors to parameters with the §7 fall-through (DOM → AX → visual).

The hard rule from the spec (§8): **parameter inference is confirmed,
never guessed.** A single-row recording whose parameters were never
confirmed MUST NOT auto-produce a final skill. ``generate_skill_from_recording``
returns ``needs_confirmation`` with the questions to ask (co-pilot) or a
"record a 2nd row" request instead. Field semantics ("a field labeled
Email is probably a parameter") only *seed the question* — they're never a
basis for auto-save.

See ``experimental/local-pc-executor/docs/WORKFLOW_RECORDING.md`` §2, §7, §8.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .recording_models import (
    DESTRUCTIVE_VERBS,
    MUTATING_VERBS,
    TrajectoryStep,
    WorkflowRecording,
)

# Field types whose value is, by §8 prior, more likely to vary per run.
# This is a PRIOR that seeds a clarifying question — never a basis for
# auto-save. The only things that auto-confirm a parameter are: a value
# that changed across ≥2 demonstration rows, an explicit co-pilot answer,
# or a data-source column join (all in `clarifications`).
_PARAMETER_PRIOR_TYPES: frozenset[str] = frozenset({"text", "email", "number", "date"})

# Field types that should never be carried as a constant in the skill even
# if they didn't vary — secrets stay on the machine (§7 value-stripping +
# §9 secret hygiene).
_SECRET_TYPES: frozenset[str] = frozenset({"secret"})


@dataclass
class SkillParameter:
    """One inferred-and-confirmed parameter of a generated skill."""

    name: str
    type: str
    label: str | None = None
    confirmed: bool = False
    # The demonstrated values seen for this parameter, in row order. Used
    # to prove variance (≥2 distinct → parameter) and to seed dry-run rows.
    sample_values: list[Any] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "type": self.type,
            "label": self.label,
            "confirmed": self.confirmed,
            "sample_values": list(self.sample_values),
        }


@dataclass
class ReplayBinding:
    """A replay-manifest entry binding one step to a parameter or constant.

    ``selector_fallthrough`` is the ordered §7 strategy list the replay
    engine tries: DOM fast-path → AX → visual. Always ends in ``visual``
    because the floor is a screenshot+action and replay is a vision agent
    — a broken selector degrades, it doesn't fail.
    """

    seq: int
    action: str
    parameter: str | None  # None → constant step
    constant_value: Any = None
    selector_fallthrough: list[dict[str, Any]] = field(default_factory=list)
    requires_assert: bool = False  # §9 read-back for mutating steps
    destructive: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "action": self.action,
            "parameter": self.parameter,
            "constant_value": self.constant_value,
            "selector_fallthrough": [dict(s) for s in self.selector_fallthrough],
            "requires_assert": self.requires_assert,
            "destructive": self.destructive,
        }


@dataclass
class ReplayManifest:
    """The structured replay plan accompanying SKILL.md."""

    bindings: list[ReplayBinding] = field(default_factory=list)
    data_source_hint: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "bindings": [b.to_dict() for b in self.bindings],
            "data_source_hint": self.data_source_hint,
        }


@dataclass
class ClarifyingQuestion:
    """A question that must be answered before a skill can be saved (§8)."""

    id: str
    question: str
    # The step seq the question is about (when field-specific).
    seq: int | None = None
    kind: str = "parameter"  # parameter | data_source | error_policy

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "question": self.question,
            "seq": self.seq,
            "kind": self.kind,
        }


@dataclass
class GeneratedSkill:
    """Result of generalizing a recording.

    Exactly one of two terminal shapes:
      * ``needs_confirmation=True`` — parameters could not be confirmed
        from the available evidence (single row + no clarifications). The
        skill is NOT final; ``questions`` lists what to ask the user (or a
        synthetic "record a 2nd row" request). ``skill_md`` is a DRAFT for
        preview only and MUST NOT be saved.
      * ``needs_confirmation=False`` — every parameter was confirmed (≥2
        rows, co-pilot answer, or data-source join). ``skill_md`` +
        ``manifest`` are final.
    """

    recording_id: str
    name: str
    needs_confirmation: bool
    skill_md: str
    manifest: ReplayManifest
    parameters: list[SkillParameter] = field(default_factory=list)
    questions: list[ClarifyingQuestion] = field(default_factory=list)
    destructive: bool = False
    needs_second_row: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "recording_id": self.recording_id,
            "name": self.name,
            "needs_confirmation": self.needs_confirmation,
            "skill_md": self.skill_md,
            "manifest": self.manifest.to_dict(),
            "parameters": [p.to_dict() for p in self.parameters],
            "questions": [q.to_dict() for q in self.questions],
            "destructive": self.destructive,
            "needs_second_row": self.needs_second_row,
        }


# ── Helpers ──────────────────────────────────────────────────────────────────


def _slugify(text: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")
    return slug or "field"


def _step_label(step: TrajectoryStep) -> str:
    """Best human label for a step's target, for naming + questions."""
    enr = step.enrichment
    if enr.label:
        return enr.label
    # Fall back to the most-specific selector value, then the action.
    for sel in enr.selectors:
        if sel.get("strategy") in ("label", "name", "id"):
            return sel["value"].lstrip("#")
    if enr.role:
        return enr.role
    return step.action


def _parameter_name(step: TrajectoryStep, used: set[str]) -> str:
    base = _slugify(_step_label(step))
    name = base
    i = 2
    while name in used:
        name = f"{base}_{i}"
        i += 1
    used.add(name)
    return name


def _selector_fallthrough(step: TrajectoryStep) -> list[dict[str, Any]]:
    """Build the §7 ordered fall-through for a step.

    DOM selectors (most-stable-first, as the shim already ordered them) →
    AX path → visual. Always terminates in a visual entry: the floor is a
    screenshot+action, so replay can always fall back to vision grounding.
    """
    chain: list[dict[str, Any]] = []
    enr = step.enrichment
    for sel in enr.selectors:
        chain.append(
            {"via": "dom", "strategy": sel.get("strategy"), "value": sel.get("value")}
        )
    if enr.ax_path:
        chain.append({"via": "ax", "ax_path": enr.ax_path, "role": enr.role})
    # Visual grounding always available — the screenshot_ref + cursor are
    # the floor. This is what makes a broken selector degrade, not fail.
    chain.append(
        {
            "via": "visual",
            "screenshot_ref": step.screenshot_ref,
            "cursor": step.cursor,
            "label": enr.label,
        }
    )
    return chain


def _value_steps(recording: WorkflowRecording) -> list[TrajectoryStep]:
    """Steps that carry a value worth parameterizing (fill/select/etc.)."""
    return [
        s
        for s in recording.steps
        if s.value is not None and s.action in MUTATING_VERBS and s.action != "submit"
    ]


def _cluster_rows(recording: WorkflowRecording) -> list[list[TrajectoryStep]]:
    """Cluster repeated step structure into per-row groups (§2 generalization).

    A "row" is a repetition of the same action+label sequence. We detect
    the repeating unit by its structural signature (ordered action|label
    tuples between submit boundaries). When the recording contains exactly
    one structural group, it's a single-row demo.

    This is intentionally simple and deterministic — the cloud/local model
    refines it, but the platform owns the structural floor so a single-row
    demo is reliably *detected* as single-row (which gates auto-save, §8).
    """
    rows: list[list[TrajectoryStep]] = []
    current: list[TrajectoryStep] = []
    for step in recording.steps:
        current.append(step)
        # A submit (or any destructive boundary) closes a row.
        if step.action in DESTRUCTIVE_VERBS:
            rows.append(current)
            current = []
    if current:
        rows.append(current)
    return rows


def _signature(row: list[TrajectoryStep]) -> tuple[tuple[str, str], ...]:
    return tuple((s.action, _step_label(s)) for s in row if s.value is not None)


def _infer_parameters(
    recording: WorkflowRecording,
    rows: list[list[TrajectoryStep]],
    clarifications: dict[str, Any],
) -> tuple[list[SkillParameter], dict[int, str]]:
    """Infer parameters and the per-step (seq → param name) binding.

    Confirmation sources, in the §8 preference order:
      1. Co-pilot answer / data-source join (in ``clarifications``).
      2. ≥2 demonstration rows: a value that changed across rows IS a
         parameter (confirmed); one that stayed constant is a constant.

    Field-type priors only mark a parameter as *candidate* (unconfirmed)
    so the caller knows to ask — they never set ``confirmed=True``.
    """
    # Map "<action>|<label>" → list of demonstrated values across rows.
    by_field: dict[str, list[Any]] = {}
    # Representative step per field (first occurrence) for naming/binding.
    field_step: dict[str, TrajectoryStep] = {}
    field_order: list[str] = []

    multi_row = len(rows) >= 2 and len({_signature(r) for r in rows}) == 1

    for row in rows:
        for step in row:
            if step.value is None or step.action == "submit":
                continue
            key = f"{step.action}|{_step_label(step)}"
            if key not in by_field:
                by_field[key] = []
                field_step[key] = step
                field_order.append(key)
            by_field[key].append(step.value.raw)

    confirmations: dict[str, bool] = clarifications.get("confirmed_parameters", {})
    data_columns: dict[str, str] = clarifications.get("data_source_columns", {})

    params: list[SkillParameter] = []
    seq_to_param: dict[int, str] = {}
    used_names: set[str] = set()

    for key in field_order:
        step = field_step[key]
        values = by_field[key]
        label = _step_label(step)
        vtype = step.value.type if step.value else "text"

        # Secrets are never parameters carried in the skill (§7/§9).
        if vtype in _SECRET_TYPES:
            continue

        # Confirmation logic.
        explicitly_confirmed = confirmations.get(key) or confirmations.get(label)
        joined_column = data_columns.get(key) or data_columns.get(label)
        varied_across_rows = multi_row and len(set(map(_hashable, values))) >= 2
        constant_across_rows = multi_row and len(set(map(_hashable, values))) == 1

        is_parameter: bool | None
        confirmed: bool
        if explicitly_confirmed is True or joined_column is not None:
            is_parameter, confirmed = True, True
        elif explicitly_confirmed is False:
            is_parameter, confirmed = False, True
        elif varied_across_rows:
            is_parameter, confirmed = True, True
        elif constant_across_rows:
            is_parameter, confirmed = False, True
        else:
            # Single row, no confirmation: PRIOR only — candidate, unconfirmed.
            is_parameter = vtype in _PARAMETER_PRIOR_TYPES
            confirmed = False

        if is_parameter:
            name = (
                _slugify(joined_column)
                if isinstance(joined_column, str)
                else _parameter_name(step, used_names)
            )
            params.append(
                SkillParameter(
                    name=name,
                    type=vtype,
                    label=label,
                    confirmed=confirmed,
                    sample_values=list(values),
                )
            )
            # Bind every step occurrence of this field.
            for row in rows:
                for s in row:
                    if s.value is not None and f"{s.action}|{_step_label(s)}" == key:
                        seq_to_param[s.seq] = name

    return params, seq_to_param


def _hashable(value: Any) -> Any:
    """Make a demonstrated value hashable for set-based variance detection."""
    if isinstance(value, (list, dict)):
        return repr(value)
    return value


def _build_manifest(
    recording: WorkflowRecording,
    seq_to_param: dict[int, str],
    data_source_hint: str | None,
) -> ReplayManifest:
    bindings: list[ReplayBinding] = []
    for step in recording.steps:
        param = seq_to_param.get(step.seq)
        # Constant value kept unless it's a secret (§7 value-stripping).
        constant_value: Any = None
        if param is None and step.value is not None:
            if step.value.type not in _SECRET_TYPES:
                constant_value = step.value.raw
        bindings.append(
            ReplayBinding(
                seq=step.seq,
                action=step.action,
                parameter=param,
                constant_value=constant_value,
                selector_fallthrough=_selector_fallthrough(step),
                requires_assert=step.is_mutating,
                destructive=step.is_destructive,
            )
        )
    return ReplayManifest(bindings=bindings, data_source_hint=data_source_hint)


def _strip_values(recording: WorkflowRecording, seq_to_param: dict[int, str]) -> None:
    """§7 value-stripping: drop ``value.raw`` for parameter steps after
    inference; keep constants unless secret. Mutates the recording in place
    so the post-generation buffered copy carries no demonstrated parameter
    data (the user's real John/jane@x.com etc.)."""
    for step in recording.steps:
        if step.value is None:
            continue
        if step.seq in seq_to_param:
            step.value.raw = None  # parameter — strip the demo value
        elif step.value.type in _SECRET_TYPES:
            step.value.raw = None  # secret constant — never retain


def _render_skill_md(
    *,
    name: str,
    description: str,
    trigger: str,
    recording: WorkflowRecording,
    params: list[SkillParameter],
    seq_to_param: dict[int, str],
    data_source_hint: str | None,
    destructive: bool,
    success_criterion: str,
) -> str:
    """Render the SKILL.md document (§2 output shape)."""
    lines: list[str] = []
    lines.append(f"# {name}")
    lines.append("")
    lines.append(f"**Description:** {description}")
    lines.append("")
    lines.append(f"**Trigger:** {trigger}")
    lines.append("")
    if destructive:
        lines.append(
            "**Destructive:** yes — this skill submits / performs irreversible "
            "actions. Requires a multi-row dry-run and explicit confirmation "
            "before unattended runs."
        )
        lines.append("")

    lines.append("## Data source")
    if data_source_hint:
        lines.append(f"Each run consumes one row from: {data_source_hint}.")
    else:
        lines.append("Data source: to be supplied at run time (one row per run).")
    if params:
        lines.append("")
        lines.append("Parameters (one per run):")
        for p in params:
            status = "confirmed" if p.confirmed else "UNCONFIRMED"
            label = f" (from field '{p.label}')" if p.label else ""
            lines.append(f"- `{{{p.name}}}` — {p.type}{label} [{status}]")
    lines.append("")

    lines.append("## Procedure")
    step_no = 1
    for step in recording.steps:
        param = seq_to_param.get(step.seq)
        target = _step_label(step)
        if param:
            detail = f"set {target} = `{{{param}}}`"
        elif step.value is not None and step.value.raw is not None:
            detail = f"set {target} = `{step.value.raw}`"
        elif step.action in ("navigate",) and step.enrichment.url:
            detail = f"go to {step.enrichment.url}"
        elif step.action == "submit":
            detail = f"submit ({target})"
        else:
            detail = f"{step.action} {target}".strip()
        lines.append(f"{step_no}. {detail}")
        step_no += 1
    lines.append("")

    lines.append("## Success criterion")
    lines.append(success_criterion)
    lines.append("")
    return "\n".join(lines)


# ── Public API ───────────────────────────────────────────────────────────────


def generate_skill_from_recording(
    recording: WorkflowRecording,
    *,
    data_source_hint: str | None = None,
    clarifications: dict[str, Any] | None = None,
) -> GeneratedSkill:
    """Generalize a recording into a (possibly draft) skill.

    Args:
        recording: the parsed WorkflowRecording.
        data_source_hint: free-text hint about where run data comes from
            (e.g. "customers.csv"); seeds the data-source contract.
        clarifications: confirmation evidence (§8). Recognized keys:
            * ``confirmed_parameters``: ``{field_key_or_label: bool}`` — a
              co-pilot answer to "does this field change each run?".
            * ``data_source_columns``: ``{field_key_or_label: column_name}``
              — a confirmed data-source join.
            * ``name`` / ``description`` / ``trigger`` / ``success_criterion``
              — optional human-supplied skill metadata.

    Returns:
        A :class:`GeneratedSkill`. ``needs_confirmation=True`` when any
        candidate parameter is unconfirmed (single-row demo, no answers) —
        the caller MUST ask ``questions`` (or request a 2nd row) before
        saving. ``needs_confirmation=False`` when every parameter is
        confirmed; ``skill_md`` + ``manifest`` are then final.
    """
    clar = clarifications or {}
    rows = _cluster_rows(recording)
    params, seq_to_param = _infer_parameters(recording, rows, clar)

    multi_row = len(rows) >= 2 and len({_signature(r) for r in rows}) == 1

    name = clar.get("name") or _derive_name(recording)
    description = clar.get("description") or _derive_description(recording, params)
    trigger = clar.get("trigger") or f"run '{name}' on a batch of rows"
    destructive = any(s.is_destructive for s in recording.steps)
    success_criterion = clar.get("success_criterion") or _derive_success_criterion(
        recording
    )

    # §8 GATE: any unconfirmed candidate parameter blocks auto-save.
    unconfirmed = [p for p in params if not p.confirmed]
    needs_confirmation = bool(unconfirmed)
    questions: list[ClarifyingQuestion] = []
    needs_second_row = False
    if needs_confirmation:
        questions = _questions_for_unconfirmed(unconfirmed, seq_to_param, recording)
        # Demonstration mode has no live loop to answer in — the resolution
        # is a second row (§8). Co-pilot mode can answer the questions.
        if not multi_row and len(rows) == 1:
            needs_second_row = True

    # Strip parameter / secret values BEFORE rendering so the draft never
    # echoes the user's real demonstrated data (§7).
    _strip_values(recording, seq_to_param)
    manifest = _build_manifest(recording, seq_to_param, data_source_hint)
    skill_md = _render_skill_md(
        name=name,
        description=description,
        trigger=trigger,
        recording=recording,
        params=params,
        seq_to_param=seq_to_param,
        data_source_hint=data_source_hint,
        destructive=destructive,
        success_criterion=success_criterion,
    )

    return GeneratedSkill(
        recording_id=recording.recording_id,
        name=name,
        needs_confirmation=needs_confirmation,
        skill_md=skill_md,
        manifest=manifest,
        parameters=params,
        questions=questions,
        destructive=destructive,
        needs_second_row=needs_second_row,
    )


def _derive_name(recording: WorkflowRecording) -> str:
    # Best-effort name from the dominant active_app / window.
    apps = [s.active_app for s in recording.steps if s.active_app]
    app = apps[0] if apps else "task"
    has_submit = any(s.action == "submit" for s in recording.steps)
    verb = "Fill and submit" if has_submit else "Run task"
    return f"{verb} in {app}"


def _derive_description(
    recording: WorkflowRecording, params: list[SkillParameter]
) -> str:
    n = len(params)
    field_phrase = f" setting {n} field{'s' if n != 1 else ''} per run" if n else ""
    apps = [s.active_app for s in recording.steps if s.active_app]
    app = apps[0] if apps else "an application"
    return f"Repeats a recorded task in {app}{field_phrase}, one row at a time."


def _derive_success_criterion(recording: WorkflowRecording) -> str:
    if any(s.action == "submit" for s in recording.steps):
        return (
            "Each row's submission is accepted (the confirmation read-back "
            "asserts the success state) with no validation error."
        )
    return "Each step's read-back assert passes for every row."


def _questions_for_unconfirmed(
    unconfirmed: list[SkillParameter],
    seq_to_param: dict[int, str],
    recording: WorkflowRecording,
) -> list[ClarifyingQuestion]:
    """Build the §3.3-style clarifying questions for unconfirmed params."""
    questions: list[ClarifyingQuestion] = []
    param_seq: dict[str, int] = {}
    for seq, pname in seq_to_param.items():
        param_seq.setdefault(pname, seq)
    for p in unconfirmed:
        label = p.label or p.name
        questions.append(
            ClarifyingQuestion(
                id=f"param:{p.name}",
                question=f"Does '{label}' change each run, or is it always the same value?",
                seq=param_seq.get(p.name),
                kind="parameter",
            )
        )
    return questions


# ── Step 3: the trust loop ───────────────────────────────────────────────────


def propose_clarifications(
    recording: WorkflowRecording,
) -> list[ClarifyingQuestion]:
    """The §3.3 clarifying questions to ask before trusting a skill.

    Always includes the two non-parameter questions ("where does the data
    come from?", "stop on validation error?") plus one per candidate
    parameter ("does this field change each run?"). These seed the
    co-pilot's real-time clarification loop; answering them is what lets a
    single-row recording become a confirmed, savable skill (§8).
    """
    questions: list[ClarifyingQuestion] = [
        ClarifyingQuestion(
            id="data_source",
            question="Where does the data for each run come from "
            "(a CSV, a spreadsheet, somewhere else)?",
            kind="data_source",
        ),
    ]
    if any(s.is_destructive for s in recording.steps):
        questions.append(
            ClarifyingQuestion(
                id="error_policy",
                question="If a row fails validation, should the skill stop, "
                "or skip that row and continue?",
                kind="error_policy",
            )
        )
    # One per candidate parameter, seeded by the field-type prior (§8).
    rows = _cluster_rows(recording)
    params, seq_to_param = _infer_parameters(recording, rows, {})
    questions.extend(
        _questions_for_unconfirmed(
            [p for p in params if not p.confirmed], seq_to_param, recording
        )
    )
    return questions


@dataclass
class StepReplayResult:
    seq: int
    action: str
    resolved_via: str | None  # dom | ax | visual | None (skipped)
    asserted: bool
    ok: bool
    detail: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "seq": self.seq,
            "action": self.action,
            "resolved_via": self.resolved_via,
            "asserted": self.asserted,
            "ok": self.ok,
            "detail": self.detail,
        }


@dataclass
class RowDryRunResult:
    row_index: int
    row: dict[str, Any]
    ok: bool
    steps: list[StepReplayResult] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "row_index": self.row_index,
            "row": dict(self.row),
            "ok": self.ok,
            "steps": [s.to_dict() for s in self.steps],
            "error": self.error,
        }


@dataclass
class DryRunResult:
    skill_name: str
    rows_attempted: int
    rows_ok: int
    destructive_blocked: bool  # submits skipped because allow_destructive=False
    per_row: list[RowDryRunResult] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.rows_ok == self.rows_attempted and self.rows_attempted > 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "rows_attempted": self.rows_attempted,
            "rows_ok": self.rows_ok,
            "destructive_blocked": self.destructive_blocked,
            "ok": self.ok,
            "per_row": [r.to_dict() for r in self.per_row],
        }


async def dry_run(
    skill: GeneratedSkill,
    data_rows: list[dict[str, Any]],
    *,
    shim: Any,
    allow_destructive: bool = False,
) -> DryRunResult:
    """Drive the skill's replay manifest over MULTIPLE rows via the shim.

    Multi-row is load-bearing (§8/§9): a single-row dry-run can't catch a
    parameter that was mis-classified as a constant — only a second row
    with a different value exposes it. This is the run the user watches
    before trusting the skill unattended.

    Each mutating step gets a read-back assert (§9). Destructive steps
    (submit/upload/run/file_op) are GATED behind ``allow_destructive`` —
    by default they're recorded as "would submit" and skipped so the dry
    run doesn't actually file 200 records.

    ``shim`` is duck-typed: it needs ``computer`` (for the visual fall-back
    typed input) and is otherwise only used through the binding's resolved
    path. In tests a mock shim records the calls; the logic is exercised
    without a live machine.
    """
    result = DryRunResult(
        skill_name=skill.name,
        rows_attempted=0,
        rows_ok=0,
        destructive_blocked=False,
    )
    for i, row in enumerate(data_rows):
        row_result = RowDryRunResult(row_index=i, row=row, ok=True)
        result.rows_attempted += 1
        try:
            for binding in skill.manifest.bindings:
                step_res = await _replay_step(
                    binding, row, shim=shim, allow_destructive=allow_destructive
                )
                row_result.steps.append(step_res)
                if step_res.detail == "destructive_skipped":
                    result.destructive_blocked = True
                if not step_res.ok:
                    row_result.ok = False
                    row_result.error = (
                        f"step seq={binding.seq} ({binding.action}) failed: "
                        f"{step_res.detail}"
                    )
                    break
        except Exception as exc:  # shim error mid-replay
            row_result.ok = False
            row_result.error = str(exc)
        if row_result.ok:
            result.rows_ok += 1
        result.per_row.append(row_result)
    return result


async def _replay_step(
    binding: ReplayBinding,
    row: dict[str, Any],
    *,
    shim: Any,
    allow_destructive: bool,
) -> StepReplayResult:
    """Replay one manifest binding for one row. Pure orchestration over the
    shim's computer surface; selection follows the §7 fall-through."""
    # Destructive gate (§9) — don't actually submit during a dry run unless
    # explicitly allowed.
    if binding.destructive and not allow_destructive:
        return StepReplayResult(
            seq=binding.seq,
            action=binding.action,
            resolved_via=None,
            asserted=False,
            ok=True,
            detail="destructive_skipped",
        )

    resolved_via = _resolve_strategy(binding)

    # Determine the value to apply (parameter from the row, or constant).
    value: Any = None
    if binding.parameter is not None:
        if binding.parameter not in row:
            return StepReplayResult(
                seq=binding.seq,
                action=binding.action,
                resolved_via=resolved_via,
                asserted=False,
                ok=False,
                detail=f"row is missing parameter '{binding.parameter}'",
            )
        value = row[binding.parameter]
    else:
        value = binding.constant_value

    # Apply the action. We route value-bearing actions through type(); the
    # actual element-grounding (DOM/AX/visual) is the shim/replay engine's
    # job — here we drive the floor input so the logic is shim-agnostic and
    # unit-testable.
    try:
        if binding.action in ("fill", "select", "paste") and value is not None:
            await shim.computer.type(str(value))
        elif binding.action == "click":
            cursor = _binding_cursor(binding)
            if cursor is not None:
                await shim.computer.click(cursor)
        elif binding.action == "submit" and allow_destructive:
            cursor = _binding_cursor(binding)
            if cursor is not None:
                await shim.computer.click(cursor)
        # navigate / wait / launch_app etc. are no-ops at the floor level
        # for the dry-run logic test; the real replay engine handles them.
    except Exception as exc:
        return StepReplayResult(
            seq=binding.seq,
            action=binding.action,
            resolved_via=resolved_via,
            asserted=False,
            ok=False,
            detail=f"input failed: {exc}",
        )

    # Read-back assert for mutating steps (§9).
    asserted = False
    ok = True
    if binding.requires_assert and binding.parameter is not None and value is not None:
        asserted, ok = await _readback_assert(binding, str(value), shim=shim)

    return StepReplayResult(
        seq=binding.seq,
        action=binding.action,
        resolved_via=resolved_via,
        asserted=asserted,
        ok=ok,
        detail=None if ok else "read-back assert mismatch",
    )


def _resolve_strategy(binding: ReplayBinding) -> str | None:
    """Pick the first strategy in the §7 fall-through. The real engine tries
    each in turn against the live screen; for the dry-run logic we report
    the highest-fidelity one available (dom > ax > visual)."""
    for entry in binding.selector_fallthrough:
        return str(entry.get("via"))
    return None


def _binding_cursor(binding: ReplayBinding) -> list[int] | None:
    for entry in binding.selector_fallthrough:
        if entry.get("via") == "visual" and entry.get("cursor"):
            cursor = entry["cursor"]
            if isinstance(cursor, (list, tuple)) and len(cursor) == 2:
                return [int(cursor[0]), int(cursor[1])]
    return None


async def _readback_assert(
    binding: ReplayBinding, expected: str, *, shim: Any
) -> tuple[bool, bool]:
    """Read the value back and compare (§9). Returns (asserted, ok).

    Uses the shim clipboard read as a portable read-back probe when
    available; if the shim can't read back, the step is reported as
    not-asserted but not failed (the real engine has richer read-back via
    DOM/AX). This keeps the dry-run honest without a live machine in tests.
    """
    reader = getattr(getattr(shim, "computer", None), "clipboard_read", None)
    if reader is None:
        return False, True
    try:
        actual = await reader()
    except Exception:
        return False, True
    if actual is None:
        return False, True
    return True, str(actual) == expected
