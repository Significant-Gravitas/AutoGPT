"""The task contract: one schema every task follows, whatever the work is.

What was asked (``goal``), what done looks like (``done_when``), how it is
going (``plan``), the proof (``evidence``), the outcome (``result``), and,
when it can't go on, the one question for the user (``ask``) or why it is
stuck (``blocker``). Task-specific content lives in the values, never in the
shape. ``raw_status`` is the detailed status and ``state`` the few lanes it
folds into; both are derived, never stored.

Nothing here insists on a size: a quick task can have one check and no plan.
Evidence kinds, blocker causes and next actions are open strings; the
``SUGGESTED_*`` values are what the UI knows how to dress up.

Everything here is pure: the store (``contract_db``) loads, applies one of
these functions and writes the result back with a version check.
"""

import json
import logging
import re
import uuid
from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.api.features.tasks.models import (
    Ask,
    Attachment,
    Blocker,
    Contract,
    ContractState,
    Criterion,
    CriterionStatus,
    Evidence,
    PendingInput,
    Plan,
    Result,
    Step,
    StepRow,
    StepStatus,
    Task,
    TaskStatus,
)

logger = logging.getLogger(__name__)

# Only a ceiling against runaway lists; the sizes worth aiming for are
# guidance in the prompt, not rules.
MAX_ITEMS = 20
MAX_OPTIONS = 6
LINE_CHARS = 200
GOAL_CHARS = 600
RESULT_CHARS = 280
SHORT_CHARS = 72
# A working task with no write for this long has gone quiet.
NO_SIGNAL_MINUTES = 15

SUGGESTED_EVIDENCE_KINDS = (
    "link",
    "file",
    "record",
    "screenshot",
    "metric",
    "quote",
    "change",
)
SUGGESTED_CAUSES = (
    "permission",
    "missing_input",
    "external_error",
    "limit",
    "no_signal",
)
SUGGESTED_ACTIONS = (
    "retry_failed_part",
    "skip",
    "allow",
    "do_in_chat",
    "change_goal",
    "take_over",
)

# A patch the contract refuses; the message tells the caller why. A plain
# ValueError, since the DatabaseManager RPC only re-raises the built-in types
# with their message.
ContractError = ValueError

__all__ = [
    "Ask",
    "Blocker",
    "Contract",
    "Criterion",
    "CriterionStatus",
    "Evidence",
    "Result",
    "Step",
    "StepRow",
    "StepStatus",
]


class ContractPatch(BaseModel):
    """One field's new value; ``append`` adds ``value`` to a list instead."""

    field: str
    value: Any = None
    op: str = "set"


class RowChange(BaseModel):
    title: str
    status: StepStatus = StepStatus.PENDING


class StepChange(BaseModel):
    id: str
    status: StepStatus | None = None
    rows: list[RowChange] | None = None


class CriterionChange(BaseModel):
    id: str
    status: CriterionStatus
    evidence_ids: list[str] = Field(default_factory=list)


class BlockerInput(BaseModel):
    what_stopped_it: str
    cause: str | None = None
    done_so_far: str | None = None
    next_actions: list[str] | None = None


class ResultInput(BaseModel):
    summary: str
    primary_evidence_id: str | None = None


class TaskPatch(BaseModel):
    """What ``update_task`` takes: any subset, applied in this order."""

    goal: str | None = None
    done_when: list[str] | None = None
    plan: list[str] | None = None
    steps: list[StepChange] = Field(default_factory=list)
    criteria: list[CriterionChange] = Field(default_factory=list)
    blocker: BlockerInput | None = None
    result: ResultInput | None = None


def one_line(text: str | None, limit: int = LINE_CHARS) -> str:
    flat = " ".join((text or "").split())
    if len(flat) <= limit:
        return flat
    cut = flat[: limit - 1]
    space = cut.rfind(" ")
    if space > limit // 2:
        cut = cut[:space]
    return cut.rstrip(" ,;:-") + "…"


def _short_id(prefix: str, taken: set[str]) -> str:
    n = len(taken) + 1
    while f"{prefix}{n}" in taken:
        n += 1
    return f"{prefix}{n}"


def _texts(values: list[str]) -> list[str]:
    return [one_line(v) for v in values if v and v.strip()][:MAX_ITEMS]


def revise(
    previous: Contract,
    goal: str | None = None,
    done_when: list[str] | None = None,
    plan: list[str] | None = None,
) -> tuple[Contract, list[ContractPatch]]:
    """Replace whichever of goal, done_when and plan are given. Criteria and
    steps whose text didn't change keep their status, and evidence and the
    open question follow them to their new ids."""
    goal_line = previous.goal if goal is None else one_line(goal, GOAL_CHARS)
    if not goal_line:
        raise ContractError("Give the goal in one sentence.")
    criteria = previous.done_when
    criteria_ids = {x.id: x.id for x in criteria}
    if done_when is not None:
        old = {x.text: x for x in previous.done_when}
        criteria, criteria_ids = [], {}
        for i, text in enumerate(_texts(done_when), 1):
            kept = old.get(text)
            if kept is not None:
                criteria_ids[kept.id] = f"d{i}"
            criteria.append(
                (kept or Criterion(id="", text=text)).model_copy(update={"id": f"d{i}"})
            )
    steps = previous.plan
    step_ids = {s.id: s.id for s in steps}
    if plan is not None:
        old_steps = {s.title: s for s in previous.plan}
        steps, step_ids = [], {}
        for i, title in enumerate(_texts(plan), 1):
            kept_step = old_steps.get(title)
            if kept_step is not None:
                step_ids[kept_step.id] = f"s{i}"
            steps.append(
                (kept_step or Step(id="", title=title)).model_copy(
                    update={"id": f"s{i}"}
                )
            )
    evidence = [
        e.model_copy(
            update={
                "criterion_ids": [
                    criteria_ids[x] for x in e.criterion_ids if x in criteria_ids
                ]
            }
        )
        for e in previous.evidence
    ]
    ask = previous.ask
    if ask is not None and ask.blocking_step_id:
        ask = ask.model_copy(
            update={"blocking_step_id": step_ids.get(ask.blocking_step_id)}
        )
    rescoped = goal_line != previous.goal and bool(previous.done_when or previous.plan)
    contract = previous.model_copy(
        update={
            "goal": goal_line,
            "previous_goal": previous.goal if rescoped else previous.previous_goal,
            "done_when": criteria,
            "plan": steps,
            "evidence": evidence,
            "ask": ask,
        }
    )
    patches = [
        ContractPatch(field=name, value=_dump(getattr(contract, name)))
        for name in ("goal", "previous_goal", "done_when", "plan")
        if _dump(getattr(previous, name)) != _dump(getattr(contract, name))
    ]
    return contract, patches


def _rows(current: list[StepRow], changes: list[RowChange]) -> list[StepRow]:
    """Rows matched by title: a known title takes the new status, a new one
    is added."""
    rows = list(current)
    index = {r.title: i for i, r in enumerate(rows)}
    for change in changes:
        title = one_line(change.title, 120)
        if not title:
            continue
        if title in index:
            rows[index[title]] = rows[index[title]].model_copy(
                update={"status": change.status}
            )
        elif len(rows) < MAX_ITEMS:
            index[title] = len(rows)
            rows.append(
                StepRow(
                    id=_short_id("r", {r.id for r in rows}),
                    title=title,
                    status=change.status,
                )
            )
    return rows


def update_step(
    contract: Contract,
    step_id: str,
    status: StepStatus | None,
    rows: list[RowChange] | None = None,
) -> tuple[Contract, list[ContractPatch]]:
    """Move one step, and its parallel rows. Starting a step finishes the one
    that was running, since a worker does one step at a time; a step whose
    rows have all settled settles with them."""
    _find(contract.plan, step_id, "step")
    plan = []
    for s in contract.plan:
        if s.id == step_id:
            new_rows = _rows(s.rows, rows) if rows else s.rows
            new_status = status or _status_from_rows(new_rows) or s.status
            plan.append(s.model_copy(update={"status": new_status, "rows": new_rows}))
        elif status == StepStatus.RUNNING and s.status == StepStatus.RUNNING:
            plan.append(s.model_copy(update={"status": StepStatus.DONE}))
        else:
            plan.append(s)
    updated = contract.model_copy(update={"plan": plan})
    return updated, [ContractPatch(field="plan", value=_dump(plan))]


_SETTLED_STEP = (StepStatus.DONE, StepStatus.SKIPPED, StepStatus.FAILED)


def _status_from_rows(rows: list[StepRow]) -> StepStatus | None:
    if not rows:
        return None
    if all(r.status in _SETTLED_STEP for r in rows):
        failed = any(r.status == StepStatus.FAILED for r in rows)
        return StepStatus.FAILED if failed else StepStatus.DONE
    if any(r.status != StepStatus.PENDING for r in rows):
        return StepStatus.RUNNING
    return None


def _settle_running(contract: Contract, to: StepStatus) -> Contract:
    """Whatever was running ends as *to*, rows included."""

    def settle(status: StepStatus) -> StepStatus:
        return to if status == StepStatus.RUNNING else status

    plan = [
        s.model_copy(
            update={
                "status": settle(s.status),
                "rows": [
                    r.model_copy(update={"status": settle(r.status)}) for r in s.rows
                ],
            }
        )
        for s in contract.plan
    ]
    return contract.model_copy(update={"plan": plan})


def add_evidence(
    contract: Contract,
    kind: str,
    title: str,
    summary: str,
    *,
    ref: str | None = None,
    url: str | None = None,
    criterion_ids: list[str] | None = None,
    attachment_id: str | None = None,
    source_activity_id: str | None = None,
    auto: bool = False,
) -> tuple[Contract, Evidence, list[ContractPatch]]:
    criterion_ids = list(dict.fromkeys(criterion_ids or []))
    for cid in criterion_ids:
        _find(contract.done_when, cid, "done_when item")
    item = Evidence(
        id=_short_id("e", {e.id for e in contract.evidence}),
        kind=one_line(kind, 40).lower() or "record",
        title=one_line(title, 120) or "Untitled",
        summary=one_line(summary) or one_line(title),
        ref=ref,
        url=url,
        criterion_ids=criterion_ids,
        attachment_id=attachment_id,
        source_activity_id=source_activity_id,
        auto=auto,
    )
    criteria = [
        (
            c.model_copy(update={"evidence_ids": [*c.evidence_ids, item.id]})
            if c.id in criterion_ids
            else c
        )
        for c in contract.done_when
    ]
    updated = contract.model_copy(
        update={"evidence": [*contract.evidence, item], "done_when": criteria}
    )
    patches = [ContractPatch(field="evidence", value=_dump(item), op="append")]
    if criterion_ids:
        patches.append(ContractPatch(field="done_when", value=_dump(criteria)))
    return updated, item, patches


def mark_criterion(
    contract: Contract,
    criterion_id: str,
    status: CriterionStatus,
    evidence_ids: list[str] | None = None,
) -> tuple[Contract, list[ContractPatch]]:
    criterion = _find(contract.done_when, criterion_id, "done_when item")
    known = {e.id for e in contract.evidence}
    missing = [e for e in evidence_ids or [] if e not in known]
    if missing:
        raise ContractError(f"No evidence with id {', '.join(missing)}.")
    linked = list(dict.fromkeys([*criterion.evidence_ids, *(evidence_ids or [])]))
    criteria = [
        (
            c.model_copy(update={"status": status, "evidence_ids": linked})
            if c.id == criterion_id
            else c
        )
        for c in contract.done_when
    ]
    updated = contract.model_copy(update={"done_when": criteria})
    return updated, [ContractPatch(field="done_when", value=_dump(criteria))]


def open_ask(
    contract: Contract,
    question: str,
    *,
    why: str | None = None,
    options: list[str] | None = None,
    default_option: str | None = None,
    blocking_step_id: str | None = None,
    expires_at: datetime | None = None,
) -> tuple[Contract, Ask, list[ContractPatch]]:
    """Open the one question, or fold a new one into the question already
    open, so the user always answers a single ask. With options, there is
    always a default: the worker's pick, else the first."""
    if blocking_step_id:
        _find(contract.plan, blocking_step_id, "step")
    text = (question or "").strip()
    if not text:
        raise ContractError("Ask one question.")
    current = contract.ask
    if current is None:
        choices = _options(options or [])
        ask = Ask(
            id=str(uuid.uuid4()),
            question=text,
            why=one_line(why) if why else None,
            options=choices,
            default_option=_default(default_option, choices),
            blocking_step_id=blocking_step_id,
            expires_at=expires_at,
        )
    else:
        reasons = [r for r in (current.why, one_line(why) if why else None) if r]
        choices = _options([*current.options, *(options or [])])
        ask = current.model_copy(
            update={
                "question": f"{current.question}\n{text}",
                "why": " ".join(dict.fromkeys(reasons)) or None,
                "options": choices,
                "default_option": _default(
                    current.default_option or default_option, choices
                ),
                "blocking_step_id": current.blocking_step_id or blocking_step_id,
                "expires_at": current.expires_at or expires_at,
            }
        )
    updated = contract.model_copy(update={"ask": ask})
    return updated, ask, [ContractPatch(field="ask", value=_dump(ask))]


def _default(pick: str | None, options: list[str]) -> str | None:
    if pick and pick.strip():
        return one_line(pick, 80)
    return options[0] if options else None


def _options(values: list[str]) -> list[str]:
    cleaned = [one_line(v, 80) for v in values if v and v.strip()]
    return list(dict.fromkeys(cleaned))[:MAX_OPTIONS]


def close_ask(contract: Contract) -> tuple[Contract, list[ContractPatch]]:
    if contract.ask is None:
        return contract, []
    return contract.model_copy(update={"ask": None}), [ContractPatch(field="ask")]


def set_blocker(
    contract: Contract, blocker: Blocker
) -> tuple[Contract, list[ContractPatch]]:
    """Record why it stopped; the step it stopped on is marked failed and
    everything finished stays."""
    updated = _settle_running(contract, StepStatus.FAILED).model_copy(
        update={"blocker": blocker}
    )
    patches = [ContractPatch(field="blocker", value=_dump(blocker))]
    if updated.plan != contract.plan:
        patches.append(ContractPatch(field="plan", value=_dump(updated.plan)))
    return updated, patches


def clear_blocker(contract: Contract) -> tuple[Contract, list[ContractPatch]]:
    if contract.blocker is None:
        return contract, []
    updated = contract.model_copy(update={"blocker": None})
    return updated, [ContractPatch(field="blocker")]


def complete(
    contract: Contract, summary: str, primary_evidence_id: str | None = None
) -> tuple[Contract, list[ContractPatch]]:
    lines = [one_line(x, RESULT_CHARS) for x in (summary or "").splitlines() if x]
    lines = [x for x in lines if x]
    if not lines:
        raise ContractError("Give the result in one or two lines.")
    if len(lines) > 2:
        lines = [lines[0], one_line(" ".join(lines[1:]), RESULT_CHARS)]
    if primary_evidence_id:
        _find(contract.evidence, primary_evidence_id, "evidence")
    elif contract.evidence:
        primary_evidence_id = contract.evidence[-1].id
    result = Result(summary="\n".join(lines), primary_evidence_id=primary_evidence_id)
    finished = _settle_running(contract, StepStatus.DONE)
    updated = finished.model_copy(
        update={"result": result, "ask": None, "blocker": None}
    )
    patches = [ContractPatch(field="result", value=_dump(result))]
    if finished.plan != contract.plan:
        patches.append(ContractPatch(field="plan", value=_dump(finished.plan)))
    if contract.ask is not None:
        patches.append(ContractPatch(field="ask"))
    if contract.blocker is not None:
        patches.append(ContractPatch(field="blocker"))
    return updated, patches


_CAUSE_LABELS = {
    "permission": "Needs your permission",
    "missing_input": "Missing something only you have",
    "external_error": "A site or service didn't respond",
    "limit": "Hit a limit",
    "no_signal": "No response for a while",
}


def short_reason(reason: str, cause: str) -> str:
    """The reason for a card, never cut mid-sentence: its first sentence when
    that fits, else a plain label for the cause."""
    first = re.split(r"(?<=[.!?])\s", reason.strip(), maxsplit=1)[0]
    if first and len(first) <= SHORT_CHARS:
        return first
    return _CAUSE_LABELS.get(cause, "Stopped before it finished")


def make_blocker(
    contract: Contract | None,
    what_stopped_it: str,
    *,
    cause: str | None = None,
    done_so_far: str | None = None,
    next_actions: list[str] | None = None,
    tool_name: str | None = None,
) -> Blocker:
    """A blocker in plain words. Whatever the caller leaves out is filled in:
    the cause from the message, what got done from the plan and evidence, and
    the next actions from the cause."""
    reason = one_line(what_stopped_it) or "The work stopped with an error."
    chosen = (
        one_line(cause, 60).lower().replace(" ", "_")
        if cause
        else classify_error(reason)
    ) or "unknown"
    return Blocker(
        done_so_far=(
            one_line(done_so_far)
            if done_so_far
            else progress_line(
                contract.plan if contract else [],
                contract.evidence if contract else [],
            )
        ),
        what_stopped_it=f"{reason} ({tool_name})" if tool_name else reason,
        short=short_reason(reason, chosen),
        cause=chosen,
        next_actions=action_keys(next_actions) or next_actions_for(chosen),
    )


_ACTION_KEY = re.compile(r"^[a-z][a-z0-9_]{1,39}$")


def action_keys(actions: list[str] | None) -> list[str]:
    """The entries that are action keys. Cards render each one as a button
    and post it back to the actions route, so sentences are dropped."""
    keys = (a.strip().lower() for a in actions or [])
    return list(dict.fromkeys(k for k in keys if _ACTION_KEY.match(k)))


def apply_patch(
    contract: Contract, patch: TaskPatch
) -> tuple[Contract, list[ContractPatch]]:
    """One ``update_task`` call: revise, move steps and criteria, then set a
    blocker or the result. Later fields win where they overlap."""
    patches: list[ContractPatch] = []

    def take(step: tuple[Contract, list[ContractPatch]]) -> Contract:
        patches.extend(step[1])
        return step[0]

    if patch.goal is not None or patch.done_when is not None or patch.plan is not None:
        contract = take(revise(contract, patch.goal, patch.done_when, patch.plan))
    for change in patch.steps:
        contract = take(update_step(contract, change.id, change.status, change.rows))
    for change in patch.criteria:
        contract = take(
            mark_criterion(contract, change.id, change.status, change.evidence_ids)
        )
    if patch.blocker is not None:
        b = patch.blocker
        blocker = make_blocker(
            contract,
            b.what_stopped_it,
            cause=b.cause,
            done_so_far=b.done_so_far,
            next_actions=b.next_actions,
        )
        contract = take(set_blocker(contract, blocker))
    if patch.result is not None:
        contract = take(
            complete(contract, patch.result.summary, patch.result.primary_evidence_id)
        )
    return contract, _squash(patches)


def _squash(patches: list[ContractPatch]) -> list[ContractPatch]:
    """Keep the last ``set`` per field, and every ``append``."""
    last = {p.field: i for i, p in enumerate(patches) if p.op == "set"}
    return [p for i, p in enumerate(patches) if p.op != "set" or last[p.field] == i]


_CLOSED = (CriterionStatus.MET, CriterionStatus.SKIPPED)


def all_met(contract: Contract) -> bool:
    return bool(contract.done_when) and all(
        c.status in _CLOSED for c in contract.done_when
    )


# Every detailed status folds into one of the few states here, and only
# here; one not listed counts as working.
STATE_OF = {
    "queued": ContractState.WORKING,
    "running": ContractState.WORKING,
    "idle": ContractState.WORKING,
    "waiting_on_subtasks": ContractState.WORKING,
    "asked": ContractState.NEEDS_YOU,
    "awaiting_accept": ContractState.NEEDS_YOU,
    "blocked": ContractState.STUCK,
    "no_signal": ContractState.STUCK,
    "done": ContractState.DONE,
    "cancelled": ContractState.STOPPED,
}


def raw_status(
    status: TaskStatus,
    contract: Contract | None,
    pending: PendingInput | None,
    *,
    running: bool = False,
    quiet_minutes: float | None = None,
    open_subtasks: int = 0,
) -> str:
    """The detailed status: what the task is doing or waiting on."""
    if status == TaskStatus.CANCELLED:
        return "cancelled"
    if status == TaskStatus.DONE:
        return "done"
    if contract is not None and contract.blocker is not None:
        return "blocked"
    if (contract is not None and contract.ask is not None) or pending is not None:
        return "asked"
    if status == TaskStatus.IN_REVIEW:
        return "awaiting_accept"
    if status == TaskStatus.BLOCKED:
        return "blocked"
    if status != TaskStatus.IN_PROGRESS:
        return "queued"
    if open_subtasks:
        return "waiting_on_subtasks"
    if quiet_minutes is not None and quiet_minutes >= NO_SIGNAL_MINUTES:
        return "no_signal"
    return "running" if running else "idle"


def state_of(raw: str) -> ContractState:
    return STATE_OF.get(raw, ContractState.WORKING)


def derive_state(
    status: TaskStatus,
    contract: Contract | None,
    pending: PendingInput | None,
    **signals: Any,
) -> ContractState:
    return state_of(raw_status(status, contract, pending, **signals))


def card_blocker(
    contract: Contract | None, raw: str, status: TaskStatus
) -> Blocker | None:
    """The blocker to show: the recorded one, or one for a task that went
    quiet or stopped without saying why."""
    if contract is not None and contract.blocker is not None:
        return contract.blocker
    if raw == "no_signal":
        return make_blocker(
            contract,
            f"No response for over {NO_SIGNAL_MINUTES} minutes.",
            cause="no_signal",
        )
    if raw == "blocked" and status == TaskStatus.BLOCKED:
        return make_blocker(contract, "The work stopped before it finished.")
    return None


def now_line(
    raw: str,
    contract: Contract | None,
    live_step: str | None = None,
    subtasks: tuple[int, int] = (0, 0),
) -> str:
    """One line about now, with real counts only: what it is doing, waiting
    on, or why it stopped."""
    plan = contract.plan if contract else []
    criteria = contract.done_when if contract else []
    if raw == "done":
        met = sum(1 for c in criteria if c.status in _CLOSED)
        return f"{met} of {len(criteria)} checks met" if criteria else "Done"
    if raw == "cancelled":
        return "Stopped"
    if raw in ("blocked", "no_signal"):
        blocker = card_blocker(contract, raw, TaskStatus.BLOCKED)
        return blocker.short if blocker else "Stuck"
    if raw == "asked":
        return "Waiting on your answer"
    if raw == "awaiting_accept":
        return "Waiting on your OK"
    if raw == "queued":
        return "Not started"
    if raw == "waiting_on_subtasks":
        done, total = subtasks
        return f"Waiting on sub-tasks · {done} of {total} done"
    step = next((s for s in plan if s.status == StepStatus.RUNNING), None)
    if step is not None and step.rows:
        settled = sum(1 for r in step.rows if r.status in _SETTLED_STEP)
        return f"{step.title} · {settled} of {len(step.rows)}"
    if step is not None:
        done = sum(1 for s in plan if s.status in (StepStatus.DONE, StepStatus.SKIPPED))
        return f"{step.title} · step {done + 1} of {len(plan)}"
    if live_step:
        return live_step
    evidence = contract.evidence if contract else []
    if evidence:
        return f"Working · {len(evidence)} result{'s' if len(evidence) != 1 else ''} so far"
    return "Working"


_ATTACHMENT_KINDS = {
    "file": "file",
    "url": "link",
    "run": "record",
    "chat": "record",
    "text": "quote",
}


def fallback_contract(
    task: Task,
    attachments: list[Attachment],
    plan: Plan | None,
    stopped_note: str | None = None,
) -> Contract:
    """The contract a task without ``update_task`` implies: its title as the
    goal, its plan's steps, its attachments as evidence and its outcome as the
    result."""
    done = task.status == TaskStatus.DONE
    step_status = StepStatus.DONE if done else StepStatus.PENDING
    steps = [
        Step(id=f"s{i}", title=one_line(s.title), status=step_status)
        for i, s in enumerate(plan.steps if plan else [], 1)
    ][:MAX_ITEMS]
    evidence = [
        Evidence(
            id=f"e{i}",
            kind=_ATTACHMENT_KINDS.get(a.kind, "record"),
            title=one_line(a.title, 120),
            summary=one_line(a.title),
            ref=a.ref_id,
            url=a.url,
            attachment_id=a.id,
        )
        for i, a in enumerate(attachments, 1)
    ]
    pending = task.pending_input
    ask = (
        Ask(
            id=pending.id,
            question=pending.question,
            options=pending.options,
            default_option=pending.options[0] if pending.options else None,
        )
        if pending
        else None
    )
    base = Contract(goal=one_line(task.title), plan=steps, evidence=evidence, ask=ask)
    blocker = None
    if task.status == TaskStatus.BLOCKED and pending is None:
        blocker = make_blocker(
            base, stopped_note or "The work stopped before it finished."
        )
    return base.model_copy(
        update={
            "blocker": blocker,
            "result": (
                Result(
                    summary=one_line(task.outcome_summary, RESULT_CHARS),
                    primary_evidence_id=evidence[-1].id if evidence else None,
                )
                if task.outcome_summary
                else None
            ),
        }
    )


def progress_line(steps: list[Step], evidence: list[Evidence]) -> str:
    done = [s.title for s in steps if s.status == StepStatus.DONE]
    parts = []
    if steps:
        parts.append(f"{len(done)} of {len(steps)} steps done")
        if done:
            parts[-1] += f" ({', '.join(done[:3])}{'…' if len(done) > 3 else ''})"
    for s in steps:
        if s.rows and s.status != StepStatus.DONE:
            finished = sum(1 for r in s.rows if r.status == StepStatus.DONE)
            parts.append(f"{s.title}: {finished} of {len(s.rows)} done")
    if evidence:
        parts.append(f"{len(evidence)} result{'s' if len(evidence) != 1 else ''} kept")
    return "; ".join(parts) + "." if parts else "Nothing finished yet."


# Ordered: the first cause whose pattern matches wins.
_CAUSES: list[tuple[str, re.Pattern[str]]] = [
    (
        "permission",
        re.compile(
            r"not (allowed|permitted|authori[sz]ed)|aren'?t allowed|isn'?t "
            r"(allowed|permitted)|permission|forbidden|unauthori[sz]ed|access "
            r"denied|only the user|started by an automation|background (task )?"
            r"runs?|cannot hire|not available in this delegated turn|approval "
            r"settings|\b40[13]\b",
            re.IGNORECASE,
        ),
    ),
    (
        "limit",
        re.compile(
            r"\blimit|quota|budget|too many|rate.?limit|no credits|out of "
            r"credits|insufficient (balance|credits)|\b429\b|capacity|is full",
            re.IGNORECASE,
        ),
    ),
    (
        "missing_input",
        re.compile(
            r"missing|is required|are required|\brequired\b|not found|no such|"
            r"no \w+ (named|with id)|not connected|connect (an?|the|your)|"
            r"credentials?|sign in|provide|which one",
            re.IGNORECASE,
        ),
    ),
    (
        "external_error",
        re.compile(
            r"timed? ?out|timeout|unavailable|connection|network|\b5\d\d\b|"
            r"service|upstream|http|failed to (fetch|connect|reach)|"
            r"an error occurred|didn'?t respond|not responding",
            re.IGNORECASE,
        ),
    ),
]

_NEXT_ACTIONS = {
    "permission": ["allow", "do_in_chat", "take_over"],
    "missing_input": ["change_goal", "retry_failed_part", "take_over"],
    "limit": ["retry_failed_part", "change_goal", "take_over"],
    "external_error": ["retry_failed_part", "skip", "take_over"],
    "no_signal": ["retry_failed_part", "take_over", "stop"],
}
_DEFAULT_ACTIONS = ["retry_failed_part", "skip", "take_over"]


def classify_error(message: str) -> str:
    for cause, pattern in _CAUSES:
        if pattern.search(message or ""):
            return cause
    return "unknown"


def next_actions_for(cause: str) -> list[str]:
    return list(_NEXT_ACTIONS.get(cause, _DEFAULT_ACTIONS))


# Causes a turn can't get past by trying again on its own, so a call that hit
# one and was never retried successfully stops the task.
BLOCKING_CAUSES = frozenset({"permission", "limit", "missing_input"})


def tool_error(output: Any) -> str | None:
    """The error a tool call returned, or None when it didn't fail. A card
    waiting on the user (approval, sign-in) is not a failure."""
    data = output
    if isinstance(output, str):
        try:
            data = json.loads(output)
        except ValueError:
            return None
    if not isinstance(data, dict) or data.get("type") != "error":
        return None
    parts = [str(data.get("message") or ""), str(data.get("error") or "")]
    return " ".join(p for p in parts if p).strip() or "The tool failed."


def unrecovered_failure(
    calls: list[tuple[str, str | None]], ignore: frozenset[str] = frozenset()
) -> tuple[str, str] | None:
    """The last failed call, as ``(tool, error)``, whose tool never succeeded
    after it in the same turn. *calls* is ``(tool, error or None)`` in order."""
    last_ok: dict[str, int] = {}
    for i, (name, error) in enumerate(calls):
        if error is None:
            last_ok[name] = i
    for i in range(len(calls) - 1, -1, -1):
        name, error = calls[i]
        if name in ignore or error is None:
            continue
        if last_ok.get(name, -1) < i:
            return name, error
    return None


def turn_blocker_reason(
    turn_failed: bool, failure: tuple[str, str] | None
) -> tuple[str, str | None] | None:
    """``(what stopped it, tool)`` when a finished turn should leave a blocker:
    the turn itself failed, or a call hit a cause retrying can't fix."""
    if failure is not None and (
        turn_failed or classify_error(failure[1]) in BLOCKING_CAUSES
    ):
        return failure[1], failure[0]
    if turn_failed:
        return "The work turn stopped with an error.", None
    return None


_EVENT_KINDS = {"FILE": "file", "RUN": "record", "INTEGRATION": "record"}


def evidence_from_event(
    category: str, event_type: str, title: str, object_id: str | None, data: dict
) -> dict[str, Any] | None:
    """The evidence a logged tool effect (a file written, a run, an
    integration call, a schedule) amounts to, or None for one that leaves
    nothing to show, such as a delete."""
    if event_type.endswith(".deleted"):
        return None
    kind = _EVENT_KINDS.get(category, "record")
    detail = data.get("path") or data.get("provider") or event_type.replace(".", " ")
    return {
        "kind": kind,
        "title": one_line(title, 120) or event_type,
        "summary": one_line(f"{title}: {detail}" if detail else title),
        "ref": object_id,
    }


def _find(items: list[Any], item_id: str, label: str) -> Any:
    found = next((i for i in items if i.id == item_id), None)
    if found is None:
        ids = ", ".join(i.id for i in items) or "none"
        raise ContractError(f"No {label} with id {item_id}. Known ids: {ids}.")
    return found


def _dump(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json")
    if isinstance(value, list):
        return [_dump(v) for v in value]
    return value


def parse_contract(raw: Any) -> Contract | None:
    if not raw or not isinstance(raw, dict) or not raw.get("goal"):
        return None
    try:
        return Contract.model_validate(raw)
    except ValidationError:
        logger.warning("Ignoring a task contract that no longer parses")
        return None


def brief(contract: Contract) -> str:
    """The contract as a few lines for a turn that starts in a fresh chat."""
    lines = [f"Goal: {contract.goal}"]
    lines += [
        f"- done when {c.id}: {c.text} [{c.status.value}]" for c in contract.done_when
    ]
    for s in contract.plan:
        lines.append(f"- step {s.id}: {s.title} [{s.status.value}]")
        lines += [f"  - {r.title} [{r.status.value}]" for r in s.rows]
    lines += [f"- evidence {e.id}: {e.title}: {e.summary}" for e in contract.evidence]
    return "\n".join(lines)
