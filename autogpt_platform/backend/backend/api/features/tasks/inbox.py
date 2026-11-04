"""Inbox view of tasks: state, recent steps, tools used and tab counts."""

import json
from datetime import UTC, datetime
from typing import Any

from backend.api.features.tasks import db
from backend.api.features.tasks.models import (
    ContractState,
    InboxActivity,
    InboxState,
    InboxStep,
    InboxTool,
    Task,
    TaskCounts,
    TaskFilters,
    TaskInbox,
    TaskStatus,
)
from backend.data.db import query_raw_with_schema

MAX_STEPS = 5
SNIPPET_LENGTH = 140
NEEDS_YOU_STATES = {ContractState.NEEDS_YOU, ContractState.STUCK}
STATUS_LABELS = {
    "backlog": "Backlog",
    "todo": "To do",
    "in_progress": "In progress",
    "blocked": "Needs you",
    "in_review": "In review",
    "done": "Done",
    "cancelled": "Cancelled",
}

_RECENT_SQL = """
SELECT "taskId", kind, "actorName", body, data, "createdAt" FROM (
  SELECT a.*, row_number() OVER (PARTITION BY a."taskId" ORDER BY a."createdAt" DESC) AS rn
  FROM {schema_prefix}"TaskActivity" a WHERE a."taskId" = ANY($1::text[])
) recent WHERE rn <= $2 ORDER BY "createdAt" ASC
"""

_LAST_TURN_SQL = """
SELECT DISTINCT ON ("taskId") "taskId", data->>'status' AS status
FROM {schema_prefix}"TaskActivity"
WHERE "taskId" = ANY($1::text[]) AND kind = 'turn'
ORDER BY "taskId", "createdAt" DESC
"""

_TOOLS_SQL = """
SELECT a."taskId", card->>'name' AS name, max(card->>'title') AS title, count(*)::int AS count
FROM {schema_prefix}"TaskActivity" a,
     jsonb_array_elements(COALESCE(a.data->'tool_call_cards', '[]'::jsonb)) card
WHERE a."taskId" = ANY($1::text[]) AND a.kind = 'turn' AND card->>'name' IS NOT NULL
GROUP BY a."taskId", card->>'name'
"""


async def with_inbox(tasks: list[Task]) -> list[Task]:
    if not tasks:
        return tasks
    ids = [t.id for t in tasks]
    recent = await query_raw_with_schema(_RECENT_SQL, ids, MAX_STEPS)
    last_turns = await _last_turn_statuses(ids)
    tools = await query_raw_with_schema(_TOOLS_SQL, ids)
    by_task: dict[str, list[dict[str, Any]]] = {}
    for row in recent:
        by_task.setdefault(row["taskId"], []).append(row)
    tools_by_task: dict[str, list[InboxTool]] = {}
    for row in tools:
        tools_by_task.setdefault(row["taskId"], []).append(
            InboxTool(
                name=row["name"],
                title=row["title"] or row["name"],
                count=int(row["count"]),
            )
        )
    return [
        _decorate(
            t,
            by_task.get(t.id, []),
            last_turns.get(t.id),
            tools_by_task.get(t.id, []),
        )
        for t in tasks
    ]


async def with_inbox_one(task: Task) -> Task:
    return (await with_inbox([task]))[0]


async def count_tasks(user_id: str, filters: TaskFilters) -> TaskCounts:
    scope = filters.model_copy(update={"status": None, "include_hidden": True})
    scope.limit = 500
    tasks = await db.list_tasks(user_id, scope)
    last_turns = await _last_turn_statuses([t.id for t in tasks])
    counts = TaskCounts(
        by_status={s.value: 0 for s in TaskStatus},
        by_state={s.value: 0 for s in ContractState},
    )
    for task in tasks:
        state = inbox_state(task.status, last_turns.get(task.id))
        counts.all += 1
        setattr(counts, state.value, getattr(counts, state.value) + 1)
        counts.needs_you += int(task.state in NEEDS_YOU_STATES)
        counts.by_status[task.status.value] += 1
        counts.by_state[task.state.value] += 1
    return counts


def inbox_state(status: TaskStatus, last_turn_status: str | None) -> InboxState:
    if status == TaskStatus.DONE:
        return InboxState.DONE
    if status == TaskStatus.CANCELLED or last_turn_status == "failed":
        return InboxState.FAILED
    if status == TaskStatus.IN_PROGRESS:
        return InboxState.RUNNING
    return InboxState.WAITING


async def _last_turn_statuses(ids: list[str]) -> dict[str, str | None]:
    if not ids:
        return {}
    rows = await query_raw_with_schema(_LAST_TURN_SQL, ids)
    return {r["taskId"]: r["status"] for r in rows}


def _decorate(
    task: Task,
    events: list[dict[str, Any]],
    last_turn_status: str | None,
    tools: list[InboxTool],
) -> Task:
    steps = _dedupe([step for event in events for step in _event_steps(event)])
    if task.live.running and task.live.current_step:
        steps.append(
            InboxStep(
                label=task.live.current_step,
                time=task.live.turn_started_at,
                kind="live",
                status="running",
            )
        )
    last = events[-1] if events else None
    task.elapsed_ms = _elapsed_ms(task)
    task.inbox = TaskInbox(
        state=inbox_state(task.status, last_turn_status),
        needs_you=task.state in NEEDS_YOU_STATES,
        last_activity=(
            InboxActivity(
                kind=last["kind"],
                text=_snippet(_event_label(last)),
                actor_name=last["actorName"],
                at=_parse_time(last["createdAt"]),
            )
            if last
            else None
        ),
        steps=steps[-MAX_STEPS:],
        tools_used=sorted(tools, key=lambda t: -t.count),
    )
    return task


def _dedupe(steps: list[InboxStep]) -> list[InboxStep]:
    return [
        step
        for i, step in enumerate(steps)
        if i == 0 or step.label != steps[i - 1].label
    ]


def _event_steps(event: dict[str, Any]) -> list[InboxStep]:
    at = _parse_time(event["createdAt"])
    data = _data(event)
    steps: list[InboxStep] = []
    if event["kind"] == "turn":
        steps = [
            InboxStep(
                label=card.get("title") or card.get("name") or "Tool call",
                time=at,
                kind="tool",
                status=card.get("status"),
            )
            for card in data.get("tool_call_cards") or []
            if isinstance(card, dict)
        ]
    status = data.get("status") if event["kind"] == "turn" else None
    steps.append(
        InboxStep(
            label=_snippet(_event_label(event)),
            time=at,
            kind=event["kind"],
            status=status,
        )
    )
    return steps


def _event_label(event: dict[str, Any]) -> str:
    kind = event["kind"]
    actor = event["actorName"] or "Someone"
    body = (event["body"] or "").strip()
    data = _data(event)
    if kind == "created":
        return "Created"
    if kind == "status_changed" and body:
        return body
    if kind == "status_changed":
        return f"Moved to {STATUS_LABELS.get(str(data.get('to')), 'a new status')}"
    if kind == "assigned":
        to = data.get("to") or {}
        return f"Assigned to {to.get('name')}" if to.get("name") else "Unassigned"
    if kind == "comment":
        who = "You" if event["actorName"] == "Me" else actor
        return f"{who} commented: {body}"
    if kind == "turn":
        return _turn_label(actor, data)
    if kind == "input_requested":
        return f"Asked: {body}"
    if kind == "input_answered":
        return f"Answered: {body}"
    if kind == "attachment_added":
        title = (data.get("attachment") or {}).get("title")
        return f"Added {title}" if title else "Added an attachment"
    if kind == "subtask_created":
        return f"Sub-task {data.get('subtask_key') or ''} created".replace("  ", " ")
    if kind == "outcome":
        return f"Outcome: {body}"
    if kind == "plan_confirmed":
        return "Plan approved"
    return body or "Update"


def _turn_label(actor: str, data: dict[str, Any]) -> str:
    status = data.get("status")
    if status == "running":
        return f"{actor} is working"
    if status == "failed":
        return f"{actor}'s turn failed"
    if data.get("summary"):
        return str(data["summary"])
    calls = int(data.get("tool_calls") or 0)
    noun = "tool call" if calls == 1 else "tool calls"
    return f"{actor} finished a turn · {calls} {noun}"


def _elapsed_ms(task: Task) -> int | None:
    if task.started_at is None:
        return None
    end = task.completed_at or datetime.now(UTC)
    start = task.started_at
    if start.tzinfo is None:
        start = start.replace(tzinfo=UTC)
    if end.tzinfo is None:
        end = end.replace(tzinfo=UTC)
    return max(0, int((end - start).total_seconds() * 1000))


def _data(event: dict[str, Any]) -> dict[str, Any]:
    raw = event.get("data")
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except ValueError:
            return {}
    return raw if isinstance(raw, dict) else {}


def _parse_time(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    return datetime.fromisoformat(str(value).replace("Z", "+00:00"))


def _snippet(text: str) -> str:
    flat = " ".join(text.split())
    if len(flat) <= SNIPPET_LENGTH:
        return flat
    return flat[: SNIPPET_LENGTH - 1].rstrip() + "…"
