"""Compose the Home "Recent work" feed: one group per actor.

The card answers "who did what this week". Every run that finished and
every durable thing produced — files written, integration actions taken,
schedules set up — is attributed to the expert, workflow, or Autopilot
that did it, so the two feeds land in the same block instead of describing
the same day from different angles.
"""

from datetime import datetime, timedelta, timezone
from typing import Literal

from backend.api.features.experts.models import Expert
from backend.blocks.llm import LLM_PROVIDER_NAMES
from backend.copilot.briefing.models import BriefingRunItem
from backend.copilot.briefing.outcome import as_utc
from backend.data.activity_event import ActivityEvent
from backend.data.execution import GraphExecutionMeta

from .briefing import live_run_items, to_outcome
from .helpers import UNKNOWN_AGENT, AgentRef, to_home_expert
from .models import (
    HomeBriefingOutcome,
    HomeRecentWork,
    HomeRecentWorkGroup,
    HomeRecentWorkItem,
    HomeWorkActor,
)

_WINDOW = timedelta(days=7)
_MAX_GROUPS = 6
_MAX_RUNS_PER_GROUP = 3
_MAX_ITEMS_PER_GROUP = 4

_ITEM_CATEGORIES: dict[str, Literal["file", "integration", "schedule"]] = {
    "FILE": "file",
    "INTEGRATION": "integration",
    "SCHEDULE": "schedule",
}

ActorKey = tuple[Literal["expert", "workflow", "autopilot"], str]


class _Bucket:
    def __init__(self, actor: HomeWorkActor) -> None:
        self.actor = actor
        self.runs: list[HomeBriefingOutcome] = []
        self.items: list[HomeRecentWorkItem] = []
        self.latest_at: datetime | None = None

    def touch(self, at: datetime | None) -> None:
        if at is None:
            return
        at = as_utc(at)
        if self.latest_at is None or at > self.latest_at:
            self.latest_at = at


def compose_recent_work(
    *,
    now: datetime,
    executions: list[GraphExecutionMeta],
    events: list[ActivityEvent],
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, AgentRef],
    session_titles: dict[str, str | None],
) -> HomeRecentWork:
    since = now - _WINDOW
    exec_by_id = {execution.id: execution for execution in executions}
    buckets: dict[ActorKey, _Bucket] = {}

    def bucket(key: ActorKey) -> _Bucket:
        found = buckets.get(key)
        if found is None:
            found = buckets[key] = _Bucket(_actor(key, expert_by_id, agent_by_graph))
        return found

    run_items = live_run_items(executions, since, now, expert_by_id, agent_by_graph)
    for item in run_items:
        target = bucket(_run_key(item))
        target.runs.append(to_outcome(item, expert_by_id))
        target.touch(item.occurred_at)

    displayable = [
        event
        for event in events
        if event.category in _ITEM_CATEGORIES
        and not _is_model_call(event)
        and as_utc(event.created_at) >= since
    ]
    for event in displayable:
        target = bucket(_event_key(event, exec_by_id))
        target.items.append(_compose_item(event, session_titles))
        target.touch(event.created_at)

    groups = sorted(
        (_compose_group(found) for found in buckets.values()),
        key=lambda group: group.latest_at,
        reverse=True,
    )
    return HomeRecentWork(
        window_started_at=since,
        completed_count=sum(item.status == "COMPLETED" for item in run_items),
        failed_count=sum(item.status == "FAILED" for item in run_items),
        groups=groups[:_MAX_GROUPS],
        total_count=len(displayable),
    )


def _compose_group(found: _Bucket) -> HomeRecentWorkGroup:
    # Runs arrive failures-first then newest-first; events arrive newest-first.
    # Both orders are worth keeping, so the caps slice rather than re-sort.
    return HomeRecentWorkGroup(
        actor=found.actor,
        latest_at=found.latest_at or datetime.min.replace(tzinfo=timezone.utc),
        runs=found.runs[:_MAX_RUNS_PER_GROUP],
        items=found.items[:_MAX_ITEMS_PER_GROUP],
        more_count=max(0, len(found.runs) - _MAX_RUNS_PER_GROUP)
        + max(0, len(found.items) - _MAX_ITEMS_PER_GROUP),
    )


def _run_key(item: BriefingRunItem) -> ActorKey:
    if item.expert_id:
        return ("expert", item.expert_id)
    return ("workflow", item.graph_id)


def _event_key(
    event: ActivityEvent, exec_by_id: dict[str, GraphExecutionMeta]
) -> ActorKey:
    if event.expert_id:
        return ("expert", event.expert_id)
    if event.graph_exec_id:
        # The executor stamps the run, not the expert, so an expert's
        # scheduled workflow lands with the expert only via its execution.
        execution = exec_by_id.get(event.graph_exec_id)
        if execution and execution.expert_id:
            return ("expert", execution.expert_id)
        return ("workflow", execution.graph_id if execution else "")
    return ("autopilot", "")


def _actor(
    key: ActorKey, expert_by_id: dict[str, Expert], agent_by_graph: dict[str, AgentRef]
) -> HomeWorkActor:
    kind, ref = key
    if kind == "expert":
        expert = expert_by_id.get(ref)
        if expert is None:
            # The expert was archived or removed after the fact; the work
            # and its attribution kind survive, only the profile is gone.
            return HomeWorkActor(kind="expert", name="Expert")
        return HomeWorkActor(
            kind="expert",
            name=expert.name,
            expert=to_home_expert(expert),
            link=f"/copilot?expertId={expert.id}",
        )
    if kind == "workflow":
        agent = agent_by_graph.get(ref, UNKNOWN_AGENT)
        return HomeWorkActor(
            kind="workflow",
            name=agent.name,
            link=(
                f"/library/agents/{agent.library_agent_id}"
                if agent.library_agent_id
                else None
            ),
        )
    return HomeWorkActor(kind="autopilot", name="Autopilot", link="/copilot")


def _is_model_call(event: ActivityEvent) -> bool:
    """Rows written before LLM credential use stopped being recorded."""
    return event.category == "INTEGRATION" and event.provider in LLM_PROVIDER_NAMES


def _compose_item(
    event: ActivityEvent, session_titles: dict[str, str | None]
) -> HomeRecentWorkItem:
    mime_type = event.data.get("mime_type")
    session_id = event.session_id
    return HomeRecentWorkItem(
        id=event.id,
        category=_ITEM_CATEGORIES[event.category],
        event_type=event.event_type,
        title=event.title,
        occurred_at=event.created_at,
        provider=event.provider,
        file_id=event.object_id if event.category == "FILE" else None,
        mime_type=mime_type if isinstance(mime_type, str) else None,
        session_title=session_titles.get(session_id) if session_id else None,
        link=f"/copilot?sessionId={session_id}" if session_id else None,
    )
