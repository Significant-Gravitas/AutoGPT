"""Server-side product analytics: the activation events GTM reports on.

One vocabulary, emitted once per real user action from the backend choke
points (execution creation, executor completion, chat-turn persistence, the
scheduler, webhook delivery), so PostHog funnels and experiments count the
same things the ``analytics.*`` SQL views count from the primary tables.

Event vocabulary (PostHog event name -> SQL equivalent), named after the
product analytics plan:

- ``agent_run_started``   human-initiated agent run (manual UI, API key, copilot
                          tool).  AgentGraphExecution.triggerSource IN
                          ('manual', 'api', 'copilot'); an expert's workflow run
                          has ``expert_id`` set and ``kind: workflow_run``.
- ``chat_message_sent``   user turn in an Autopilot or expert chat
                          (``expert_id`` set).  ChatMessage role='user' on a
                          session with interactive origin.
- ``agent_run_finished``  terminal run outcome (``status``: completed | failed),
                          with ``trigger`` so failures can be split by how they
                          started. Includes subgraph and automated runs; filter
                          to human triggers when comparing outcomes with
                          run-start events. A top-level expert run is
                          ``expert_id`` set and ``is_subgraph_run`` false.
- ``schedule_created``    a schedule was registered (``target``: agent | autopilot |
                          expert).  ActivityEvent category SCHEDULE / schedule.created.
- ``schedule_fired``      a schedule produced work.  For agent/expert targets the run
                          row carries triggerSource='schedule' and triggerRef=schedule
                          id; for autopilot targets the session has origin='automation'.
- ``trigger_fired``       a webhook produced a run (triggerSource='webhook').
- ``expert_hired``        a user hired an expert from a template (sent from
                          ``experts_db.hire_expert``, so every surface counts).
- ``integration_connected`` a user connected a credential (OAuth or manual).
                          IntegrationCredential rows by createdByUserId.

``agent_run_started`` is not the same as a "task": the ``analytics.*`` views
count a copilot-started run through the chat turn that asked for it, so a
task is ``agent_run_started`` with ``trigger`` other than ``copilot``, plus
every ``chat_message_sent``. Event names live in
``backend.util.posthog_events``; the full list and the task filter are in
``docs/platform/tracking-plan.md``.

Every emitter is best-effort: tracking can never break the work it describes.
"""

import logging
from datetime import datetime
from typing import TYPE_CHECKING, Any, Literal

from backend.util import posthog_client
from backend.util.posthog_events import PostHogEvent

if TYPE_CHECKING:
    from backend.data.execution import GraphExecutionEntry, GraphExecutionMeta
    from backend.data.model import GraphExecutionStats

logger = logging.getLogger(__name__)


ScheduleTarget = Literal["agent", "autopilot", "expert"]

# Triggers that mean "a person asked for this run now". Schedule and webhook
# runs are reported as schedule_fired / trigger_fired by their own emitters,
# and nested sub-graph runs are an implementation detail of the parent run.
HUMAN_RUN_TRIGGERS = frozenset({"manual", "api", "copilot"})


def _enum_value(value: Any) -> Any:
    return getattr(value, "value", value)


def track(
    user_id: str | None,
    event: PostHogEvent,
    properties: dict[str, Any] | None = None,
    *,
    dedup_key: str | None = None,
) -> None:
    """Send one event for *user_id*, dropping unset properties. Silently
    no-ops when analytics is off or there is no user."""
    posthog_client.capture(
        user_id,
        event,
        {k: v for k, v in (properties or {}).items() if v is not None},
        dedup_key=dedup_key,
    )


def track_agent_run_started(
    *,
    user_id: str,
    graph_id: str,
    graph_exec_id: str,
    trigger: str | None,
    trigger_ref: str | None = None,
    expert_id: str | None = None,
    preset_id: str | None = None,
    is_dry_run: bool = False,
) -> None:
    trigger_value = _enum_value(trigger)
    if is_dry_run or trigger_value not in HUMAN_RUN_TRIGGERS:
        return
    properties = {
        "graph_id": graph_id,
        "graph_exec_id": graph_exec_id,
        "trigger": trigger_value,
        "trigger_ref": trigger_ref,
        "expert_id": expert_id,
        "preset_id": preset_id,
    }
    if expert_id:
        properties["kind"] = "workflow_run"
    track(user_id, PostHogEvent.AGENT_RUN_STARTED, properties)


def track_agent_run_finished(
    *,
    user_id: str,
    graph_id: str,
    graph_exec_id: str,
    status: Any,
    trigger: str | None,
    expert_id: str | None = None,
    failure_reason: Any = None,
    cost_cents: int | None = None,
    duration_seconds: float | None = None,
    is_dry_run: bool = False,
    is_subgraph_run: bool = False,
) -> None:
    if is_dry_run:
        return
    status_value = _enum_value(status)
    if status_value not in ("COMPLETED", "FAILED"):
        return
    # Keyed on the run alone: the event fires before the terminal status is
    # persisted, so a failed persist requeues and finishes the run again,
    # possibly with another status. COMPLETED/FAILED are absorbing, so one
    # run never legitimately finishes twice.
    track(
        user_id,
        PostHogEvent.AGENT_RUN_FINISHED,
        {
            "status": status_value.lower(),
            "graph_id": graph_id,
            "graph_exec_id": graph_exec_id,
            "trigger": _enum_value(trigger),
            "expert_id": expert_id,
            "failure_reason": _enum_value(failure_reason),
            "cost_cents": cost_cents,
            "duration_seconds": duration_seconds,
            "is_subgraph_run": is_subgraph_run,
        },
        dedup_key=graph_exec_id,
    )


def handle_run_finished(
    graph_exec: "GraphExecutionEntry",
    exec_meta: "GraphExecutionMeta",
    exec_stats: "GraphExecutionStats",
) -> None:
    """Executor completion hook. Mirrors activity_events.handle_run_completed."""
    try:
        track_agent_run_finished(
            user_id=graph_exec.user_id,
            graph_id=graph_exec.graph_id,
            graph_exec_id=graph_exec.graph_exec_id,
            status=exec_meta.status,
            trigger=exec_meta.trigger_source,
            expert_id=exec_meta.expert_id,
            failure_reason=exec_stats.failure_reason,
            cost_cents=exec_stats.cost,
            duration_seconds=exec_stats.walltime,
            is_dry_run=exec_stats.is_dry_run,
            is_subgraph_run=(
                graph_exec.execution_context.parent_execution_id is not None
            ),
        )
    except Exception:
        logger.warning(
            "Failed to track run outcome for %s",
            graph_exec.graph_exec_id,
            exc_info=True,
        )


def track_chat_turn(
    *,
    user_id: str | None,
    session_id: str,
    expert_id: str | None = None,
    origin: str | None = None,
    surface: str | None = None,
    message_length: int | None = None,
) -> None:
    """A person sent a chat message. Model-authored turns are not activation."""
    if origin == "automation":
        return
    track(
        user_id,
        PostHogEvent.CHAT_MESSAGE_SENT,
        {
            "session_id": session_id,
            "expert_id": expert_id,
            "origin": origin,
            "surface": surface or "chat",
            "kind": "chat_turn",
            "message_length": message_length,
        },
    )


def schedule_target(*, expert_id: str | None, is_copilot_turn: bool) -> ScheduleTarget:
    if expert_id:
        return "expert"
    return "autopilot" if is_copilot_turn else "agent"


def track_schedule_created(
    *,
    user_id: str,
    schedule_id: str,
    target: ScheduleTarget,
    expert_id: str | None = None,
    cron: str | None = None,
    run_at: datetime | None = None,
    graph_id: str | None = None,
    session_id: str | None = None,
) -> None:
    track(
        user_id,
        PostHogEvent.SCHEDULE_CREATED,
        {
            "schedule_id": schedule_id,
            "target": target,
            "expert_id": expert_id,
            "cron": cron,
            "is_recurring": cron is not None,
            "run_at": run_at.isoformat() if run_at else None,
            "graph_id": graph_id,
            "session_id": session_id,
        },
    )


def track_schedule_fired(
    *,
    user_id: str,
    schedule_id: str | None,
    target: ScheduleTarget,
    expert_id: str | None = None,
    graph_id: str | None = None,
    graph_exec_id: str | None = None,
    session_id: str | None = None,
) -> None:
    track(
        user_id,
        PostHogEvent.SCHEDULE_FIRED,
        {
            "schedule_id": schedule_id,
            "target": target,
            "expert_id": expert_id,
            "graph_id": graph_id,
            "graph_exec_id": graph_exec_id,
            "session_id": session_id,
        },
    )


def track_trigger_fired(
    *,
    user_id: str,
    webhook_id: str,
    graph_id: str,
    graph_exec_id: str,
    expert_id: str | None = None,
    preset_id: str | None = None,
) -> None:
    track(
        user_id,
        PostHogEvent.TRIGGER_FIRED,
        {
            "webhook_id": webhook_id,
            "graph_id": graph_id,
            "graph_exec_id": graph_exec_id,
            "expert_id": expert_id,
            "preset_id": preset_id,
            "target": "expert" if expert_id else "agent",
        },
    )


def track_integration_connected(
    *,
    user_id: str,
    provider: str,
    credential_type: str,
    method: Literal["oauth", "manual", "device_code"],
) -> None:
    track(
        user_id,
        PostHogEvent.INTEGRATION_CONNECTED,
        {
            "provider": _enum_value(provider),
            "credential_type": _enum_value(credential_type),
            "method": method,
        },
    )
