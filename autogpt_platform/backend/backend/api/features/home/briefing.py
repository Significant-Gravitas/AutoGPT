"""The "what happened overnight" card on /home.

The 9am job persists a `UserBriefing` and posts it into the copilot thread, so
that row — not a fresh 24h recompute — is what the card anchors on. Runs that
reached a terminal state *after* the row was generated are appended live, so a
run finishing at 10:30am shows up on the next poll rather than tomorrow morning.
"""

from datetime import datetime, timedelta
from typing import Literal

from backend.api.features.experts.models import Expert
from backend.copilot.briefing.models import BriefingContent, BriefingRunItem
from backend.copilot.briefing.outcome import (
    as_utc,
    as_utc_or_none,
    compose_run_outcome,
    outcome_fallbacks,
)
from backend.data.execution import ExecutionStatus, GraphExecutionMeta

from .helpers import UNKNOWN_AGENT, AgentRef, to_home_expert
from .models import HomeBriefing, HomeBriefingOutcome, HomeExpert

_MAX_OUTCOMES = 4
_BRIEFING_WINDOW = timedelta(hours=24)


def compose_briefing(
    *,
    now: datetime,
    executions: list[GraphExecutionMeta],
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, AgentRef],
    persisted: BriefingContent | None = None,
) -> HomeBriefing:
    if persisted is None:
        window_start = now - _BRIEFING_WINDOW
        return _briefing(
            generated_at=now,
            window_started_at=window_start,
            outcomes=_live_outcomes(
                executions, window_start, now, expert_by_id, agent_by_graph
            ),
            source="live",
        )

    generated_at = as_utc(persisted.generated_at)
    anchored = [_outcome(item, expert_by_id) for item in persisted.run_items]
    anchored_ids = {outcome.id for outcome in anchored}
    fresh = [
        outcome
        for outcome in _live_outcomes(
            executions, generated_at, now, expert_by_id, agent_by_graph
        )
        if outcome.id not in anchored_ids
    ]
    # Stable sort: failures need a decision so they lead, and within a status
    # group the briefing's own ordering survives ahead of the newer runs.
    merged = anchored + fresh
    merged.sort(key=lambda outcome: outcome.status == "completed")
    # `run_items` is capped by the job, so the anchor is a slice of the night
    # rather than all of it. The stored totals carry the rest; without them a
    # 12-run night would report the 10 that fit.
    anchored_completed = sum(outcome.status == "completed" for outcome in anchored)
    return _briefing(
        generated_at=generated_at,
        window_started_at=generated_at - _BRIEFING_WINDOW,
        outcomes=merged,
        source="persisted",
        omitted_completed=max(0, persisted.completed_total - anchored_completed),
        omitted_failed=max(
            0, persisted.failed_total - len(anchored) + anchored_completed
        ),
    )


def without_summaries(content: BriefingContent) -> BriefingContent:
    """Strip the AI-written text out of a stored briefing.

    Summaries are persisted regardless of `AI_ACTIVITY_STATUS`, and the live
    path scrubs them per-execution when the flag is off — so the persisted path
    has to scrub them too, or the card would leak what the gate hides.

    An item without a summary carries no AI text at all: its `title`/`detail`
    already came from the non-AI fallback, and for a failure that detail is the
    run's own error — which the live gate keeps (it drops only
    `activity_status` and `correctness_score`). Clearing those too would
    downgrade a real error to the generic retry line the moment home switched
    to the persisted row.
    """
    return content.model_copy(
        update={
            "run_items": [
                (
                    item.model_copy(update={"summary": None, "title": "", "detail": ""})
                    if item.summary
                    else item
                )
                for item in content.run_items
            ]
        }
    )


def _briefing(
    *,
    generated_at: datetime,
    window_started_at: datetime,
    outcomes: list[HomeBriefingOutcome],
    source: Literal["persisted", "live"],
    omitted_completed: int = 0,
    omitted_failed: int = 0,
) -> HomeBriefing:
    """`omitted_*` are runs the briefing covered but did not list, so they
    count toward the totals (and the routine overflow) without a card."""
    listed_completed = sum(outcome.status == "completed" for outcome in outcomes)
    completed = listed_completed + omitted_completed
    shown = outcomes[:_MAX_OUTCOMES]
    shown_completed = sum(outcome.status == "completed" for outcome in shown)
    return HomeBriefing(
        generated_at=generated_at,
        window_started_at=window_started_at,
        completed_count=completed,
        failed_count=len(outcomes) - listed_completed + omitted_failed,
        routine_count=max(0, completed - shown_completed),
        outcomes=shown,
        source=source,
    )


def _live_outcomes(
    executions: list[GraphExecutionMeta],
    since: datetime,
    now: datetime,
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, AgentRef],
) -> list[HomeBriefingOutcome]:
    terminal = [
        execution
        for execution in executions
        if execution.status in {ExecutionStatus.COMPLETED, ExecutionStatus.FAILED}
        and _occurred_at(execution, now) >= since
    ]
    # Failures first (they need a decision), then most recent within each group.
    terminal.sort(
        key=lambda execution: (
            execution.status == ExecutionStatus.COMPLETED,
            -_occurred_at(execution, now).timestamp(),
        )
    )
    return [
        _outcome(_run_item(execution, expert_by_id, agent_by_graph), expert_by_id)
        for execution in terminal
    ]


def _run_item(
    execution: GraphExecutionMeta,
    expert_by_id: dict[str, Expert],
    agent_by_graph: dict[str, AgentRef],
) -> BriefingRunItem:
    agent = agent_by_graph.get(execution.graph_id, UNKNOWN_AGENT)
    return compose_run_outcome(
        execution,
        agent_name=agent.name,
        library_agent_id=agent.library_agent_id,
        expert=expert_by_id.get(execution.expert_id or ""),
    )


def _outcome(
    item: BriefingRunItem, expert_by_id: dict[str, Expert]
) -> HomeBriefingOutcome:
    failed = item.status == "FAILED"
    fallback_title, fallback_detail = outcome_fallbacks(
        item.agent_name, failed=failed, error=None
    )
    return HomeBriefingOutcome(
        id=item.execution_id,
        status="failed" if failed else "completed",
        title=item.title or fallback_title,
        summary=item.detail or fallback_detail,
        expert=_expert(item, expert_by_id),
        agent_name=item.agent_name,
        occurred_at=as_utc_or_none(item.occurred_at),
        duration_seconds=item.duration_seconds,
        cost_cents=item.cost_cents,
        link=item.link,
    )


def _expert(
    item: BriefingRunItem, expert_by_id: dict[str, Expert]
) -> HomeExpert | None:
    """Prefer the live expert record; a stored briefing keeps its own copy of
    the display fields for an expert that has since been archived or renamed."""
    live = expert_by_id.get(item.expert_id or "")
    if live:
        return to_home_expert(live)
    if not item.expert_id or not item.expert_name:
        return None
    return HomeExpert(
        id=item.expert_id,
        name=item.expert_name,
        role=item.expert_role or "",
        avatar_url=item.expert_avatar_url,
    )


def _occurred_at(execution: GraphExecutionMeta, now: datetime) -> datetime:
    return as_utc(execution.ended_at or execution.started_at or now)
