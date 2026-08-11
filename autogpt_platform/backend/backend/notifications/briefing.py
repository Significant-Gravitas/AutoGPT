"""Assembling the Briefing.

An inverted pyramid under a hard length budget: the lede, anything that needs
the user, at most three highlights, a six-row ledger, and a footer with the
volume knob. It is never sent empty — a period with nothing to say produces no
email, and a period with only a whisper of activity produces a deliberately
tiny email rather than a padded one.
"""

import logging

from prisma.enums import BriefingFrequency, NotificationType
from pydantic import BaseModel

from backend.data import alerts as alerts_db
from backend.data import briefing_data
from backend.data.alerts import AlertConditionDTO
from backend.data.briefing_data import AgentPeriodStats, ScoredRun
from backend.data.execution import get_graph_execution
from backend.data.notifications import (
    BriefingAttentionItem,
    BriefingData,
    BriefingHighlight,
    BriefingLedgerRow,
    BriefingTotals,
    NotificationEventModel,
)
from backend.notifications.alert_causes import SEVERITY, parse_cause
from backend.notifications.briefing_period import period_window
from backend.notifications.gist import build_gist, fallback_gist
from backend.util.logging import TruncatedLogger
from backend.util.settings import Settings

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Briefing]")
settings = Settings()

MAX_HIGHLIGHTS = 3
# Under this much activity the digest shrinks instead of padding.
QUIET_RUN_THRESHOLD = 3
# Roughly what a credit costs, used only for the parenthetical dollar estimate
# under the ledger.
CREDITS_PER_USD = 100


class BuiltBriefing(BaseModel):
    """The briefing plus the alert conditions its attention block reported, so
    only those are marked once the email is safely on the queue."""

    data: BriefingData
    attention_condition_ids: list[str]


async def build_briefing(
    user_id: str, frequency: BriefingFrequency, timezone_name: str, now
) -> BuiltBriefing | None:
    """Assemble one user's briefing, or None when there is nothing to send."""
    window = period_window(frequency, timezone_name, now)
    agents = await briefing_data.get_agent_period_stats(user_id, window.start, window.end)
    if not agents:
        # Never send empty: a zero-run period sends nothing at all.
        return None

    conditions = await alerts_db.get_briefing_conditions(user_id)
    attention = _attention_block(conditions)
    highlights = await _highlights(user_id, window.start, window.end, agents)
    ledger = [_ledger_row(agent, attention) for agent in agents]

    runs = sum(agent.runs for agent in agents)
    failed = sum(agent.failed for agent in agents)
    credits_used = sum(agent.credits for agent in agents)
    totals = BriefingTotals(
        # Invariants: runs is the ledger's sum, and agents_active is its length.
        runs=runs,
        agents_active=len(ledger),
        agents_idle=max(
            await briefing_data.count_active_agents(user_id) - len(agents), 0
        ),
        failed=failed,
        credits_used=credits_used,
        credits_balance=await briefing_data.get_credit_balance(user_id),
        usd_estimate=round(credits_used / CREDITS_PER_USD, 2) if credits_used else None,
    )

    quiet = _is_quiet(totals, attention)
    return BuiltBriefing(
        data=BriefingData(
            mode="quiet" if quiet else "standard",
            period=window.period,
            totals=totals,
            standout=None if quiet else _standout(agents, totals),
            subject_note=_subject_note(highlights),
            attention=attention,
            highlights=highlights,
            ledger=ledger,
            only_agent=ledger[0].agent if quiet and len(ledger) == 1 else None,
            quiet_summary=_quiet_summary(totals) if quiet else None,
        ),
        attention_condition_ids=[c.id for c in conditions],
    )


def briefing_event(
    user_id: str, data: BriefingData
) -> NotificationEventModel[BriefingData]:
    return NotificationEventModel[BriefingData](
        user_id=user_id, type=NotificationType.BRIEFING, data=data
    )


async def mark_attention_reported(condition_ids: list[str]) -> None:
    """Called once the briefing is actually queued, so the same conditions are
    not reported again next period."""
    await alerts_db.mark_briefed(condition_ids)


def _attention_block(
    conditions: list[AlertConditionDTO],
) -> list[BriefingAttentionItem]:
    """Absorbs every alert condition that was capped or deduped during the
    period, plus anything still unresolved — so nothing actionable is ever
    silently dropped. Sorted by severity, because the first card gets the
    strong amber rule."""
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    causes = sorted(
        (parse_cause(c.cause, c.data) for c in conditions),
        key=lambda c: SEVERITY[c.cause],
    )
    return [cause.attention_item(base_url) for cause in causes]


async def _highlights(
    user_id: str, start, end, agents: list[AgentPeriodStats]
) -> list[BriefingHighlight]:
    """At most three notable outputs from the whole period, each a gist and a
    deep link. This is what replaces the old per-run email."""
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    runs = await briefing_data.get_top_scored_runs(
        user_id, start, end, limit=MAX_HIGHLIGHTS
    )
    runs_by_graph = {agent.graph_id: agent.runs for agent in agents}
    return [
        BriefingHighlight(
            agent=run.agent,
            gist=await _gist_for(user_id, run, runs_by_graph.get(run.graph_id, 1)),
            link_label="See the run",
            url=f"{base_url}/library/agents/{run.graph_id}/runs/{run.execution_id}",
        )
        for run in runs
    ]


async def _gist_for(user_id: str, run: ScoredRun, agent_runs: int) -> str:
    """Summarisation pass, then a truncated first line, then counts — but never
    a raw output. The inbox gets the gist and the link; the platform is where
    outputs live."""
    outputs = await _run_outputs(user_id, run.execution_id)
    return build_gist(outputs, run.activity_status) or fallback_gist(agent_runs)


async def _run_outputs(user_id: str, execution_id: str) -> dict[str, list]:
    try:
        execution = await get_graph_execution(user_id, execution_id)
    except Exception:
        logger.warning(
            "Could not load outputs for run %s; falling back to counts",
            execution_id,
            exc_info=True,
        )
        return {}
    return dict(execution.outputs) if execution else {}


def _ledger_row(
    agent: AgentPeriodStats, attention: list[BriefingAttentionItem]
) -> BriefingLedgerRow:
    if agent.failed:
        return BriefingLedgerRow(
            agent=agent.agent,
            runs=agent.runs,
            credits=agent.credits,
            issues_label=f"{agent.failed} failed",
            issues_kind="fail",
        )
    # An agent with no failures can still be blocked — a skipped schedule shows
    # as amber, not red.
    waiting = next((item for item in attention if item.agent == agent.agent), None)
    return BriefingLedgerRow(
        agent=agent.agent,
        runs=agent.runs,
        credits=agent.credits,
        issues_label=waiting.tag if waiting and waiting.tag else None,
        issues_kind="warn" if waiting and waiting.tag else None,
    )


def _is_quiet(totals: BriefingTotals, attention: list[BriefingAttentionItem]) -> bool:
    return (
        totals.runs <= QUIET_RUN_THRESHOLD and totals.failed == 0 and not attention
    )


def _quiet_summary(totals: BriefingTotals) -> str:
    times = "once" if totals.runs == 1 else f"{totals.runs} times"
    return f"It ran {times}, and nothing needed your attention."


def _standout(agents: list[AgentPeriodStats], totals: BriefingTotals) -> str | None:
    """One rule-built sentence of colour, or nothing. A standout that has to be
    invented is worse than no standout."""
    if len(agents) < 3 or totals.runs == 0:
        return None

    busiest = max(agents, key=lambda a: a.runs)
    if busiest.runs * 2 >= totals.runs:
        return (
            f"{busiest.agent} did most of the work — {busiest.runs:,} of "
            f"{totals.runs:,} runs."
        )

    if totals.credits_used > 0:
        costliest = max(agents, key=lambda a: a.credits)
        if costliest.credits * 2 >= totals.credits_used:
            return f"{costliest.agent} accounted for most of your credits."
    return None


def _subject_note(highlights: list[BriefingHighlight]) -> str | None:
    """A short phrase for the subject, taken from the top highlight — the
    subject is the email for most people, so it carries the payload."""
    if not highlights:
        return None
    top = highlights[0]
    note = f"{top.agent} {top.gist.rstrip('.')}"
    return note if len(note) <= 60 else f"{note[:59].rstrip()}…"
