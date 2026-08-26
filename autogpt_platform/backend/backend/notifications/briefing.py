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

from backend.data.alerts import AlertConditionDTO
from backend.data.briefing_data import AgentPeriodStats, ScoredRun
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
from backend.util.clients import get_database_manager_async_client
from backend.util.logging import TruncatedLogger
from backend.util.settings import Settings

logger = TruncatedLogger(logging.getLogger(__name__), prefix="[Briefing]")
settings = Settings()

MAX_HIGHLIGHTS = 3
# The email shows six rows and a "+ N quieter agents" line. Capping here
# rather than in the template keeps the queued payload the size of the email
# instead of the size of the account.
MAX_LEDGER_ROWS = 6
# Likewise for the attention block: everything beyond this stays unbriefed and
# is carried by the next period rather than bloating this one.
MAX_ATTENTION_ITEMS = 6
# Under this much activity the digest shrinks instead of padding.
QUIET_RUN_THRESHOLD = 3
# Roughly what a credit costs, used only for the parenthetical dollar estimate
# under the ledger.
CREDITS_PER_USD = 100


def _db():
    """No Prisma connection in the notification service; go via the RPC."""
    return get_database_manager_async_client()


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
    agents = await _db().get_agent_period_stats(user_id, window.start, window.end)
    conditions = await _db().get_briefing_alert_conditions(user_id)

    if not agents and not conditions:
        # Never send empty — but a period with no runs can still carry an
        # unresolved condition, and the Briefing is the only place those show.
        return None
    attention, reported_condition_ids = _attention_block(conditions)
    highlights = await _highlights(user_id, window.start, window.end, agents)
    all_rows = [_ledger_row(agent, attention) for agent in agents]
    ledger, overflow = all_rows[:MAX_LEDGER_ROWS], all_rows[MAX_LEDGER_ROWS:]

    runs = sum(agent.runs for agent in agents)
    failed = sum(agent.failed for agent in agents)
    credits_used = sum(agent.credits for agent in agents)
    totals = BriefingTotals(
        # Invariants: runs is the sum over every active agent, and
        # agents_active counts them all — the ledger is the displayed slice,
        # so the totals are taken before the cap.
        runs=runs,
        agents_active=len(all_rows),
        agents_idle=max(await _db().count_active_agents(user_id) - len(agents), 0),
        failed=failed,
        credits_used=credits_used,
        credits_balance=await _db().get_briefing_credit_balance(user_id),
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
            ledger_overflow=len(overflow),
            ledger_overflow_runs=sum(row.runs for row in overflow),
            ledger_overflow_issues=sum(1 for row in overflow if row.issues_label),
            only_agent=all_rows[0].agent if quiet and len(all_rows) == 1 else None,
            quiet_summary=_quiet_summary(totals) if quiet else None,
        ),
        attention_condition_ids=reported_condition_ids,
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
    await _db().mark_alert_conditions_briefed(condition_ids)


def _attention_block(
    conditions: list[AlertConditionDTO],
) -> tuple[list[BriefingAttentionItem], list[str]]:
    """Absorbs every alert condition that was capped or deduped during the
    period, plus anything still unresolved. Sorted by severity, because the
    first card gets the strong amber rule.

    Returns the items alongside the ids of exactly the conditions they came
    from. The block is capped, and marking the overflow briefed would retire
    conditions the reader never saw — they are filtered out of the next
    period's candidates, so they would never surface in any briefing.
    """
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    ranked = sorted(
        ((parse_cause(c.cause, c.data), c) for c in conditions),
        key=lambda pair: SEVERITY[pair[0].cause],
    )[:MAX_ATTENTION_ITEMS]
    return (
        [cause.attention_item(base_url) for cause, _ in ranked],
        [condition.id for _, condition in ranked],
    )


async def _highlights(
    user_id: str, start, end, agents: list[AgentPeriodStats]
) -> list[BriefingHighlight]:
    """At most three notable outputs from the whole period, each a gist and a
    deep link. This is what replaces the old per-run email."""
    base_url = settings.config.frontend_base_url or settings.config.platform_base_url
    runs = await _db().get_top_scored_runs(user_id, start, end, limit=MAX_HIGHLIGHTS)
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
        execution = await _db().get_graph_execution(user_id, execution_id)
    except Exception:
        logger.warning(
            f"Could not load outputs for run {execution_id}; falling back to counts",
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
    return totals.runs <= QUIET_RUN_THRESHOLD and totals.failed == 0 and not attention


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
