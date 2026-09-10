"""The Briefing's load-bearing invariants.

Never sent empty, totals equal the ledger's sums, the attention block absorbs
everything the alert engine deferred, and a whisper of activity shrinks the
email instead of padding it.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import AlertCause, AlertConditionStatus, BriefingFrequency

from backend.data.alerts import AlertConditionDTO
from backend.data.briefing_data import AgentPeriodStats, ScoredRun
from backend.notifications.alert_causes import AuthExpiredCause
from backend.notifications.briefing import (
    MAX_ATTENTION_ITEMS,
    MAX_HIGHLIGHTS,
    build_briefing,
)
from backend.notifications.conftest import make_db_client

USER = "user-1"
NOW = datetime(2026, 8, 3, 11, 30, tzinfo=timezone.utc)

AUTH = AuthExpiredCause(
    cta_path="/integrations/gmail/reconnect",
    agent="Invoice Chaser",
    provider="Gmail",
    expired_at_label="9:14",
    runs_skipped=5,
    next_try_label="16:00",
)


def _agents(*specs: tuple[str, int, int, float]) -> list[AgentPeriodStats]:
    return [
        AgentPeriodStats(
            graph_id=f"g-{name}",
            agent=name,
            runs=runs,
            failed=failed,
            credits=credits,
            top_score=1.0,
        )
        for name, runs, failed, credits in specs
    ]


def _deferred_condition(condition_id: str = "c1") -> AlertConditionDTO:
    return AlertConditionDTO(
        id=condition_id,
        user_id=USER,
        cause=AlertCause.AUTH_EXPIRED,
        cause_key="auth_expired:gmail:g1",
        data=AUTH.model_dump(mode="json"),
        status=AlertConditionStatus.DEFERRED,
        created_at=NOW,
        sent_at=None,
        briefed_at=None,
    )


def _db_client(agents, conditions=None, runs=None, balance=100.0, idle=0):
    """Patch the Briefing's RPC client.

    `idle` is the user's total active-agent count; the Briefing subtracts the
    ones that ran.
    """
    client = make_db_client(
        get_agent_period_stats=AsyncMock(return_value=agents),
        get_top_scored_runs=AsyncMock(return_value=runs or []),
        count_active_agents=AsyncMock(return_value=idle),
        get_briefing_credit_balance=AsyncMock(return_value=balance),
        get_briefing_alert_conditions=AsyncMock(return_value=conditions or []),
    )
    return patch("backend.notifications.briefing._db", return_value=client)


async def _built(agents, **kwargs):
    with _db_client(agents, **kwargs):
        return await build_briefing(USER, BriefingFrequency.WEEKLY, "UTC", NOW)


async def _build(agents, **kwargs):
    built = await _built(agents, **kwargs)
    return built.data if built else None


@pytest.mark.asyncio
async def test_a_period_with_no_runs_sends_nothing_at_all():
    assert await _build([]) is None


@pytest.mark.asyncio
async def test_totals_equal_the_ledgers_sums():
    data = await _build(_agents(("A", 10, 1, 2.0), ("B", 5, 0, 1.0)))
    assert data is not None
    assert data.totals.runs == sum(row.runs for row in data.ledger)
    assert data.totals.agents_active == len(data.ledger)
    assert data.totals.failed == 1


@pytest.mark.asyncio
async def test_a_whisper_of_activity_shrinks_instead_of_padding():
    data = await _build(_agents(("Lead Scout", 2, 0, 0.1)))
    assert data is not None
    assert data.mode == "quiet"
    assert data.only_agent == "Lead Scout"
    assert data.quiet_summary is not None


@pytest.mark.asyncio
async def test_a_failure_keeps_the_standard_form():
    data = await _build(_agents(("Lead Scout", 2, 1, 0.1)))
    assert data is not None
    assert data.mode == "standard"


@pytest.mark.asyncio
async def test_attention_absorbs_what_the_alert_engine_deferred():
    data = await _build(
        _agents(("Invoice Chaser", 6, 0, 0.25)),
        conditions=[_deferred_condition()],
    )
    assert data is not None
    assert [item.title for item in data.attention] == ["Invoice Chaser is stuck"]
    assert data.attention[0].cta_label == "Reconnect Gmail"
    # A blocked agent with no failures reads amber, not red.
    row = next(r for r in data.ledger if r.agent == "Invoice Chaser")
    assert row.issues_kind == "warn"
    assert row.issues_label == "5 runs skipped"


@pytest.mark.asyncio
async def test_a_failing_agent_reads_red():
    data = await _build(_agents(("Price Watch", 42, 3, 2.6)))
    assert data is not None
    row = data.ledger[0]
    assert row.issues_kind == "fail"
    assert row.issues_label == "3 failed"


@pytest.mark.asyncio
async def test_highlights_are_capped_and_carry_a_gist_and_a_link():
    scored = [
        ScoredRun(
            execution_id=f"x{i}",
            graph_id="g-A",
            agent="A",
            interestingness=1.0,
            activity_status=f"Did thing {i}.",
        )
        for i in range(MAX_HIGHLIGHTS)
    ]
    data = await _build(_agents(("A", 30, 0, 1.0)), runs=scored)
    assert data is not None
    assert len(data.highlights) == MAX_HIGHLIGHTS
    assert all(h.gist and h.url for h in data.highlights)
    # The subject carries the payload, taken from the top highlight.
    assert data.subject_note is not None and data.subject_note.startswith("A ")


@pytest.mark.asyncio
async def test_the_briefing_reports_exactly_the_conditions_it_carried():
    built = await _built(
        _agents(("Invoice Chaser", 6, 0, 0.25)),
        conditions=[_deferred_condition()],
    )
    assert built is not None
    # Only these are marked briefed, so a condition raised while the email was
    # being built still gets its turn next period.
    assert built.attention_condition_ids == ["c1"]


@pytest.mark.asyncio
async def test_conditions_past_the_display_cap_keep_their_turn():
    """Only what the attention block actually showed is marked briefed.

    Marking the overflow too would retire conditions the reader never saw:
    they are filtered out of the next period's candidates, so they would never
    appear in any briefing.
    """
    conditions = [_deferred_condition(f"c{i}") for i in range(1, 9)]
    built = await _built(_agents(("Invoice Chaser", 6, 0, 0.25)), conditions=conditions)
    assert built is not None
    assert len(built.data.attention) == MAX_ATTENTION_ITEMS
    assert built.attention_condition_ids == [f"c{i}" for i in range(1, 7)]


@pytest.mark.asyncio
async def test_a_run_with_nothing_to_say_still_gets_an_honest_gist():
    scored = [
        ScoredRun(
            execution_id="x1",
            graph_id="g-A",
            agent="A",
            interestingness=1.0,
            activity_status=None,
        )
    ]
    data = await _build(_agents(("A", 7, 0, 1.0)), runs=scored)
    assert data is not None
    assert data.highlights[0].gist == "completed 7 runs."


@pytest.mark.asyncio
async def test_a_quiet_period_still_reports_what_is_waiting():
    """A deferred condition is reported by the Briefing and nowhere else.

    Returning early on a zero-run period read the conditions *after* the
    return, so anything the Alert engine had capped or deduped was stranded:
    never emailed, and never marked briefed, so it could sit unreported
    indefinitely. Nothing actionable is silently dropped is the whole promise.
    """
    built = await _built([], conditions=[_deferred_condition()])

    assert built is not None
    assert [item.agent for item in built.data.attention] == ["Invoice Chaser"]
    assert built.attention_condition_ids == ["c1"]
    assert built.data.totals.runs == 0


@pytest.mark.asyncio
async def test_a_period_with_neither_runs_nor_conditions_sends_nothing():
    assert await _built([], conditions=[]) is None
