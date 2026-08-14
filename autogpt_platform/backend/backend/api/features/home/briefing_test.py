from datetime import datetime, timedelta, timezone

from backend.api.features.experts.models import Expert
from backend.copilot.briefing.generate import AgentInfo
from backend.copilot.briefing.generate import compose_briefing as compose_job_briefing
from backend.copilot.briefing.models import BriefingContent, BriefingRunItem
from backend.data.execution import ExecutionStatus, GraphExecutionMeta

from .briefing import compose_briefing, without_summaries
from .helpers import AgentRef

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)
TRIAGE = {"graph": AgentRef(name="Inbox triage", library_agent_id="library-agent")}
NO_LINK = {"graph": AgentRef(name="Inbox triage", library_agent_id=None)}


def _execution(
    *,
    exec_id: str,
    status: ExecutionStatus,
    ended_at: datetime,
    activity_status: str | None = None,
    error: str | None = None,
    expert_id: str | None = None,
    graph_id: str = "graph",
) -> GraphExecutionMeta:
    return GraphExecutionMeta(
        id=exec_id,
        user_id="user",
        graph_id=graph_id,
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=status,
        started_at=ended_at - timedelta(minutes=1),
        ended_at=ended_at,
        expert_id=expert_id,
        stats=GraphExecutionMeta.Stats(
            activity_status=activity_status, error=error, duration=42.0, cost=7
        ),
    )


def _expert(expert_id: str = "expert-1") -> Expert:
    return Expert(
        id=expert_id,
        name="Ana",
        avatar_url="https://a/x.png",
        role="Researcher",
        tagline=None,
        bio=None,
        skills=[],
        identity="",
        voice_preferences="",
        boundaries="",
        protected_soul_rules=[],
        is_template=False,
        source_template_id=None,
        is_archived=False,
        workflows=[],
    )


def _stored(
    *items: BriefingRunItem,
    generated_at: datetime = NOW,
    completed_total: int = 0,
    failed_total: int = 0,
) -> BriefingContent:
    return BriefingContent(
        generated_at=generated_at,
        timezone="UTC",
        zero_expert_fallback=False,
        run_items=list(items),
        decision_items=[],
        decision_total=0,
        completed_total=completed_total,
        failed_total=failed_total,
    )


def _stored_item(
    execution_id: str,
    *,
    status: str = "COMPLETED",
    title: str = "Sorted 12 emails.",
    detail: str = "Nothing needed a reply.",
) -> BriefingRunItem:
    return BriefingRunItem(
        expert_id="expert-1",
        expert_name="Ana",
        expert_role="Researcher",
        expert_avatar_url="https://a/x.png",
        agent_name="Inbox triage",
        graph_id="graph",
        execution_id=execution_id,
        library_agent_id="library-agent",
        status=status,
        summary=f"{title} {detail}",
        title=title,
        detail=detail,
        occurred_at=NOW - timedelta(hours=2),
        duration_seconds=42.0,
        cost_cents=7,
        link="/library/agents/library-agent?activeTab=runs&activeItem=" + execution_id,
    )


def test_briefing_ignores_runs_outside_the_24h_window() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="fresh",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=2),
            ),
            _execution(
                exec_id="stale",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=30),
            ),
        ],
        expert_by_id={},
        agent_by_graph=NO_LINK,
    )

    assert briefing.source == "live"
    assert briefing.window_started_at == NOW - timedelta(hours=24)
    assert briefing.completed_count == 1
    assert [outcome.id for outcome in briefing.outcomes] == ["fresh"]


def test_briefing_lists_failures_before_successes() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="ok",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=1),
                activity_status="Sorted 12 emails. Nothing needed a reply.",
            ),
            _execution(
                exec_id="broken",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=3),
            ),
        ],
        expert_by_id={},
        agent_by_graph=NO_LINK,
    )

    assert [outcome.status for outcome in briefing.outcomes] == ["failed", "completed"]
    assert briefing.failed_count == 1
    assert briefing.outcomes[0].title == "Inbox triage needs a retry"
    assert briefing.outcomes[1].title == "Sorted 12 emails."
    assert briefing.outcomes[1].summary == "Nothing needed a reply."


def test_briefing_counts_unlisted_successes_as_routine() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id=f"run-{index}",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=1),
            )
            for index in range(6)
        ],
        expert_by_id={},
        agent_by_graph=NO_LINK,
    )

    assert briefing.completed_count == 6
    assert len(briefing.outcomes) == 4
    assert briefing.routine_count == 2


def test_briefing_keeps_a_raw_error_out_of_the_headline() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="broken",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=1),
                error="KeyError: 'recipient'",
            )
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
    )

    assert briefing.outcomes[0].title == "Inbox triage needs a retry"
    assert briefing.outcomes[0].summary == "KeyError: 'recipient'"


def test_briefing_orders_each_status_group_most_recent_first() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[
            _execution(
                exec_id="old-failure",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=8),
            ),
            _execution(
                exec_id="new-failure",
                status=ExecutionStatus.FAILED,
                ended_at=NOW - timedelta(hours=1),
            ),
            _execution(
                exec_id="old-success",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=9),
            ),
            _execution(
                exec_id="new-success",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=2),
            ),
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
    )

    assert [outcome.id for outcome in briefing.outcomes] == [
        "new-failure",
        "old-failure",
        "new-success",
        "old-success",
    ]


def test_persisted_briefing_anchors_the_card_on_the_stored_row() -> None:
    """The stored row is what the copilot thread was posted from, so the card
    must tell that story — not a fresh 24h recompute of it."""
    briefing = compose_briefing(
        now=NOW + timedelta(hours=2),
        # Present in the live window but absent from the stored row (the job
        # only briefs expert-owned runs) — it must not become an anchor.
        executions=[
            _execution(
                exec_id="not-in-the-briefing",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=3),
            )
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(_stored_item("stored-run")),
    )

    assert briefing.source == "persisted"
    assert briefing.generated_at == NOW
    assert briefing.window_started_at == NOW - timedelta(hours=24)
    assert [outcome.id for outcome in briefing.outcomes] == ["stored-run"]
    outcome = briefing.outcomes[0]
    assert (outcome.title, outcome.summary) == (
        "Sorted 12 emails.",
        "Nothing needed a reply.",
    )
    assert (outcome.duration_seconds, outcome.cost_cents) == (42.0, 7)


def test_persisted_briefing_appends_runs_that_finished_after_it() -> None:
    """A run that finishes at 10:30am belongs on the next poll, not tomorrow."""
    briefing = compose_briefing(
        now=NOW + timedelta(hours=2),
        executions=[
            _execution(
                exec_id="after-the-briefing",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW + timedelta(hours=1, minutes=30),
                activity_status="Booked the flight.",
            ),
            # Already told in the stored row — it must not be listed twice.
            _execution(
                exec_id="stored-run",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW - timedelta(hours=2),
            ),
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(_stored_item("stored-run")),
    )

    assert [outcome.id for outcome in briefing.outcomes] == [
        "stored-run",
        "after-the-briefing",
    ]
    assert briefing.completed_count == 2
    assert briefing.outcomes[1].title == "Booked the flight."


def test_persisted_briefing_sorts_a_later_failure_above_stored_successes() -> None:
    briefing = compose_briefing(
        now=NOW + timedelta(hours=3),
        executions=[
            _execution(
                exec_id="broke-later",
                status=ExecutionStatus.FAILED,
                ended_at=NOW + timedelta(hours=2),
            )
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(_stored_item("stored-run")),
    )

    assert [outcome.id for outcome in briefing.outcomes] == [
        "broke-later",
        "stored-run",
    ]
    assert (briefing.failed_count, briefing.completed_count) == (1, 1)


def test_persisted_briefing_caps_outcomes_and_absorbs_the_overflow() -> None:
    stored = _stored(*[_stored_item(f"stored-{index}") for index in range(3)])
    briefing = compose_briefing(
        now=NOW + timedelta(hours=4),
        executions=[
            _execution(
                exec_id=f"later-{index}",
                status=ExecutionStatus.COMPLETED,
                ended_at=NOW + timedelta(hours=1, minutes=index),
            )
            for index in range(3)
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=stored,
    )

    assert briefing.completed_count == 6
    assert len(briefing.outcomes) == 4
    assert briefing.routine_count == 2
    assert [outcome.id for outcome in briefing.outcomes][:3] == [
        "stored-0",
        "stored-1",
        "stored-2",
    ]


def test_persisted_briefing_prefers_the_live_expert_record() -> None:
    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={"expert-1": _expert()},
        agent_by_graph=TRIAGE,
        persisted=_stored(_stored_item("stored-run")),
    )

    expert = briefing.outcomes[0].expert
    assert expert is not None
    assert (expert.id, expert.name, expert.role) == ("expert-1", "Ana", "Researcher")


def test_persisted_briefing_drops_attribution_without_a_stored_name() -> None:
    """An id with nothing to render beside it is not an attribution — the card
    would otherwise show an expert row with a blank name."""
    nameless = _stored_item("stored-run").model_copy(update={"expert_name": None})

    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(nameless),
    )

    assert briefing.outcomes[0].expert is None


def test_persisted_failures_keep_their_order_ahead_of_later_ones() -> None:
    """Both halves are failures, so the status key can't separate them: the
    stable sort is what keeps the briefing's own story first."""
    briefing = compose_briefing(
        now=NOW + timedelta(hours=2),
        executions=[
            _execution(
                exec_id="broke-later",
                status=ExecutionStatus.FAILED,
                ended_at=NOW + timedelta(hours=1),
            )
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(_stored_item("broke-overnight", status="FAILED")),
    )

    assert [outcome.id for outcome in briefing.outcomes] == [
        "broke-overnight",
        "broke-later",
    ]
    assert briefing.failed_count == 2


def test_persisted_briefing_keeps_stored_attribution_for_an_unknown_expert() -> None:
    """An expert archived since this morning still owns the run it produced."""
    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(_stored_item("stored-run")),
    )

    expert = briefing.outcomes[0].expert
    assert expert is not None
    assert (expert.id, expert.name, expert.role) == ("expert-1", "Ana", "Researcher")


def test_a_row_written_before_the_composer_was_unified_falls_back_to_the_agent() -> (
    None
):
    """`title`/`detail` were added with the shared composer — an older row has
    neither, and must still render a usable headline."""
    legacy = BriefingRunItem(
        expert_id=None,
        expert_name=None,
        expert_avatar_url=None,
        agent_name="Inbox triage",
        graph_id="graph",
        execution_id="legacy-run",
        library_agent_id=None,
        status="FAILED",
        summary="Sorted 12 emails.",
        link=None,
    )

    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(legacy),
    )

    outcome = briefing.outcomes[0]
    assert outcome.title == "Inbox triage needs a retry"
    assert outcome.summary == (
        "Open the run to inspect the failure and choose the next step."
    )
    assert outcome.expert is None


def test_without_summaries_drops_the_ai_written_text() -> None:
    stripped = without_summaries(_stored(_stored_item("stored-run")))

    item = stripped.run_items[0]
    assert (item.summary, item.title, item.detail) == (None, "", "")
    assert item.agent_name == "Inbox triage"


def test_without_summaries_keeps_an_error_the_ai_never_wrote() -> None:
    """The live gate drops only `activity_status`/`correctness_score`, so a
    failure with no summary keeps showing its real error. The persisted path
    must not downgrade that same failure to the generic retry line."""
    failure = _stored_item("broke", status="FAILED").model_copy(
        update={
            "summary": None,
            "title": "Inbox triage needs a retry",
            "detail": "SMTP connection refused",
        }
    )

    item = without_summaries(_stored(failure)).run_items[0]

    assert (item.title, item.detail) == (
        "Inbox triage needs a retry",
        "SMTP connection refused",
    )


def test_persisted_briefing_counts_runs_the_job_did_not_list() -> None:
    """`run_items` is capped at 10 by the job. A 12-completion night must
    still report 12 completed, with the unlisted ones falling into routine."""
    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(
            *(_stored_item(f"stored-{i}") for i in range(10)),
            completed_total=12,
            failed_total=0,
        ),
    )

    assert briefing.completed_count == 12
    assert briefing.failed_count == 0
    assert len(briefing.outcomes) == 4
    assert briefing.routine_count == 8


def test_persisted_run_totals_absorb_runs_that_finished_later() -> None:
    briefing = compose_briefing(
        now=NOW + timedelta(hours=2),
        executions=[
            _execution(
                exec_id="broke-later",
                status=ExecutionStatus.FAILED,
                ended_at=NOW + timedelta(hours=1),
            )
        ],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(
            *(_stored_item(f"stored-{i}") for i in range(10)),
            completed_total=12,
            failed_total=0,
        ),
    )

    assert (briefing.completed_count, briefing.failed_count) == (12, 1)


def test_persisted_briefing_falls_back_to_the_listed_runs_without_totals() -> None:
    """Rows written before the totals existed default them to 0 — the counts
    then have to come off `run_items` rather than reporting nothing happened."""
    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=_stored(
            _stored_item("stored-run"),
            _stored_item("broke", status="FAILED"),
        ),
    )

    assert (briefing.completed_count, briefing.failed_count) == (1, 1)


def test_the_job_and_home_describe_the_same_run_identically() -> None:
    """One composer, two consumers: the copilot thread and the home card must
    not be able to tell different stories about the same execution."""
    execution = _execution(
        exec_id="run-1",
        status=ExecutionStatus.COMPLETED,
        ended_at=NOW - timedelta(hours=1),
        activity_status="Sorted 12 emails. Nothing needed a reply.",
        expert_id="expert-1",
        graph_id="g-1",
    )
    expert = _expert()

    job_content = compose_job_briefing(
        experts=[expert],
        executions=[execution],
        reviews=[],
        agent_info_by_graph_id={"g-1": AgentInfo("Inbox triage", "library-agent")},
        generated_at=NOW,
        tz_name="UTC",
    )
    assert job_content is not None

    home = compose_briefing(
        now=NOW,
        executions=[execution],
        expert_by_id={"expert-1": expert},
        agent_by_graph={
            "g-1": AgentRef(name="Inbox triage", library_agent_id="library-agent")
        },
    )
    anchored = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={"expert-1": expert},
        agent_by_graph={},
        persisted=job_content,
    )

    assert home.outcomes == anchored.outcomes


def test_persisted_briefing_carries_the_stored_narrative() -> None:
    stored = _stored(_stored_item("stored-run")).model_copy(
        update={"narrative": "I cleared your inbox and found one thing to decide."}
    )

    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
        persisted=stored,
    )

    assert briefing.narrative == ("I cleared your inbox and found one thing to decide.")


def test_live_briefing_has_no_narrative() -> None:
    """Nothing generated it — home never makes the call itself."""
    briefing = compose_briefing(
        now=NOW,
        executions=[],
        expert_by_id={},
        agent_by_graph=TRIAGE,
    )

    assert briefing.narrative is None


def test_without_summaries_drops_the_narrative() -> None:
    """The narrative is written from the summaries, so the same gate hides it."""
    stored = _stored(_stored_item("stored-run")).model_copy(
        update={"narrative": "I read all 12 of your emails."}
    )

    assert without_summaries(stored).narrative is None
