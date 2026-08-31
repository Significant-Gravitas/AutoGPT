from datetime import datetime, timedelta, timezone

from backend.api.features.experts.models import Expert
from backend.data.execution import ExecutionStatus, GraphExecutionMeta

from .outcome import as_utc, compose_run_outcome, run_link, split_summary

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)


def _execution(
    *,
    status: ExecutionStatus = ExecutionStatus.COMPLETED,
    activity_status: str | None = None,
    error: str | None = None,
) -> GraphExecutionMeta:
    return GraphExecutionMeta(
        id="run-1",
        user_id="user",
        graph_id="graph-1",
        graph_version=1,
        inputs=None,
        credential_inputs=None,
        nodes_input_masks=None,
        preset_id=None,
        status=status,
        started_at=NOW - timedelta(minutes=5),
        ended_at=NOW,
        expert_id="expert-1",
        stats=GraphExecutionMeta.Stats(
            activity_status=activity_status, error=error, duration=30.0, cost=9
        ),
    )


def _expert() -> Expert:
    return Expert(
        id="expert-1",
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


def test_outcome_splits_the_summary_and_carries_stats() -> None:
    item = compose_run_outcome(
        _execution(activity_status="Sorted 12 emails. Nothing needed a reply."),
        agent_name="Inbox triage",
        library_agent_id="lib-1",
        expert=_expert(),
    )

    assert (item.title, item.detail) == ("Sorted 12 emails.", "Nothing needed a reply.")
    assert item.summary == "Sorted 12 emails. Nothing needed a reply."
    assert (item.status, item.duration_seconds, item.cost_cents) == (
        "COMPLETED",
        30.0,
        9,
    )
    assert item.occurred_at == NOW
    assert item.link == "/library/agents/lib-1?activeTab=runs&activeItem=run-1"
    assert (item.expert_name, item.expert_role) == ("Ana", "Researcher")


def test_a_raw_error_stays_out_of_the_headline() -> None:
    item = compose_run_outcome(
        _execution(status=ExecutionStatus.FAILED, error="KeyError: 'recipient'"),
        agent_name="Inbox triage",
        library_agent_id=None,
        expert=None,
    )

    assert item.status == "FAILED"
    assert item.title == "Inbox triage needs a retry"
    assert item.detail == "KeyError: 'recipient'"
    assert item.link is None
    assert item.expert_id is None


def test_a_failure_without_an_error_gets_a_next_step_detail() -> None:
    item = compose_run_outcome(
        _execution(status=ExecutionStatus.FAILED),
        agent_name="Inbox triage",
        library_agent_id=None,
        expert=None,
    )

    assert item.detail == (
        "Open the run to inspect the failure and choose the next step."
    )


def test_split_summary_uses_fallbacks_for_empty_input() -> None:
    assert split_summary(None, fallback_title="Ran", fallback_detail="All good") == (
        "Ran",
        "All good",
    )
    assert split_summary("   ", fallback_title="Ran", fallback_detail="All good") == (
        "Ran",
        "All good",
    )


def test_split_summary_clips_a_single_sentence_title() -> None:
    title, detail = split_summary(
        "x" * 200, fallback_title="Ran", fallback_detail="All good"
    )

    assert title == "x" * 120
    assert detail == "All good"


def test_split_summary_clips_a_long_first_sentence() -> None:
    """A period past the limit must not smuggle an unbounded headline through
    the split branch."""
    title, detail = split_summary(
        "y" * 200 + ". And then some more.",
        fallback_title="Ran",
        fallback_detail="All good",
    )

    assert title == "y" * 119 + "."
    assert len(title) == 120
    assert detail == "And then some more."


def test_split_summary_splits_on_the_first_sentence() -> None:
    assert split_summary(
        "Sorted 12 emails.  Nothing needed a reply.",
        fallback_title="Ran",
        fallback_detail="All good",
    ) == ("Sorted 12 emails.", "Nothing needed a reply.")


def test_run_link_needs_a_library_agent() -> None:
    assert run_link(None, "execution") is None
    assert run_link("library agent", "exec-1") == (
        "/library/agents/library%20agent?activeTab=runs&activeItem=exec-1"
    )


def test_run_link_keeps_each_id_a_single_url_component() -> None:
    """`quote` keeps `/` by default; an id carrying one would otherwise move
    the boundary between the path and the route it addresses."""
    assert run_link("lib/1", "exec/1") == (
        "/library/agents/lib%2F1?activeTab=runs&activeItem=exec%2F1"
    )


def test_as_utc_pins_naive_timestamps() -> None:
    assert as_utc(NOW.replace(tzinfo=None)) == NOW
    assert as_utc(NOW) == NOW
