from datetime import datetime, timezone

from backend.api.features.experts.models import Expert, ExpertWorkflowRef

from .agents import compose_agent_statuses, compose_team_summary

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)


def _expert(
    *,
    expert_id: str,
    schedules_paused_at: datetime | None = None,
    last_run_status: str | None = None,
    needs_setup: bool = False,
) -> Expert:
    return Expert(
        id=expert_id,
        name=f"Expert {expert_id}",
        avatar_url=None,
        role="Assistant",
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
        workflows=[
            ExpertWorkflowRef(
                id=f"workflow-{expert_id}",
                store_listing_version_id=None,
                library_agent_id=None,
                graph_id=f"graph-{expert_id}",
                name="Inbox triage",
                description=None,
                schedule_cron="0 9 * * *" if needs_setup else None,
                schedule_id=None,
            )
        ],
        last_run_status=last_run_status,
        schedules_paused_at=schedules_paused_at,
    )


def test_working_outranks_paused_and_setup() -> None:
    statuses = compose_agent_statuses(
        experts=[
            _expert(expert_id="busy", schedules_paused_at=NOW, needs_setup=True),
            _expert(expert_id="idle"),
        ],
        running_expert_ids={"busy"},
        next_run_by_expert={},
    )

    assert [status.status for status in statuses] == ["working", "ready"]


def test_status_precedence_walks_down_to_failed() -> None:
    statuses = compose_agent_statuses(
        experts=[
            _expert(expert_id="ready"),
            _expert(expert_id="failed", last_run_status="FAILED"),
            _expert(expert_id="setup", needs_setup=True),
            _expert(expert_id="paused", schedules_paused_at=NOW, needs_setup=True),
        ],
        running_expert_ids=set(),
        next_run_by_expert={},
    )

    assert [status.status for status in statuses] == [
        "paused",
        "needs_setup",
        "failed",
        "ready",
    ]


def test_team_summary_groups_non_ready_states_as_attention() -> None:
    statuses = compose_agent_statuses(
        experts=[
            _expert(expert_id="ready"),
            _expert(expert_id="working"),
            _expert(expert_id="paused", schedules_paused_at=NOW),
            _expert(expert_id="setup", needs_setup=True),
        ],
        running_expert_ids={"working"},
        next_run_by_expert={},
    )

    summary = compose_team_summary(statuses)

    assert summary.total == 4
    assert summary.ready == 1
    assert summary.working == 1
    assert summary.needs_attention == 2
