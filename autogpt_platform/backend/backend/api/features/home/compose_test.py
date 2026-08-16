from datetime import datetime, timezone

from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.data.execution_cost_summary import UserExecutionCostSummary

from .compose import compose_home_dashboard

NOW = datetime(2026, 8, 10, 9, 0, tzinfo=timezone.utc)


def _expert(
    expert_id: str, *, is_template: bool = False, is_archived: bool = False
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
        is_template=is_template,
        source_template_id=None,
        is_archived=is_archived,
        workflows=[
            ExpertWorkflowRef(
                id=f"workflow-{expert_id}",
                store_listing_version_id=None,
                library_agent_id=None,
                graph_id=f"graph-{expert_id}",
                name="Inbox triage",
                description=None,
                schedule_cron="0 9 * * *",
                schedule_id=None,
            )
        ],
    )


def _cost_summary() -> UserExecutionCostSummary:
    return UserExecutionCostSummary(
        total_cents=0,
        run_count=0,
        billable_run_count=0,
        failed_cost_cents=0,
        by_agent=[],
        top_runs=[],
        daily=[],
    )


def test_templates_and_archived_experts_are_left_off_the_dashboard() -> None:
    dashboard = compose_home_dashboard(
        now=NOW,
        experts=[
            _expert("hired"),
            _expert("template", is_template=True),
            _expert("archived", is_archived=True),
        ],
        executions=[],
        reviews=[],
        schedules=[],
        library_refs=[],
        cost_summary=_cost_summary(),
        credits_balance=100,
        timezone_name="UTC",
    )

    assert [agent.expert.id for agent in dashboard.agents] == ["hired"]
    assert dashboard.team.total == 1
    assert [item.id for item in dashboard.attention] == ["setup-hired"]
