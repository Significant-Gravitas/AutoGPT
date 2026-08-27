from datetime import datetime, timezone

from backend.api.features.experts.models import (
    Expert,
    ExpertWorkItem,
    ExpertWorkflowRef,
)
from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata, PendingQuestion
from backend.data.execution_cost_summary import (
    UserExecutionCostSummary,
    UserExpertCostRollup,
)

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


def _cost_summary(
    by_expert: list[UserExpertCostRollup] | None = None,
) -> UserExecutionCostSummary:
    return UserExecutionCostSummary(
        total_cents=0,
        run_count=0,
        billable_run_count=0,
        failed_cost_cents=0,
        by_agent=[],
        by_expert=by_expert or [],
        top_runs=[],
        daily=[],
    )


def _work_item(status: str, *, completed_at: datetime | None = None) -> ExpertWorkItem:
    return ExpertWorkItem(
        id=f"work-{status}",
        expert_id="hired",
        manager_session_id="manager-1",
        delegated_session_id="delegated-1",
        project_phase="Launch",
        task_title="Find launch partners",
        expected_deliverable="A verified partner shortlist",
        deliverable_mode="message",
        success_criteria=[],
        dependencies=[],
        source_artifacts=[],
        constraints=[],
        approval_boundaries=[],
        estimate_minutes=30,
        progress=60 if status == "running" else 100,
        status=status,
        result="Two verified partners" if status == "delivered" else None,
        blocker=None,
        confidence="verified" if status == "delivered" else "unknown",
        artifacts=[],
        created_at=NOW,
        updated_at=NOW,
        started_at=NOW,
        completed_at=completed_at,
        link=f"/copilot?sessionId=delegated-1&workItemId=work-{status}",
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


def test_active_delegated_work_marks_expert_working_and_reaches_active_tasks() -> None:
    dashboard = compose_home_dashboard(
        now=NOW,
        experts=[_expert("hired")],
        executions=[],
        reviews=[],
        schedules=[],
        library_refs=[],
        cost_summary=_cost_summary(),
        credits_balance=100,
        timezone_name="UTC",
        work_items=[_work_item("running")],
    )

    assert dashboard.agents[0].status == "working"
    assert dashboard.team.working == 1
    assert dashboard.active_tasks[0].work_item_id == "work-running"
    assert dashboard.work_items[0].progress == 60


def test_delivered_delegated_work_appears_in_home_outcomes() -> None:
    dashboard = compose_home_dashboard(
        now=NOW,
        experts=[_expert("hired")],
        executions=[],
        reviews=[],
        schedules=[],
        library_refs=[],
        cost_summary=_cost_summary(),
        credits_balance=100,
        timezone_name="UTC",
        work_items=[_work_item("delivered", completed_at=NOW)],
    )

    outcome = dashboard.briefing.outcomes[0]
    assert outcome.work_item_id == "work-delivered"
    assert outcome.summary == "Two verified partners"
    assert outcome.confidence == "verified"


def test_expert_spend_reaches_the_agent_rows_and_the_team_total() -> None:
    dashboard = compose_home_dashboard(
        now=NOW,
        experts=[_expert("hired"), _expert("quiet"), _expert("gone", is_archived=True)],
        executions=[],
        reviews=[],
        schedules=[],
        library_refs=[],
        cost_summary=_cost_summary(
            [
                UserExpertCostRollup(expert_id="hired", cost_cents=730, run_count=4),
                # Attributed to an expert the dashboard doesn't list, so it is
                # dropped rather than silently inflating the team total.
                UserExpertCostRollup(expert_id="gone", cost_cents=500, run_count=2),
            ]
        ),
        credits_balance=100,
        timezone_name="UTC",
    )

    spend = {agent.expert.id: agent.spend_cents for agent in dashboard.agents}
    assert spend == {"hired": 730, "quiet": 0}
    assert dashboard.team.spend_cents == 730


def _pending_question_session(session_id: str = "sess-1") -> ChatSessionInfo:
    return ChatSessionInfo(
        session_id=session_id,
        user_id="user-1",
        usage=[],
        started_at=NOW,
        updated_at=NOW,
        metadata=ChatSessionMetadata(
            pending_question=PendingQuestion(text="Which vendor?", asked_at=NOW)
        ),
    )


def test_pending_question_reaches_the_attention_list() -> None:
    """A chat session with a pending question must surface on Home's
    "Needs You" surface — not merely be accepted as an argument."""
    dashboard = compose_home_dashboard(
        now=NOW,
        experts=[],
        executions=[],
        reviews=[],
        schedules=[],
        library_refs=[],
        cost_summary=_cost_summary(),
        credits_balance=100,
        timezone_name="UTC",
        questions=[_pending_question_session("sess-1")],
    )

    assert [item.id for item in dashboard.attention] == ["question-sess-1"]
    assert dashboard.attention[0].kind == "question"


def test_no_questions_means_no_question_attention_items() -> None:
    dashboard = compose_home_dashboard(
        now=NOW,
        experts=[],
        executions=[],
        reviews=[],
        schedules=[],
        library_refs=[],
        cost_summary=_cost_summary(),
        credits_balance=100,
        timezone_name="UTC",
        questions=[],
    )

    assert dashboard.attention == []
