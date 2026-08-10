from datetime import datetime, timedelta, timezone

from prisma.enums import ReviewStatus

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.executor.scheduler import GraphExecutionJobInfo

from .attention import compose_attention_items

NOW = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)


def _review(
    created_at: datetime, *, node_exec_id: str = "node-execution"
) -> PendingHumanReviewModel:
    return PendingHumanReviewModel(
        node_exec_id=node_exec_id,
        node_id="node",
        user_id="user",
        graph_exec_id="graph-execution",
        graph_id="graph",
        graph_version=1,
        payload={"recipient": "friend@example.com"},
        instructions="Send the prepared message",
        editable=True,
        status=ReviewStatus.WAITING,
        created_at=created_at,
    )


def _expert(*, needs_setup: bool = False) -> Expert:
    return Expert(
        id="expert",
        name="Ada",
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
                id="workflow",
                store_listing_version_id=None,
                library_agent_id=None,
                graph_id="graph",
                name="Inbox triage",
                description=None,
                schedule_cron="0 9 * * *" if needs_setup else None,
                schedule_id=None,
            )
        ],
    )


def _schedule() -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        id="schedule",
        name="Daily triage",
        next_run_time="2026-08-10T09:00:00+00:00",
        user_id="user",
        graph_id="graph",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )


def test_review_priority_uses_dashboard_generation_time() -> None:
    now = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)

    items = compose_attention_items(
        now=now,
        experts=[],
        reviews=[_review(now - timedelta(hours=25))],
        schedules=[],
        credits_balance=None,
    )

    assert items[0].priority == "high"


def test_review_payload_is_compacted_for_preview() -> None:
    now = datetime(2026, 8, 9, 12, 0, tzinfo=timezone.utc)

    items = compose_attention_items(
        now=now,
        experts=[],
        reviews=[_review(now - timedelta(hours=1))],
        schedules=[],
        credits_balance=None,
    )

    assert items[0].priority == "normal"
    assert items[0].preview == '{"recipient": "friend@example.com"}'


def test_high_priority_sorts_first_then_oldest() -> None:
    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[
            _review(NOW - timedelta(hours=2), node_exec_id="recent-normal"),
            _review(NOW - timedelta(hours=5), node_exec_id="older-normal"),
            _review(NOW - timedelta(hours=30), node_exec_id="stale-high"),
        ],
        schedules=[],
        credits_balance=None,
    )

    assert [item.id for item in items] == [
        "approval-stale-high",
        "approval-older-normal",
        "approval-recent-normal",
    ]


def test_setup_item_is_raised_for_workflows_without_a_schedule() -> None:
    items = compose_attention_items(
        now=NOW,
        experts=[_expert(needs_setup=True)],
        reviews=[],
        schedules=[],
        credits_balance=None,
    )

    assert [(item.kind, item.priority) for item in items] == [("setup", "normal")]
    assert items[0].description == "1 scheduled workflow needs setup."


def test_empty_balance_only_warns_when_schedules_exist() -> None:
    without_schedules = compose_attention_items(
        now=NOW, experts=[], reviews=[], schedules=[], credits_balance=0
    )
    with_schedules = compose_attention_items(
        now=NOW, experts=[], reviews=[], schedules=[_schedule()], credits_balance=0
    )

    assert without_schedules == []
    assert [item.kind for item in with_schedules] == ["credits"]
    assert with_schedules[0].primary_action.href == "/profile/credits"


def test_positive_balance_raises_no_credits_item() -> None:
    items = compose_attention_items(
        now=NOW, experts=[], reviews=[], schedules=[_schedule()], credits_balance=500
    )

    assert items == []


def test_naive_timestamps_are_normalised_to_utc() -> None:
    naive_created = (NOW - timedelta(hours=30)).replace(tzinfo=None)
    paused_expert = _expert()
    paused_expert.schedules_paused_at = (NOW - timedelta(hours=2)).replace(tzinfo=None)

    items = compose_attention_items(
        now=NOW,
        experts=[paused_expert, _expert(needs_setup=True)],
        reviews=[_review(naive_created, node_exec_id="naive")],
        schedules=[],
        credits_balance=None,
    )

    assert [item.id for item in items] == [
        "approval-naive",
        "paused-expert",
        "setup-expert",
    ]
    assert items[0].priority == "high"
    assert items[0].created_at == NOW - timedelta(hours=30)
