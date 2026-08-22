from datetime import datetime, timedelta, timezone

from prisma.enums import ReviewStatus

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts.models import Expert, ExpertWorkflowRef
from backend.copilot.model import ChatSessionInfo, ChatSessionMetadata, PendingQuestion
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


def _paused_expert(*, weekly_budget: int | None, weekly_spend: int) -> Expert:
    expert = _expert()
    expert.schedules_paused_at = NOW - timedelta(hours=3)
    expert.weekly_budget = weekly_budget
    expert.weekly_spend = weekly_spend
    return expert


def test_paused_expert_explains_a_breached_budget() -> None:
    items = compose_attention_items(
        now=NOW,
        experts=[_paused_expert(weekly_budget=500, weekly_spend=500)],
        reviews=[],
        schedules=[],
        credits_balance=None,
    )

    assert items[0].kind == "paused"
    assert items[0].description == "Weekly budget reached: 500 of 500 credits."


def test_paused_expert_without_a_breached_budget_stays_generic() -> None:
    items = compose_attention_items(
        now=NOW,
        experts=[_paused_expert(weekly_budget=500, weekly_spend=10)],
        reviews=[],
        schedules=[],
        credits_balance=None,
    )

    assert items[0].description == "Scheduled work is paused."


def test_long_payloads_are_truncated_with_an_ellipsis() -> None:
    review = _review(NOW - timedelta(hours=1))
    review.payload = {"body": "x" * 400}

    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[review],
        schedules=[],
        credits_balance=None,
    )

    preview = items[0].preview
    assert preview is not None
    assert len(preview) == 138
    assert preview.endswith("…")


def test_review_link_prefers_the_copilot_session() -> None:
    review = _review(NOW - timedelta(hours=1))
    review.session_id = "session 1"

    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[review],
        schedules=[],
        credits_balance=None,
    )

    assert items[0].primary_action.href == "/copilot?sessionId=session%201"


def test_review_link_falls_back_to_the_library_run_then_the_library() -> None:
    with_agent = _review(NOW - timedelta(hours=1), node_exec_id="with-agent")
    with_agent.library_agent_id = "library-agent"
    bare = _review(NOW - timedelta(hours=1), node_exec_id="bare")

    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[with_agent, bare],
        schedules=[],
        credits_balance=None,
    )

    hrefs = {item.id: item.primary_action.href for item in items}
    assert hrefs["approval-with-agent"] == (
        "/library/agents/library-agent?activeTab=runs&activeItem=graph-execution"
    )
    assert hrefs["approval-bare"] == "/library"


def test_review_without_expert_details_has_no_expert() -> None:
    review = _review(NOW - timedelta(hours=1))
    review.expert_id = "expert"
    review.expert_name = None

    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[review],
        schedules=[],
        credits_balance=None,
    )

    assert items[0].expert is None


def _asking_session(
    *,
    session_id: str = "sess-1",
    expert_id: str | None = None,
    text: str | None = "Which?",
    asked_at: datetime | None = None,
) -> ChatSessionInfo:
    """A chat waiting on the user. ``text=None`` is a session whose question
    was already answered, which clears ``pending_question``."""
    return ChatSessionInfo(
        session_id=session_id,
        user_id="user",
        usage=[],
        started_at=NOW,
        updated_at=NOW,
        expert_id=expert_id,
        metadata=ChatSessionMetadata(
            pending_question=(
                PendingQuestion(text=text, asked_at=asked_at or NOW)
                if text is not None
                else None
            )
        ),
    )


def test_pending_question_becomes_an_item_linking_back_to_the_chat() -> None:
    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=[_asking_session(text="Monday or Friday?")],
    )

    assert [item.kind for item in items] == ["question"]
    assert items[0].id == "question-sess-1"
    assert items[0].title == "Autopilot has a question"
    assert items[0].description == "Monday or Friday?"
    assert items[0].primary_action.href == "/copilot?sessionId=sess-1"


def test_answered_question_leaves_nothing_behind() -> None:
    """Answering clears ``pending_question``. The session is still a recent
    chat and can still be handed to the composer, but it must no longer
    raise a "has a question" card."""
    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=[_asking_session(text=None)],
    )

    assert items == []


def test_expert_question_is_titled_and_avatared_by_its_expert() -> None:
    items = compose_attention_items(
        now=NOW,
        experts=[_expert()],
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=[_asking_session(expert_id="expert")],
    )

    assert items[0].title == "Ada has a question"
    assert items[0].expert is not None and items[0].expert.name == "Ada"


def test_question_from_an_archived_expert_is_dropped() -> None:
    """The expert map only holds *active* experts, so a missing entry means
    the asker was archived. Its thread now refuses every turn before the
    reply clears ``pending_question``, so a card here could never be answered
    away — and the item carries no dismiss action."""
    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=[_asking_session(expert_id="archived-expert")],
    )

    assert items == []


def test_archived_askers_question_does_not_hide_an_answerable_one() -> None:
    """Dropping the dead card must not cost the user a live one."""
    items = compose_attention_items(
        now=NOW,
        experts=[_expert()],
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=[
            _asking_session(session_id="dead", expert_id="archived-expert"),
            _asking_session(session_id="live", expert_id="expert"),
        ],
    )

    assert [item.id for item in items] == ["question-live"]


def test_one_session_asking_twice_yields_one_item() -> None:
    """A session holds one pending question, latest wins. Even handed the
    same session twice, the composer must emit a single card — two would
    collide on the ``question-<session_id>`` item id."""
    earlier = _asking_session(text="first question", asked_at=NOW - timedelta(hours=2))
    latest = _asking_session(text="latest question only", asked_at=NOW)

    items = compose_attention_items(
        now=NOW,
        experts=[],
        reviews=[],
        schedules=[],
        credits_balance=None,
        questions=[earlier, latest],
    )

    assert len(items) == 1
    assert items[0].description == "latest question only"
