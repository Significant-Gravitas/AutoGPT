"""Hand-offs on Home: questions and held cards under Needs you, finished
hand-offs under Recent work."""

from datetime import UTC, datetime, timedelta

from backend.api.features.experts.delegations import DelegationSummary
from backend.api.features.experts.models import Expert
from backend.api.features.graph_executions.review.model import PendingHumanReviewModel
from backend.copilot.tools.models import DelegatedExpertInfo

from .attention import compose_attention_items
from .recent_work import compose_recent_work

_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=UTC)
_ALEX = DelegatedExpertInfo(
    id="expert-alex", name="Alex", role="Product Manager", avatar_url=None
)


def _expert() -> Expert:
    return Expert(
        id="expert-alex",
        name="Alex",
        avatar_url=None,
        role="Product Manager",
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


def _delegation(status: str, **overrides) -> DelegationSummary:
    base = {
        "sub_session_id": "sub-1",
        "parent_session_id": "otto-chat",
        "expert": _ALEX,
        "title": "Onboarding revamp PRD",
        "brief": "Onboarding revamp PRD\nKeep it short",
        "status": status,
        "created_at": _NOW - timedelta(minutes=20),
        "finished_at": None,
        "elapsed_seconds": None,
        "cost_usd": 0.31,
        "files_count": 0,
        "question": None,
        "question_options": [],
    }
    return DelegationSummary.model_validate({**base, **overrides})


def test_a_delegated_question_asks_from_the_chat_that_delegated():
    asking = _delegation(
        "needs_input",
        question="Q4 release train or December?",
        asked_at=_NOW - timedelta(minutes=3),
    )

    [item] = compose_attention_items(
        now=_NOW,
        experts=[_expert()],
        reviews=[],
        schedules=[],
        credits_balance=None,
        delegated_questions=[asking],
    )

    assert item.kind == "question"
    assert item.id == "question-sub-1"
    assert item.title == "Q4 release train or December?"
    assert item.description == (
        "Alex, working for Otto on “Onboarding revamp PRD” · asked 3m ago"
    )
    assert item.primary_action.href == "/copilot?sessionId=otto-chat"
    assert item.expert is not None and item.expert.name == "Alex"


def test_a_question_from_an_archived_teammate_is_dropped():
    asking = _delegation("needs_input", question="Which?", asked_at=_NOW)

    items = compose_attention_items(
        now=_NOW,
        experts=[],
        reviews=[],
        schedules=[],
        credits_balance=None,
        delegated_questions=[asking],
    )

    assert items == []


def test_a_held_hand_off_is_titled_by_its_brief_and_teammate():
    review = PendingHumanReviewModel.model_validate(
        {
            "node_exec_id": "copilot-node-gate-delegate_to_expert:abc",
            "node_id": "copilot-node-gate-delegate_to_expert",
            "user_id": "u",
            "session_id": "otto-chat",
            "payload": {
                "tool": "delegate_to_expert",
                "arguments": {"expert_id": "expert-alex", "prompt": "Revamp"},
                "subject": {
                    "key": "delegate_to_expert",
                    "name": "x",
                    "effect": "platform",
                },
                "headline": {"ask": "Hand a task to a teammate", "object": "Alex"},
                "handoff": {
                    "expert_id": "expert-alex",
                    "expert_name": "Alex",
                    "expert_role": "Product Manager",
                    "expert_avatar_url": "https://example.com/a.png",
                    "brief": "Onboarding revamp PRD\nDetails follow",
                    "why": "Alex owns onboarding",
                },
            },
            "instructions": "Hand a task to a teammate",
            "editable": True,
            "status": "WAITING",
            "created_at": _NOW,
        }
    )

    [item] = compose_attention_items(
        now=_NOW, experts=[], reviews=[review], schedules=[], credits_balance=None
    )

    assert item.kind == "approval"
    assert item.title == "Hand off “Onboarding revamp PRD” to Alex"
    assert item.expert is not None
    assert (item.expert.name, item.expert.role) == ("Alex", "Product Manager")
    assert item.primary_action.href == "/copilot?sessionId=otto-chat"


def test_finished_hand_offs_land_under_the_teammate_who_took_them():
    done = _delegation(
        "completed",
        created_at=datetime(2026, 9, 28, 10, 41, tzinfo=UTC),
        finished_at=datetime(2026, 9, 28, 10, 48, tzinfo=UTC),
        files_count=1,
    )
    stopped = _delegation(
        "cancelled",
        sub_session_id="sub-2",
        title="Email Dana",
        created_at=datetime(2026, 9, 28, 9, 0, tzinfo=UTC),
        finished_at=datetime(2026, 9, 28, 9, 5, tzinfo=UTC),
    )
    working = _delegation("running", sub_session_id="sub-3")
    old = _delegation(
        "completed",
        sub_session_id="sub-4",
        finished_at=_NOW - timedelta(days=9),
    )

    work = compose_recent_work(
        now=_NOW,
        executions=[],
        events=[],
        expert_by_id={"expert-alex": _expert()},
        agent_by_graph={},
        session_titles={},
        delegations=[done, stopped, working, old],
        timezone_name="UTC",
    )

    [group] = work.groups
    assert group.actor.name == "Alex"
    assert group.delegation_count == 2
    assert [
        (d.sub_session_id, d.status, d.description, d.link) for d in group.delegations
    ] == [
        (
            "sub-1",
            "completed",
            "Delegated 10:41 · returned 10:48 · 1 file",
            "/copilot?sessionId=otto-chat",
        ),
        (
            "sub-2",
            "cancelled",
            "Delegated 09:00 · stopped 09:05",
            "/copilot?sessionId=otto-chat",
        ),
    ]
