"""A held hand-off: the card carries who and what, and an edited card runs
the edited hand-off."""

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import check_action, held
from backend.copilot.gate.reads import is_held_read
from backend.copilot.model import (
    ChatSession,
    get_chat_session,
    update_session_autopilot_mode,
    upsert_chat_session,
)
from backend.copilot.tools.base import BaseTool
from backend.copilot.tools.models import ResponseType, ToolResponseBase
from backend.data.db_accessors import review_db

_TOOL = "delegate_to_expert"
_ARGS = {"expert_id": "expert-b", "prompt": "Draft the PRD", "reason": "Bea owns PRDs"}


class _Delegate(BaseTool):
    def __init__(self) -> None:
        self.runs: list[dict[str, Any]] = []

    @property
    def name(self) -> str:
        return _TOOL

    @property
    def description(self) -> str:
        return "delegates"

    @property
    def parameters(self) -> dict:
        return {"type": "object", "properties": {}}

    async def _execute(self, user_id, session, **kwargs) -> ToolResponseBase:
        self.runs.append(kwargs)
        return ToolResponseBase(type=ResponseType.ERROR, message="delegated")


def _expert(expert_id: str, name: str) -> MagicMock:
    expert = MagicMock(id=expert_id, role="Product Manager", color="violet")
    expert.name = name
    expert.avatar_url = f"https://example.com/{expert_id}.png"
    return expert


@pytest.fixture
def wiring():
    tool = _Delegate()
    team = {
        "expert-b": _expert("expert-b", "Bea"),
        "expert-c": _expert("expert-c", "Cy"),
    }

    async def resolve(_user_id: str, ref: str):
        return team.get(ref)

    with (
        patch("backend.copilot.gate.is_feature_enabled", AsyncMock(return_value=True)),
        patch("backend.copilot.tools.get_tool", return_value=tool),
        patch(
            "backend.copilot.tools.expert_delegation.resolve_target_expert",
            AsyncMock(side_effect=resolve),
        ),
    ):
        yield tool


async def _ask_first_session(user_id: str) -> ChatSession:
    session = await upsert_chat_session(ChatSession.new(user_id=user_id, dry_run=False))
    await update_session_autopilot_mode(session.session_id, user_id, "ask_first")
    reloaded = await get_chat_session(session.session_id, user_id)
    assert reloaded is not None
    return reloaded


async def _hold(session: ChatSession, user_id: str) -> str:
    decision = await check_action(_TOOL, dict(_ARGS), user_id, session, "call-1")
    assert not decision.allowed and decision.review_id
    return decision.review_id


async def _row(review_id: str, user_id: str):
    rows = await review_db().get_reviews_by_node_exec_ids([review_id], user_id)
    return rows[review_id]


@pytest.mark.asyncio(loop_scope="session")
async def test_the_card_names_the_teammate_the_brief_and_why(
    setup_test_user, test_user_id, wiring
):
    session = await _ask_first_session(test_user_id)

    row = await _row(await _hold(session, test_user_id), test_user_id)

    assert row.payload["handoff"] == {
        "expert_id": "expert-b",
        "expert_name": "Bea",
        "expert_role": "Product Manager",
        "expert_avatar_url": "https://example.com/expert-b.png",
        "expert_color": "violet",
        "brief": "Draft the PRD",
        "why": "Bea owns PRDs",
    }
    assert row.editable is True


@pytest.mark.asyncio(loop_scope="session")
async def test_an_edited_card_runs_the_edited_hand_off(
    setup_test_user, test_user_id, wiring
):
    session = await _ask_first_session(test_user_id)
    review_id = await _hold(session, test_user_id)
    edited = dict((await _row(review_id, test_user_id)).payload)
    edited["handoff"] = {
        **edited["handoff"],
        "brief": "Draft the PRD, Q4 scope only",
        "expert_id": "expert-c",
    }
    await review_db().process_all_reviews_for_execution(
        user_id=test_user_id,
        review_decisions={review_id: (ReviewStatus.APPROVED, edited, None)},
    )

    results = await held.resolve_answered(test_user_id, session)

    assert [r.metadata["held_call"]["outcome"] for r in results] == ["approved"]
    assert wiring.runs == [
        {**_ARGS, "prompt": "Draft the PRD, Q4 scope only", "expert_id": "expert-c"}
    ]
    # The edit was the user's approval: it ran, it was not held again. (The
    # teammate's reply is still screened as a read, which is its own card.)
    waiting = await review_db().get_pending_reviews_for_chat_session(
        session.session_id, test_user_id
    )
    assert [r for r in waiting if not is_held_read(r.node_exec_id)] == []


@pytest.mark.asyncio(loop_scope="session")
async def test_an_approved_card_left_as_is_runs_the_original_call(
    setup_test_user, test_user_id, wiring
):
    session = await _ask_first_session(test_user_id)
    review_id = await _hold(session, test_user_id)
    await review_db().process_all_reviews_for_execution(
        user_id=test_user_id,
        review_decisions={review_id: (ReviewStatus.APPROVED, None, None)},
    )

    await held.resolve_answered(test_user_id, session)

    assert wiring.runs == [_ARGS]
