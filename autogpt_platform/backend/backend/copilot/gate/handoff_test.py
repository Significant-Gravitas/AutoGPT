"""The hand-off card and reading an edit back from it, without a database."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.gate import handoff
from backend.copilot.gate.handoff import (
    BRIEF_MAX_CHARS,
    HandoffCard,
    approved_edit,
    edited_args,
    handoff_card,
    is_approved_edit,
)

_ARGS = {"expert_id": "expert-b", "prompt": "Draft the PRD", "reason": "owns it"}


def _payload(**card) -> dict:
    base = {"expert_id": "expert-b", "expert_name": "Bea", "brief": "Draft the PRD"}
    return {"handoff": {**base, **card}}


@pytest.mark.asyncio
async def test_only_a_hand_off_gets_a_card():
    assert await handoff_card("post_to_chat_platform", _ARGS, "u1") is None


@pytest.mark.asyncio
async def test_an_unknown_teammate_is_named_by_the_reference():
    with patch.object(handoff, "_resolve", AsyncMock(return_value=None)):
        card = await handoff_card("delegate_to_expert", _ARGS, "u1")

    assert card == HandoffCard(
        expert_id="expert-b",
        expert_name="expert-b",
        brief="Draft the PRD",
        why="owns it",
    )


@pytest.mark.asyncio
async def test_a_failed_lookup_still_builds_the_card():
    with patch(
        "backend.copilot.tools.expert_delegation.resolve_target_expert",
        AsyncMock(side_effect=RuntimeError("down")),
    ):
        card = await handoff_card("delegate_to_expert", {"prompt": "x"}, "u1")

    assert card is not None and (card.expert_id, card.why) == ("", None)


def test_an_unedited_card_is_not_an_edit():
    assert edited_args("delegate_to_expert", _ARGS, _payload()) is None


def test_a_cut_brief_read_back_is_not_an_edit():
    long_args = {**_ARGS, "prompt": "x" * (BRIEF_MAX_CHARS + 10)}
    payload = _payload(brief="x" * BRIEF_MAX_CHARS)

    assert edited_args("delegate_to_expert", long_args, payload) is None


def test_the_brief_and_teammate_are_the_only_edits():
    edited = edited_args(
        "delegate_to_expert",
        _ARGS,
        {**_payload(brief=" Q4 only ", expert_id="expert-c"), "reason": "ignored"},
    )

    assert edited == {**_ARGS, "prompt": "Q4 only", "expert_id": "expert-c"}


@pytest.mark.parametrize(
    "tool,payload",
    [
        ("post_to_chat_platform", _payload(brief="changed")),
        ("delegate_to_expert", {"handoff": {"brief": "no teammate"}}),
        ("delegate_to_expert", {}),
    ],
)
def test_nothing_else_reads_as_an_edit(tool, payload):
    assert edited_args(tool, _ARGS, payload) is None


def test_an_approved_edit_passes_only_while_its_card_runs():
    assert not is_approved_edit("review-1")
    with approved_edit("review-1"):
        assert is_approved_edit("review-1")
        assert not is_approved_edit("review-2")
    assert not is_approved_edit("review-1")
