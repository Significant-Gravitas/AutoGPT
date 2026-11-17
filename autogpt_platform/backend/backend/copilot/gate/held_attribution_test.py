"""Approval delivery retains each call's original usage generation."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.gate import held
from backend.copilot.model import ChatSession
from backend.copilot.trial_cost_context import (
    TrialCostContext,
    get_trial_cost_context,
    restore_cost_context,
)


async def test_mixed_trial_and_paid_deliveries_keep_their_own_origin(monkeypatch):
    trial = TrialCostContext(user_id="user-1", trial_id="trial-1")
    paid = TrialCostContext(user_id="user-1", trial_id=None, generation="paid-1")
    calls = [
        held.HeldCall(
            review_id=f"review-{i}",
            tool_name="post_to_chat_platform",
            tool_call_id=f"call-{i}",
            args={},
            cost_context=context,
        )
        for i, context in enumerate((trial, paid))
    ]
    monkeypatch.setattr(held, "answered", AsyncMock(return_value=calls))
    monkeypatch.setattr(held, "_claim", AsyncMock(return_value=True))
    monkeypatch.setattr(
        held,
        "review_db",
        lambda: MagicMock(get_reviews_by_node_exec_ids=AsyncMock(return_value={})),
    )
    origins = []

    async def outcome(*args):
        origins.append(get_trial_cost_context("user-1"))
        return "approved", "done"

    monkeypatch.setattr(held, "_settled", outcome)
    delivered = []
    with restore_cost_context("user-1", paid):
        async for _status in held.resolve_answered(
            "user-1", ChatSession.new(user_id="user-1", dry_run=False), delivered.append
        ):
            pass
        assert get_trial_cost_context("user-1") == paid
    assert len(delivered) == 2
    assert origins == [trial, paid]
    assert get_trial_cost_context("user-1") is None


@pytest.mark.parametrize(
    "snapshot", [None, TrialCostContext(user_id="another-user", trial_id="trial-1")]
)
async def test_unknown_held_origin_closes_without_running(monkeypatch, snapshot):
    call = held.HeldCall(
        review_id="review-1",
        tool_name="post_to_chat_platform",
        tool_call_id="call-1",
        args={},
        cost_context=snapshot,
    )
    monkeypatch.setattr(held, "answered", AsyncMock(return_value=[call]))
    monkeypatch.setattr(held, "_claim", AsyncMock(return_value=True))
    monkeypatch.setattr(
        held,
        "review_db",
        lambda: MagicMock(get_reviews_by_node_exec_ids=AsyncMock(return_value={})),
    )
    consume = AsyncMock()
    run = AsyncMock(return_value=("approved", "done"))
    settled = AsyncMock(return_value=("approved", "done"))
    monkeypatch.setattr(held.review_store, "consume", consume)
    monkeypatch.setattr(held, "_run", run)
    monkeypatch.setattr(held, "_settled", settled)
    delivered = []
    async for _status in held.resolve_answered(
        "user-1", ChatSession.new(user_id="user-1", dry_run=False), delivered.append
    ):
        pass
    [result] = delivered
    run.assert_not_awaited()
    settled.assert_not_awaited()
    consume.assert_awaited_once_with("review-1", "user-1")
    assert result.metadata["held_call"]["outcome"] == "closed"
    assert "send the request again" in result.content
