"""Tests for the bounded re-plan escalation controller."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.planner import replan as replan_mod
from backend.copilot.planner.models import Plan
from backend.copilot.planner.replan import (
    MAX_REPLANS,
    REPLAN_ERROR_THRESHOLD,
    ReplanController,
    evaluate_replan_trigger,
)
from backend.copilot.planner.service import REPLAN_SIGNAL, PlannerUsage


def _plan(n: int = 1) -> Plan:
    return Plan.model_validate(
        {
            "steps": [
                {"id": f"step-{i + 1}", "description": "d", "success_criteria": "s"}
                for i in range(n)
            ]
        }
    )


def _revise_result(plan, cost=0.03):
    """Shape a ``revise_plan`` return: ``(plan, PlannerUsage)``."""
    return plan, PlannerUsage(prompt_tokens=50, completion_tokens=10, cost_usd=cost)


def _controller(plan) -> ReplanController:
    return ReplanController(
        plan,
        planner_model="anthropic/claude-opus-4.7",
        user_id="u",
        session_id="s",
        config=ChatConfig(),
    )


class TestEvaluateReplanTrigger:
    def test_explicit_signal_returns_reason(self):
        reason = evaluate_replan_trigger(f"stuck\n{REPLAN_SIGNAL} tool missing", 0)
        assert reason == "tool missing"

    def test_signal_without_reason_has_default(self):
        assert evaluate_replan_trigger(REPLAN_SIGNAL, 0)

    def test_error_threshold(self):
        assert evaluate_replan_trigger("", REPLAN_ERROR_THRESHOLD) is not None

    def test_below_threshold_no_signal(self):
        assert evaluate_replan_trigger("all good", REPLAN_ERROR_THRESHOLD - 1) is None


class TestReplanController:
    @pytest.mark.asyncio
    async def test_no_plan_is_noop(self):
        c = _controller(None)
        out = await c.maybe_revise(
            executor_text=f"{REPLAN_SIGNAL} x", consecutive_tool_errors=9, tools=[]
        )
        assert out.action == "none"

    @pytest.mark.asyncio
    async def test_signal_triggers_revision(self):
        revised = _plan(2)
        with patch.object(
            replan_mod,
            "revise_plan",
            new=AsyncMock(return_value=_revise_result(revised)),
        ):
            c = _controller(_plan(1))
            out = await c.maybe_revise(
                executor_text=f"working\n{REPLAN_SIGNAL} tool gone",
                consecutive_tool_errors=0,
                tools=[],
            )
        assert out.action == "revised"
        assert out.plan is revised
        assert out.system_message and "<execution_plan>" in out.system_message
        assert c.replans_used == 1
        # The re-plan call's usage is surfaced so the caller can bill it.
        assert out.usage.cost_usd == 0.03

    @pytest.mark.asyncio
    async def test_scan_cursor_prevents_double_trigger(self):
        with patch.object(
            replan_mod,
            "revise_plan",
            new=AsyncMock(return_value=_revise_result(_plan(1))),
        ):
            c = _controller(_plan(1))
            text = f"working\n{REPLAN_SIGNAL} gone"
            first = await c.maybe_revise(
                executor_text=text, consecutive_tool_errors=0, tools=[]
            )
            # Same cumulative text — the signal is already past the cursor.
            second = await c.maybe_revise(
                executor_text=text, consecutive_tool_errors=0, tools=[]
            )
        assert first.action == "revised"
        assert second.action == "none"

    @pytest.mark.asyncio
    async def test_error_threshold_triggers_revision(self):
        with patch.object(
            replan_mod,
            "revise_plan",
            new=AsyncMock(return_value=_revise_result(_plan(1))),
        ):
            c = _controller(_plan(1))
            out = await c.maybe_revise(
                executor_text="no signal",
                consecutive_tool_errors=REPLAN_ERROR_THRESHOLD,
                tools=[],
            )
        assert out.action == "revised"

    @pytest.mark.asyncio
    async def test_cap_after_max_replans(self):
        with patch.object(
            replan_mod,
            "revise_plan",
            new=AsyncMock(return_value=_revise_result(_plan(1))),
        ):
            c = _controller(_plan(1))
            actions = []
            cumulative = ""
            for i in range(MAX_REPLANS + 2):
                cumulative += f"\nround{i} {REPLAN_SIGNAL} r{i}"
                out = await c.maybe_revise(
                    executor_text=cumulative, consecutive_tool_errors=0, tools=[]
                )
                actions.append(out.action)
        assert actions.count("revised") == MAX_REPLANS
        assert actions[MAX_REPLANS] == "capped"
        # Sticky: once capped, always a no-op.
        assert actions[MAX_REPLANS + 1] == "none"
        assert c.capped is True

    @pytest.mark.asyncio
    async def test_revision_failure_does_not_consume_cap(self):
        with patch.object(
            replan_mod,
            "revise_plan",
            new=AsyncMock(return_value=(None, PlannerUsage(prompt_tokens=50))),
        ):
            c = _controller(_plan(1))
            out = await c.maybe_revise(
                executor_text=f"{REPLAN_SIGNAL} nope",
                consecutive_tool_errors=0,
                tools=[],
            )
        assert out.action == "revision_failed"
        assert c.replans_used == 0
        # A failed revision still bills the call it made.
        assert out.usage.prompt_tokens == 50
