"""Tests for planner LLM calls, parsing, and executor prompt building."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.planner import service as svc
from backend.copilot.planner.models import Plan
from backend.copilot.planner.service import (
    REPLAN_SIGNAL,
    PlannerUsage,
    generate_plan,
    parse_plan,
    plan_to_executor_prompt,
)

_VALID_JSON = (
    '{"steps":[{"id":"step-1","description":"Fetch issues",'
    '"expected_tools":["run_block"],"success_criteria":"issues fetched"}]}'
)


def _call_result(text, prompt=100, completion=20, cost=0.05):
    """Shape a ``_call_planner_model`` return: ``(text, PlannerUsage)``."""
    return text, PlannerUsage(
        prompt_tokens=prompt, completion_tokens=completion, cost_usd=cost
    )


def _cfg() -> ChatConfig:
    return ChatConfig(
        planner_model="anthropic/claude-opus-4.7",
        executor_model="anthropic/claude-sonnet-4-6",
    )


def _plan(n: int = 2) -> Plan:
    return Plan.model_validate(
        {
            "steps": [
                {"id": f"step-{i + 1}", "description": "d", "success_criteria": "s"}
                for i in range(n)
            ]
        }
    )


class TestParsePlan:
    def test_fenced_json(self):
        assert parse_plan("```json\n" + _VALID_JSON + "\n```") is not None

    def test_prose_wrapped_json(self):
        assert parse_plan("Sure! " + _VALID_JSON + " Hope that helps.") is not None

    @pytest.mark.parametrize("bad", ["", "not json", "{not valid}", '{"foo": 1}'])
    def test_invalid_returns_none(self, bad):
        assert parse_plan(bad) is None


class TestPlanToExecutorPrompt:
    def test_contains_key_directives(self):
        prompt = plan_to_executor_prompt(_plan())
        assert "<execution_plan>" in prompt and "</execution_plan>" in prompt
        assert "TodoWrite" in prompt
        assert REPLAN_SIGNAL in prompt

    def test_revised_header(self):
        assert "REVISED" in plan_to_executor_prompt(_plan(), revised=True)


class TestGeneratePlan:
    @pytest.mark.asyncio
    async def test_valid_first_try(self):
        with patch.object(
            svc,
            "_call_planner_model",
            new=AsyncMock(return_value=_call_result(_VALID_JSON)),
        ):
            plan, usage = await generate_plan(
                message="build an agent",
                conversation=[],
                tools=[],
                planner_model="anthropic/claude-opus-4.7",
                user_id="u",
                config=_cfg(),
            )
        assert plan is not None and len(plan.steps) == 1
        # Planner usage is threaded back so the caller can bill it.
        assert usage.prompt_tokens == 100 and usage.cost_usd == 0.05

    @pytest.mark.asyncio
    async def test_retries_once_then_succeeds_and_accumulates_usage(self):
        mock = AsyncMock(
            side_effect=[_call_result("garbage"), _call_result(_VALID_JSON)]
        )
        with patch.object(svc, "_call_planner_model", new=mock):
            plan, usage = await generate_plan(
                message="build an agent",
                conversation=[],
                tools=[],
                planner_model="anthropic/claude-opus-4.7",
                user_id="u",
                config=_cfg(),
            )
        assert plan is not None
        assert mock.call_count == 2
        # Both attempts' tokens/cost are summed — the failed attempt still bills.
        assert usage.prompt_tokens == 200 and usage.cost_usd == 0.10

    @pytest.mark.asyncio
    async def test_two_invalid_outputs_fail_open(self):
        with patch.object(
            svc, "_call_planner_model", new=AsyncMock(return_value=_call_result("nope"))
        ):
            plan, usage = await generate_plan(
                message="build an agent",
                conversation=[],
                tools=[],
                planner_model="anthropic/claude-opus-4.7",
                user_id="u",
                config=_cfg(),
            )
        assert plan is None
        # Failing to produce a plan still bills both attempts.
        assert usage.prompt_tokens == 200

    @pytest.mark.asyncio
    async def test_transport_error_fails_open_without_retry(self):
        mock = AsyncMock(return_value=(None, PlannerUsage()))
        with patch.object(svc, "_call_planner_model", new=mock):
            plan, _usage = await generate_plan(
                message="build an agent",
                conversation=[],
                tools=[],
                planner_model="anthropic/claude-opus-4.7",
                user_id="u",
                config=_cfg(),
            )
        assert plan is None
        assert mock.call_count == 1

    @pytest.mark.asyncio
    async def test_unnormalizable_model_fails_open(self):
        # A non-Anthropic slug under a direct-Anthropic transport can't be
        # normalized — planner must fail open rather than raise.
        cfg = ChatConfig(
            planner_model="moonshotai/kimi-k2.6",
            use_openrouter=False,
            direct_anthropic_api_key="sk-test",
        )
        with patch.object(svc, "_call_planner_model", new=AsyncMock()) as mock:
            plan, usage = await generate_plan(
                message="build an agent",
                conversation=[],
                tools=[],
                planner_model="moonshotai/kimi-k2.6",
                user_id="u",
                config=cfg,
            )
        assert plan is None
        assert usage.prompt_tokens == 0
        mock.assert_not_called()
