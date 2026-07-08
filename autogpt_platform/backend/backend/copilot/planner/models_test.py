"""Tests for the plan schema."""

import pytest
from pydantic import ValidationError

from backend.copilot.planner.models import MAX_PLAN_STEPS, Plan, PlanStep


def _step(id: str = "step-1", **over):
    data = {"id": id, "description": "Do the thing", "success_criteria": "It is done"}
    data.update(over)
    return data


class TestPlanStep:
    def test_valid(self):
        s = PlanStep.model_validate(_step(expected_tools=["run_block"]))
        assert s.id == "step-1"
        assert s.expected_tools == ["run_block"]

    def test_expected_tools_default_empty(self):
        assert PlanStep.model_validate(_step()).expected_tools == []

    def test_strips_and_cleans_tools(self):
        s = PlanStep.model_validate(_step(expected_tools=[" run_block ", "", "  "]))
        assert s.expected_tools == ["run_block"]

    @pytest.mark.parametrize("field", ["id", "description", "success_criteria"])
    def test_blank_required_field_rejected(self, field):
        with pytest.raises(ValidationError):
            PlanStep.model_validate(_step(**{field: "   "}))


class TestPlan:
    def test_valid_plan(self):
        p = Plan.model_validate({"steps": [_step("step-1"), _step("step-2")]})
        assert len(p.steps) == 2

    def test_empty_plan_rejected(self):
        with pytest.raises(ValidationError):
            Plan.model_validate({"steps": []})

    def test_duplicate_ids_rejected(self):
        with pytest.raises(ValidationError):
            Plan.model_validate({"steps": [_step("dup"), _step("dup")]})

    def test_over_cap_rejected(self):
        steps = [_step(f"step-{i}") for i in range(MAX_PLAN_STEPS + 1)]
        with pytest.raises(ValidationError):
            Plan.model_validate({"steps": steps})

    def test_at_cap_allowed(self):
        steps = [_step(f"step-{i}") for i in range(MAX_PLAN_STEPS)]
        assert len(Plan.model_validate({"steps": steps}).steps) == MAX_PLAN_STEPS
