"""Structured plan schema for the two-phase planner/executor split.

The planner model emits JSON validated into :class:`Plan`.  The schema is
deliberately small — ordered steps, each with an id, a short description, the
tool(s) expected, and a success criterion — so the executor can turn it into a
``TodoWrite`` checklist (reusing the existing frontend checklist UX) and know
when each step is done.  Persisted inside ``ChatSessionMetadata.plan`` (a JSON
column, no migration).
"""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator, model_validator

# Upper bound on plan size — a planner that emits hundreds of steps is
# hallucinating structure, not planning.  Keeps the persisted metadata and the
# injected system-prompt block bounded.
MAX_PLAN_STEPS = 20


class PlanStep(BaseModel):
    """A single ordered step in a plan."""

    id: str = Field(description="Stable short id for the step, e.g. 'step-1'.")
    description: str = Field(
        description="Short imperative description of the step, e.g. "
        "'Fetch the latest issues from GitHub'."
    )
    expected_tools: list[str] = Field(
        default_factory=list,
        description="Names of the tool(s) the executor is expected to use for "
        "this step. Advisory only — the executor is not forced to use them.",
    )
    success_criteria: str = Field(
        description="How to tell the step has succeeded, e.g. "
        "'A non-empty list of issues is retrieved'."
    )

    @field_validator("id", "description", "success_criteria")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("must be a non-empty string")
        return v.strip()

    @field_validator("expected_tools")
    @classmethod
    def _clean_tools(cls, v: list[str]) -> list[str]:
        return [t.strip() for t in v if isinstance(t, str) and t.strip()]


class Plan(BaseModel):
    """An ordered, validated plan produced by the planner model."""

    steps: list[PlanStep] = Field(description="Ordered list of plan steps.")

    @field_validator("steps")
    @classmethod
    def _validate_step_list(cls, v: list[PlanStep]) -> list[PlanStep]:
        if not v:
            raise ValueError("plan must contain at least one step")
        if len(v) > MAX_PLAN_STEPS:
            raise ValueError(f"plan exceeds the {MAX_PLAN_STEPS}-step limit")
        return v

    @model_validator(mode="after")
    def _unique_ids(self) -> "Plan":
        ids = [s.id for s in self.steps]
        if len(set(ids)) != len(ids):
            raise ValueError("plan step ids must be unique")
        return self


class TurnTokenBreakdown(BaseModel):
    """Per-phase token / cost split for one baseline turn.

    Lets a test or the ``planner_executor`` eval suite compare where the tokens
    go under the two-phase split: the up-front ``planner`` call, the
    ``executor`` tool-call loop, and any ``replan`` calls.  Compare the
    ``total_*`` here against a flag-OFF (single-loop) run's total to see the
    overhead the split adds.
    """

    planner_prompt_tokens: int = 0
    planner_completion_tokens: int = 0
    executor_prompt_tokens: int = 0
    executor_completion_tokens: int = 0
    replan_prompt_tokens: int = 0
    replan_completion_tokens: int = 0
    planner_cost_usd: float | None = None
    executor_cost_usd: float | None = None
    replan_cost_usd: float | None = None

    @property
    def planner_tokens(self) -> int:
        return self.planner_prompt_tokens + self.planner_completion_tokens

    @property
    def executor_tokens(self) -> int:
        return self.executor_prompt_tokens + self.executor_completion_tokens

    @property
    def replan_tokens(self) -> int:
        return self.replan_prompt_tokens + self.replan_completion_tokens

    @property
    def total_tokens(self) -> int:
        return self.planner_tokens + self.executor_tokens + self.replan_tokens

    @property
    def total_prompt_tokens(self) -> int:
        return (
            self.planner_prompt_tokens
            + self.executor_prompt_tokens
            + self.replan_prompt_tokens
        )

    @property
    def total_completion_tokens(self) -> int:
        return (
            self.planner_completion_tokens
            + self.executor_completion_tokens
            + self.replan_completion_tokens
        )

    @property
    def total_cost_usd(self) -> float | None:
        parts = [
            c
            for c in (
                self.planner_cost_usd,
                self.executor_cost_usd,
                self.replan_cost_usd,
            )
            if c is not None
        ]
        return sum(parts) if parts else None

    @property
    def overhead_tokens(self) -> int:
        """Tokens spent outside the executor loop (planner + re-plans).

        This is the extra cost the split adds on top of the tool-call loop
        that would run anyway — the number to weigh against any executor-side
        savings from running on the cheaper model with a plan.
        """
        return self.planner_tokens + self.replan_tokens
