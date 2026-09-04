"""The reasoner↔executor wire format.

A split-brain turn is two transcripts talking through exactly two payloads: the
reasoner sends an :class:`Intent` (what to achieve, what counts as done), the
executor sends back a :class:`Report` (what happened, what is in the way). The
point of typing them is that nothing else crosses — no tool call, no schema, no
raw error text — so the reasoner's context cannot be grown by the executor's
work.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

ReportStatus = Literal["done", "partial", "blocked"]


class Intent(BaseModel):
    """One unit of work the reasoner hands down."""

    goal: str = Field(description="What to achieve, in one or two sentences.")
    acceptance: str = Field(description="What the executor must see to call it done.")
    context: str = Field(
        default="",
        description="Facts the executor cannot discover with its own tools.",
    )
    max_steps: int = Field(default=10, description="Tool-call budget for this intent.")


class Report(BaseModel):
    """What comes back. Every field is bounded on purpose."""

    status: ReportStatus
    summary: str = Field(description="What was done and what the state now is.")
    problems: list[str] = Field(
        default_factory=list,
        description="Blockers the reasoner must decide about. Empty when none.",
    )
    artifacts: dict[str, Any] = Field(
        default_factory=dict,
        description="Handles (ids, counts, names) — never payloads.",
    )
    steps_used: int = 0


INTENT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "Agent",
    "description": (
        "Hand one unit of work to your executor, which has the tools and does "
        "the work in its own context. Returns its report. This is the only way "
        "you can affect anything."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "goal": {"type": "string", "description": "What to achieve."},
            "acceptance": {
                "type": "string",
                "description": "What the executor must observe to call it done.",
            },
            "context": {
                "type": "string",
                "description": (
                    "Facts it cannot discover itself (decisions you made, "
                    "constraints from the user). Keep it short."
                ),
                "default": "",
            },
            "max_steps": {
                "type": "integer",
                "description": "Tool-call budget for this intent.",
                "default": 10,
            },
        },
        "required": ["goal", "acceptance"],
    },
}

REPORT_TOOL_SCHEMA: dict[str, Any] = {
    "name": "report",
    "description": (
        "Hand the result of this intent back to the reasoner and end your turn. "
        "The reasoner sees ONLY this — it cannot see your tool calls, the "
        "schemas you read, or the errors you got. Anything it must know to "
        "decide the next step has to be in here."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "status": {"type": "string", "enum": ["done", "partial", "blocked"]},
            "summary": {
                "type": "string",
                "description": "What you did and what the state is now.",
            },
            "problems": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Blockers needing a decision, each with enough detail to "
                    "decide on. Empty when there are none."
                ),
                "default": [],
            },
            "artifacts": {
                "type": "object",
                "description": "Handles only — ids, names, counts. Never payloads.",
                "default": {},
            },
        },
        "required": ["status", "summary"],
    },
}
