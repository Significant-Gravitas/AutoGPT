"""Read the rubric's verdict off the tool calls a turn made.

The first call that commits to an outcome decides: creating a task or plan,
proposing a hire (the hire card puts it on the board as one task), changing
an open task, or scheduling. A question doesn't decide: the user answers it
and the turn goes on, so what Otto does next is the outcome, and the
question is recorded beside it. A turn that commits to nothing answered
inline, which is "Not a task".
"""

import functools
from typing import Any

from pydantic import BaseModel, Field

from backend.blocks import get_blocks
from backend.blocks._base import BlockType

from .examples import Verdict

ASK_TOOL = "ask_question"
ROUTINE_TOOLS = frozenset(
    {"schedule_routine", "schedule_followup", "setup_agent_webhook_trigger"}
)
UPDATE_TOOLS = frozenset({"comment_on_task", "update_task_status"})
HIRE_TOOLS = frozenset({"hire_expert", "raise_expert"})
BUILD_TOOLS = frozenset({"create_agent", "edit_agent"})
TRIGGER_BLOCK_TYPES = (BlockType.WEBHOOK, BlockType.WEBHOOK_MANUAL)


class ToolCall(BaseModel):
    name: str
    args: dict[str, Any] = Field(default_factory=dict)


class TurnVerdict(BaseModel):
    outcome: Verdict
    asked_first: bool


def call_verdict(call: ToolCall) -> Verdict | None:
    """The outcome this call commits the turn to, or None if it doesn't."""
    if call.name in ROUTINE_TOOLS:
        return Verdict.ROUTINE
    if call.name == "run_agent" and call.args.get("cron"):
        return Verdict.ROUTINE
    if call.name in BUILD_TOOLS and _starts_from_a_trigger(call.args):
        return Verdict.ROUTINE
    if call.name in UPDATE_TOOLS:
        return Verdict.UPDATE_EXISTING
    if call.name in HIRE_TOOLS:
        return Verdict.TASK
    if call.name == "create_task":
        if call.args.get("parent_task_id"):
            return Verdict.UPDATE_EXISTING
        return Verdict.TASK
    if call.name == "propose_plan":
        if call.args.get("split"):
            return Verdict.PARENT_WITH_SUBTASKS
        return Verdict.TASK
    return None


def _starts_from_a_trigger(args: dict[str, Any]) -> bool:
    """A workflow built to run "whenever…": its graph has a trigger block.
    ``schedule_routine`` only takes crons, so an event routine is built."""
    graph = args.get("agent_json")
    nodes = graph.get("nodes") if isinstance(graph, dict) else None
    if not isinstance(nodes, list):
        return False
    return any(
        isinstance(n, dict) and n.get("block_id") in trigger_block_ids() for n in nodes
    )


@functools.cache
def trigger_block_ids() -> frozenset[str]:
    return frozenset(
        block_id
        for block_id, block in get_blocks().items()
        if block().block_type in TRIGGER_BLOCK_TYPES
    )


def turn_verdict(calls: list[ToolCall]) -> TurnVerdict:
    asked = False
    for call in calls:
        if call.name == ASK_TOOL:
            asked = True
            continue
        if (outcome := call_verdict(call)) is not None:
            return TurnVerdict(outcome=outcome, asked_first=asked)
    return TurnVerdict(outcome=Verdict.NOT_A_TASK, asked_first=asked)


def passes(expected: Verdict, got: TurnVerdict) -> bool:
    """The right outcome; for "Ask then task", a question and then a task."""
    if expected == Verdict.ASK_THEN_TASK:
        return got.asked_first and got.outcome in (
            Verdict.TASK,
            Verdict.PARENT_WITH_SUBTASKS,
        )
    return got.outcome == expected


def asked_needlessly(expected: Verdict, got: TurnVerdict) -> bool:
    return got.asked_first and expected != Verdict.ASK_THEN_TASK
