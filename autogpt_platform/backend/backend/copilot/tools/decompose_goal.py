"""DecomposeGoalTool - Breaks agent-building goals into sub-instructions."""

import logging
from typing import Any

from backend.copilot.model import ChatSession

from .base import BaseTool
from .models import (
    DecompositionStepModel,
    ErrorResponse,
    TaskDecompositionResponse,
    ToolResponseBase,
)

logger = logging.getLogger(__name__)

DEFAULT_ACTION = "add_block"
VALID_ACTIONS = {"add_block", "connect_blocks", "configure", "add_input", "add_output"}


class DecomposeGoalTool(BaseTool):
    """Tool for decomposing an agent goal into sub-instructions."""

    @property
    def name(self) -> str:
        return "decompose_goal"

    @property
    def description(self) -> str:
        return (
            "Show the user a plain-English plan of what the agent will do, "
            "as a step-by-step card before constructing it. Display-only — "
            "the build continues in the same turn without pausing for user "
            "input."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "goal": {
                    "type": "string",
                    "description": "The user's agent-building goal.",
                },
                "steps": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {
                                "type": "string",
                                "description": (
                                    "Plain-English description of what "
                                    "this step does for the user. Do not "
                                    "put block class names or wiring verbs "
                                    "here — block_name and action carry "
                                    "that technical detail."
                                ),
                            },
                            "action": {
                                "type": "string",
                                "description": (
                                    "Action type: 'add_block', 'connect_blocks', "
                                    "'configure', 'add_input', 'add_output'."
                                ),
                                "enum": list(VALID_ACTIONS),
                            },
                            "block_name": {
                                "type": "string",
                                "description": "Block name if adding a block.",
                            },
                        },
                        "required": ["description", "action"],
                    },
                    "description": "List of sub-instructions for the plan.",
                },
            },
            "required": ["goal", "steps"],
        }

    async def _execute(
        self,
        user_id: str | None,
        session: ChatSession,
        goal: str | None = None,
        steps: list[Any] | None = None,
        **kwargs,
    ) -> ToolResponseBase:
        session_id = session.session_id if session else None

        if not goal:
            return ErrorResponse(
                message="Please provide a goal to decompose.",
                error="missing_goal",
                session_id=session_id,
            )

        if not steps:
            return ErrorResponse(
                message="Please provide at least one step in the plan.",
                error="missing_steps",
                session_id=session_id,
            )

        decomposition_steps: list[DecompositionStepModel] = []
        for i, step in enumerate(steps):
            if not isinstance(step, dict):
                return ErrorResponse(
                    message=f"Step {i + 1} is malformed — expected an object.",
                    error="invalid_step",
                    session_id=session_id,
                )
            description = step.get("description", "")
            if not description or not description.strip():
                return ErrorResponse(
                    message=f"Step {i + 1} is missing a description.",
                    error="empty_description",
                    session_id=session_id,
                )
            action = step.get("action", DEFAULT_ACTION)
            if action not in VALID_ACTIONS:
                action = DEFAULT_ACTION
            decomposition_steps.append(
                DecompositionStepModel(
                    step_id=f"step_{i + 1}",
                    description=description,
                    action=action,
                    block_name=step.get("block_name"),
                    status="pending",
                )
            )

        return TaskDecompositionResponse(
            message=f"Here's the plan to build your agent ({len(decomposition_steps)} steps):",
            goal=goal,
            steps=decomposition_steps,
            step_count=len(decomposition_steps),
            session_id=session_id,
        )
