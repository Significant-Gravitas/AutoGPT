"""The planning system organizes the Agent's activities."""
from autogpt.core.planning.base import (
    ModelMessage,
    ModelPrompt,
    ModelRole,
    Planner,
    PlanningPromptContext,
    SelfFeedbackPromptContext,
)
from autogpt.core.planning.simple import PlannerConfiguration, SimplePlanner
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Basic example needs to be created.\n"
        "5/10: SimplePlanner started and has an implementation for the objective prompt, which is all we need for bootstrapping.\n"
        "5/14: Use pydantic models for configuration.\n"
    ),
)
