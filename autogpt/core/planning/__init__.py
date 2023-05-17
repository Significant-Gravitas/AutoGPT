"""The planning system organizes the Agent's activities."""
from autogpt.core.planning.base import Planner
from autogpt.core.planning.schema import (
    LanguageModelClassification,
    LanguageModelPrompt,
    LanguageModelResponse,
    PlanningContext,
    ReflectionContext,
)
from autogpt.core.planning.simple import PlannerSettings, SimplePlanner
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.INTERFACE_DONE,
    handoff_notes=(
        "Before times: Interface has been created. Basic example needs to be created.\n"
        "5/10: SimplePlanner started and has an implementation for the objective prompt, which is all we need for bootstrapping.\n"
        "5/14: Use pydantic models for configuration.\n"
        "5/16: Planner is working for the objective prompt. Templates have been extracted from across the code\n"
        "      for the execution prompt. Additionally, there's an unresolved token count relationship\n"
        "      between the language model and the planner class. Solution TBD.\n"
    ),
)
