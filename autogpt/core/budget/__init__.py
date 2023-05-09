"""The budget subsystem manages resource limits for the Agent."""
from autogpt.core.budget.base import BudgetManager
from autogpt.core.status import ShortStatus, Status

status = Status(
    module_name=__name__,
    short_status=ShortStatus.BASIC_DONE,
    handoff_notes="Interface has been completed and a basic implementation has been created.",
)
