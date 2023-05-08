"""The budget subsystem manages resource limits for the Agent."""
from autogpt.core.budget.base import BudgetManager

import autogpt.core.status

status = autogpt.core.status.Status.BASIC_DONE
handover_notes = "Interface has been completed and a basic implementation has been created."
