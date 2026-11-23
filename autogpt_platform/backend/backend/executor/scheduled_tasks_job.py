"""The Scheduler's due-task sweep job (``backend.copilot.task_due``).

Registered by reference (``START_SCHEDULED_TASKS_JOB``): the sweep imports
copilot modules that import the scheduler, so the scheduler cannot import
this module at load time.
"""

from backend.copilot.task_due import start_due_tasks
from backend.executor.scheduler import run_async


def start_scheduled_tasks() -> None:
    """Start the tasks whose start time has come; ``max_instances`` keeps one
    pass at a time."""
    run_async(start_due_tasks(), timeout=None)
