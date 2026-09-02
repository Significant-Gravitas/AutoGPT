"""Domain errors for the task spine.

Kept Prisma-free so copilot tools (which run without a Prisma client) can
import and catch them. Mirrors ``experts/errors.py``.
"""


class DelegatedTaskNotFoundError(Exception):
    def __init__(self, task_id: str):
        super().__init__(f"Task {task_id} not found")
        self.task_id = task_id
