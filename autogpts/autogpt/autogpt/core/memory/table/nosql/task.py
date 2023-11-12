from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from .base import BaseNoSQLTable


class TasksTable(BaseNoSQLTable):
    table_name = "tasks"
    primary_key = "task_id"
    secondary_key = "plan_id"
    third_key = "task_composed_id"

    from autogpts.AFAAS.app.lib.tasks import Task
    
    
    def add(self, value: dict, id: str = Task.generate_uuid()) -> str:
        return super().add(value, id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    def update(self, task_id: str, plan_id: str, value: dict):
        key = TasksTable.Key(
            primary_key=str(task_id),
            secondary_key=str(plan_id),
        )
        return super().update(key=key, value=value)

    def delete(self, task_id: str, plan_id: str):
        key = TasksTable.Key(
            primary_key=str(task_id),
            secondary_key=str(plan_id),
        )
        return super().delete(key=key)

    def get(self, task_id: str, plan_id: str) -> Task:
        key = TasksTable.Key(
            primary_key=str(task_id),
            secondary_key=str(plan_id),
        )
        return super().get(key=key)
