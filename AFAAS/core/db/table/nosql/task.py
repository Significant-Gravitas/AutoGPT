from __future__ import annotations

from .....interfaces.db.table.nosql.base import BaseNoSQLTable


class TasksTable(BaseNoSQLTable):
    table_name = "tasks"
    primary_key = "task_id"
    secondary_key = "plan_id"
    third_key = "task_composed_id"

    from AFAAS.lib.task.task import Task

    async def add(self, value: dict, id: str = Task.generate_uuid()) -> str:
        return await super().add(value=value, id=id)

    # NOTE : overwrite parent update
    # Perform any custom logic needed for updating an agent
    async def update(self, task_id: str, plan_id: str, value: dict):
        key = TasksTable.Key(
            primary_key=str(task_id),
            secondary_key=str(plan_id),
        )
        return await super().update(key=key, value=value)

    async def delete(self, task_id: str, plan_id: str):
        key = TasksTable.Key(
            primary_key=str(task_id),
            secondary_key=str(plan_id),
        )
        return await super().delete(key=key)

    async def get(self, task_id: str, plan_id: str) -> Task:
        key = TasksTable.Key(
            primary_key=str(task_id),
            secondary_key=str(plan_id),
        )
        return await super().get(key=key)
