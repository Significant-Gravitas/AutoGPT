from __future__ import annotations

import copy
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest

from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import Task

from .dataset.agent_planner import agent_dataset
from .dataset.plan_familly_dinner import (
    _plan_familly_dinner,
    _plan_step_3,
    _plan_step_10,
    _plan_step_11,
    _plan_step_12,
    _plan_step_13,
    _plan_step_14,
    _plan_step_15,
    _plan_step_16,
    _plan_step_17,
    _plan_step_18,
    _plan_step_19,
    _plan_step_20,
    _plan_step_21,
    _plan_step_22,
    _plan_step_23,
    _plan_step_24,
    default_task,
    plan_familly_dinner_with_tasks_saved_in_db,
    plan_step_0,
    plan_step_1,
    plan_step_2,
    plan_step_3,
    plan_step_4,
    plan_step_5,
    plan_step_6,
    plan_step_7,
    plan_step_8,
    plan_step_9,
    plan_step_10,
    plan_step_11,
    plan_step_12,
    plan_step_13,
    plan_step_14,
    task_awaiting_preparation,
    task_ready_no_predecessors_or_subtasks,
    task_with_mixed_predecessors,
    task_with_ongoing_subtasks,
    task_with_unmet_predecessors,
)
from .utils.ascii_tree import make_tree, print_tree, pytest_terminal_summary, test_trees


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "plan_step, current_task_id, current_task_status, expected_next_task_id",
    [
        # Current Task Provided
        (
            plan_familly_dinner_with_tasks_saved_in_db,
            "101",
            TaskStatusList.READY,
            "101",
        ),
        (plan_familly_dinner_with_tasks_saved_in_db, "101", TaskStatusList.DONE, "102"),
        (_plan_step_10, "300.1.1", TaskStatusList.DONE, "300.1.2"),
        (_plan_step_11, "300.1.2", TaskStatusList.DONE, "300.2.1"),
        (_plan_step_12, "300.2.1", TaskStatusList.READY, "300.2.1"),
        (_plan_step_12, "300.2.1", TaskStatusList.DONE, "300.2.2"),
        (_plan_step_13, "300.2.2", TaskStatusList.DONE, "300.3"),
        (_plan_step_15, "300.3.1", TaskStatusList.READY, "300.3.1"),
        (_plan_step_15, "300.3.1", TaskStatusList.DONE, "300.3.2"),
        (_plan_step_16, "300.3.2", TaskStatusList.READY, "300.3.2"),
        (_plan_step_16, "300.3.2", TaskStatusList.DONE, "300.3.3"),
        (_plan_step_17, "300.3.3", TaskStatusList.READY, "300.3.3"),
        (_plan_step_17, "300.3.3", TaskStatusList.DONE, "300.4"),
        (_plan_step_18, "300.4", TaskStatusList.READY, "300.4"),
        (_plan_step_18, "300.4", TaskStatusList.DONE, "300.5"),
        (_plan_step_19, "300.5", TaskStatusList.READY, "300.5"),
        (_plan_step_19, "300.5", TaskStatusList.DONE, "300.6"),
        (_plan_step_20, "300.6", TaskStatusList.READY, "300.6"),
        (_plan_step_20, "300.6", TaskStatusList.DONE, "201"),
        # (_plan_step_21, '300.4' , TaskStatusList.READY, '300.4'),
        # (_plan_step_21, '300.4' , TaskStatusList.DONE, '300.5'),
        (_plan_step_21, "201", TaskStatusList.READY, "201"),
        (_plan_step_21, "201", TaskStatusList.DONE, "300"),
        (_plan_step_22, "300", TaskStatusList.READY, "300"),
        (_plan_step_22, "300", TaskStatusList.DONE, "100"),
        # (_plan_step_22, '100' , TaskStatusList.READY,  '100'),
        # (_plan_step_22, '100' , TaskStatusList.DONE, None),
        #     (plan_familly_dinner_with_tasks_saved_in_db, None, '101'),
    ],
)
async def test_get_next_task(
    plan_step, current_task_id, current_task_status, expected_next_task_id, request
):
    plan: Plan = await plan_step()

    for task_id in plan.get_ready_tasks_ids():
        ready_task = await plan.get_task(task_id=task_id)
        ready_task.state = TaskStatusList.BACKLOG

    plan._ready_task_ids = []

    if current_task_id is not None:
        current_task: Task = await plan.get_task(task_id=current_task_id)
        if current_task_status is not TaskStatusList.DONE:
            current_task.state = current_task_status
        else:
            await current_task.close_task()
    else:
        current_task = None

    next_task = await plan.get_next_task(task=current_task)

    try:
        if "100" == expected_next_task_id:
            assert next_task is None
        else:
            assert next_task.task_id == expected_next_task_id

    except AssertionError:
        raise AssertionError(
            f"current_task_id  = {current_task_id} \n"
            f"next_task.task_id = {next_task.task_id} \n "
            f"expected_next_task_id = {expected_next_task_id} \n "
            f"{await make_tree(plan)}"
        )
