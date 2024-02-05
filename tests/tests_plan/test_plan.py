from __future__ import annotations

import copy
import json
import uuid
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError

from AFAAS.interfaces.task.meta import TaskStatusList
from AFAAS.lib.task.plan import Plan
from AFAAS.lib.task.task import Task

from tests.dataset.agent_planner import agent_dataset
from tests.dataset.plan_familly_dinner import (
    _plan_familly_dinner,
    _plan_step_3,
    _plan_step_10,
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
    plan_step_15,
    plan_step_16,
    plan_step_17,
    plan_step_18,
    plan_step_19,
    plan_step_20,
    plan_step_21,
    plan_step_22,
    plan_step_23,
    plan_step_24,
    plan_step_25,
    plan_step_26,
    plan_step_27,
    plan_step_28,
    plan_step_29,
    plan_step_30,
    plan_step_31,
    plan_step_32,
    plan_step_33,
    plan_with_no_task,
    task_awaiting_preparation,
    task_ready_no_predecessors_or_subtasks,
    task_with_mixed_predecessors,
    task_with_ongoing_subtasks,
    task_with_unmet_predecessors,
)
from .utils.ascii_tree import make_tree, print_tree, pytest_terminal_summary, test_trees


@pytest.mark.asyncio
async def test_plan_hashable(plan_step_2: Plan):
    # Check if the plan can be used as a dictionary key
    other_plan = await _plan_step_10()
    assert hash(plan_step_2) == hash(other_plan), "Hashes of same plans should match"
    other_plan = "new_id"
    assert hash(plan_step_2) != hash(
        other_plan
    ), "Hashes of different plans should not match"


def test_plan_length(plan_with_no_task: Plan, default_task: Task):
    plan = plan_with_no_task
    assert len(plan) == 0, "Plan length should be 0 when no tasks are added"

    # Add a task and check if length updates
    plan.add_task(default_task)
    assert len(plan) == 1, "Plan length should be 1 after adding a task"

    # Optionally, add more tasks and verify the length increases accordingly
