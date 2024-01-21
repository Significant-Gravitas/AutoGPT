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
    default_task,
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
    plan_with_no_task,
    task_awaiting_preparation,
    task_ready_no_predecessors_or_subtasks,
    task_with_mixed_predecessors,
    task_with_ongoing_subtasks,
    task_with_unmet_predecessors,
)
from .utils.ascii_tree import print_tree, pytest_terminal_summary, test_trees


@pytest.mark.asyncio
async def test_new_plan_instance_creation(agent):
    # Test case for creating a new Plan instance with a given agent_id
    plan = Plan(agent=agent, task_goal=agent.agent_goal_sentence)
    assert plan.agent.agent_id == "pytest_A639f7cda-c88c-44d7-b0b2-a4a4abbd4a6c"


@pytest.mark.asyncio
async def test_existing_plan_instance_retrieval(agent):
    # Test case for retrieving an existing Plan instance with an agent_id
    first_plan = Plan(agent=agent, task_goal=agent.agent_goal_sentence)
    second_plan = Plan(agent=agent, task_goal=agent.agent_goal_sentence + "12345")
    assert first_plan is second_plan


async def mock_plan_data(agent, tasks):
    """
    Generates mock plan data for testing.

    Args:
        agent_id (str): The agent ID associated with the plan.
        tasks (list): A list of mock tasks to include in the plan.

    Returns:
        dict: A dictionary representing the mock plan data.
    """
    return {
        "plan_id": "mock_plan_" + str(uuid.uuid4()),
        "agent": agent,  # Assuming BaseAgent can be instantiated like this
        "tasks": tasks,
        "task_goal": agent.agent_goal_sentence
        # Add other necessary plan attributes here
    }


async def mock_plan_with_task_states(agent):
    """
    Generates a mock plan with tasks in various states.

    Args:
        agent_id (str): The agent ID associated with the plan.

    Returns:
        dict: A dictionary representing the mock plan with tasks in various states.
    """
    tasks = [
        {"task_id": "task1", "state": TaskStatusList.READY},
        {"task_id": "task2", "state": TaskStatusList.IN_PROGRESS_WITH_SUBTASKS},
        {"task_id": "task3", "state": TaskStatusList.DONE},
        # Add more tasks as needed
    ]
    return await mock_plan_data(agent, tasks)


@pytest.mark.asyncio
async def test_load_plan_no_tasks(plan_with_no_task):
    assert len(plan_with_no_task.get_all_tasks_ids()) == 0


@pytest.mark.asyncio
async def test_load_plan_with_various_task_states(plan_with_no_task):
    # Mock plan data with tasks in different states
    # prepare : Plan = await Plan._load(plan_id = plan_with_no_task.plan_id, agent = plan_with_no_task.agent)

    # assert prepare is None

    await plan_with_no_task.db_create()
    plan_with_no_task.add_tasks(
        tasks=[
            Task(
                agent=plan_with_no_task.agent,
                plan_id=plan_with_no_task.plan_id,
                **{
                    "task_id": "task1",
                    "state": TaskStatusList.READY,
                    "task_goal": "Task Goal 1",
                },
            ),
            Task(
                agent=plan_with_no_task.agent,
                plan_id=plan_with_no_task.plan_id,
                **{
                    "task_id": "task2",
                    "state": TaskStatusList.IN_PROGRESS_WITH_SUBTASKS,
                    "task_goal": "Task Goal 2",
                },
            ),
            Task(
                agent=plan_with_no_task.agent,
                plan_id=plan_with_no_task.plan_id,
                **{
                    "task_id": "task3",
                    "state": TaskStatusList.DONE,
                    "task_goal": "Task Goal 3",
                },
            ),
        ]
    )
    len(plan_with_no_task.get_all_tasks_ids()) == 3
    await plan_with_no_task.db_save()
    len(plan_with_no_task.get_all_tasks_ids()) == 3

    plan_id = plan_with_no_task.plan_id
    agent = plan_with_no_task.agent

    Plan._instance = {}
    # plan_data = await mock_plan_with_task_states(agent = plan_with_no_task.agent)
    loaded_plan = await Plan.get_plan_from_db(plan_id=plan_id, agent=agent)
    assert len(loaded_plan.get_all_tasks_ids()) == 3


@pytest.mark.asyncio
@patch("AFAAS.lib.task.plan.Plan.db_create")
async def test_db_create_plan_with_db_error(mock_db_create, agent):
    # Mock a database error when db_create is called
    mock_db_create.side_effect = Exception("Database error")

    # Test creating a plan when there's a database error
    with pytest.raises(Exception) as exc_info:
        await Plan.db_create(agent)

    # Assert that the exception message is as expected
    assert str(exc_info.value) == "Database error"


@pytest.mark.asyncio
async def test_get_existing_plan_from_db(default_task: Task):
    # Test retrieving an existing plan from the database
    await default_task.agent.plan.db_create()
    Plan._instance = {}
    existing_plan = await Plan.get_plan_from_db(
        default_task.agent.plan.plan_id, default_task.agent
    )
    assert existing_plan is not None


# @pytest.mark.asyncio
# async def test_get_nonexistent_plan_from_db(agent):
#     # Test retrieving an existing plan from the database
#     nonexistent_plan =    await Plan.get_plan_from_db("nonexistent_plan_id", agent)
#     assert nonexistent_plan is None


@pytest.mark.asyncio
async def test_get_nonexistent_plan_from_db_with_exception(agent):
    # Test behavior when the plan does not exist in the database
    with pytest.raises(Exception):
        await Plan.get_plan_from_db("nonexistent_plan_id", agent)


def test_generate_uuid_format_and_uniqueness():
    uuid1 = Plan.generate_uuid()
    uuid2 = Plan.generate_uuid()
    assert isinstance(uuid1, str) and uuid1.startswith("PL")
    assert isinstance(uuid2, str) and uuid2.startswith("PL")
    assert uuid1 != uuid2


@pytest.mark.asyncio
async def test_retrieve_existing_task_in_plan(plan_step_0: Plan):
    # Assuming plan_fixture is a fixture with a predefined plan
    task = await plan_step_0.get_task("201")
    assert task is not None
    assert task.task_id == "201"


@pytest.mark.asyncio
async def test_retrieve_nonexistent_task_in_plan(plan_step_0: Plan):
    # Test behavior when the task does not exist in the plan
    with pytest.raises(Exception):
        await plan_step_0.get_task("nonexistent_task_id")


@pytest.mark.asyncio
async def test_retrieve_plan_itself(plan_step_0: Plan):
    # Test retrieving the plan itself using the plan_id
    retrieved_plan = await plan_step_0.get_task(plan_step_0.plan_id)
    assert retrieved_plan is plan_step_0
