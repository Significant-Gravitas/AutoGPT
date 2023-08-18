import os
import sqlite3

import pytest

from autogpt.db import AgentDB, DataNotFoundError


def test_table_creation():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)

    conn = sqlite3.connect("test_db.sqlite3")
    cursor = conn.cursor()

    # Test for tasks table existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='tasks'")
    assert cursor.fetchone() is not None

    # Test for steps table existence
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='steps'")
    assert cursor.fetchone() is not None

    # Test for artifacts table existence
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='artifacts'"
    )
    assert cursor.fetchone() is not None

    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_create_task():
    # Having issues with pytest fixture so added setup and teardown in each test as a rapid workaround
    # TODO: Fix this!
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)

    task = await agent_db.create_task("task_input")
    assert task.input == "task_input"
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_create_and_get_task():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)
    await agent_db.create_task("task_input")
    task = await agent_db.get_task(1)
    assert task.input == "task_input"
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_create_and_get_task_with_steps():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)
    await agent_db.create_task("task_input")
    task = await agent_db.get_task(1)

    assert task.input == "task_input"
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_get_task_not_found():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)
    with pytest.raises(DataNotFoundError):
        await agent_db.get_task(9999)
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_create_and_get_step():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)
    await agent_db.create_task("task_input")
    await agent_db.create_step(1, "step_name")
    step = await agent_db.get_step(1, 1)
    assert step.name == "step_name"
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_updating_step():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)
    await agent_db.create_task("task_input")
    await agent_db.create_step(1, "step_name")
    await agent_db.update_step(1, 1, "completed")

    step = await agent_db.get_step(1, 1)
    assert step.status.value == "completed"
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_get_step_not_found():
    db_name = "sqlite:///test_db.sqlite3"
    agent_db = AgentDB(db_name)
    with pytest.raises(DataNotFoundError):
        await agent_db.get_step(9999, 9999)
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_get_artifact():
    db_name = "sqlite:///test_db.sqlite3"
    db = AgentDB(db_name)

    # Given: A task and its corresponding artifact
    task = await db.create_task("test_input")
    step = await db.create_step(task.task_id, "step_name")

    # Create an artifact
    artifact = await db.create_artifact(
        task_id=task.task_id,
        file_name="sample_file.txt",
        uri="file:///path/to/sample_file.txt",
        agent_created=True,
        step_id=step.step_id,
    )

    # When: The artifact is fetched by its ID
    fetched_artifact = await db.get_artifact(int(task.task_id), artifact.artifact_id)

    # Then: The fetched artifact matches the original
    assert fetched_artifact.file_name == "sample_file.txt"
    assert fetched_artifact.uri == "file:///path/to/sample_file.txt"

    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_list_tasks():
    db_name = "sqlite:///test_db.sqlite3"
    db = AgentDB(db_name)

    # Given: Multiple tasks in the database
    task1 = await db.create_task("test_input_1")
    task2 = await db.create_task("test_input_2")

    # When: All tasks are fetched
    fetched_tasks = await db.list_tasks()

    # Then: The fetched tasks list includes the created tasks
    task_ids = [task.task_id for task in fetched_tasks]
    assert task1.task_id in task_ids
    assert task2.task_id in task_ids
    os.remove(db_name.split("///")[1])


@pytest.mark.asyncio
async def test_list_steps():
    db_name = "sqlite:///test_db.sqlite3"
    db = AgentDB(db_name)

    # Given: A task and multiple steps for that task
    task = await db.create_task("test_input")
    step1 = await db.create_step(task.task_id, "step_1")
    step2 = await db.create_step(task.task_id, "step_2")

    # When: All steps for the task are fetched
    fetched_steps = await db.list_steps(task.task_id)

    # Then: The fetched steps list includes the created steps
    step_ids = [step.step_id for step in fetched_steps]
    assert str(step1.step_id) in step_ids
    assert str(step2.step_id) in step_ids
    os.remove(db_name.split("///")[1])
