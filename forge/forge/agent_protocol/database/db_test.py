import os
import sqlite3
from datetime import datetime

import pytest

from forge.agent_protocol.database.db import (
    AgentDB,
    ArtifactModel,
    StepModel,
    TaskModel,
    convert_to_artifact,
    convert_to_step,
    convert_to_task,
)
from forge.agent_protocol.models import (
    Artifact,
    Step,
    StepRequestBody,
    StepStatus,
    Task,
)
from forge.utils.exceptions import NotFoundError as DataNotFoundError

TEST_DB_FILENAME = "test_db.sqlite3"
TEST_DB_URL = f"sqlite:///{TEST_DB_FILENAME}"


@pytest.fixture
def agent_db():
    db = AgentDB(TEST_DB_URL)
    yield db
    db.close()
    os.remove(TEST_DB_FILENAME)


@pytest.fixture
def raw_db_connection(agent_db: AgentDB):
    connection = sqlite3.connect(TEST_DB_FILENAME)
    yield connection
    connection.close()


def test_table_creation(raw_db_connection: sqlite3.Connection):
    cursor = raw_db_connection.cursor()

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


@pytest.mark.asyncio
async def test_task_schema():
    now = datetime.now()
    task = Task(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        input="Write the words you receive to the file 'output.txt'.",
        created_at=now,
        modified_at=now,
        artifacts=[
            Artifact(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                agent_created=True,
                file_name="main.py",
                relative_path="python/code/",
                created_at=now,
                modified_at=now,
            )
        ],
    )
    assert task.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert task.input == "Write the words you receive to the file 'output.txt'."
    assert len(task.artifacts) == 1
    assert task.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"


@pytest.mark.asyncio
async def test_step_schema():
    now = datetime.now()
    step = Step(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        step_id="6bb1801a-fd80-45e8-899a-4dd723cc602e",
        created_at=now,
        modified_at=now,
        name="Write to file",
        input="Write the words you receive to the file 'output.txt'.",
        status=StepStatus.created,
        output=(
            "I am going to use the write_to_file command and write Washington "
            "to a file called output.txt <write_to_file('output.txt', 'Washington')>"
        ),
        artifacts=[
            Artifact(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                file_name="main.py",
                relative_path="python/code/",
                created_at=now,
                modified_at=now,
                agent_created=True,
            )
        ],
        is_last=False,
    )
    assert step.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert step.step_id == "6bb1801a-fd80-45e8-899a-4dd723cc602e"
    assert step.name == "Write to file"
    assert step.status == StepStatus.created
    assert step.output == (
        "I am going to use the write_to_file command and write Washington "
        "to a file called output.txt <write_to_file('output.txt', 'Washington')>"
    )
    assert len(step.artifacts) == 1
    assert step.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
    assert step.is_last is False


@pytest.mark.asyncio
async def test_convert_to_task():
    now = datetime.now()
    task_model = TaskModel(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        created_at=now,
        modified_at=now,
        input="Write the words you receive to the file 'output.txt'.",
        additional_input={},
        artifacts=[
            ArtifactModel(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                created_at=now,
                modified_at=now,
                relative_path="file:///path/to/main.py",
                agent_created=True,
                file_name="main.py",
            )
        ],
    )
    task = convert_to_task(task_model)
    assert task.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert task.input == "Write the words you receive to the file 'output.txt'."
    assert len(task.artifacts) == 1
    assert task.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"


@pytest.mark.asyncio
async def test_convert_to_step():
    now = datetime.now()
    step_model = StepModel(
        task_id="50da533e-3904-4401-8a07-c49adf88b5eb",
        step_id="6bb1801a-fd80-45e8-899a-4dd723cc602e",
        created_at=now,
        modified_at=now,
        name="Write to file",
        status="created",
        input="Write the words you receive to the file 'output.txt'.",
        additional_input={},
        artifacts=[
            ArtifactModel(
                artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
                created_at=now,
                modified_at=now,
                relative_path="file:///path/to/main.py",
                agent_created=True,
                file_name="main.py",
            )
        ],
        is_last=False,
    )
    step = convert_to_step(step_model)
    assert step.task_id == "50da533e-3904-4401-8a07-c49adf88b5eb"
    assert step.step_id == "6bb1801a-fd80-45e8-899a-4dd723cc602e"
    assert step.name == "Write to file"
    assert step.status == StepStatus.created
    assert len(step.artifacts) == 1
    assert step.artifacts[0].artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
    assert step.is_last is False


@pytest.mark.asyncio
async def test_convert_to_artifact():
    now = datetime.now()
    artifact_model = ArtifactModel(
        artifact_id="b225e278-8b4c-4f99-a696-8facf19f0e56",
        created_at=now,
        modified_at=now,
        relative_path="file:///path/to/main.py",
        agent_created=True,
        file_name="main.py",
    )
    artifact = convert_to_artifact(artifact_model)
    assert artifact.artifact_id == "b225e278-8b4c-4f99-a696-8facf19f0e56"
    assert artifact.relative_path == "file:///path/to/main.py"
    assert artifact.agent_created is True


@pytest.mark.asyncio
async def test_create_task(agent_db: AgentDB):
    task = await agent_db.create_task("task_input")
    assert task.input == "task_input"


@pytest.mark.asyncio
async def test_create_and_get_task(agent_db: AgentDB):
    task = await agent_db.create_task("test_input")
    fetched_task = await agent_db.get_task(task.task_id)
    assert fetched_task.input == "test_input"


@pytest.mark.asyncio
async def test_get_task_not_found(agent_db: AgentDB):
    with pytest.raises(DataNotFoundError):
        await agent_db.get_task("9999")


@pytest.mark.asyncio
async def test_create_and_get_step(agent_db: AgentDB):
    task = await agent_db.create_task("task_input")
    step_input = {"type": "python/code"}
    request = StepRequestBody(input="test_input debug", additional_input=step_input)
    step = await agent_db.create_step(task.task_id, request)
    step = await agent_db.get_step(task.task_id, step.step_id)
    assert step.input == "test_input debug"


@pytest.mark.asyncio
async def test_updating_step(agent_db: AgentDB):
    created_task = await agent_db.create_task("task_input")
    step_input = {"type": "python/code"}
    request = StepRequestBody(input="test_input debug", additional_input=step_input)
    created_step = await agent_db.create_step(created_task.task_id, request)
    await agent_db.update_step(created_task.task_id, created_step.step_id, "completed")

    step = await agent_db.get_step(created_task.task_id, created_step.step_id)
    assert step.status.value == "completed"


@pytest.mark.asyncio
async def test_get_step_not_found(agent_db: AgentDB):
    with pytest.raises(DataNotFoundError):
        await agent_db.get_step("9999", "9999")


@pytest.mark.asyncio
async def test_get_artifact(agent_db: AgentDB):
    # Given: A task and its corresponding artifact
    task = await agent_db.create_task("test_input debug")
    step_input = {"type": "python/code"}
    requst = StepRequestBody(input="test_input debug", additional_input=step_input)

    step = await agent_db.create_step(task.task_id, requst)

    # Create an artifact
    artifact = await agent_db.create_artifact(
        task_id=task.task_id,
        file_name="test_get_artifact_sample_file.txt",
        relative_path="file:///path/to/test_get_artifact_sample_file.txt",
        agent_created=True,
        step_id=step.step_id,
    )

    # When: The artifact is fetched by its ID
    fetched_artifact = await agent_db.get_artifact(artifact.artifact_id)

    # Then: The fetched artifact matches the original
    assert fetched_artifact.artifact_id == artifact.artifact_id
    assert (
        fetched_artifact.relative_path
        == "file:///path/to/test_get_artifact_sample_file.txt"
    )


@pytest.mark.asyncio
async def test_list_tasks(agent_db: AgentDB):
    # Given: Multiple tasks in the database
    task1 = await agent_db.create_task("test_input_1")
    task2 = await agent_db.create_task("test_input_2")

    # When: All tasks are fetched
    fetched_tasks, pagination = await agent_db.list_tasks()

    # Then: The fetched tasks list includes the created tasks
    task_ids = [task.task_id for task in fetched_tasks]
    assert task1.task_id in task_ids
    assert task2.task_id in task_ids


@pytest.mark.asyncio
async def test_list_steps(agent_db: AgentDB):
    step_input = {"type": "python/code"}
    request = StepRequestBody(input="test_input debug", additional_input=step_input)

    # Given: A task and multiple steps for that task
    task = await agent_db.create_task("test_input")
    step1 = await agent_db.create_step(task.task_id, request)
    request = StepRequestBody(input="step two")
    step2 = await agent_db.create_step(task.task_id, request)

    # When: All steps for the task are fetched
    fetched_steps, pagination = await agent_db.list_steps(task.task_id)

    # Then: The fetched steps list includes the created steps
    step_ids = [step.step_id for step in fetched_steps]
    assert step1.step_id in step_ids
    assert step2.step_id in step_ids
