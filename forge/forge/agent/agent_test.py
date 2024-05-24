from pathlib import Path

import pytest

from forge.agent_protocol.database.db import AgentDB
from forge.agent_protocol.models.task import (
    StepRequestBody,
    Task,
    TaskListResponse,
    TaskRequestBody,
)
from forge.file_storage.base import FileStorageConfiguration
from forge.file_storage.local import LocalFileStorage

from .agent import Agent


@pytest.fixture
def agent():
    db = AgentDB("sqlite:///test.db")
    config = FileStorageConfiguration(root=Path("./test_workspace"))
    workspace = LocalFileStorage(config)
    return Agent(db, workspace)


@pytest.mark.skip
@pytest.mark.asyncio
async def test_create_task(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task: Task = await agent.create_task(task_request)
    assert task.input == "test_input"


@pytest.mark.skip
@pytest.mark.asyncio
async def test_list_tasks(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    tasks = await agent.list_tasks()
    assert isinstance(tasks, TaskListResponse)


@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_task(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    retrieved_task = await agent.get_task(task.task_id)
    assert retrieved_task.task_id == task.task_id


@pytest.mark.skip
@pytest.mark.asyncio
async def test_create_and_execute_step(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    step = await agent.create_and_execute_step(task.task_id, step_request)
    assert step.input == "step_input"
    assert step.additional_input == {"input": "additional_test_input"}


@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_step(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    step = await agent.create_and_execute_step(task.task_id, step_request)
    retrieved_step = await agent.get_step(task.task_id, step.step_id)
    assert retrieved_step.step_id == step.step_id


@pytest.mark.skip
@pytest.mark.asyncio
async def test_list_artifacts(agent):
    artifacts = await agent.list_artifacts()
    assert isinstance(artifacts, list)


@pytest.mark.skip
@pytest.mark.asyncio
async def test_create_artifact(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    artifact_request = ArtifactRequestBody(file=None, uri="test_uri")
    artifact = await agent.create_artifact(task.task_id, artifact_request)
    assert artifact.uri == "test_uri"


@pytest.mark.skip
@pytest.mark.asyncio
async def test_get_artifact(agent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    artifact_request = ArtifactRequestBody(file=None, uri="test_uri")
    artifact = await agent.create_artifact(task.task_id, artifact_request)
    retrieved_artifact = await agent.get_artifact(task.task_id, artifact.artifact_id)
    assert retrieved_artifact.artifact_id == artifact.artifact_id
