from pathlib import Path

import pytest
from fastapi import UploadFile

from forge.file_storage.base import FileStorageConfiguration
from forge.file_storage.local import LocalFileStorage

from .agent import ProtocolAgent
from .database.db import AgentDB
from .models.task import StepRequestBody, Task, TaskListResponse, TaskRequestBody


@pytest.fixture
def agent(tmp_project_root: Path):
    db = AgentDB("sqlite:///test.db")
    config = FileStorageConfiguration(root=tmp_project_root)
    workspace = LocalFileStorage(config)
    return ProtocolAgent(db, workspace)


@pytest.fixture
def file_upload():
    this_file = Path(__file__)
    file_handle = this_file.open("rb")
    yield UploadFile(file_handle, filename=this_file.name)
    file_handle.close()


@pytest.mark.asyncio
async def test_create_task(agent: ProtocolAgent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task: Task = await agent.create_task(task_request)
    assert task.input == "test_input"


@pytest.mark.asyncio
async def test_list_tasks(agent: ProtocolAgent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    await agent.create_task(task_request)
    tasks = await agent.list_tasks()
    assert isinstance(tasks, TaskListResponse)


@pytest.mark.asyncio
async def test_get_task(agent: ProtocolAgent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    retrieved_task = await agent.get_task(task.task_id)
    assert retrieved_task.task_id == task.task_id


@pytest.mark.xfail(reason="execute_step is not implemented")
@pytest.mark.asyncio
async def test_execute_step(agent: ProtocolAgent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    step = await agent.execute_step(task.task_id, step_request)
    assert step.input == "step_input"
    assert step.additional_input == {"input": "additional_test_input"}


@pytest.mark.xfail(reason="execute_step is not implemented")
@pytest.mark.asyncio
async def test_get_step(agent: ProtocolAgent):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    step_request = StepRequestBody(
        input="step_input", additional_input={"input": "additional_test_input"}
    )
    step = await agent.execute_step(task.task_id, step_request)
    retrieved_step = await agent.get_step(task.task_id, step.step_id)
    assert retrieved_step.step_id == step.step_id


@pytest.mark.asyncio
async def test_list_artifacts(agent: ProtocolAgent):
    tasks = await agent.list_tasks()
    assert tasks.tasks, "No tasks in test.db"

    artifacts = await agent.list_artifacts(tasks.tasks[0].task_id)
    assert isinstance(artifacts.artifacts, list)


@pytest.mark.asyncio
async def test_create_artifact(agent: ProtocolAgent, file_upload: UploadFile):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)
    artifact = await agent.create_artifact(
        task_id=task.task_id,
        file=file_upload,
        relative_path=f"a_dir/{file_upload.filename}",
    )
    assert artifact.file_name == file_upload.filename
    assert artifact.relative_path == f"a_dir/{file_upload.filename}"


@pytest.mark.asyncio
async def test_create_and_get_artifact(agent: ProtocolAgent, file_upload: UploadFile):
    task_request = TaskRequestBody(
        input="test_input", additional_input={"input": "additional_test_input"}
    )
    task = await agent.create_task(task_request)

    artifact = await agent.create_artifact(
        task_id=task.task_id,
        file=file_upload,
        relative_path=f"b_dir/{file_upload.filename}",
    )
    await file_upload.seek(0)
    file_upload_content = await file_upload.read()

    retrieved_artifact = await agent.get_artifact(task.task_id, artifact.artifact_id)
    retrieved_artifact_content = bytearray()
    async for b in retrieved_artifact.body_iterator:
        retrieved_artifact_content.extend(b)  # type: ignore
    assert retrieved_artifact_content == file_upload_content
