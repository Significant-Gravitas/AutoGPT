import asyncio
import os
from uuid import uuid4
import tempfile
import aiofiles
from fastapi import APIRouter, UploadFile, Form, File
from fastapi.responses import FileResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from typing import Callable, List, Optional, Annotated, Coroutine, Any

from .db import InMemorySeparatedDB, Task, AbstractDB, Step

# from agent_protocol.server import app
from .models.task_request_body import TaskRequestBody
from .models.step_request_body import StepRequestBody
from .models.artifact import Artifact
from .models.artifacts import Artifacts
from .models.pagination import Pagination
from .legacy_models import Status
from pydantic import BaseModel


class TaskListResponse(BaseModel):
    tasks: List[Task]
    pagination: Pagination


class TaskStepsListResponse(BaseModel):
    steps: List[Step]
    pagination: Pagination


StepHandler = Callable[[Step], Coroutine[Any, Any, Step]]
TaskHandler = Callable[[Task], Coroutine[Any, Any, None]]


_task_handler: Optional[TaskHandler]
_step_handler: Optional[StepHandler]


base_router = APIRouter()


@base_router.post("/ap/v1/agent/tasks", response_model=Task, tags=["agent"])
async def create_agent_task(body: TaskRequestBody | None = None) -> Task:
    """
    Creates a task for the agent.
    """
    if not _task_handler:
        raise Exception("Task handler not defined")

    task = await Agent.db.create_task(
        input=body.input if body else None,
        additional_input=body.additional_input if body else None,
    )
    await _task_handler(task)

    return task


@base_router.get("/ap/v1/agent/tasks", response_model=TaskListResponse, tags=["agent"])
async def list_agent_tasks_ids(
    page_size: int = 10, current_page: int = 1
) -> TaskListResponse:
    """
    List all tasks that have been created for the agent.
    """
    tasks = await Agent.db.list_tasks()
    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size
    return TaskListResponse(
        tasks=tasks[start_index:end_index],
        pagination=Pagination(
            total_items=len(tasks),
            total_pages=len(tasks) // page_size,
            current_page=current_page,
            page_size=page_size,
        ),
    )


@base_router.get("/ap/v1/agent/tasks/{task_id}", response_model=Task, tags=["agent"])
async def get_agent_task(task_id: str) -> Task:
    """
    Get details about a specified agent task.
    """
    return await Agent.db.get_task(task_id)


@base_router.get(
    "/ap/v1/agent/tasks/{task_id}/steps",
    response_model=TaskStepsListResponse,
    tags=["agent"],
)
async def list_agent_task_steps(
    task_id: str, page_size: int = 10, current_page: int = 1
) -> TaskStepsListResponse:
    """
    List all steps for the specified task.
    """
    # task = await Agent.db.get_task(task_id)
    steps = await Agent.db.list_steps(task_id)
    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size
    return TaskStepsListResponse(
        steps=steps[start_index:end_index],
        pagination=Pagination(
            total_items=len(steps),
            total_pages=len(steps) // page_size,
            current_page=current_page,
            page_size=page_size,
        ),
    )


@base_router.post(
    "/ap/v1/agent/tasks/{task_id}/steps",
    response_model=Step,
    tags=["agent"],
)
async def execute_agent_task_step(
    task_id: str,
    body: StepRequestBody | None = None,
) -> Step:
    """
    Execute a step in the specified agent task.
    """
    if not _step_handler:
        raise Exception("Step handler not defined")

    task = await Agent.db.get_task(task_id)
    step_list = await Agent.db.list_steps(task_id)
    if len(step_list) == 0:
        raise Exception("No steps exist")
    step = next(filter(lambda x: x.status == Status["created"], step_list), None)

    if not step:
        print("Last step already executed")
        return next(filter(lambda x: x.is_last, step_list), None)

    # step.status = Status["running"]

    step.input = body.input if body else None
    if body:
        if body.additional_input:
            step.additional_input = body.additional_input.update(step.additional_input)

    step = await _step_handler(step)

    step.status = Status["completed"]
    return step


@base_router.get(
    "/ap/v1/agent/tasks/{task_id}/steps/{step_id}",
    response_model=Step,
    tags=["agent"],
)
async def get_agent_task_step(task_id: str, step_id: str) -> Step:
    """
    Get details about a specified task step.
    """
    return await Agent.db.get_step(task_id, step_id)


@base_router.get(
    "/ap/v1/agent/tasks/{task_id}/artifacts",
    response_model=Artifacts,
    tags=["agent"],
)
async def list_agent_task_artifacts(
    task_id: str, page_size: int = 1000, current_page: int = 1
) -> Artifacts:
    """
    List all artifacts for the specified task.
    """
    start_index = (current_page - 1) * page_size
    end_index = start_index + page_size
    artifacts = (await Agent.db.list_artifacts(task_id))[start_index:end_index]
    artifacts = Artifacts(
        artifacts=artifacts,
        pagination=Pagination(
            total_items=len(artifacts),
            total_pages=len(artifacts),
            current_page=current_page,
            page_size=page_size,
        ),
    )
    return artifacts


@base_router.post(
    "/ap/v1/agent/tasks/{task_id}/artifacts",
    response_model=Artifact,
    tags=["agent"],
)
async def upload_agent_task_artifacts(
    task_id: str,
    file: Annotated[UploadFile, File()],
    relative_path: Annotated[Optional[str], Form()] = None,
) -> Artifact:
    """
    Upload an artifact for the specified task.
    """
    file_name = file.filename or str(uuid4())
    await Agent.db.get_task(task_id)
    artifact = await Agent.db.create_artifact(
        task_id=task_id,
        agent_created=False,
        file_name=file_name,
        relative_path=relative_path,
    )

    path = Agent.get_artifact_folder(task_id, artifact)
    if not os.path.exists(path):
        os.makedirs(path)

    async with aiofiles.open(os.path.join(path, file_name), "wb") as f:
        while content := await file.read(1024 * 1024):  # async read chunk ~1MiB
            await f.write(content)

    return artifact


@base_router.get(
    "/ap/v1/agent/tasks/{task_id}/artifacts/{artifact_id}",
    tags=["agent"],
)
async def download_agent_task_artifacts(task_id: str, artifact_id: str) -> FileResponse:
    """
    Download the specified artifact.
    """
    artifact = await Agent.db.get_artifact(task_id, artifact_id)
    path = Agent.get_artifact_path(task_id, artifact)
    with open(path, "r") as file:
        content = file.read()
        print(content)
    response = FileResponse(
        path=path, media_type="application/octet-stream", filename=artifact.file_name
    )
    return response


class Agent:
    db: AbstractDB = InMemorySeparatedDB()
    workspace: str = tempfile.gettempdir()

    @staticmethod
    def setup_agent(task_handler: TaskHandler, step_handler: StepHandler):
        """
        Set the agent's task and step handlers.
        """
        global _task_handler
        _task_handler = task_handler

        global _step_handler
        _step_handler = step_handler

        return Agent

    @staticmethod
    def get_workspace(task_id: str) -> str:
        """
        Get the workspace path for the specified task.
        """
        return os.path.join(Agent.workspace, task_id)

    @staticmethod
    def get_artifact_folder(task_id: str, artifact: Artifact) -> str:
        """
        Get the artifact path for the specified task and artifact.
        """
        workspace_path = Agent.get_workspace(task_id)
        relative_path = artifact.relative_path or ""
        return os.path.join(workspace_path, relative_path)

    @staticmethod
    def get_artifact_path(task_id: str, artifact: Artifact) -> str:
        """
        Get the artifact path for the specified task and artifact.
        """
        return os.path.join(
            Agent.get_artifact_folder(task_id, artifact), artifact.file_name
        )

    # @staticmethod
    # def start(port: int = 8000, router: APIRouter = base_router):
    #     """
    #     Start the agent server.
    #     """
    #     config = Config()
    #     config.bind = [f"localhost:{port}"]  # As an example configuration setting
    #     app.include_router(router)
    #     asyncio.run(serve(app, config))
