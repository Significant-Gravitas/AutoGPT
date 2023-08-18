import asyncio
from typing import List
from uuid import uuid4

from fastapi import APIRouter, Request, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from hypercorn.asyncio import serve
from hypercorn.config import Config

from .db import Step, Task, TaskDB
from .middlewares import AgentMiddleware
from .models import Artifact, Status, StepRequestBody, TaskRequestBody
from .server import app

base_router = APIRouter()


@base_router.get("/heartbeat")
async def heartbeat() -> Response:
    """
    Heartbeat endpoint to check if the server is running.
    """
    return Response(status_code=200)


@base_router.post("/agent/tasks", response_model=Task, tags=["agent"])
async def create_agent_task(
    request: Request, body: TaskRequestBody | None = None
) -> Task:
    """
    Creates a task for the agent.
    """
    agent: Agent = request["agent"]

    task = await agent.db.create_task(
        input=body.input if body else None,
        additional_input=body.additional_input if body else None,
    )
    print(task)
    await agent.create_task(task)

    return task


@base_router.get("/agent/tasks", response_model=List[str], tags=["agent"])
async def list_agent_tasks_ids(request: Request) -> List[str]:
    """
    List all tasks that have been created for the agent.
    """
    agent: Agent = request["agent"]
    return [task.task_id for task in await agent.db.list_tasks()]


@base_router.get("/agent/tasks/{task_id}", response_model=Task, tags=["agent"])
async def get_agent_task(request: Request, task_id: str) -> Task:
    """
    Get details about a specified agent task.
    """
    agent: Agent = request["agent"]
    return await agent.db.get_task(task_id)


@base_router.get(
    "/agent/tasks/{task_id}/steps",
    response_model=List[str],
    tags=["agent"],
)
async def list_agent_task_steps(request: Request, task_id: str) -> List[str]:
    """
    List all steps for the specified task.
    """
    agent: Agent = request["agent"]
    task = await agent.db.get_task(task_id)
    return [s.step_id for s in task.steps]


@base_router.post(
    "/agent/tasks/{task_id}/steps",
    response_model=Step,
    tags=["agent"],
)
async def execute_agent_task_step(
    request: Request,
    task_id: str,
    body: StepRequestBody | None = None,
) -> Step:
    """
    Execute a step in the specified agent task.
    """
    agent: Agent = request["agent"]

    if body.input != "y":
        step = await agent.db.create_step(
            task_id=task_id,
            input=body.input if body else None,
            additional_properties=body.additional_input if body else None,
        )
        step = await agent.run_step(step)
        step.output = "Task completed"
        step.is_last = True
    else:
        steps = await agent.db.list_steps(task_id)
        artifacts = await agent.db.list_artifacts(task_id)
        step = steps[-1]
        step.artifacts = artifacts
        step.output = "No more steps to run."
        step.is_last = True
    if isinstance(step.status, Status):
        step.status = step.status.value
    return JSONResponse(content=step.dict(), status_code=200)


@base_router.get(
    "/agent/tasks/{task_id}/steps/{step_id}",
    response_model=Step,
    tags=["agent"],
)
async def get_agent_task_step(request: Request, task_id: str, step_id: str) -> Step:
    """
    Get details about a specified task step.
    """
    agent: Agent = request["agent"]
    return await agent.db.get_step(task_id, step_id)


@base_router.get(
    "/agent/tasks/{task_id}/artifacts",
    response_model=List[Artifact],
    tags=["agent"],
)
async def list_agent_task_artifacts(request: Request, task_id: str) -> List[Artifact]:
    """
    List all artifacts for the specified task.
    """
    agent: Agent = request["agent"]
    task = await agent.db.get_task(task_id)
    return task.artifacts


@base_router.post(
    "/agent/tasks/{task_id}/artifacts",
    response_model=Artifact,
    tags=["agent"],
)
async def upload_agent_task_artifacts(
    request: Request,
    task_id: str,
    file: UploadFile | None = None,
    uri: str | None = None,
) -> Artifact:
    """
    Upload an artifact for the specified task.
    """
    agent: Agent = request["agent"]
    if not file and not uri:
        return Response(status_code=400, content="No file or uri provided")
    data = None
    if not uri:
        file_name = file.filename or str(uuid4())
        try:
            data = b""
            while contents := file.file.read(1024 * 1024):
                data += contents
        except Exception as e:
            return Response(status_code=500, content=str(e))

    artifact = await agent.save_artifact(task_id, artifact, data)
    agent.db.add_artifact(task_id, artifact)

    return artifact


@base_router.get(
    "/agent/tasks/{task_id}/artifacts/{artifact_id}",
    tags=["agent"],
)
async def download_agent_task_artifacts(
    request: Request, task_id: str, artifact_id: str
) -> FileResponse:
    """
    Download the specified artifact.
    """
    agent: Agent = request["agent"]
    artifact = await agent.db.get_artifact(task_id, artifact_id)
    retrieved_artifact: Artifact = await agent.retrieve_artifact(task_id, artifact)
    path = artifact.file_name
    with open(path, "wb") as f:
        f.write(retrieved_artifact)
    return FileResponse(
        # Note: mimetype is guessed in the FileResponse constructor
        path=path,
        filename=artifact.file_name,
    )


class Agent:
    def __init__(self, db: TaskDB):
        self.name = self.__class__.__name__
        self.db = db

    def start(self, port: int = 8000, router: APIRouter = base_router):
        """
        Start the agent server.
        """
        config = Config()
        config.bind = [f"localhost:{port}"]  # As an example configuration setting
        app.title = f"{self.name} - Agent Communication Protocol"
        app.include_router(router)
        app.add_middleware(AgentMiddleware, agent=self)
        asyncio.run(serve(app, config))

    async def create_task(self, task: Task):
        """
        Handles a new task
        """
        return task

    async def run_step(self, step: Step) -> Step:
        return step

    async def retrieve_artifact(self, task_id: str, artifact: Artifact) -> bytes:
        """
        Retrieve the artifact data from wherever it is stored and return it as bytes.
        """
        raise NotImplementedError("")

    async def save_artifact(
        self, task_id: str, artifact: Artifact, data: bytes | None = None
    ) -> Artifact:
        """
        Save the artifact data to the agent's workspace, loading from uri if bytes are not available.
        """
        raise NotImplementedError()
