import logging
import os
import pathlib
from io import BytesIO
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from forge.agent_protocol.api_router import base_router
from forge.agent_protocol.database.db import AgentDB
from forge.agent_protocol.middlewares import AgentMiddleware
from forge.agent_protocol.models.task import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from forge.file_storage.base import FileStorage

logger = logging.getLogger(__name__)


class ProtocolAgent:
    def __init__(self, database: AgentDB, workspace: FileStorage):
        self.db = database
        self.workspace = workspace

    def get_agent_app(self, router: APIRouter = base_router):
        """
        Start the agent server.
        """

        app = FastAPI(
            title="AutoGPT Forge",
            description="Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # Add CORS middleware
        origins = [
            "http://localhost:5000",
            "http://127.0.0.1:5000",
            "http://localhost:8000",
            "http://127.0.0.1:8000",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
            # Add any other origins you want to whitelist
        ]

        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        app.include_router(router, prefix="/ap/v1")
        script_dir = os.path.dirname(os.path.realpath(__file__))
        frontend_path = pathlib.Path(
            os.path.join(script_dir, "../../../classic/frontend/build/web")
        ).resolve()

        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            logger.warning(
                f"Frontend not found. {frontend_path} does not exist. "
                "The frontend will not be served."
            )
        app.add_middleware(AgentMiddleware, agent=self)

        return app

    def start(self, port):
        uvicorn.run(
            "forge.app:app", host="localhost", port=port, log_level="error", reload=True
        )

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input,
        )
        return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        tasks, pagination = await self.db.list_tasks(page, pageSize)
        response = TaskListResponse(tasks=tasks, pagination=pagination)
        return response

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        task = await self.db.get_task(task_id)
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        steps, pagination = await self.db.list_steps(task_id, page, pageSize)
        response = TaskStepsListResponse(steps=steps, pagination=pagination)
        return response

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        Create a step for the task.
        """
        raise NotImplementedError

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        step = await self.db.get_step(task_id, step_id)
        return step

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        artifacts, pagination = await self.db.list_artifacts(task_id, page, pageSize)
        return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str = ""
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        file_name = file.filename or str(uuid4())
        data = b""
        while contents := file.file.read(1024 * 1024):
            data += contents
        # Check if relative path ends with filename
        if relative_path.endswith(file_name):
            file_path = relative_path
        else:
            file_path = os.path.join(relative_path, file_name)

        await self.workspace.write_file(file_path, data)

        artifact = await self.db.create_artifact(
            task_id=task_id,
            file_name=file_name,
            relative_path=relative_path,
            agent_created=False,
        )
        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> StreamingResponse:
        """
        Get an artifact by ID.
        """
        artifact = await self.db.get_artifact(artifact_id)
        if artifact.file_name not in artifact.relative_path:
            file_path = os.path.join(artifact.relative_path, artifact.file_name)
        else:
            file_path = artifact.relative_path
        retrieved_artifact = self.workspace.read_file(file_path, binary=True)

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )
