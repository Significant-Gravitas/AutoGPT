import asyncio
import os

from fastapi import APIRouter, FastAPI, HTTPException, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config
from prometheus_fastapi_instrumentator import Instrumentator

from .db import AgentDB
from .forge_log import CustomLogger
from .middlewares import AgentMiddleware
from .routes.agent_protocol import base_router
from .schema import *
from .tracing import setup_tracing
from .utils import run
from .workspace import Workspace, load_from_uri

LOG = CustomLogger(__name__)


class Agent:
    def __init__(self, database: AgentDB, workspace: Workspace):
        self.db = database
        self.workspace = workspace

    def start(self, port: int = 8000, router: APIRouter = base_router):
        """
        Start the agent server.
        """
        config = Config()
        config.bind = [f"localhost:{port}"]
        app = FastAPI(
            title="Auto-GPT Forge",
            description="Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # Add Prometheus metrics to the agent
        # https://github.com/trallnag/prometheus-fastapi-instrumentator
        instrumentator = Instrumentator().instrument(app)

        @app.on_event("startup")
        async def _startup():
            instrumentator.expose(app)

        app.include_router(router)
        app.add_middleware(AgentMiddleware, agent=self)
        setup_tracing(app)
        config.loglevel = "ERROR"
        config.bind = [f"0.0.0.0:{port}"]

        LOG.info(f"Agent server starting on {config.bind}")
        asyncio.run(serve(app, config))

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        try:
            task = await self.db.create_task(
                input=task_request.input if task_request.input else None,
                additional_input=task_request.additional_input
                if task_request.additional_input
                else None,
            )
            LOG.info(task.json())
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        try:
            tasks, pagination = await self.db.list_tasks(page, pageSize)
            response = TaskListResponse(tasks=tasks, pagination=pagination)
            return Response(content=response.json(), media_type="application/json")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        if not task_id:
            return Response(status_code=400, content="Task ID is required.")
        if not isinstance(task_id, str):
            return Response(status_code=400, content="Task ID must be a string.")
        try:
            task = await self.db.get_task(task_id)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        if not task_id:
            return Response(status_code=400, content="Task ID is required.")
        if not isinstance(task_id, str):
            return Response(status_code=400, content="Task ID must be a string.")
        try:
            steps, pagination = await self.db.list_steps(task_id, page, pageSize)
            response = TaskStepsListResponse(steps=steps, pagination=pagination)
            return Response(content=response.json(), media_type="application/json")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    async def create_and_execute_step(
        self, task_id: str, step_request: StepRequestBody
    ) -> Step:
        """
        Create a step for the task.
        """
        if step_request.input != "y":
            step = await self.db.create_step(
                task_id=task_id,
                input=step_request.input if step_request else None,
                additional_properties=step_request.additional_input
                if step_request
                else None,
            )
            # utils.run
            artifacts = run(step.input)
            for artifact in artifacts:
                art = await self.db.create_artifact(
                    task_id=step.task_id,
                    file_name=artifact["file_name"],
                    uri=artifact["uri"],
                    agent_created=True,
                    step_id=step.step_id,
                )
                assert isinstance(
                    art, Artifact
                ), f"Artifact not instance of Artifact {type(art)}"
                step.artifacts.append(art)
            step.status = "completed"
        else:
            steps, steps_pagination = await self.db.list_steps(
                task_id, page=1, per_page=100
            )
            artifacts, artifacts_pagination = await self.db.list_artifacts(
                task_id, page=1, per_page=100
            )
            step = steps[-1]
            step.artifacts = artifacts
            step.output = "No more steps to run."
            # The step is the last step on this page so checking if this is the
            # last page is sufficent to know if it is the last step
            step.is_last = steps_pagination.current_page == steps_pagination.total_pages
        if isinstance(step.status, Status):
            step.status = step.status.value
        step.output = "Done some work"
        return JSONResponse(content=step.dict(), status_code=200)

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        if not task_id or not step_id:
            return Response(
                status_code=400, content="Task ID and step ID are required."
            )
        if not isinstance(task_id, str) or not isinstance(step_id, str):
            return Response(
                status_code=400, content="Task ID and step ID must be strings."
            )
        try:
            step = await self.db.get_step(task_id, step_id)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return step

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        if not task_id:
            return Response(status_code=400, content="Task ID is required.")
        if not isinstance(task_id, str):
            return Response(status_code=400, content="Task ID must be a string.")
        try:
            artifacts, pagination = await self.db.list_artifacts(
                task_id, page, pageSize
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        response = TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)
        return Response(content=response.json(), media_type="application/json")

    async def create_artifact(
        self,
        task_id: str,
        file: UploadFile | None = None,
        uri: str | None = None,
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
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
        else:
            try:
                data = await load_from_uri(uri, task_id)
                file_name = uri.split("/")[-1]
            except Exception as e:
                return Response(status_code=500, content=str(e))

        file_path = os.path.join(task_id / file_name)
        self.write(file_path, data)
        self.db.save_artifact(task_id, artifact)

        artifact = await self.create_artifact(
            task_id=task_id,
            file_name=file_name,
            uri=f"file://{file_path}",
            agent_created=False,
        )

        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        try:
            artifact = await self.db.get_artifact(task_id, artifact_id)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        try:
            retrieved_artifact = await self.load_from_uri(artifact.uri, artifact_id)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        path = artifact.file_name
        try:
            with open(path, "wb") as f:
                f.write(retrieved_artifact)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return FileResponse(
            # Note: mimetype is guessed in the FileResponse constructor
            path=path,
            filename=artifact.file_name,
        )
