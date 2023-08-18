import asyncio
import os
import typing

from fastapi import APIRouter, FastAPI, Response, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from hypercorn.asyncio import serve
from hypercorn.config import Config

from .db import AgentDB
from .middlewares import AgentMiddleware
from .routes.agent_protocol import base_router
from .schema import Artifact, Status, Step, StepRequestBody, Task, TaskRequestBody
from .utils import run
from .workspace import Workspace


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
        app.include_router(router)
        app.add_middleware(AgentMiddleware, agent=self)
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
            print(task)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        print(task)
        return task

    async def list_tasks(self) -> typing.List[str]:
        """
        List the IDs of all tasks that the agent has created.
        """
        try:
            task_ids = [task.task_id for task in await self.db.list_tasks()]
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return task_ids

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

    async def list_steps(self, task_id: str) -> typing.List[str]:
        """
        List the IDs of all steps that the task has created.
        """
        if not task_id:
            return Response(status_code=400, content="Task ID is required.")
        if not isinstance(task_id, str):
            return Response(status_code=400, content="Task ID must be a string.")
        try:
            steps_ids = [step.step_id for step in await self.db.list_steps(task_id)]
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return steps_ids

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
            steps = await self.db.list_steps(task_id)
            artifacts = await self.db.list_artifacts(task_id)
            step = steps[-1]
            step.artifacts = artifacts
            step.output = "No more steps to run."
            step.is_last = True
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

    async def list_artifacts(self, task_id: str) -> typing.List[Artifact]:
        """
        List the artifacts that the task has created.
        """
        if not task_id:
            return Response(status_code=400, content="Task ID is required.")
        if not isinstance(task_id, str):
            return Response(status_code=400, content="Task ID must be a string.")
        try:
            artifacts = await self.db.list_artifacts(task_id)
        except Exception as e:
            return Response(status_code=500, content=str(e))
        return artifacts

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
        artifact = await self.db.get_artifact(task_id, artifact_id)
        if not artifact.uri.startswith("file://"):
            return Response(
                status_code=500, content="Loading from none file uri not implemented"
            )
        file_path = artifact.uri.split("file://")[1]
        if not self.workspace.exists(file_path):
            return Response(status_code=500, content="File not found")
        retrieved_artifact = self.workspace.read(file_path)
        path = artifact.file_name
        with open(path, "wb") as f:
            f.write(retrieved_artifact)
        return FileResponse(
            # Note: mimetype is guessed in the FileResponse constructor
            path=path,
            filename=artifact.file_name,
        )
