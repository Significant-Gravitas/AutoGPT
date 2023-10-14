import asyncio
import logging
import os
import pathlib
from io import BytesIO
from uuid import uuid4

from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from forge.sdk.db import AgentDB
from forge.sdk.errors import NotFoundError
from forge.sdk.routes.agent_protocol import base_router
from forge.sdk.schema import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HypercornConfig

from autogpt.agent_factory.configurators import configure_agent_with_state
from autogpt.agent_factory.generators import generate_agent_for_task
from autogpt.agent_manager import AgentManager
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.file_workspace import FileWorkspace

logger = logging.getLogger(__name__)


class AgentProtocolServer:
    def __init__(
        self,
        app_config: Config,
        database: AgentDB,
        llm_provider: ChatModelProvider,
    ):
        self.app_config = app_config
        self.db = database
        self.llm_provider = llm_provider
        self.agent_manager = AgentManager(app_data_dir=app_config.app_data_dir)

    def start(self, port: int = 8000, router: APIRouter = base_router):
        """Start the agent server."""
        config = HypercornConfig()
        config.bind = [f"localhost:{port}"]
        app = FastAPI(
            title="AutoGPT Server",
            description="Forked from AutoGPT Forge; Modified version of The Agent Protocol.",
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
        frontend_path = (
            pathlib.Path(script_dir)
            .joinpath("../../../../frontend/build/web")
            .resolve()
        )

        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            logger.warning(
                f"Frontend not found. {frontend_path} does not exist. The frontend will not be available."
            )

        config.loglevel = "ERROR"
        config.bind = [f"0.0.0.0:{port}"]

        logger.info(f"Agent server starting on http://localhost:{port}")
        asyncio.run(hypercorn_serve(app, config))

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        task_agent = await generate_agent_for_task(
            task=task_request.input,
            app_config=self.app_config,
            llm_provider=self.llm_provider,
        )
        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input,
        )
        agent_id = task_agent.state.agent_id = task_agent_id(task.task_id)
        task_agent.attach_fs(self.app_config.app_data_dir / "agents" / agent_id)
        return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        tasks, pagination = await self.db.list_tasks(page, pageSize)
        response = TaskListResponse(tasks=tasks, pagination=pagination)
        return response

    async def get_task(self, task_id: int) -> Task:
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
        agent = configure_agent_with_state(
            state=self.agent_manager.retrieve_state(task_agent_id(task_id)),
            app_config=self.app_config,
            llm_provider=self.llm_provider,
        )
        agent.workspace.on_write_file = lambda path: await self.db.create_artifact(
            task_id=task_id,
            file_name=path.parts[-1],
            relative_path=str(path),
        )

        # Save step request
        step = await self.db.create_step(task_id=task_id, input=step_request)

        # According to the Agent Protocol spec, the first execute_step request contains
        #  the same task input as the parent create_task request.
        # To prevent this from interfering with the agent's process, we ignore the input
        #  of this first step request, and just generate the first step proposal.
        is_init_step = not bool(agent.event_history)
        execute_result = None
        if is_init_step:
            step_request.input = ""
        elif (
            agent.event_history.current_episode
            and not agent.event_history.current_episode.result
        ):
            if not step_request.input:
                step = await self.db.update_step(
                    task_id=task_id,
                    step_id=step.step_id,
                    status="running",
                )
                # Execute previously proposed action
                execute_result = await agent.execute(
                    command_name=agent.event_history.current_episode.action.name,
                    command_args=agent.event_history.current_episode.action.args,
                )
            else:
                execute_result = await agent.execute(
                    command_name="human_feedback",  # HACK
                    command_args={},
                    user_input=step_request.input,
                )

        # Propose next action
        thought_process_output = await agent.propose_action(
            additional_input=step_request.input or ""
        )
        step = await self.db.update_step(
            task_id=task_id,
            step_id=step.step_id,
            status="completed",
            additional_input=(
                {
                    "action_result": execute_result,
                }
                if not is_init_step
                else {}
            ).update(
                thought_process_output[2]
            ),  # HACK
        )
        return step

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
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        data = None
        file_name = file.filename or str(uuid4())
        data = b""
        while contents := file.file.read(1024 * 1024):
            data += contents
        # Check if relative path ends with filename
        if relative_path.endswith(file_name):
            file_path = relative_path
        else:
            file_path = os.path.join(relative_path, file_name)

        workspace = get_task_agent_file_workspace(task_id, self.agent_manager)
        workspace.write_file(file_path, data)

        artifact = await self.db.create_artifact(
            task_id=task_id,
            file_name=file_name,
            relative_path=relative_path,
            agent_created=False,
        )
        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> Artifact:
        """
        Get an artifact by ID.
        """
        try:
            artifact = await self.db.get_artifact(artifact_id)
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            workspace = get_task_agent_file_workspace(task_id, self.agent_manager)
            retrieved_artifact = workspace.read_file(file_path)
        except NotFoundError as e:
            raise
        except FileNotFoundError as e:
            raise

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )


def task_agent_id(task_id: str | int) -> str:
    return f"AutoGPT-{task_id}"


def get_task_agent_file_workspace(
    task_id: str | int,
    agent_manager: AgentManager,
) -> FileWorkspace:
    return FileWorkspace(
        root=agent_manager.get_agent_dir(
            agent_id=task_agent_id(task_id),
            must_exist=True,
        ),
        restrict_to_root=True,
    )
