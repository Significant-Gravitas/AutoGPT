import logging
import os
import pathlib
from io import BytesIO
from uuid import uuid4

import orjson
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from forge.sdk.db import AgentDB
from forge.sdk.errors import NotFoundError
from forge.sdk.middlewares import AgentMiddleware
from forge.sdk.model import (
    Artifact,
    Step,
    StepRequestBody,
    Task,
    TaskArtifactsListResponse,
    TaskListResponse,
    TaskRequestBody,
    TaskStepsListResponse,
)
from forge.sdk.routes.agent_protocol import base_router
from hypercorn.asyncio import serve as hypercorn_serve
from hypercorn.config import Config as HypercornConfig
from sentry_sdk import set_user

from autogpt.agent_factory.configurators import configure_agent_with_state
from autogpt.agent_factory.generators import generate_agent_for_task
from autogpt.agent_manager import AgentManager
from autogpt.agents.utils.exceptions import AgentFinished
from autogpt.app.utils import is_port_free
from autogpt.commands.system import finish
from autogpt.commands.user_interaction import ask_user
from autogpt.config import Config
from autogpt.core.resource.model_providers import ChatModelProvider
from autogpt.core.resource.model_providers.openai import OpenAIProvider
from autogpt.core.resource.model_providers.schema import ModelProviderBudget
from autogpt.file_storage import FileStorage
from autogpt.logs.utils import fmt_kwargs
from autogpt.models.action_history import ActionErrorResult, ActionSuccessResult

logger = logging.getLogger(__name__)


class AgentProtocolServer:
    _task_budgets: dict[str, ModelProviderBudget]

    def __init__(
        self,
        app_config: Config,
        database: AgentDB,
        file_storage: FileStorage,
        llm_provider: ChatModelProvider,
    ):
        self.app_config = app_config
        self.db = database
        self.file_storage = file_storage
        self.llm_provider = llm_provider
        self.agent_manager = AgentManager(file_storage)
        self._task_budgets = {}

    async def start(self, port: int = 8000, router: APIRouter = base_router):
        """Start the agent server."""
        logger.debug("Starting the agent server...")
        if not is_port_free(port):
            logger.error(f"Port {port} is already in use.")
            logger.info(
                "You can specify a port by either setting the AP_SERVER_PORT "
                "environment variable or defining AP_SERVER_PORT in the .env file."
            )
            return

        config = HypercornConfig()
        config.bind = [f"localhost:{port}"]
        app = FastAPI(
            title="AutoGPT Server",
            description="Forked from AutoGPT Forge; "
            "Modified version of The Agent Protocol.",
            version="v0.4",
        )

        # Configure CORS middleware
        default_origins = [f"http://localhost:{port}"]  # Default only local access
        configured_origins = [
            origin
            for origin in os.getenv("AP_SERVER_CORS_ALLOWED_ORIGINS", "").split(",")
            if origin  # Empty list if not configured
        ]
        origins = configured_origins or default_origins

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
                f"Frontend not found. {frontend_path} does not exist. "
                "The frontend will not be available."
            )

        # Used to access the methods on this class from API route handlers
        app.add_middleware(AgentMiddleware, agent=self)

        config.loglevel = "ERROR"
        config.bind = [f"0.0.0.0:{port}"]

        logger.info(f"AutoGPT server starting on http://localhost:{port}")
        await hypercorn_serve(app, config)

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        if user_id := (task_request.additional_input or {}).get("user_id"):
            set_user({"id": user_id})

        task = await self.db.create_task(
            input=task_request.input,
            additional_input=task_request.additional_input,
        )
        logger.debug(f"Creating agent for task: '{task.input}'")
        task_agent = await generate_agent_for_task(
            agent_id=task_agent_id(task.task_id),
            task=task.input,
            app_config=self.app_config,
            file_storage=self.file_storage,
            llm_provider=self._get_task_llm_provider(task),
        )
        await task_agent.save_state()

        return task

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        logger.debug("Listing all tasks...")
        tasks, pagination = await self.db.list_tasks(page, pageSize)
        response = TaskListResponse(tasks=tasks, pagination=pagination)
        return response

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        logger.debug(f"Getting task with ID: {task_id}...")
        task = await self.db.get_task(task_id)
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        logger.debug(f"Listing all steps created by task with ID: {task_id}...")
        steps, pagination = await self.db.list_steps(task_id, page, pageSize)
        response = TaskStepsListResponse(steps=steps, pagination=pagination)
        return response

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """Create a step for the task."""
        logger.debug(f"Creating a step for task with ID: {task_id}...")

        # Restore Agent instance
        task = await self.get_task(task_id)
        agent = configure_agent_with_state(
            state=self.agent_manager.load_agent_state(task_agent_id(task_id)),
            app_config=self.app_config,
            file_storage=self.file_storage,
            llm_provider=self._get_task_llm_provider(task),
        )

        if user_id := (task.additional_input or {}).get("user_id"):
            set_user({"id": user_id})

        # According to the Agent Protocol spec, the first execute_step request contains
        #  the same task input as the parent create_task request.
        # To prevent this from interfering with the agent's process, we ignore the input
        #  of this first step request, and just generate the first step proposal.
        is_init_step = not bool(agent.event_history)
        execute_command, execute_command_args, execute_result = None, None, None
        execute_approved = False

        # HACK: only for compatibility with AGBenchmark
        if step_request.input == "y":
            step_request.input = ""

        user_input = step_request.input if not is_init_step else ""

        if (
            not is_init_step
            and agent.event_history.current_episode
            and not agent.event_history.current_episode.result
        ):
            execute_command = agent.event_history.current_episode.action.name
            execute_command_args = agent.event_history.current_episode.action.args
            execute_approved = not user_input

            logger.debug(
                f"Agent proposed command"
                f" {execute_command}({fmt_kwargs(execute_command_args)})."
                f" User input/feedback: {repr(user_input)}"
            )

        # Save step request
        step = await self.db.create_step(
            task_id=task_id,
            input=step_request,
            is_last=execute_command == finish.__name__ and execute_approved,
        )
        agent.llm_provider = self._get_task_llm_provider(task, step.step_id)

        # Execute previously proposed action
        if execute_command:
            assert execute_command_args is not None
            agent.workspace.on_write_file = lambda path: self._on_agent_write_file(
                task=task, step=step, relative_path=path
            )

            if execute_command == ask_user.__name__:  # HACK
                execute_result = ActionSuccessResult(outputs=user_input)
                agent.event_history.register_result(execute_result)
            elif not execute_command:
                execute_result = None
            elif execute_approved:
                step = await self.db.update_step(
                    task_id=task_id,
                    step_id=step.step_id,
                    status="running",
                )

                try:
                    # Execute previously proposed action
                    execute_result = await agent.execute(
                        command_name=execute_command,
                        command_args=execute_command_args,
                    )
                except AgentFinished:
                    additional_output = {}
                    task_total_cost = agent.llm_provider.get_incurred_cost()
                    if task_total_cost > 0:
                        additional_output["task_total_cost"] = task_total_cost
                        logger.info(
                            f"Total LLM cost for task {task_id}: "
                            f"${round(task_total_cost, 2)}"
                        )

                    step = await self.db.update_step(
                        task_id=task_id,
                        step_id=step.step_id,
                        output=execute_command_args["reason"],
                        additional_output=additional_output,
                    )
                    await agent.save_state()
                    return step
            else:
                assert user_input
                execute_result = await agent.execute(
                    command_name="human_feedback",  # HACK
                    command_args={},
                    user_input=user_input,
                )

        # Propose next action
        try:
            next_command, next_command_args, raw_output = await agent.propose_action()
            logger.debug(f"AI output: {raw_output}")
        except Exception as e:
            step = await self.db.update_step(
                task_id=task_id,
                step_id=step.step_id,
                status="completed",
                output=f"An error occurred while proposing the next action: {e}",
            )
            return step

        # Format step output
        output = (
            (
                f"`{execute_command}({fmt_kwargs(execute_command_args)})` returned:"
                + ("\n\n" if "\n" in str(execute_result) else " ")
                + f"{execute_result}\n\n"
            )
            if execute_command_args and execute_command != ask_user.__name__
            else ""
        )
        output += f"{raw_output['thoughts']['speak']}\n\n"
        output += (
            f"Next Command: {next_command}({fmt_kwargs(next_command_args)})"
            if next_command != ask_user.__name__
            else next_command_args["question"]
        )

        additional_output = {
            **(
                {
                    "last_action": {
                        "name": execute_command,
                        "args": execute_command_args,
                        "result": (
                            ""
                            if execute_result is None
                            else (
                                orjson.loads(execute_result.json())
                                if not isinstance(execute_result, ActionErrorResult)
                                else {
                                    "error": str(execute_result.error),
                                    "reason": execute_result.reason,
                                }
                            )
                        ),
                    },
                }
                if not is_init_step
                else {}
            ),
            **raw_output,
        }

        task_cumulative_cost = agent.llm_provider.get_incurred_cost()
        if task_cumulative_cost > 0:
            additional_output["task_cumulative_cost"] = task_cumulative_cost
        logger.debug(
            f"Running total LLM cost for task {task_id}: "
            f"${round(task_cumulative_cost, 3)}"
        )

        step = await self.db.update_step(
            task_id=task_id,
            step_id=step.step_id,
            status="completed",
            output=output,
            additional_output=additional_output,
        )

        await agent.save_state()
        return step

    async def _on_agent_write_file(
        self, task: Task, step: Step, relative_path: pathlib.Path
    ) -> None:
        """
        Creates an Artifact for the written file, or updates the Artifact if it exists.
        """
        if relative_path.is_absolute():
            raise ValueError(f"File path '{relative_path}' is not relative")
        for a in task.artifacts or []:
            if a.relative_path == str(relative_path):
                logger.debug(f"Updating Artifact after writing to existing file: {a}")
                if not a.agent_created:
                    await self.db.update_artifact(a.artifact_id, agent_created=True)
                break
        else:
            logger.debug(f"Creating Artifact for new file '{relative_path}'")
            await self.db.create_artifact(
                task_id=step.task_id,
                step_id=step.step_id,
                file_name=relative_path.parts[-1],
                agent_created=True,
                relative_path=str(relative_path),
            )

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
        file_name = file.filename or str(uuid4())
        data = b""
        while contents := file.file.read(1024 * 1024):
            data += contents
        # Check if relative path ends with filename
        if relative_path.endswith(file_name):
            file_path = relative_path
        else:
            file_path = os.path.join(relative_path, file_name)

        workspace = self._get_task_agent_file_workspace(task_id)
        await workspace.write_file(file_path, data)

        artifact = await self.db.create_artifact(
            task_id=task_id,
            file_name=file_name,
            relative_path=relative_path,
            agent_created=False,
        )
        return artifact

    async def get_artifact(self, task_id: str, artifact_id: str) -> StreamingResponse:
        """
        Download a task artifact by ID.
        """
        try:
            workspace = self._get_task_agent_file_workspace(task_id)
            artifact = await self.db.get_artifact(artifact_id)
            if artifact.file_name not in artifact.relative_path:
                file_path = os.path.join(artifact.relative_path, artifact.file_name)
            else:
                file_path = artifact.relative_path
            retrieved_artifact = workspace.read_file(file_path, binary=True)
        except NotFoundError:
            raise
        except FileNotFoundError:
            raise

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{artifact.file_name}"'
            },
        )

    def _get_task_agent_file_workspace(self, task_id: str | int) -> FileStorage:
        agent_id = task_agent_id(task_id)
        return self.file_storage.clone_with_subroot(f"agents/{agent_id}/workspace")

    def _get_task_llm_provider(
        self, task: Task, step_id: str = ""
    ) -> ChatModelProvider:
        """
        Configures the LLM provider with headers to link outgoing requests to the task.
        """
        task_llm_budget = self._task_budgets.get(
            task.task_id, self.llm_provider.default_settings.budget.copy(deep=True)
        )

        task_llm_provider_config = self.llm_provider._configuration.copy(deep=True)
        _extra_request_headers = task_llm_provider_config.extra_request_headers
        _extra_request_headers["AP-TaskID"] = task.task_id
        if step_id:
            _extra_request_headers["AP-StepID"] = step_id
        if task.additional_input and (user_id := task.additional_input.get("user_id")):
            _extra_request_headers["AutoGPT-UserID"] = user_id

        task_llm_provider = None
        if isinstance(self.llm_provider, OpenAIProvider):
            settings = self.llm_provider._settings.copy()
            settings.budget = task_llm_budget
            settings.configuration = task_llm_provider_config  # type: ignore
            task_llm_provider = OpenAIProvider(
                settings=settings,
                logger=logger.getChild(f"Task-{task.task_id}_OpenAIProvider"),
            )

        if task_llm_provider and task_llm_provider._budget:
            self._task_budgets[task.task_id] = task_llm_provider._budget

        return task_llm_provider or self.llm_provider


def task_agent_id(task_id: str | int) -> str:
    return f"AutoGPT-{task_id}"
