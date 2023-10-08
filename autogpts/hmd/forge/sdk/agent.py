import os
import pathlib
from io import BytesIO
from uuid import uuid4

import uvicorn
from fastapi import APIRouter, FastAPI, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .abilities.registry import AbilityRegister
from .db import AgentDB
from .errors import NotFoundError
from .forge_log import ForgeLogger
from .middlewares import AgentMiddleware
from .routes.agent_protocol import base_router
from .schema import *
from .workspace import Workspace

from .prompting import PromptEngine

LOG = ForgeLogger(__name__)


class Agent:
    def __init__(self, database: AgentDB, workspace: Workspace):
        self.db = database
        self.workspace = workspace
        self.abilities = AbilityRegister(self)

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
            os.path.join(script_dir, "../../../../frontend/build/web")
        ).resolve()

        if os.path.exists(frontend_path):
            app.mount("/app", StaticFiles(directory=frontend_path), name="app")

            @app.get("/", include_in_schema=False)
            async def root():
                return RedirectResponse(url="/app/index.html", status_code=307)

        else:
            LOG.warning(
                f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
            )
        app.add_middleware(AgentMiddleware, agent=self)

        return app

    def start(self, port):
        uvicorn.run(
            "forge.app:app", host="localhost", port=port, log_level="error", reload=True
        )

    # async def create_task(self, task_request: TaskRequestBody) -> Task:
    #     """
    #     The agent protocol, which is the core of the Forge, works by creating a task and then
    #     executing steps for that task. This method is called when the agent is asked to create
    #     a task.

    #     We are hooking into function to add a custom log message. Though you can do anything you
    #     want here.
    #     """
    #     task = await super().create_task(task_request)
    #     LOG.info(
    #         f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
    #     )
    #     return task

    async def create_task(self, task_request: TaskRequestBody) -> Task:
        """
        Create a task for the agent.
        """
        try:
            task = await self.db.create_task(
                input=task_request.input,
                additional_input=task_request.additional_input,
            )

            LOG.info(
                f"📦 Task created: {task.task_id} input: {task.input[:40]}{'...' if len(task.input) > 40 else ''}"
            )

            return task
        except Exception as e:
            raise

    async def list_tasks(self, page: int = 1, pageSize: int = 10) -> TaskListResponse:
        """
        List all tasks that the agent has created.
        """
        try:
            tasks, pagination = await self.db.list_tasks(page, pageSize)
            response = TaskListResponse(tasks=tasks, pagination=pagination)
            return response
        except Exception as e:
            raise

    async def get_task(self, task_id: str) -> Task:
        """
        Get a task by ID.
        """
        try:
            task = await self.db.get_task(task_id)
        except Exception as e:
            raise
        return task

    async def list_steps(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskStepsListResponse:
        """
        List the IDs of all steps that the task has created.
        """
        try:
            steps, pagination = await self.db.list_steps(task_id, page, pageSize)
            response = TaskStepsListResponse(steps=steps, pagination=pagination)
            return response
        except Exception as e:
            raise

    async def execute_step(self, task_id: str, step_request: StepRequestBody) -> Step:
        """
        The agent protocol, which is the core of the Forge, works by creating a task and then
        executing steps for that task. This method is called when the agent is asked to execute
        a step.

        The task that is created contains an input string, for the bechmarks this is the task
        the agent has been asked to solve and additional input, which is a dictionary and
        could contain anything.

        If you want to get the task use:

        ```
        task = await self.db.get_task(task_id)
        ```

        The step request body is essentailly the same as the task request and contains an input
        string, for the bechmarks this is the task the agent has been asked to solve and
        additional input, which is a dictionary and could contain anything.

        You need to implement logic that will take in this step input and output the completed step
        as a step object. You can do everything in a single step or you can break it down into
        multiple steps. Returning a request to continue in the step output, the user can then decide
        if they want the agent to continue or not.
        """

        task = await self.db.get_task(task_id)

        step = await self.db.create_step(
            task_id=task_id, input=step_request, is_last=True
        )

        # Log the message
        LOG.info(f"\t✅ Final Step completed: {step.step_id} input: {step.input[:19]}")

        # Initialize the PromptEngine with the "gpt-3.5-turbo" model
        prompt_engine = PromptEngine("gpt-3.5-turbo")

        # Load the system and task prompts
        system_prompt = prompt_engine.load_prompt("system-format")

        # Initialize the messages list with the system prompt
        messages = [
            {"role": "system", "content": system_prompt},
        ]

        # Define the task parameters
        task_kwargs = {
            "task": task.input,
            "abilities": self.abilities.list_abilities_for_prompt(),
        }

        # Load the task prompt with the defined task parameters
        task_prompt = prompt_engine.load_prompt("task-step", **task_kwargs)

        # Append the task prompt to the messages list
        messages.append({"role": "user", "content": task_prompt})

        try:
            # Define the parameters for the chat completion request
            chat_completion_kwargs = {
                "messages": messages,
                "model": "gpt-3.5-turbo",
            }
            # Make the chat completion request and parse the response
            chat_response = await chat_completion_request(**chat_completion_kwargs)
            answer = json.loads(chat_response["choices"][0]["message"]["content"])

            # Log the answer for debugging purposes
            LOG.info(pprint.pformat(answer))

        except json.JSONDecodeError as e:
            # Handle JSON decoding errors
            LOG.error(f"Unable to decode chat response: {chat_response}")
        except Exception as e:
            # Handle other exceptions
            LOG.error(f"Unable to generate chat response: {e}")  # Step 7: Executing the Derived Ability

        # self.abilities = AbilityRegister(self)

        # Extract the ability from the answer
        ability = answer["ability"]

        # Run the ability and get the output
        # We don't actually use the output in this example
        output = await self.abilities.run_ability(
            task_id, ability["name"], **ability["args"]
        )

        # Set the step output to the "speak" part of the answer
        step.output = answer["thoughts"]["speak"]

        return step

        raise NotImplementedError

    async def get_step(self, task_id: str, step_id: str) -> Step:
        """
        Get a step by ID.
        """
        try:
            step = await self.db.get_step(task_id, step_id)
            return step
        except Exception as e:
            raise

    async def list_artifacts(
        self, task_id: str, page: int = 1, pageSize: int = 10
    ) -> TaskArtifactsListResponse:
        """
        List the artifacts that the task has created.
        """
        try:
            artifacts, pagination = await self.db.list_artifacts(
                task_id, page, pageSize
            )
            return TaskArtifactsListResponse(artifacts=artifacts, pagination=pagination)

        except Exception as e:
            raise

    async def create_artifact(
        self, task_id: str, file: UploadFile, relative_path: str
    ) -> Artifact:
        """
        Create an artifact for the task.
        """
        data = None
        file_name = file.filename or str(uuid4())
        try:
            data = b""
            while contents := file.file.read(1024 * 1024):
                data += contents
            # Check if relative path ends with filename
            if relative_path.endswith(file_name):
                file_path = relative_path
            else:
                file_path = os.path.join(relative_path, file_name)

            self.workspace.write(task_id, file_path, data)

            artifact = await self.db.create_artifact(
                task_id=task_id,
                file_name=file_name,
                relative_path=relative_path,
                agent_created=False,
            )
        except Exception as e:
            raise
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
            retrieved_artifact = self.workspace.read(task_id=task_id, path=file_path)
        except NotFoundError as e:
            raise
        except FileNotFoundError as e:
            raise
        except Exception as e:
            raise

        return StreamingResponse(
            BytesIO(retrieved_artifact),
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f"attachment; filename={artifact.file_name}"
            },
        )
