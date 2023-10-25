import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
print(sys.path)
import asyncio
from pathlib import Path
import pathlib
from typing import Callable, Optional, Coroutine, Any

from gpt_engineer.core.db import DB
from gpt_engineer.cli.main import main
from gpt_engineer.API.agent import base_router, Agent
from gpt_engineer.API.db import NotFoundException, not_found_exception_handler

from models.step import Step
from models.task import Task

from openai.error import AuthenticationError

from fastapi import FastAPI, APIRouter
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from hypercorn.asyncio import serve
from hypercorn.config import Config


StepHandler = Callable[[Step], Coroutine[Any, Any, Step]]
TaskHandler = Callable[[Task], Coroutine[Any, Any, None]]

_task_handler: Optional[TaskHandler]
_step_handler: Optional[StepHandler]

app = FastAPI(
    title="AutoGPT Forge",
    description="Modified version of The Agent Protocol.",
    version="v0.4",
)

app.add_exception_handler(NotFoundException, not_found_exception_handler)


async def task_handler(task: Task) -> None:
    """
    Process the given task to set up the initial steps based on the task input.

    Parameters:
    - task (Task): An object containing task details, including the input prompt.

    Behavior:
    - Checks if the task has valid input.
    - Reads additional input properties and sets up a workspace directory.
    - Writes the prompt to a file.
    - Registers the workspace directory as an artifact.
    - Creates an initial step named "create_code".

    Exceptions:
    - Raises an exception if no input prompt is provided in the 'input' field of the task.

    Returns:
    - None
    """

    # Validate that we have a prompt or terminate.
    if task.input is None:
        raise Exception("No input prompt in the 'input' field.")

    # Extract additional properties from the task if available.
    additional_input = dict()
    if hasattr(task, "additional_input"):
        if hasattr(task.additional_input, "__root__"):
            additional_input = task.additional_input.__root__

    # Set up the root directory for the agent, defaulting to tempdir. Setting the workspace of the global Agent is POTENTIALLY PROBLEMATIC
    Agent.workspace = additional_input.get("root_dir", Agent.workspace)

    workspace = DB(Agent.get_workspace(task.task_id))

    # Write prompt to a file in the workspace.
    workspace["prompt"] = f"{task.input}\n"

    # only for the test=WriteFile This should acutally
    # workspace["random_file.txt"] = f"Washington D.C"

    # Ensure no prompt hang by writing to the consent file.
    consent_file = Path(os.getcwd()) / ".gpte_consent"
    consent_file.write_text("false")

    # await Agent.db.create_artifact(
    #     task_id=task.task_id,
    #     relative_path=root_dir,
    #     file_name=os.path.join(root_dir, task.task_id),
    # )

    await Agent.db.create_step(
        task_id=task.task_id,
        name="create_code",
        is_last=True,
        additional_input=additional_input,
    )


async def step_handler(step: Step) -> Step:
    """
    Handle the provided step by triggering code generation or other operations.

    Parameters:
    - step (Step): An object containing step details and properties.

    Behavior:
    - If not a dummy step, triggers the main code generation process.
    - Handles potential authentication errors during code generation.
    - Creates a dummy step if it's the last step to ensure continuity.

    Returns:
    - step (Step): Returns the processed step, potentially with modifications.
    """

    # if not step.name == "Dummy step":
    workspace_dir = Agent.get_workspace(step.task_id)
    artifacts_in = await Agent.db.list_artifacts(step.task_id)
    # if any python files in the artifacts in
    python_file_list = [".py" in artifact.file_name for artifact in artifacts_in]
    if any(python_file_list):
        steps_config = "simple_enhanced"
    else:
        steps_config = "simple"
    try:
        main(
            workspace_dir,
            step.additional_input.get("model", "gpt-4"),
            step.additional_input.get("temperature", 0.1),
            steps_config,
            False,
            step.additional_input.get("azure_endpoint", ""),
            step.additional_input.get("verbose", False),
        )
    except AuthenticationError:
        print("The agent lacks a valid OPENAI_API_KEY to execute the requested step.")

    # check if new files have been created and make artifacts for those (CURRENTLY ONLY CONSIDERS TOP LEVEL DIRECTORY AND WILL MAKE FALSE OVERWRITES IF THERE ARE MULTIPLE FILES WITH THE SAME NAME IN THE TREE).
    artifacts = await Agent.db.list_artifacts(step.task_id)
    existing_artifacts = {
        os.path.join(artifact.relative_path, artifact.file_name)
        if artifact.agent_created
        else artifact.file_name
        for artifact in artifacts
    }

    # HACK SOLVING A TEMPORARY PROBLEM: CURRENTLY GPT-ENGINEER WRITES AND EXECUTES CODE IN A SUBDIR CALLED WORKSPACE BY DEFAULT, WHICH NOW IS A DIRECTORY INSIDE 'workspace_dir'. FOR CORRECT REPORTING, WE MUST COPY ALL FILES TO 'workspace_dir

    # gpte_workspace_path = Path(os.path.join(workspace_dir, "workspace"))
    # if os.path.exists(gpte_workspace_path):
    #     for item in gpte_workspace_path.iterdir():
    #         if item.is_dir():
    #             (Path(workspace_dir) / item.name).mkdir(parents=True, exist_ok=True)
    #             shutil.copytree(item, Path(workspace_dir) / item.name, dirs_exist_ok=True)
    #         else:
    #             shutil.copy2(item, workspace_dir)
    #     shutil.rmtree(gpte_workspace_path)

    # create artifacts, enabling agbenchmark to know about the existence of the files. LAST TIME I CHECKED, ANY NON-EMPTY RELATIVE PATHS GAVE RUNTIME ERRORS IN agbenchmark
    for item in os.listdir(workspace_dir):
        full_path = os.path.join(workspace_dir, item)
        if os.path.isfile(full_path):
            if item not in existing_artifacts:
                existing_artifacts.add(item)
                await Agent.db.create_artifact(
                    task_id=step.task_id,
                    relative_path="",
                    file_name=item,
                )
    # additionally, if the file pre-execution files exists, add the paths inside of it too
    if os.path.exists(os.path.join(workspace_dir, "pre-execution-files.txt")):
        with open(os.path.join(workspace_dir, "pre-execution-files.txt"), "r") as file:
            # Iterate over each line in the file
            for line in file:
                line = line.strip()
                if line not in existing_artifacts:
                    directory, filename = os.path.split(line)
                    existing_artifacts.add(line)
                    await Agent.db.create_artifact(
                        task_id=step.task_id,
                        relative_path=directory,
                        file_name=filename,
                    )

    # path_black_list = list()
    # for dirpath, dirnames, filenames in os.walk(workspace_dir):
    #     if "pyvenv.cfg" in filenames or "pip-selfcheck.json" in filenames:
    #         path_black_list.append(dirpath)
    #         continue
    #     bools = [dirpath in path_name for path_name in path_black_list]
    #     if any(bools):
    #         continue
    #     for filename in filenames:
    #         full_path = os.path.join(dirpath, filename)
    #
    #         if (not full_path in existing_artifacts):
    #             existing_artifacts.add(full_path)
    #             if os.path.isfile(full_path):
    #                 rel_path = os.path.relpath(dirpath, workspace_dir)
    #                 # BENCHMARK_TEMP = "/home/axel/Software/Auto-GPT/benchmark/agbenchmark_config/temp_folder/"
    #                 # Path(os.path.join(BENCHMARK_TEMP, rel_path)).mkdir(exist_ok=True, parents=True)
    #                 await Agent.db.create_artifact(
    #                     task_id=step.task_id,
    #                     relative_path=str(rel_path),
    #                     file_name=filename,
    #                     )

    return step


class AgentMiddleware:
    """
    Middleware that injects the agent instance into the request scope.
    """

    def __init__(self, app: FastAPI, agent: "Agent"):
        """

        Args:
            app: The FastAPI app - automatically injected by FastAPI.
            agent: The agent instance to inject into the request scope.
        """
        self.app = app
        self.agent = agent

    async def __call__(self, scope, receive, send):
        scope["agent"] = self.agent
        await self.app(scope, receive, send)


def simplified_fast_API(port):
    Agent.setup_agent(task_handler, step_handler)

    config = Config()
    config.bind = [f"localhost:{port}"]  # As an example configuration setting
    app.include_router(base_router)
    asyncio.run(serve(app, config))


def run_fast_API_app(port):
    Agent.setup_agent(task_handler, step_handler)

    app = FastAPI(
        title="Agent Communication Protocol",
        description="Specification of the API protocol for communication with an agent.",
        version="v1",
    )

    app.add_exception_handler(NotFoundException, not_found_exception_handler)

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

    # app.include_router(router, prefix="/ap/v1")
    app.include_router(base_router)
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
        print(
            f"Frontend not found. {frontend_path} does not exist. The frontend will not be served"
        )
    app.add_middleware(AgentMiddleware, agent=Agent)
    config = Config()
    config.loglevel = "ERROR"
    config.bind = [f"localhost:{port}"]

    print(f"Agent server starting on http://localhost:{port}")
    asyncio.run(serve(app, config))
    return app


if __name__ == "__main__":
    custom = True
    port = 8000
    if custom:
        run_fast_API_app(port)
    else:
        simplified_fast_API(port)
