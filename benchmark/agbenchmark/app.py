import datetime
import glob
import json
import logging
import os
import sys
import uuid
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Optional

import httpx
import psutil
from agent_protocol_client import AgentApi, ApiClient, ApiException, Configuration
from agent_protocol_client.models import Task, TaskRequestBody
from fastapi import APIRouter, FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Extra, ValidationError

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.execute_sub_process import execute_subprocess
from agbenchmark.reports.processing.report_types_v2 import (
    BenchmarkRun,
    Metrics,
    RepositoryInfo,
    RunDetails,
    TaskInfo,
)
from agbenchmark.schema import TaskEvalRequestBody
from agbenchmark.utils.data_types import ChallengeData
from agbenchmark.utils.utils import write_pretty_json

sys.path.append(str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

CHALLENGES: dict[str, ChallengeData] = {}
challenges_path = Path(__file__).parent / "challenges"
challenge_spec_files = deque(
    glob.glob(
        f"{challenges_path}/**/data.json",
        recursive=True,
    )
)

logger.debug("Loading challenges...")
while challenge_spec_files:
    challenge_spec_file = Path(challenge_spec_files.popleft())
    challenge_relpath = challenge_spec_file.relative_to(challenges_path.parent)
    if challenge_relpath.is_relative_to("challenges/deprecated"):
        continue

    logger.debug(f"Loading {challenge_relpath}...")
    try:
        challenge_info = ChallengeData.parse_file(challenge_spec_file)
    except ValidationError as e:
        if logging.getLogger().level == logging.DEBUG:
            logger.warning(f"Spec file {challenge_relpath} failed to load:\n{e}")
        logger.debug(f"Invalid challenge spec: {challenge_spec_file.read_text()}")
        continue
    challenge_info.spec_file = challenge_spec_file

    if not challenge_info.eval_id:
        challenge_info.eval_id = str(uuid.uuid4())
        # this will sort all the keys of the JSON systematically
        # so that the order is always the same
        write_pretty_json(challenge_info.dict(), challenge_spec_file)

    CHALLENGES[challenge_info.eval_id] = challenge_info

task_informations = defaultdict(dict[str, Any])


def find_agbenchmark_without_uvicorn():
    pids = []
    for process in psutil.process_iter(
        attrs=[
            "pid",
            "cmdline",
            "name",
            "username",
            "status",
            "cpu_percent",
            "memory_info",
            "create_time",
            "cwd",
            "connections",
        ]
    ):
        try:
            # Convert the process.info dictionary values to strings and concatenate them
            full_info = " ".join([str(v) for k, v in process.as_dict().items()])

            if "agbenchmark" in full_info and "uvicorn" not in full_info:
                pids.append(process.pid)
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
            pass
    return pids


class CreateReportRequest(BaseModel):
    test: str = None
    test_run_id: str = None
    # category: Optional[str] = []
    mock: Optional[bool] = False

    class Config:
        extra = Extra.forbid  # this will forbid any extra fields


updates_list = []

origins = [
    "http://localhost:8000",
    "http://localhost:8080",
    "http://127.0.0.1:5000",
    "http://localhost:5000",
]


def stream_output(pipe):
    for line in pipe:
        print(line, end="")


def setup_fastapi_app(agbenchmark_config: AgentBenchmarkConfig) -> FastAPI:
    from agbenchmark.agent_api_interface import (
        copy_agent_artifacts_into_folder,
        upload_artifacts,
    )
    from agbenchmark.agent_interface import copy_artifacts_into_temp_folder
    from agbenchmark.generate_test import create_challenge_from_spec_file

    configuration = Configuration(
        host=agbenchmark_config.host or "http://localhost:8000"
    )
    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    router = APIRouter()

    @router.post("/reports")
    def run_single_test(body: CreateReportRequest) -> dict:
        pids = find_agbenchmark_without_uvicorn()
        logger.info(f"pids already running with agbenchmark: {pids}")

        logger.debug(f"Request to /reports: {body.dict()}")

        _initialize_updates_file(agbenchmark_config)

        # it's a hack because other parts of the code are using sys.argv
        logger.debug(f"Current working directory: {os.getcwd()}")
        command_options = ["agbenchmark"]
        # if body.category:
        #     sys.argv.append(f"--category={body.category}")
        command_options.append(f"--test={body.test}")
        if body.mock:
            command_options.append("--mock")

        logger.debug(f"Running {command_options}")
        execute_subprocess(command_options, 200)
        logger.debug("Finished running")

        # List all folders in the current working directory
        path_reports = agbenchmark_config.reports_folder
        folders = [folder for folder in path_reports.iterdir() if folder.is_dir()]

        # Sort the folders based on their names
        sorted_folders = sorted(folders, key=lambda x: x.name)

        # Get the last folder
        latest_folder = sorted_folders[-1] if sorted_folders else None

        # Read report.json from this folder
        if latest_folder:
            report_path = latest_folder / "report.json"
            logger.debug(f"Getting latest report from {report_path}")
            if report_path.exists():
                with report_path.open() as file:
                    data = json.load(file)
                logger.debug(f"Report data: {data}")
            else:
                logger.error(
                    "Could not get result after running benchmark: "
                    f"'report.json' does not exist in '{latest_folder}'"
                )
        else:
            logger.error(
                "Could not get result after running benchmark: no reports found"
            )

        return data

    @router.get("/updates")
    def get_updates(request: Request) -> list[dict]:
        try:
            # Read data from the "update.json" file (provide the correct file path)
            logger.debug(
                f"Reading updates from {agbenchmark_config.updates_json_file}..."
            )
            with open(agbenchmark_config.updates_json_file, "r") as file:
                data = json.load(file)

            # Get the last_update_time from the query parameter
            query_param = request.query_params.get("last_update_time")

            if query_param is None:
                # Handle the case when last_update_time is not provided
                logger.error("last_update_time parameter is missing")
                raise HTTPException(
                    status_code=400,
                    detail="last_update_time parameter is missing",
                )

            # Parse query_param as a Unix timestamp (seconds)
            query_timestamp = int(query_param)

            # Filter the data, keep items newer than query_timestamp
            filtered_data = [
                item for item in data if item["timestamp"] > query_timestamp
            ]

            # Extract only the "content" field from each item
            filtered_data = [item["content"] for item in filtered_data]

            logger.debug("Returning filtered data to the client")
            return filtered_data
        except FileNotFoundError:
            logger.error("File not found: updates.json")
            raise HTTPException(
                status_code=404,
                detail="File not found",
            )

    @router.post("/agent/tasks", tags=["agent"])
    async def create_agent_task(task_eval_request: TaskEvalRequestBody) -> Task:
        """
        Creates a new task using the provided TaskEvalRequestBody and returns a Task.

        Args:
            task_eval_request: `TaskRequestBody` including an eval_id.

        Returns:
            Task: A new task with task_id, input, additional_input,
                and empty lists for artifacts and steps.

        Example:
            Request (TaskEvalRequestBody defined in schema.py):
                {
                    ...,
                    "eval_id": "50da533e-3904-4401-8a07-c49adf88b5eb"
                }

            Response (Task defined in `agent_protocol_client.models`):
                {
                    "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                    "input": "Write the word 'Washington' to a .txt file",
                    "artifacts": []
                }
        """
        try:
            async with ApiClient(configuration) as api_client:
                api_instance = AgentApi(api_client)
                task_input = CHALLENGES[task_eval_request.eval_id].task

                task_request_body = TaskRequestBody(input=task_input)
                task_response = await api_instance.create_agent_task(
                    task_request_body=task_request_body
                )
                task_informations[task_response.task_id][
                    "benchmark_start_time"
                ] = datetime.datetime.now(datetime.timezone.utc).strftime(
                    "%Y-%m-%dT%H:%M:%S+00:00"
                )
                task_informations[task_response.task_id][
                    "eval_id"
                ] = task_eval_request.eval_id
                await upload_artifacts(
                    api_instance,
                    str(CHALLENGES[task_eval_request.eval_id].spec_file.parent),
                    task_response.task_id,
                    "artifacts_in",
                )
                return task_response
        except ApiException as e:
            logger.error(f"Error whilst trying to create a task:\n{e}")
            logger.error(
                "The above error was caused while processing request: "
                f"{task_eval_request}"
            )
            raise HTTPException(500)

    @router.post("/agent/tasks/{task_id}/steps")
    async def proxy(request: Request, task_id: str):
        timeout = httpx.Timeout(300.0, read=300.0)  # 5 minutes
        async with httpx.AsyncClient(timeout=timeout) as client:
            # Construct the new URL
            new_url = f"{configuration.host}/ap/v1/agent/tasks/{task_id}/steps"

            # Forward the request
            response = await client.post(
                new_url,
                data=await request.body(),
                headers=dict(request.headers),
            )

            # Return the response from the forwarded request
            return Response(content=response.content, status_code=response.status_code)

    @router.post("/agent/tasks/{task_id}/evaluations")
    async def create_evaluation(task_id: str) -> BenchmarkRun:
        challenge_info = CHALLENGES[task_informations[task_id]["eval_id"]]
        workspace = agbenchmark_config.temp_folder
        try:
            async with ApiClient(configuration) as api_client:
                api_instance = AgentApi(api_client)
                await copy_agent_artifacts_into_folder(api_instance, task_id, workspace)

            artifact_path = challenge_info.spec_file.parent
            copy_artifacts_into_temp_folder(workspace, "custom_python", artifact_path)

            challenge = create_challenge_from_spec_file(challenge_info.spec_file)
            scores = challenge.get_scores(workspace)
            is_score_100 = 1 in scores["values"]

            eval_info = BenchmarkRun(
                repository_info=RepositoryInfo(),
                run_details=RunDetails(
                    command=f"agbenchmark --test={challenge_info.name}",
                    benchmark_start_time=(
                        task_informations[task_id]["benchmark_start_time"]
                    ),
                    test_name=challenge_info.name,
                ),
                task_info=TaskInfo(
                    data_path=str(
                        challenge_info.spec_file.relative_to(challenges_path.parent)
                    ),
                    is_regression=None,
                    category=[c.value for c in challenge_info.category],
                    task=challenge_info.task,
                    answer=challenge_info.ground.answer,
                    description=challenge_info.info.description,
                ),
                metrics=Metrics(
                    success=is_score_100,
                    attempted=True,
                ),
                config={},
            )

            logger.debug(f"Returning evaluation data:\n{eval_info.json(indent=4)}")
            return eval_info
        except ApiException as e:
            logger.error(f"Error {e} whilst trying to evaluate task: {task_id}")
            raise HTTPException(500)

    app.include_router(router, prefix="/ap/v1")

    return app


def _initialize_updates_file(config: AgentBenchmarkConfig):
    if config.updates_json_file.exists():
        logger.debug("Resetting updates.json to an empty array")
    else:
        logger.debug("Creating updates.json with an empty array")

    with open(config.updates_json_file, "w") as file:
        json.dump([], file, indent=2)
