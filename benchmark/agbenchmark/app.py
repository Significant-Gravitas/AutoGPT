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
from pydantic import BaseModel, Extra

from agbenchmark.execute_sub_process import execute_subprocess
from agbenchmark.reports.processing.report_types_v2 import (
    BenchmarkRun,
    Metrics,
    RepositoryInfo,
    RunDetails,
    TaskInfo,
)
from agbenchmark.schema import TaskEvalRequestBody
from agbenchmark.utils.path_manager import PATH_MANAGER
from agbenchmark.utils.utils import write_pretty_json

sys.path.append(str(Path(__file__).parent.parent))

configuration = Configuration(host="http://localhost:8000")
logger = logging.getLogger(__name__)
router = APIRouter()

# Change the current working directory to the benchmark path
# home_path = find_absolute_benchmark_path()
# os.chdir(home_path)

general_command = ["poetry", "run", "agbenchmark", "start", "--backend"]

challenges_path = Path(__file__).parent / "challenges"

json_files = deque(
    glob.glob(
        f"{challenges_path}/**/data.json",
        recursive=True,
    )
)

CHALLENGES = {}
task_informations = defaultdict(dict[str, Any])

while json_files:
    json_file = json_files.popleft()

    with open(json_file, "r") as file:
        data = json.load(file)

        if "eval_id" not in data:
            data["eval_id"] = str(uuid.uuid4())
        # this will sort all the keys of the JSON systematically
        # so that the order is always the same
        write_pretty_json(data, json_file)
        # ok
        CHALLENGES[data["eval_id"]] = data
        CHALLENGES[data["eval_id"]]["path"] = json_file


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
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def stream_output(pipe):
    for line in pipe:
        print(line, end="")


@router.post("/reports")
def run_single_test(body: CreateReportRequest) -> dict:
    pids = find_agbenchmark_without_uvicorn()
    logger.info(f"pids already running with agbenchmark: {pids}")

    logger.debug(f"Request to /reports: {body.dict()}")

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
    path_reports = Path.cwd() / "agbenchmark_config" / "reports"
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
        logger.error("Could not get result after running benchmark: no reports found")

    return data


@router.get("/updates")
def get_updates(request: Request) -> list[dict]:
    try:
        # Read data from the "update.json" file (provide the correct file path)
        logger.debug(f"Reading updates from {PATH_MANAGER.updates_json_file}...")
        with open(PATH_MANAGER.updates_json_file, "r") as file:
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

        # Convert query_param to a Unix timestamp (assuming it's in seconds as a string)
        query_timestamp = int(query_param)

        # Filter the data based on the timestamp: keep items newer than query_timestamp
        filtered_data = [item for item in data if item["timestamp"] > query_timestamp]

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
        task_eval_request (TaskEvalRequestBody): `TaskRequestBody` including an eval_id.

    Returns:
        Task: A new task with task_id, input, additional_input, and empty lists for artifacts and steps.

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
    from agbenchmark.agent_api_interface import upload_artifacts

    try:
        async with ApiClient(configuration) as api_client:
            api_instance = AgentApi(api_client)
            task_input = CHALLENGES[task_eval_request.eval_id]["task"]

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
                str(Path(CHALLENGES[task_eval_request.eval_id]["path"]).parent),
                task_response.task_id,
                "artifacts_in",
            )
            return task_response
    except ApiException as e:
        logger.error(f"Error whilst trying to create a task:\n{e}")
        logger.error(
            f"The above error was caused while processing request: {task_eval_request}"
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
    from agbenchmark.agent_api_interface import copy_agent_artifacts_into_temp_folder
    from agbenchmark.agent_interface import copy_artifacts_into_temp_folder
    from agbenchmark.generate_test import create_challenge

    try:
        async with ApiClient(configuration) as api_client:
            api_instance = AgentApi(api_client)
            await copy_agent_artifacts_into_temp_folder(api_instance, task_id)
        # add custom python
        challenge_info = CHALLENGES[task_informations[task_id]["eval_id"]]

        artifact_path = str(Path(challenge_info["path"]).parent)
        copy_artifacts_into_temp_folder(
            PATH_MANAGER.temp_folder, "custom_python", artifact_path
        )
        json_file = challenge_info["path"]
        json_files = deque()

        _, challenge_class = create_challenge(challenge_info, json_file, json_files)
        challenge_instance = challenge_class()
        scores = challenge_instance.get_scores(config={})
        test_name = "Test" + challenge_info["name"]
        is_score_100 = 1 in scores["values"]

        eval_info = BenchmarkRun(
            repository_info=RepositoryInfo(),
            run_details=RunDetails(
                command=f"agbenchmark --test={test_name}",
                benchmark_start_time=task_informations[task_id]["benchmark_start_time"],
                test_name=challenge_info["name"],
            ),
            task_info=TaskInfo(
                data_path=challenge_info["path"].split("benchmark/", 1)[-1],
                is_regression=None,
                category=challenge_info["category"],
                task=challenge_info["task"],
                answer=challenge_info["ground"]["answer"],
                description=challenge_info["info"]["description"],
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
