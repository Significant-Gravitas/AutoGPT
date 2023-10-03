import datetime
import uuid
from collections import defaultdict, deque
from pathlib import Path

import httpx

from agbenchmark.agent_protocol_client import (
    AgentApi,
    ApiClient,
    ApiException,
    Configuration,
)
from agbenchmark.reports.processing.report_types_v2 import BenchmarkRun
from agbenchmark.schema import TaskEvalRequestBody
from agbenchmark.utils.utils import write_pretty_json

configuration = Configuration(host="http://localhost:8000" + "/ap/v1")

import json
import os
import sys
from typing import Any, Optional

import psutil
from fastapi import APIRouter, FastAPI
from fastapi import (
    HTTPException as FastAPIHTTPException,  # Import HTTPException from FastAPI
)
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware

from agbenchmark.execute_sub_process import execute_subprocess
from agbenchmark.schema import Task, TaskRequestBody

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from pydantic import BaseModel, Extra

router = APIRouter()
import glob

# Change the current working directory to the benchmark path
# home_path = find_absolute_benchmark_path()
# os.chdir(home_path)

general_command = ["poetry", "run", "agbenchmark", "start", "--backend"]

import psutil

challenges_path = os.path.join(os.path.dirname(__file__), "challenges")

json_files = deque(
    glob.glob(
        f"{challenges_path}/**/data.json",
        recursive=True,
    )
)

CHALLENGES = {}
task_informations = defaultdict(dict)

while json_files:
    json_file = json_files.popleft()

    with open(json_file, "r") as file:
        data = json.load(file)

        if "eval_id" not in data:
            data["eval_id"] = str(uuid.uuid4())
        # this will sort all the keys of the JSON systematically so that the order is always the same
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
            full_info = " ".join([str(v) for k, v in process.info.items()])

            if "agbenchmark" in full_info and "uvicorn" not in full_info:
                pids.append(process.info["pid"])
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

updates_list = []

import json

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
def run_single_test(body: CreateReportRequest) -> Any:
    pids = find_agbenchmark_without_uvicorn()
    print(f"pids already running with agbenchmark: {pids}")
    print(body.dict())
    # it's a hack because other parts of the code are using sys.argv
    print(os.getcwd())
    command_options = ["agbenchmark"]
    # if body.category:
    #     sys.argv.append(f"--category={body.category}")
    command_options.append(f"--test={body.test}")
    if body.mock:
        command_options.append("--mock")

    execute_subprocess(command_options, 200)
    import json
    from pathlib import Path

    print("finished running")
    # List all folders in the current working directory
    path_reports = Path.cwd() / "agbenchmark_config" / "reports"
    folders = [folder for folder in path_reports.iterdir() if folder.is_dir()]

    # Sort the folders based on their names
    sorted_folders = sorted(folders, key=lambda x: x.name)

    # Get the last folder
    last_folder = sorted_folders[-1] if sorted_folders else None

    # Read report.json from this folder
    if last_folder:
        report_path = last_folder / "report.json"
        print(report_path)
        if report_path.exists():
            with report_path.open() as file:
                data = json.load(file)
            print(data)
        else:
            print(f"'report.json' does not exist in '{last_folder}'")
    else:
        print("No folders found.")

    return Response(
        content=json.dumps(data),
        status_code=200,
        media_type="application/json",
    )


import json
from typing import Any

from fastapi import FastAPI, Request, Response


@router.get("/updates")
def get_updates(request: Request) -> Any:
    from agbenchmark.__main__ import UPDATES_JSON_PATH

    try:
        # Read data from the "update.json" file (provide the correct file path)
        with open(UPDATES_JSON_PATH, "r") as file:
            data = json.load(file)

        # Get the last_update_time from the query parameter
        query_param = request.query_params.get("last_update_time")

        if query_param is None:
            # Handle the case when last_update_time is not provided
            print("ERROR: last_update_time parameter is missing")
            return Response(
                content=json.dumps({"error": "last_update_time parameter is missing"}),
                status_code=400,
                media_type="application/json",
                headers={"Content-Type": "application/json"},
            )

        # Convert query_param to a Unix timestamp (assuming it's in seconds as a string)
        query_timestamp = int(query_param)

        # Filter the data based on the timestamp (keep timestamps before query_timestamp)
        filtered_data = [item for item in data if item["timestamp"] > query_timestamp]

        # Extract only the "content" field from each item
        filtered_data = [item["content"] for item in filtered_data]

        # Convert the filtered data to JSON
        filtered_json = json.dumps(filtered_data, indent=2)

        print("INFO: Returning filtered data to the client")
        return Response(
            content=filtered_json,
            status_code=200,
            media_type="application/json",
            headers={"Content-Type": "application/json"},
        )
    except FileNotFoundError:
        print("ERROR: File not found: updates.json")
        return Response(
            content=json.dumps({"error": "File not found"}),
            status_code=404,
            media_type="application/json",
            headers={"Content-Type": "application/json"},
        )


@router.post("/agent/tasks", tags=["agent"], response_model=Task)
async def create_agent_task(task_eval_request: TaskEvalRequestBody) -> Task:
    """
    Creates a new task using the provided TaskRequestBody and returns a Task.

    Args:
        request (Request): FastAPI request object.
        task (TaskRequestBody): The task request containing input and additional input data.

    Returns:
        Task: A new task with task_id, input, additional_input, and empty lists for artifacts and steps.

    Example:
        Request (TaskRequestBody defined in schema.py):
            {
                "input": "Write the words you receive to the file 'output.txt'.",
                "additional_input": "python/code"
            }

        Response (Task defined in schema.py):
            {
                "task_id": "50da533e-3904-4401-8a07-c49adf88b5eb",
                "input": "Write the word 'Washington' to a .txt file",
                "additional_input": "python/code",
                "artifacts": [],
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
            return Response(
                content=task_response.json(),
                status_code=200,
                media_type="application/json",
            )
    except ApiException as e:
        print(f"Error whilst trying to create a task: {task_eval_request}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )


@router.post("/agent/tasks/{task_id}/steps")
async def proxy(request: Request, task_id: str):
    timeout = httpx.Timeout(300.0, read=300.0)  # 5 minutes
    async with httpx.AsyncClient(timeout=timeout) as client:
        # Construct the new URL
        new_url = f"http://localhost:8000/ap/v1/agent/tasks/{task_id}/steps"

        # Forward the request
        response = await client.post(
            new_url,
            data=await request.body(),
            headers=dict(request.headers),
        )

        # Return the response from the forwarded request
        return Response(content=response.content, status_code=response.status_code)


@router.post("/agent/tasks/{task_id}/evaluations")
async def create_evaluation(task_id: str) -> deque:
    from agbenchmark.__main__ import TEMP_FOLDER_ABS_PATH
    from agbenchmark.agent_api_interface import copy_agent_artifacts_into_temp_folder
    from agbenchmark.agent_interface import copy_artifacts_into_temp_folder
    from agbenchmark.generate_test import create_challenge

    try:
        async with ApiClient(configuration) as api_client:
            api_instance = AgentApi(api_client)
            await copy_agent_artifacts_into_temp_folder(api_instance, task_id)
        # add custom python
        data = CHALLENGES[task_informations[task_id]["eval_id"]]

        artifact_path = str(Path(data["path"]).parent)
        copy_artifacts_into_temp_folder(
            TEMP_FOLDER_ABS_PATH, "custom_python", artifact_path
        )
        json_file = CHALLENGES[task_informations[task_id]["eval_id"]]["path"]
        json_files = deque()

        _, challenge_class = create_challenge(data, json_file, json_files)
        challenge_instance = challenge_class()
        scores = challenge_instance.get_scores(config={})
        test_name = "Test" + data["name"]
        is_score_100 = 1 in scores["values"]

        info_details = {
            "repository_info": {
                "repo_url": None,
                "team_name": None,
                "benchmark_git_commit_sha": None,
                "agent_git_commit_sha": None,
            },
            "run_details": {
                "run_id": None,
                "command": "agbenchmark" + " --test=" + test_name,
                "completion_time": None,
                "benchmark_start_time": task_informations[task_id][
                    "benchmark_start_time"
                ],
                "test_name": data["name"],
            },
            "task_info": {
                "data_path": data["path"].split("benchmark/", 1)[-1],
                "is_regression": None,
                "category": data["category"],
                "task": data["task"],
                "answer": data["ground"]["answer"],
                "description": data["info"]["description"],
            },
            "metrics": {
                "difficulty": None,
                "success": is_score_100,
                "attempted": True,
                "success_percentage": None,
                "cost": None,
                "run_time": None,
            },
            "reached_cutoff": None,
            "config": {},
        }

        BenchmarkRun.parse_obj(info_details)

        print(json.dumps(info_details, indent=4))
        return Response(
            content=json.dumps(info_details),
            status_code=200,
            media_type="application/json",
        )
    except ApiException as e:
        print(f"Error whilst trying to evaluate the task: {task_id}")
        return Response(
            content=json.dumps({"error": "Internal server error"}),
            status_code=500,
            media_type="application/json",
        )
    # path = Path(json_file).resolve()


app.include_router(router, prefix="/ap/v1")
