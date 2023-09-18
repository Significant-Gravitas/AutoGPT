import json
import os
import sys
from typing import Any, Optional

import psutil
from fastapi import FastAPI
from fastapi import (
    HTTPException as FastAPIHTTPException,  # Import HTTPException from FastAPI
)
from fastapi import Request, Response
from fastapi.middleware.cors import CORSMiddleware

from agbenchmark.execute_sub_process import execute_subprocess

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI
from pydantic import BaseModel, Extra

# Change the current working directory to the benchmark path
# home_path = find_absolute_benchmark_path()
# os.chdir(home_path)

general_command = ["poetry", "run", "agbenchmark", "start", "--backend"]

import psutil


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


@app.post("/reports")
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


@app.get("/updates")
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
