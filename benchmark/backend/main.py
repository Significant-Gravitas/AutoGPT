import ast
import json
import os
import subprocess
import sys
from importlib import reload
from typing import Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from agbenchmark.utils.utils import find_absolute_benchmark_path
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["http://localhost:3000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Change the current working directory to the benchmark path
home_path = find_absolute_benchmark_path()
os.chdir(home_path)

general_command = ["poetry", "run", "agbenchmark", "start", "--backend"]


@app.get("/run_single_test")
def run_single_test(
    test: str = Query(...),
    mock: bool = Query(False),
    nc: bool = Query(False),
    cutoff: int = Query(None),
) -> Any:
    command = list(general_command)  # Make a copy of the general command

    # Always add the --test flag, since test is a required parameter
    command.extend(["--test", test])

    # Conditionally add other flags
    if mock:
        command.append("--mock")
    if nc:
        command.extend(["--nc", str(nc)])
    if cutoff is not None:
        command.extend(["--cutoff", str(cutoff)])

    print(f"Running command: {' '.join(command)}")  # Debug print

    result = subprocess.run(command, capture_output=True, text=True)

    stdout_dict = ast.literal_eval(result.stdout)

    return {
        "returncode": result.returncode,
        "stdout": json.dumps(stdout_dict),
        "stderr": result.stderr,
    }


@app.get("/run_suite")
def run_suite(
    suite: str = Query(...),
    mock: bool = Query(False),
    nc: bool = Query(False),
    cutoff: int = Query(None),
) -> Any:
    command = list(general_command)  # Make a copy of the general command

    # Always add the --test flag, since test is a required parameter
    command.extend(["--suite", suite])

    # Conditionally add other flags
    if mock:
        command.append("--mock")
    if nc:
        command.extend(["--nc", str(nc)])
    if cutoff is not None:
        command.extend(["--cutoff", str(cutoff)])

    print(f"Running command: {' '.join(command)}")  # Debug print

    result = subprocess.run(command, capture_output=True, text=True)

    stdout_dict = ast.literal_eval(result.stdout)

    return {
        "returncode": result.returncode,
        "stdout": json.dumps(stdout_dict),
        "stderr": result.stderr,
    }


@app.get("/run_by_category")
def run_by_category(
    category: list[str] = Query(...),  # required
    mock: bool = Query(False),
    nc: bool = Query(False),
    cutoff: int = Query(None),
) -> Any:
    command = list(general_command)  # Make a copy of the general command

    # Always add the --test flag, since test is a required parameter
    command.extend(["--category", *category])

    # Conditionally add other flags
    if mock:
        command.append("--mock")
    if nc:
        command.extend(["--nc", str(nc)])
    if cutoff is not None:
        command.extend(["--cutoff", str(cutoff)])

    print(f"Running command: {' '.join(command)}")  # Debug print

    result = subprocess.run(command, capture_output=True, text=True)

    stdout_dict = ast.literal_eval(result.stdout)

    return {
        "returncode": result.returncode,
        "stdout": json.dumps(stdout_dict),
        "stderr": result.stderr,
    }


@app.get("/run")
def run(
    maintain: bool = Query(False),
    improve: bool = Query(False),
    explore: bool = Query(False),
    mock: bool = Query(False),
    no_dep: bool = Query(False),
    nc: bool = Query(False),
    category: list[str] = Query(None),
    skip_category: list[str] = Query(None),
    test: str = Query(None),
    suite: str = Query(None),
    cutoff: int = Query(None),
) -> Any:
    command = list(general_command)  # Make a copy of the general command

    # Conditionally add other flags
    if mock:
        command.append("--mock")
    if nc:
        command.extend(["--nc", str(nc)])
    if cutoff is not None:
        command.extend(["--cutoff", str(cutoff)])
    if maintain:
        command.append("--maintain")
    if improve:
        command.append("--improve")
    if explore:
        command.append("--explore")
    if no_dep:
        command.append("--no_dep")

    if category:
        for cat in category:
            command.extend(["-c", cat])

    if skip_category:
        for skip_cat in skip_category:
            command.extend(["-s", skip_cat])

    if test:
        command.extend(["--test", test])

    if suite:
        command.extend(["--suite", suite])

    print(f"Running command: {' '.join(command)}")  # Debug print

    result = subprocess.run(command, capture_output=True, text=True)

    stdout_dict = ast.literal_eval(result.stdout)

    return {
        "returncode": result.returncode,
        "stdout": json.dumps(stdout_dict),
        "stderr": result.stderr,
    }
