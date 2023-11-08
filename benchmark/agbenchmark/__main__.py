import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import pytest
import toml
from dotenv import load_dotenv
from helicone.lock import HeliconeLockManager

from agbenchmark.app import app
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.data_types import AgentBenchmarkConfig

load_dotenv()

BENCHMARK_START_TIME_DT = datetime.now(timezone.utc)
BENCHMARK_START_TIME = BENCHMARK_START_TIME_DT.strftime("%Y-%m-%dT%H:%M:%S+00:00")
TEMP_FOLDER_ABS_PATH = Path.cwd() / "agbenchmark_config" / "temp_folder"
CHALLENGES_ALREADY_BEATEN = (
    Path.cwd() / "agbenchmark_config" / "challenges_already_beaten.json"
)
UPDATES_JSON_PATH = Path.cwd() / "agbenchmark_config" / "updates.json"


if os.environ.get("HELICONE_API_KEY"):
    HeliconeLockManager.write_custom_property(
        "benchmark_start_time", BENCHMARK_START_TIME
    )

with open(
    Path(__file__).resolve().parent / "challenges" / "optional_categories.json"
) as f:
    OPTIONAL_CATEGORIES = json.load(f)["optional_categories"]


def get_unique_categories() -> set[str]:
    """Find all data.json files in the directory relative to this file and its subdirectories,
    read the "category" field from each file, and return a set of unique categories."""
    categories = set()

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(this_dir, "./challenges/**/data.json")
    # Use it as the base for the glob pattern
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            try:
                data = json.load(f)
                categories.update(data.get("category", []))
            except json.JSONDecodeError:
                print(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                print(f"IOError: file could not be read: {data_file}")
                continue

    return categories


def run_benchmark(
    maintain: bool = False,
    improve: bool = False,
    explore: bool = False,
    mock: bool = False,
    no_dep: bool = False,
    nc: bool = False,
    keep_answers: bool = False,
    category: Optional[tuple[str]] = None,
    skip_category: Optional[tuple[str]] = None,
    test: Optional[str] = None,
    cutoff: Optional[int] = None,
    server: bool = False,
) -> int:
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""
    # Check if configuration file exists and is not empty

    initialize_updates_file()
    SingletonReportManager()
    agent_benchmark_config_path = str(Path.cwd() / "agbenchmark_config" / "config.json")
    try:
        with open(agent_benchmark_config_path, "r") as f:
            agent_benchmark_config = AgentBenchmarkConfig(**json.load(f))
            agent_benchmark_config.agent_benchmark_config_path = (
                agent_benchmark_config_path
            )
    except json.JSONDecodeError:
        print("Error: benchmark_config.json is not a valid JSON file.")
        return 1

    if maintain and improve and explore:
        print(
            "Error: You can't use --maintain, --improve or --explore at the same time. Please choose one."
        )
        return 1

    if test and (category or skip_category or maintain or improve or explore):
        print(
            "Error: If you're running a specific test make sure no other options are selected. Please just pass the --test."
        )
        return 1

    assert agent_benchmark_config.host, "Error: host needs to be added to the config."

    print("Current configuration:")
    for key, value in vars(agent_benchmark_config).items():
        print(f"{key}: {value}")

    pytest_args = ["-vs"]
    if keep_answers:
        pytest_args.append("--keep-answers")

    if test:
        print("Running specific test:", test)
    else:
        # Categories that are used in the challenges
        categories = get_unique_categories()
        if category:
            invalid_categories = set(category) - categories
            assert (
                not invalid_categories
            ), f"Invalid categories: {invalid_categories}. Valid categories are: {categories}"

        if category:
            categories_to_run = set(category)
            if skip_category:
                categories_to_run = categories_to_run.difference(set(skip_category))
                assert categories_to_run, "Error: You can't skip all categories"
            pytest_args.extend(["-m", " or ".join(categories_to_run), "--category"])
            print("Running tests of category:", categories_to_run)
        elif skip_category:
            categories_to_run = categories - set(skip_category)
            assert categories_to_run, "Error: You can't skip all categories"
            pytest_args.extend(["-m", " or ".join(categories_to_run), "--category"])
            print("Running tests of category:", categories_to_run)
        else:
            print("Running all categories")

        if maintain:
            print("Running only regression tests")
            pytest_args.append("--maintain")
        elif improve:
            print("Running only non-regression tests")
            pytest_args.append("--improve")
        elif explore:
            print("Only attempt challenges that have never been beaten")
            pytest_args.append("--explore")

    if mock:
        pytest_args.append("--mock")
        os.environ[
            "IS_MOCK"
        ] = "True"  # ugly hack to make the mock work when calling from API

    if no_dep:
        pytest_args.append("--no_dep")

    if nc and cutoff:
        print(
            "Error: You can't use both --nc and --cutoff at the same time. Please choose one."
        )
        return 1

    if nc:
        pytest_args.append("--nc")
    if cutoff:
        pytest_args.append("--cutoff")
        print(f"Setting cuttoff override to {cutoff} seconds.")
    current_dir = Path(__file__).resolve().parent
    print(f"Current directory: {current_dir}")
    pytest_args.extend((str(current_dir), "--cache-clear"))
    exit_code = pytest.main(pytest_args)
    SingletonReportManager().clear_instance()


@click.group(invoke_without_command=True)
@click.option("--backend", is_flag=True, help="If it's being run from the cli")
@click.option("-c", "--category", multiple=True, help="Specific category to run")
@click.option(
    "-s",
    "--skip-category",
    multiple=True,
    help="Skips preventing the tests from this category from running",
)
@click.option("--test", multiple=True, help="Specific test to run")
@click.option("--maintain", is_flag=True, help="Runs only regression tests")
@click.option("--improve", is_flag=True, help="Run only non-regression tests")
@click.option(
    "--explore",
    is_flag=True,
    help="Only attempt challenges that have never been beaten",
)
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option(
    "--no_dep",
    is_flag=True,
    help="Run without dependencies",
)
@click.option("--nc", is_flag=True, help="Run without cutoff")
@click.option("--keep-answers", is_flag=True, help="Keep answers")
@click.option("--cutoff", help="Set or override tests cutoff (seconds)")
@click.argument("value", type=str, required=False)
def cli(
    maintain: bool,
    improve: bool,
    explore: bool,
    mock: bool,
    no_dep: bool,
    nc: bool,
    keep_answers: bool,
    category: Optional[list[str]] = None,
    skip_category: Optional[list[str]] = None,
    test: Optional[str] = None,
    cutoff: Optional[int] = None,
    backend: Optional[bool] = False,
    value: Optional[str] = None,
) -> Any:
    # Redirect stdout if backend is True
    if value == "start":
        raise ("`agbenchmark start` is removed. Run `agbenchmark` instead.")
    if value == "serve":
        return serve()
    original_stdout = sys.stdout  # Save the original standard output
    exit_code = None

    if backend:
        with open("backend/backend_stdout.txt", "w") as f:
            sys.stdout = f
            exit_code = run_benchmark(
                maintain=maintain,
                improve=improve,
                explore=explore,
                mock=mock,
                no_dep=no_dep,
                nc=nc,
                keep_answers=keep_answers,
                category=category,
                skip_category=skip_category,
                test=test,
                cutoff=cutoff,
            )

        sys.stdout = original_stdout

    else:
        exit_code = run_benchmark(
            maintain=maintain,
            improve=improve,
            explore=explore,
            mock=mock,
            no_dep=no_dep,
            nc=nc,
            keep_answers=keep_answers,
            category=category,
            skip_category=skip_category,
            test=test,
            cutoff=cutoff,
        )

        sys.exit(exit_code)


@cli.command()
def version():
    """Print the version of the benchmark tool."""
    current_directory = Path(__file__).resolve().parent
    version = toml.load(current_directory / ".." / "pyproject.toml")["tool"]["poetry"][
        "version"
    ]
    print(f"Benchmark Tool Version {version}")


def serve():
    import uvicorn

    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


def initialize_updates_file():
    if os.path.exists(UPDATES_JSON_PATH):
        # If the file already exists, overwrite it with an empty list
        with open(UPDATES_JSON_PATH, "w") as file:
            json.dump([], file, indent=2)
        print("Initialized updates.json by overwriting with an empty array")
    else:
        # If the file doesn't exist, create it and write an empty list
        with open(UPDATES_JSON_PATH, "w") as file:
            json.dump([], file, indent=2)
        print("Created updates.json and initialized it with an empty array")


if __name__ == "__main__":
    cli()
