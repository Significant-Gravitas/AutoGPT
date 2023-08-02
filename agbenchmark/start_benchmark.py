import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import click
import pytest
from helicone.lock import HeliconeLockManager

from agbenchmark.utils.utils import (
    AGENT_NAME,
    calculate_dynamic_paths,
    get_git_commit_sha,
)

CURRENT_DIRECTORY = Path(__file__).resolve().parent
BENCHMARK_START_TIME = datetime.now().strftime("%Y-%m-%d-%H:%M")

HeliconeLockManager.write_custom_property("benchmark_start_time", BENCHMARK_START_TIME)

(
    HOME_DIRECTORY,
    CONFIG_PATH,
    REGRESSION_TESTS_PATH,
    REPORTS_PATH,
    SUCCESS_RATE_PATH,
    CHALLENGES_PATH,
) = calculate_dynamic_paths()
BENCHMARK_GIT_COMMIT_SHA = get_git_commit_sha(HOME_DIRECTORY / ".." / "..")
AGENT_GIT_COMMIT_SHA = get_git_commit_sha(HOME_DIRECTORY)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--category", default=None, help="Specific category to run")
@click.option("--test", default=None, help="Specific test to run")
@click.option("--maintain", is_flag=True, help="Runs only regression tests")
@click.option("--improve", is_flag=True, help="Run only non-regression tests")
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option("--suite", default=None, help="Run a suite of related tests")
@click.option(
    "--no_dep",
    is_flag=True,
    help="Run without dependencies (can be useful for a suite run)",
)
@click.option("--nc", is_flag=True, help="Run without cutoff")
def start(
    category: str,
    test: str,
    maintain: bool,
    improve: bool,
    mock: bool,
    suite: str,
    no_dep: bool,
    nc: bool,
) -> int:
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""
    # Check if configuration file exists and is not empty

    if maintain and improve:
        print(
            "Error: You can't use both --maintain and --improve at the same time. Please choose one."
        )
        return 1

    if test and (category or maintain or improve or suite):
        print(
            "Error: If you're running a specific test make sure no other options are selected. Please just pass the --test."
        )
        return 1

    # TODO: test and ensure that this functionality works before removing
    # change elif suite below if removing
    if suite and (category or maintain or improve):
        print(
            "Error: If you're running a specific suite make sure no other options are selected. Please just pass the --suite."
        )
        return 1

    if os.path.join("Auto-GPT-Benchmarks") in str(HOME_DIRECTORY) and not AGENT_NAME:
        print(
            "If you are running from the Auto-GPT-Benchmarks repo, you must have AGENT_NAME defined."
        )
        return 1

    if not os.path.exists(CONFIG_PATH) or os.stat(CONFIG_PATH).st_size == 0:
        config = {}

        config["workspace"] = click.prompt(
            "Please enter a new workspace path",
            default=os.path.join("workspace"),
            show_default=True,
        )

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)
    else:
        # If the configuration file exists and is not empty, load it
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    print("Current configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    os.environ["MOCK_TEST"] = "True" if mock else "False"

    pytest_args = ["-vs"]
    if test:
        print("Running specific test:", test)
        pytest_args.extend(["-k", test, "--test"])
    elif suite:
        print("Running specific suite:", suite)
        pytest_args.extend(["--suite"])
    else:
        if category:
            pytest_args.extend(["-m", category, "--category"])
            print("Running tests of category:", category)
        else:
            print("Running all categories")

        if maintain:
            print("Running only regression tests")
            pytest_args.append("--maintain")
        elif improve:
            print("Running only non-regression tests")
            pytest_args.append("--improve")

    if mock:
        pytest_args.append("--mock")

    if no_dep:
        pytest_args.append("--no_dep")
    if nc:
        pytest_args.append("--nc")

    # when used as a library, the pytest directory to execute is in the CURRENT_DIRECTORY
    pytest_args.append(str(CURRENT_DIRECTORY))

    return sys.exit(pytest.main(pytest_args))


def get_regression_data() -> Any:
    with open(REGRESSION_TESTS_PATH, "r") as file:
        data = json.load(file)

    return data


if __name__ == "__main__":
    start()
