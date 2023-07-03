import json
import os
import sys
from pathlib import Path
from typing import List

import click
import pytest
from dotenv import load_dotenv, set_key

load_dotenv()

CURRENT_DIRECTORY = Path(__file__).resolve().parent


CONFIG_PATH = str(Path(os.getcwd()) / "config.json")

REGRESSION_TESTS_PATH = str(Path(os.getcwd()) / "regression_tests.json")


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--category", default=None, help="Specific category to run")
@click.option("--reg", is_flag=True, help="Runs only regression tests")
@click.option("--mock", is_flag=True, help="Run with mock")
def start(category: str, reg: bool, mock: bool) -> int:
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""
    # Check if configuration file exists and is not empty
    if not os.path.exists(CONFIG_PATH) or os.stat(CONFIG_PATH).st_size == 0:
        config = {}

        config["workspace"] = click.prompt(
            "Please enter a new workspace path",
            default=os.path.join(Path.home(), "workspace"),
        )

        config["func_path"] = click.prompt(
            "Please enter a the path to your run_specific_agent function implementation",
            default="/benchmarks.py",
        )

        config["cutoff"] = click.prompt(
            "Please enter a hard cutoff runtime for your agent",
            default="60",
        )

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)
    else:
        # If the configuration file exists and is not empty, load it
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    set_key(".env", "MOCK_TEST", "True" if mock else "False")
    if mock:
        config["workspace"] = "agbenchmark/mocks/workspace"

    # create workspace directory if it doesn't exist
    workspace_path = os.path.abspath(config["workspace"])
    if not os.path.exists(workspace_path):
        os.makedirs(workspace_path, exist_ok=True)

    if not os.path.exists(REGRESSION_TESTS_PATH):
        with open(REGRESSION_TESTS_PATH, "a"):
            pass

    print("Current configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    print("Starting benchmark tests...", category)
    tests_to_run = []
    pytest_args = ["-vs"]
    if category:
        pytest_args.extend(["-m", category])
    else:
        if reg:
            print("Running all regression tests")
            tests_to_run = get_regression_tests()
        else:
            print("Running all categories")

    if mock:
        pytest_args.append("--mock")

    # Run pytest with the constructed arguments
    if not tests_to_run:
        tests_to_run = [str(CURRENT_DIRECTORY)]
    pytest_args.extend(tests_to_run)

    return sys.exit(pytest.main(pytest_args))


def get_regression_tests() -> List[str]:
    if not Path(REGRESSION_TESTS_PATH).exists():
        with open(REGRESSION_TESTS_PATH, "w") as file:
            json.dump({}, file)

    with open(REGRESSION_TESTS_PATH, "r") as file:
        data = json.load(file)

    regression_tests = [
        str(CURRENT_DIRECTORY / ".." / value["test"]) for key, value in data.items()
    ]

    return regression_tests


if __name__ == "__main__":
    start()
