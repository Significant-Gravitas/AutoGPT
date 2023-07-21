import json
import os
import sys
from pathlib import Path
from typing import Any

import click
import pytest

from agbenchmark.utils import calculate_dynamic_paths

CURRENT_DIRECTORY = Path(__file__).resolve().parent

(
    HOME_DIRECTORY,
    CONFIG_PATH,
    REGRESSION_TESTS_PATH,
    INFO_TESTS_PATH,
) = calculate_dynamic_paths()


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--category", default=None, help="Specific category to run")
@click.option("--test", default=None, help="Specific test to run")
@click.option("--maintain", is_flag=True, help="Runs only regression tests")
@click.option("--improve", is_flag=True, help="Run only non-regression tests")
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option("--nc", is_flag=True, help="Run without cutoff")
def start(
    category: str, test: str, maintain: bool, improve: bool, mock: bool, nc: bool
) -> int:
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""
    # Check if configuration file exists and is not empty

    if maintain and improve:
        print(
            "Error: You can't use both --maintain and --improve at the same time. Please choose one."
        )
        return 1

    if test and (category or maintain or improve):
        print(
            "Error: If you're running a specific test make sure no other options are selected. Please just pass the --test."
        )
        return 1

    print(CONFIG_PATH, os.path.exists(CONFIG_PATH), os.stat(CONFIG_PATH).st_size)
    if not os.path.exists(CONFIG_PATH) or os.stat(CONFIG_PATH).st_size == 0:
        config = {}

        config["workspace"] = click.prompt(
            "Please enter a new workspace path",
            default=os.path.join(Path.home(), "workspace"),
        )

        config["entry_path"] = click.prompt(
            "Please enter a the path to your run_specific_agent function implementation within the benchmarks folder",
            default="agbenchmark/benchmarks.py",
        )

        config["cutoff"] = click.prompt(
            "Please enter a hard cutoff runtime for your agent per test",
            default="60",
        )

        with open(CONFIG_PATH, "w") as f:
            json.dump(config, f)
    else:
        # If the configuration file exists and is not empty, load it
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)

    os.environ["MOCK_TEST"] = "True" if mock else "False"

    if not os.path.exists(REGRESSION_TESTS_PATH):
        with open(REGRESSION_TESTS_PATH, "w"):
            pass

    if not os.path.exists(INFO_TESTS_PATH):
        with open(INFO_TESTS_PATH, "w"):
            pass

    print("Current configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    pytest_args = ["-vs"]
    if test:
        print("Running specific test:", test)
        pytest_args.extend(["-k", test, "--test"])
    else:
        if category:
            pytest_args.extend(["-m", category])
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
