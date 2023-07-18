# radio charts, logs, helper functions for tests, anything else relevant.
import glob
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from agbenchmark.challenges.define_task_types import DIFFICULTY_MAP, DifficultyLevel

AGENT_NAME = os.getenv("AGENT_NAME")
HOME_ENV = os.getenv("HOME_ENV")


def calculate_info_test_path(reports_path: Path) -> str:
    report_location = os.getenv("REPORT_LOCATION", ".")
    if report_location:
        reports_path = Path(os.getcwd()) / report_location

    command = sys.argv

    if not reports_path.exists():
        reports_path.mkdir(parents=True, exist_ok=True)

    json_files = glob.glob(str(reports_path / "*.json"))

    # Default naming scheme
    file_count = len(json_files)
    run_name = f"file{file_count + 1}_{datetime.now().strftime('%m-%d-%H-%M')}.json"

    # # If "--test" is in command
    if "--test" in command:
        test_index = command.index("--test")
        try:
            test_arg = command[test_index + 1]  # Argument after --test
        except IndexError:
            raise ValueError("Expected an argument after --test")

        # Get all files that include the string that is the argument after --test
        related_files = [f for f in json_files if test_arg in f]
        related_file_count = len(related_files)

        # Determine the prefix based on the existing files
        if related_file_count == 0:
            # Try to find the highest prefix number among all files, then increment it
            all_prefix_numbers = []
            for f in json_files:
                number = float(Path(f).stem.split("_")[0])
                all_prefix_numbers.append(math.floor(number))

            max_prefix = max(all_prefix_numbers, default=0)
            print("HEY WE ARE HERE BIG DAWG", max_prefix)
            run_name = f"{max_prefix + 1}_{test_arg}.json"
        else:
            # Take the number from before the _ and add the .{number}
            prefix_str = Path(related_files[0]).stem.rsplit("_", 1)[0].split(".")[0]
            prefix = math.floor(float(prefix_str))
            run_name = f"{prefix}.{related_file_count}_{test_arg}.json"

    print("run_namerun_namerun_name", run_name)
    new_file_path = reports_path / run_name
    return str(new_file_path)


def replace_backslash(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(
            r"\\+", "/", value
        )  # replace one or more backslashes with a forward slash
    elif isinstance(value, list):
        return [replace_backslash(i) for i in value]
    elif isinstance(value, dict):
        return {k: replace_backslash(v) for k, v in value.items()}
    else:
        return value


def calculate_success_percentage(results: list[bool]) -> float:
    # Take the last 10 results or all if less than 10
    last_results = results[-10:] if len(results) > 10 else results
    success_count = last_results.count(True)
    total_count = len(last_results)
    if total_count == 0:
        return 0
    success_percentage = (success_count / total_count) * 100  # as a percentage
    return round(success_percentage, 2)


def get_highest_success_difficulty(data: dict) -> str:
    highest_difficulty = None
    highest_difficulty_level = 0

    for test_name, test_data in data.items():
        if test_data["metrics"]["success"]:
            # Replace 'medium' with 'intermediate' for this example
            difficulty_str = test_data["metrics"]["difficulty"]

            try:
                difficulty_enum = DifficultyLevel[difficulty_str.lower()]
                difficulty_level = DIFFICULTY_MAP[difficulty_enum]

                if difficulty_level > highest_difficulty_level:
                    highest_difficulty = difficulty_enum
                    highest_difficulty_level = difficulty_level
            except KeyError:
                print(
                    f"Unexpected difficulty level '{difficulty_str}' in test '{test_name}'"
                )

    if highest_difficulty is not None:
        highest_difficulty_str = highest_difficulty.name  # convert enum to string
    else:
        highest_difficulty_str = ""

    return f"{highest_difficulty_str}: {highest_difficulty_level}"


def assign_paths(folder_path: Path) -> tuple[str, str, str]:
    CONFIG_PATH = str(folder_path / "config.json")
    REGRESSION_TESTS_PATH = str(folder_path / "regression_tests.json")

    if HOME_ENV == "ci" and AGENT_NAME:
        INFO_TESTS_PATH = calculate_info_test_path(
            Path(os.getcwd()) / "agbenchmark" / "reports" / AGENT_NAME
        )
    else:
        INFO_TESTS_PATH = calculate_info_test_path(folder_path / "reports")

    return CONFIG_PATH, REGRESSION_TESTS_PATH, INFO_TESTS_PATH


def calculate_dynamic_paths() -> tuple[Path, str, str, str]:
    # the default home is where you're running from
    HOME_DIRECTORY = Path(os.getcwd())
    benchmarks_folder_path = HOME_DIRECTORY / "agbenchmark"

    if AGENT_NAME and HOME_ENV == "ci":
        if "/Auto-GPT-Benchmarks/agent" in str(HOME_DIRECTORY):
            raise Exception("Must run from root of benchmark repo if HOME_ENV is ci")

        # however if the env is local and the agent name is defined, we want to run that agent from the repo and then get the data in the internal agbenchmark directory
        # this is for the ci/cd pipeline
        benchmarks_folder_path = HOME_DIRECTORY / "agent" / AGENT_NAME / "agbenchmark"

        CONFIG_PATH, REGRESSION_TESTS_PATH, INFO_TESTS_PATH = assign_paths(
            benchmarks_folder_path
        )

        # we want to run the agent from the submodule
        HOME_DIRECTORY = Path(os.getcwd()) / "agent" / AGENT_NAME

    elif AGENT_NAME and not os.path.join("Auto-GPT-Benchmarks", "agent") in str(
        HOME_DIRECTORY
    ):
        # if the agent name is defined but the run is not from the agent repo, then home is the agent repo
        # used for development of both a benchmark and an agent
        HOME_DIRECTORY = Path(os.getcwd()) / "agent" / AGENT_NAME
        benchmarks_folder_path = HOME_DIRECTORY / "agbenchmark"

        CONFIG_PATH, REGRESSION_TESTS_PATH, INFO_TESTS_PATH = assign_paths(
            benchmarks_folder_path
        )

    else:
        # otherwise the default is when home is an agent (running agbenchmark from agent/agent_repo)
        # used when its just a pip install
        CONFIG_PATH, REGRESSION_TESTS_PATH, INFO_TESTS_PATH = assign_paths(
            benchmarks_folder_path
        )

    if not benchmarks_folder_path.exists():
        benchmarks_folder_path.mkdir(exist_ok=True)

    return (
        HOME_DIRECTORY,
        CONFIG_PATH,
        REGRESSION_TESTS_PATH,
        INFO_TESTS_PATH,
    )
