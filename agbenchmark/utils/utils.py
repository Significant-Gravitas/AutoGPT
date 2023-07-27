# radio charts, logs, helper functions for tests, anything else relevant.
import glob
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

load_dotenv()

from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel

AGENT_NAME = os.getenv("AGENT_NAME")
HOME_ENV = os.getenv("HOME_ENV")
report_location = os.getenv("REPORT_LOCATION", None)


def calculate_info_test_path(reports_path: Path) -> str:
    if report_location:
        reports_path = Path(os.getcwd()) / report_location

    command = sys.argv

    if not reports_path.exists():
        reports_path.mkdir(parents=True, exist_ok=True)

    json_files = glob.glob(str(reports_path / "*.json"))

    # Default naming scheme
    file_count = len(json_files)
    run_name = f"file{file_count + 1}_{datetime.now().strftime('%m-%d-%H-%M')}.json"

    test_index = None
    test_arg = None
    if "--test" in command:
        test_index = command.index("--test")
    elif "--suite" in command:
        test_index = command.index("--suite")
    elif "--category" in command:
        test_index = command.index("--category")
    elif "--maintain" in command:
        test_index = command.index("--maintain")
        test_arg = "maintain"
    elif "--improve" in command:
        test_index = command.index("--improve")
        test_arg = "improve"

    # # If "--test" is in command
    if test_index:
        if not test_arg:
            test_arg = command[test_index + 1]  # Argument after --

        # Try to find the highest prefix number among all files, then increment it
        all_prefix_numbers = []
        # count related files and assign the correct file number
        related_files = []
        prefix_number = 0.0

        # Get all files that include the string that is the argument after --test
        for file in json_files:
            file_name = Path(file).name.rsplit(".", 1)[0]
            file_parts = file_name.split("_")
            try:
                if "file" in file_parts[0]:
                    # default files are called file{num}
                    number = float(file_parts[0][4:]) + 1
                else:
                    number = float(file_parts[0]) + 1
            except:
                number = file_count + 1
            test_name = "_".join(file_parts[1:])
            all_prefix_numbers.append(math.floor(number))
            if test_arg == test_name:
                prefix_number = number
                related_files.append(test_name)

        related_file_count = len(related_files)

        # Determine the prefix based on the existing files
        if related_file_count == 0:
            max_prefix = max(all_prefix_numbers, default=0)
            run_name = f"{max_prefix + 1}_{test_arg}.json"
        else:
            print(f"Found {related_file_count} files with '{test_arg}' in the name")
            # Take the number from before the _ and add the .{number}

            prefix = 0
            prefix = math.floor(prefix_number)

            run_name = f"{prefix}.{related_file_count}_{test_arg}.json"

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


def get_test_path(json_file: str | Path) -> str:
    if isinstance(json_file, str):
        json_file = Path(json_file)

    # Find the index of "agbenchmark" in the path parts
    try:
        agbenchmark_index = json_file.parts.index("agbenchmark")
    except ValueError:
        raise ValueError("Invalid challenge location.")

    # Create the path from "agbenchmark" onwards
    challenge_location = Path(*json_file.parts[agbenchmark_index:])

    formatted_location = replace_backslash(str(challenge_location))
    if isinstance(formatted_location, str):
        return formatted_location
    else:
        return str(challenge_location)


def get_highest_success_difficulty(
    data: dict, just_string: Optional[bool] = None
) -> str:
    highest_difficulty = None
    highest_difficulty_level = 0

    for test_name, test_data in data.items():
        try:
            if test_data.get("tests", None):
                highest_difficulty_str = test_data["metrics"]["highest_difficulty"]
                try:
                    highest_difficulty = DifficultyLevel[highest_difficulty_str]
                    highest_difficulty_level = DIFFICULTY_MAP[highest_difficulty]
                except KeyError:
                    print(
                        f"Unexpected difficulty level '{highest_difficulty_str}' in test '{test_name}'"
                    )
                    continue
            else:
                if test_data["metrics"]["success"]:
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
                        continue
        except Exception:
            print(f"Make sure you selected the right test, no reports were generated.")
            break

    if highest_difficulty is not None:
        highest_difficulty_str = highest_difficulty.name  # convert enum to string
    else:
        highest_difficulty_str = ""

    if highest_difficulty_level and not just_string:
        return f"{highest_difficulty_str}: {highest_difficulty_level}"
    elif highest_difficulty_str:
        return highest_difficulty_str
    return "No successful tests"


def assign_paths(folder_path: Path) -> tuple[str, str, str]:
    CONFIG_PATH = str(folder_path / "config.json")
    REGRESSION_TESTS_PATH = str(folder_path / "regression_tests.json")

    INFO_TESTS_PATH = calculate_info_test_path(folder_path / "reports")

    return CONFIG_PATH, REGRESSION_TESTS_PATH, INFO_TESTS_PATH


def calculate_dynamic_paths() -> tuple[Path, str, str, str]:
    # the default home is where you're running from
    HOME_DIRECTORY = Path(os.getcwd())
    benchmarks_folder_path = HOME_DIRECTORY / "agbenchmark"

    if AGENT_NAME and not os.path.join("Auto-GPT-Benchmarks", "agent") in str(
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
