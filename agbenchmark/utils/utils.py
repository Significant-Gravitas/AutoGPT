# radio charts, logs, helper functions for tests, anything else relevant.
import math
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional

import git
from dotenv import load_dotenv

load_dotenv()

from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel

AGENT_NAME = os.getenv("AGENT_NAME")
REPORT_LOCATION = os.getenv("REPORT_LOCATION", None)


def calculate_info_test_path(reports_path: Path) -> str:
    command = sys.argv

    if not reports_path.exists():
        reports_path.mkdir(parents=True, exist_ok=True)

    # Collect all directories in reports_path
    all_items = os.listdir(str(reports_path))
    dirs = [item for item in all_items if os.path.isdir(reports_path / item)]

    # Default naming scheme
    dir_count = len(dirs)
    run_name = f"folder{dir_count + 1}_{datetime.now().strftime('%m-%d-%H-%M')}"

    test_index = None
    test_arg = None
    # Check command-line arguments
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
    elif "--improve" in command:
        test_index = command.index("--explore")
        test_arg = "explore"

    if test_index:
        if not test_arg:
            test_arg = command[test_index + 1]

        # collect related directories and their prefix numbers
        related_dirs = []
        prefix_numbers = []

        for dir in dirs:
            dir_parts = dir.split("_")
            try:
                prefix = float(dir_parts[0][6:])
                test_name = "_".join(dir_parts[1:])
                if test_arg == test_name:
                    prefix_numbers.append(math.floor(prefix))
                    related_dirs.append(test_name)
            except:
                pass

        related_dir_count = len(related_dirs)

        if related_dir_count == 0:
            max_prefix = max(prefix_numbers, default=0)
            run_name = f"folder{max_prefix + 1}_{test_arg}"
        else:
            print(
                f"Found {related_dir_count} directories with '{test_arg}' in the name"
            )
            prefix = max(prefix_numbers)
            run_name = f"folder{prefix}.{related_dir_count}_{test_arg}"

    new_dir_path = reports_path / run_name

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    return str(new_dir_path)


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


def assign_paths(folder_path: Path) -> tuple[str, str, str, str, str]:
    CONFIG_PATH = str(folder_path / "config.json")

    reports_location = folder_path / "reports"

    # if the user has a locally defined challenges path that they've added tests to
    CHALLENGES_PATH = str(folder_path / "challenges")
    if not os.path.exists(CHALLENGES_PATH):
        CHALLENGES_PATH = str(Path(__file__).parent.parent / "challenges")

    if not os.path.exists(reports_location):
        os.makedirs(reports_location)

    # from the ci
    if REPORT_LOCATION:
        reports_location = Path.cwd() / REPORT_LOCATION

    REPORTS_PATH = calculate_info_test_path(reports_location)

    REGRESSION_TESTS_PATH = str(reports_location / "regression_tests.json")

    SUCCESS_RATE_PATH = str(reports_location / "success_rate.json")

    return (
        CONFIG_PATH,
        REGRESSION_TESTS_PATH,
        REPORTS_PATH,
        SUCCESS_RATE_PATH,
        CHALLENGES_PATH,
    )


def calculate_dynamic_paths() -> tuple[Path, str, str, str, str, str]:
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

        (
            CONFIG_PATH,
            REGRESSION_TESTS_PATH,
            REPORTS_PATH,
            SUCCESS_RATE_PATH,
            CHALLENGES_PATH,
        ) = assign_paths(benchmarks_folder_path)
    else:
        # otherwise the default is when home is an agent (running agbenchmark from agent/agent_repo)
        # used when its just a pip install
        (
            CONFIG_PATH,
            REGRESSION_TESTS_PATH,
            REPORTS_PATH,
            SUCCESS_RATE_PATH,
            CHALLENGES_PATH,
        ) = assign_paths(benchmarks_folder_path)

    if not benchmarks_folder_path.exists():
        benchmarks_folder_path.mkdir(exist_ok=True)

    if not os.path.exists(benchmarks_folder_path / "reports"):
        os.makedirs(benchmarks_folder_path / "reports")

    if not os.path.exists(REGRESSION_TESTS_PATH):
        with open(REGRESSION_TESTS_PATH, "w"):
            pass

    if not os.path.exists(SUCCESS_RATE_PATH):
        with open(SUCCESS_RATE_PATH, "w"):
            pass

    if not os.path.exists(Path(REPORTS_PATH) / "report.json"):
        with open(Path(REPORTS_PATH) / "report.json", "w"):
            pass

    return (
        HOME_DIRECTORY,
        CONFIG_PATH,
        REGRESSION_TESTS_PATH,
        REPORTS_PATH,
        SUCCESS_RATE_PATH,
        CHALLENGES_PATH,
    )


def get_git_commit_sha(directory: Path) -> Optional[str]:
    try:
        repo = git.Repo(directory)
        remote_url = repo.remotes.origin.url
        if remote_url.endswith(".git"):
            remote_url = remote_url[:-4]
        git_commit_sha = f"{remote_url}/tree/{repo.head.commit.hexsha}"

        print(f"GIT_COMMIT_SHA: {git_commit_sha}")
        return git_commit_sha
    except Exception:
        print(f"{directory} is not a git repository!")
        return None


def agent_eligibible_for_optional_categories(
    optional_challenge_categories: List, agent_categories: List
) -> bool:
    for element in optional_challenge_categories:
        if element not in agent_categories:
            return False
    return True
