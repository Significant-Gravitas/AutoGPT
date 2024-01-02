# radio charts, logs, helper functions for tests, anything else relevant.
import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv

from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel

load_dotenv()

AGENT_NAME = os.getenv("AGENT_NAME")
REPORT_LOCATION = os.getenv("REPORT_LOCATION", None)

logger = logging.getLogger(__name__)


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
        agbenchmark_index = json_file.parts.index("benchmark")
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
                    logger.warning(
                        f"Unexpected difficulty level '{highest_difficulty_str}' "
                        f"in test '{test_name}'"
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
                        logger.warning(
                            f"Unexpected difficulty level '{difficulty_str}' "
                            f"in test '{test_name}'"
                        )
                        continue
        except Exception as e:
            logger.warning(
                "An unexpected error [1] occurred while analyzing report [2]."
                "Please notify a maintainer.\n"
                f"Report data [1]: {data}\n"
                f"Error [2]: {e}"
            )
            logger.warning(
                "Make sure you selected the right test, no reports were generated."
            )
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


# def get_git_commit_sha(directory: Path) -> Optional[str]:
#     try:
#         repo = git.Repo(directory)
#         remote_url = repo.remotes.origin.url
#         if remote_url.endswith(".git"):
#             remote_url = remote_url[:-4]
#         git_commit_sha = f"{remote_url}/tree/{repo.head.commit.hexsha}"

#         # logger.debug(f"GIT_COMMIT_SHA: {git_commit_sha}")
#         return git_commit_sha
#     except Exception:
#         # logger.error(f"{directory} is not a git repository!")
#         return None


def write_pretty_json(data, json_file):
    sorted_data = deep_sort(data)
    json_graph = json.dumps(sorted_data, indent=4)
    with open(json_file, "w") as f:
        f.write(json_graph)
        f.write("\n")


def deep_sort(obj):
    """
    Recursively sort the keys in JSON object
    """
    if isinstance(obj, dict):
        return {k: deep_sort(v) for k, v in sorted(obj.items())}
    if isinstance(obj, list):
        return [deep_sort(elem) for elem in obj]
    return obj
