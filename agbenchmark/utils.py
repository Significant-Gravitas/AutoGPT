# radio charts, logs, helper functions for tests, anything else relevant.
import glob
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agbenchmark.challenges.define_task_types import DIFFICULTY_MAP, DifficultyLevel


def calculate_info_test_path(benchmarks_folder_path: Path) -> str:
    INFO_TESTS_PATH = benchmarks_folder_path / "reports"

    if not INFO_TESTS_PATH.exists():
        INFO_TESTS_PATH.mkdir(parents=True, exist_ok=True)
        return str(
            INFO_TESTS_PATH / f"file1_{datetime.now().strftime('%m-%d-%H-%M')}.json"
        )
    else:
        json_files = glob.glob(str(INFO_TESTS_PATH / "*.json"))
        file_count = len(json_files)
        run_name = f"file{file_count + 1}_{datetime.now().strftime('%m-%d-%H-%M')}.json"
        new_file_path = INFO_TESTS_PATH / run_name
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
