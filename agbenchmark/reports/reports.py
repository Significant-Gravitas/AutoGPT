import json
import os
import sys
from pathlib import Path
from typing import Any

from agbenchmark.agent_interface import MOCK_FLAG
from agbenchmark.reports.ReportManager import ReportManager
from agbenchmark.start_benchmark import (
    CONFIG_PATH,
    REGRESSION_TESTS_PATH,
    REPORTS_PATH,
    SUCCESS_RATE_PATH,
)
from agbenchmark.utils.data_types import DIFFICULTY_MAP, DifficultyLevel, SuiteConfig
from agbenchmark.utils.get_data_from_helicone import get_data_from_helicone
from agbenchmark.utils.utils import (
    calculate_success_percentage,
    get_highest_success_difficulty,
    get_test_path,
    replace_backslash,
)

# tests that consistently pass are considered regression tests
regression_manager = ReportManager(REGRESSION_TESTS_PATH)

# user facing reporting information
info_manager = ReportManager(str(Path(REPORTS_PATH) / "report.json"))

# internal db step in replacement track pass/fail rate
internal_info = ReportManager(SUCCESS_RATE_PATH)


def generate_combined_suite_report(
    item: Any, challenge_data: dict, challenge_location: str
) -> None:
    root_path = Path(__file__).parent.parent.parent
    suite_config = SuiteConfig.deserialize(
        root_path / Path(challenge_location) / "suite.json"
    )
    item.test_name = suite_config.prefix

    data_paths = suite_config.get_data_paths(root_path / Path(challenge_location))
    scores = getattr(item, "scores", {})
    mock = "--mock" in sys.argv  # Check if --mock is in sys.argv

    tests = {}
    num_highest_difficulty: int = 0
    str_highest_difficulty: str = "No successful tests"
    for i, test_name in enumerate(challenge_data["ground"]):
        raw_difficulty = challenge_data["info"][test_name]["difficulty"]
        test_details = {
            "difficulty": raw_difficulty.value,
            "data_path": challenge_location,
        }

        test_info_details = {
            "data_path": replace_backslash(data_paths[i]),
            "is_regression": False,
            "category": challenge_data["category"],
            "answer": challenge_data["ground"][test_name]["answer"],
            "description": challenge_data["info"][test_name]["description"],
            "metrics": {
                "difficulty": raw_difficulty.value,
                "success": False,
                "attempted": False,
            },
        }

        if 1 in scores.get("scores_obj", {}).get(test_name, []):
            # add dependency successful here

            test_info_details["metrics"]["success"] = True
            test_info_details["metrics"]["attempted"] = True

            # replace the highest difficulty if needed
            if DIFFICULTY_MAP[raw_difficulty] > num_highest_difficulty:
                num_highest_difficulty = DIFFICULTY_MAP[raw_difficulty]
                str_highest_difficulty = raw_difficulty.value
        else:
            # add dependency fail here

            if not mock:  # don't remove if it's a mock test
                regression_manager.remove_test(test_name)

        prev_test_results: list[bool] = get_previous_test_results(
            test_name, test_info_details
        )

        update_regression_tests(
            prev_test_results, test_info_details, test_name, test_details
        )

        tests[test_name] = test_info_details

    info_details: Any = {
        "data_path": challenge_location,
        "task": challenge_data["task"],
        "category": suite_config.shared_category,
        "metrics": {
            "percentage": scores.get("percentage", 0),
            "highest_difficulty": str_highest_difficulty,
        },
        "tests": tests,
    }

    # user facing reporting
    item.info_details = info_details


def get_previous_test_results(
    test_name: str, info_details: dict[str, Any]
) -> list[bool]:
    agent_tests: dict[str, list[bool]] = {}
    mock = "--mock" in sys.argv  # Check if --mock is in sys.argv

    prev_test_results = internal_info.tests.get(test_name, [])

    if not mock:
        # only add if it's an actual test
        prev_test_results.append(info_details["metrics"]["success"])
        internal_info.add_test(test_name, prev_test_results)

    # can calculate success rate regardless of mock
    info_details["metrics"]["success_%"] = calculate_success_percentage(
        prev_test_results
    )

    return prev_test_results


def update_regression_tests(
    prev_test_results: list[bool],
    info_details: dict,
    test_name: str,
    test_details: dict,
) -> None:
    if len(prev_test_results) >= 3 and prev_test_results[-3:] == [True, True, True]:
        # if the last 3 tests were successful, add to the regression tests
        info_details["is_regression"] = True
        regression_manager.add_test(test_name, test_details)


def generate_single_call_report(
    item: Any, call: Any, challenge_data: dict[str, Any]
) -> None:
    difficulty = challenge_data["info"]["difficulty"]

    if isinstance(difficulty, DifficultyLevel):
        difficulty = difficulty.value

    # Extract the challenge_location from the class
    challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
    test_name = item.nodeid.split("::")[1]
    item.test_name = test_name

    test_details = {
        "difficulty": difficulty,
        "data_path": challenge_location,
    }

    info_details: Any = {
        "data_path": challenge_location,
        "is_regression": False,
        "category": challenge_data["category"],
        "task": challenge_data["task"],
        "answer": challenge_data["ground"]["answer"],
        "description": challenge_data["info"]["description"],
        "metrics": {
            "difficulty": difficulty,
            "success": False,
            "attempted": True,
        },
    }

    mock = "--mock" in sys.argv  # Check if --mock is in sys.argv

    if call.excinfo is None:
        info_details["metrics"]["success"] = True
    else:
        if not mock:  # don't remove if it's a mock test
            regression_manager.remove_test(test_name)
        info_details["metrics"]["fail_reason"] = str(call.excinfo.value)
        if call.excinfo.typename == "Skipped":
            info_details["metrics"]["attempted"] = False

    prev_test_results: list[bool] = get_previous_test_results(test_name, info_details)

    update_regression_tests(prev_test_results, info_details, test_name, test_details)

    # user facing reporting
    item.info_details = info_details


def finalize_reports(item: Any, challenge_data: dict[str, Any]) -> None:
    run_time = dict(item.user_properties).get("run_time")

    info_details = getattr(item, "info_details", {})
    test_name = getattr(item, "test_name", "")

    if info_details and test_name:
        if run_time:
            cost = None
            if not MOCK_FLAG and os.environ.get("HELICONE_API_KEY"):
                print("Getting cost from Helicone")
                cost = get_data_from_helicone(test_name)
            else:
                print("Helicone not setup or mock flag set, not getting cost")

            info_details["metrics"]["cost"] = cost

            if info_details["metrics"].get("success", None) is None:
                info_details["metrics"]["attempted"] = False
                info_details["metrics"]["success"] = False
            elif (
                info_details["metrics"].get("success") is False
                and "attempted" not in info_details["metrics"]
            ):
                info_details["metrics"]["attempted"] = False

            info_details["metrics"]["run_time"] = f"{str(round(run_time, 3))} seconds"

            info_details["reached_cutoff"] = float(run_time) > challenge_data["cutoff"]

        info_manager.add_test(test_name, info_details)


def generate_separate_suite_reports(suite_reports: dict) -> None:
    for prefix, suite_file_datum in suite_reports.items():
        successes = []
        run_time = 0.0
        data = {}

        info_details: Any = {
            "data_path": "",
            "metrics": {
                "percentage": 0,
                "highest_difficulty": "",
                "run_time": "0 seconds",
            },
            "tests": {},
        }

        for name in suite_file_datum:
            test_data = info_manager.tests[name]  # get the individual test reports
            data[name] = test_data  # this is for calculating highest difficulty
            info_manager.remove_test(name)

            successes.append(test_data["metrics"]["success"])
            run_time += float(test_data["metrics"]["run_time"].split(" ")[0])

            info_details["tests"][name] = test_data

        info_details["metrics"]["percentage"] = round(
            (sum(successes) / len(successes)) * 100, 2
        )
        info_details["metrics"]["run_time"] = f"{str(round(run_time, 3))} seconds"
        info_details["metrics"]["highest_difficulty"] = get_highest_success_difficulty(
            data, just_string=True
        )
        suite_path = (
            Path(next(iter(data.values()))["data_path"]).resolve().parent.parent
        )
        info_details["data_path"] = get_test_path(suite_path)
        info_manager.add_test(prefix, info_details)


def session_finish(suite_reports: dict) -> None:
    flags = "--test" in sys.argv or "--maintain" in sys.argv or "--improve" in sys.argv
    if not flags:
        generate_separate_suite_reports(suite_reports)

    with open(CONFIG_PATH, "r") as f:
        config = json.load(f)

    internal_info.save()
    info_manager.end_info_report(config)
    regression_manager.save()
