import json
import os
import sys
from typing import Any, Dict

from agbenchmark.__main__ import CHALLENGES_ALREADY_BEATEN
from agbenchmark.reports.agent_benchmark_config import get_agent_benchmark_config
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.data_types import DifficultyLevel
from agbenchmark.utils.get_data_from_helicone import get_data_from_helicone
from agbenchmark.utils.utils import calculate_success_percentage


def get_previous_test_results(
    test_name: str, info_details: dict[str, Any]
) -> list[bool]:
    agent_tests: dict[str, list[bool]] = {}
    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv

    prev_test_results = SingletonReportManager().INTERNAL_INFO_MANAGER.tests.get(
        test_name, []
    )

    if not mock:
        # only add if it's an actual test
        prev_test_results.append(info_details["metrics"]["success"])
        SingletonReportManager().INTERNAL_INFO_MANAGER.add_test(
            test_name, prev_test_results
        )

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
        SingletonReportManager().REGRESSION_MANAGER.add_test(test_name, test_details)


def generate_single_call_report(
    item: Any,
    call: Any,
    challenge_data: dict[str, Any],
    answers: dict[str, Any],
    challenge_location,
    test_name,
) -> None:
    try:
        difficulty = challenge_data["info"]["difficulty"]
    except KeyError:
        return None

    if isinstance(difficulty, DifficultyLevel):
        difficulty = difficulty.value

    # Extract the challenge_location from the class
    # challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
    # test_name = item.nodeid.split("::")[1]
    # item.test_name = test_name

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
        # "answers": answers,
    }
    if answers:
        info_details["answers"] = answers

    if "metadata" in challenge_data:
        info_details["metadata"] = challenge_data["metadata"]

    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv
    if call:
        if call.excinfo is None:
            info_details["metrics"]["success"] = True
        else:
            if not mock:  # don't remove if it's a mock test
                SingletonReportManager().REGRESSION_MANAGER.remove_test(test_name)
            info_details["metrics"]["fail_reason"] = str(call.excinfo.value)
            if call.excinfo.typename == "Skipped":
                info_details["metrics"]["attempted"] = False

    prev_test_results: list[bool] = get_previous_test_results(test_name, info_details)

    update_regression_tests(prev_test_results, info_details, test_name, test_details)

    # user facing reporting
    if item:
        item.info_details = info_details

    return info_details


def finalize_reports(item: Any, challenge_data: dict[str, Any]) -> None:
    run_time = dict(item.user_properties).get("run_time")

    info_details = getattr(item, "info_details", {})
    test_name = getattr(item, "test_name", "")

    if info_details and test_name:
        if run_time is not None:
            cost = None
            if "--mock" not in sys.argv and os.environ.get("HELICONE_API_KEY"):
                print("Getting cost from Helicone")
                cost = get_data_from_helicone(test_name)

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

            if "--mock" not in sys.argv:
                update_challenges_already_beaten(info_details, test_name)
                if info_details.get("tests") is not None:
                    for nested_test_name, nested_test_info in info_details[
                        "tests"
                    ].items():
                        update_challenges_already_beaten(
                            nested_test_info, nested_test_name
                        )

        SingletonReportManager().INFO_MANAGER.add_test(test_name, info_details)


def update_challenges_already_beaten(
    info_details: Dict[str, Any], test_name: str
) -> None:
    current_run_successful = info_details["metrics"]["success"]
    try:
        with open(CHALLENGES_ALREADY_BEATEN, "r") as f:
            challenge_data = json.load(f)
    except:
        challenge_data = {}
    challenge_beaten_in_the_past = challenge_data.get(test_name)

    challenge_data[test_name] = True
    if challenge_beaten_in_the_past is None and not current_run_successful:
        challenge_data[test_name] = False

    with open(CHALLENGES_ALREADY_BEATEN, "w") as f:
        json.dump(challenge_data, f, indent=4)


def session_finish(suite_reports: dict) -> None:
    agent_benchmark_config = get_agent_benchmark_config()

    SingletonReportManager().INTERNAL_INFO_MANAGER.save()
    SingletonReportManager().INFO_MANAGER.end_info_report(agent_benchmark_config)
    SingletonReportManager().REGRESSION_MANAGER.save()
