import json
import logging
import os
import sys
from pathlib import Path

import pytest

from agbenchmark.challenges import ChallengeInfo
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types import Metrics, Test
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.data_types import DifficultyLevel
from agbenchmark.utils.utils import calculate_success_percentage

# from agbenchmark.utils.get_data_from_helicone import get_data_from_helicone

logger = logging.getLogger(__name__)


def get_and_update_success_history(test_name: str, info_details: Test) -> list[bool]:
    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv

    prev_test_results = SingletonReportManager().SUCCESS_RATE_TRACKER.tests.get(
        test_name, []
    )

    if not mock and info_details.metrics.success is not None:
        # only add if it's an actual test
        prev_test_results.append(info_details.metrics.success)
        SingletonReportManager().SUCCESS_RATE_TRACKER.update(
            test_name, prev_test_results
        )

    # can calculate success rate regardless of mock
    info_details.metrics.success_percentage = calculate_success_percentage(
        prev_test_results
    )

    return prev_test_results


def update_regression_tests(
    prev_test_results: list[bool],
    info_details: Test,
    test_name: str,
) -> None:
    if len(prev_test_results) >= 3 and prev_test_results[-3:] == [True, True, True]:
        # if the last 3 tests were successful, add to the regression tests
        info_details.is_regression = True
        SingletonReportManager().REGRESSION_MANAGER.add_test(
            test_name, info_details.dict(include={"difficulty", "data_path"})
        )


def initialize_test_report(
    item: pytest.Item,
    challenge_info: ChallengeInfo,
):
    difficulty = challenge_info.difficulty
    if isinstance(difficulty, DifficultyLevel):
        difficulty = difficulty.value

    # Extract the challenge_location from the class
    # challenge_location: str = getattr(item.cls, "CHALLENGE_LOCATION", "")
    # test_name = item.nodeid.split("::")[1]
    # item.test_name = test_name

    test_info = dict(item.user_properties).get("info_details") or Test(
        data_path=challenge_info.source_uri,
        is_regression=False,
        category=[c.value for c in challenge_info.category],
        task=challenge_info.task,
        answer=challenge_info.reference_answer or "",
        description=challenge_info.description or "",
        metrics=Metrics(
            difficulty=difficulty,
            attempted=False,
        ),
    )

    # user facing reporting
    if item:
        item.user_properties.append(("info_details", test_info))

    return test_info


def finalize_test_report(
    item: pytest.Item, call: pytest.CallInfo, config: AgentBenchmarkConfig
) -> None:
    user_properties: dict = dict(item.user_properties)

    info_details: Test = user_properties.get("info_details", {})
    test_name: str = user_properties.get("test_name", "")

    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv

    logger.debug(f"Finalizing report with CallInfo: {vars(call)}")
    if call.excinfo is None:
        info_details.metrics.success = True
    else:
        if not mock:  # don't remove if it's a mock test
            SingletonReportManager().REGRESSION_MANAGER.remove_test(test_name)
        info_details.metrics.fail_reason = str(call.excinfo.value)
        if call.excinfo.typename == "Skipped":
            info_details.metrics.attempted = False
    info_details.metrics.attempted = True
    info_details.metrics.run_time = f"{str(round(call.duration, 3))} seconds"
    info_details.reached_cutoff = user_properties.get("timed_out", False)

    prev_test_results: list[bool] = get_and_update_success_history(
        test_name, info_details
    )

    update_regression_tests(prev_test_results, info_details, test_name)

    if info_details and test_name:
        # if "--mock" not in sys.argv and os.environ.get("HELICONE_API_KEY"):
        #     logger.debug("Getting cost from Helicone")
        #     info_details.metrics.cost = get_data_from_helicone(test_name)
        #     logger.debug(f"Cost: {cost}")

        if "--mock" not in sys.argv:
            update_challenges_already_beaten(
                config.challenges_already_beaten_file, info_details, test_name
            )

        SingletonReportManager().INFO_MANAGER.add_test_report(test_name, info_details)


def update_challenges_already_beaten(
    challenges_already_beaten_file: Path, info_details: Test, test_name: str
) -> None:
    current_run_successful = info_details.metrics.success
    try:
        with open(challenges_already_beaten_file, "r") as f:
            challenge_data = json.load(f)
    except FileNotFoundError:
        challenge_data = {}
    challenge_beaten_in_the_past = challenge_data.get(test_name)

    challenge_data[test_name] = True
    if challenge_beaten_in_the_past is None and not current_run_successful:
        challenge_data[test_name] = False

    with open(challenges_already_beaten_file, "w") as f:
        json.dump(challenge_data, f, indent=4)


def session_finish(agbenchmark_config: AgentBenchmarkConfig) -> None:
    SingletonReportManager().INFO_MANAGER.finalize_session_report(agbenchmark_config)
    SingletonReportManager().REGRESSION_MANAGER.save()
    SingletonReportManager().SUCCESS_RATE_TRACKER.save()
