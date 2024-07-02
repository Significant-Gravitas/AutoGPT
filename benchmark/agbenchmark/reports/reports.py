import json
import logging
import os
from pathlib import Path

import pytest
from pydantic import ValidationError

from agbenchmark.challenges import ChallengeInfo
from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.report_types import Test, TestMetrics, TestResult
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.data_types import DifficultyLevel

# from agbenchmark.utils.get_data_from_helicone import get_data_from_helicone

logger = logging.getLogger(__name__)


def get_and_update_success_history(
    test_name: str, success: bool | None
) -> list[bool | None]:
    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv

    prev_test_results = SingletonReportManager().SUCCESS_RATE_TRACKER.tests.get(
        test_name, []
    )

    if not mock:
        # only add if it's an actual test
        prev_test_results.append(success)
        SingletonReportManager().SUCCESS_RATE_TRACKER.update(
            test_name, prev_test_results
        )

    return prev_test_results


def update_regression_tests(
    prev_test_results: list[bool | None],
    test_report: Test,
    test_name: str,
) -> None:
    if len(prev_test_results) >= 3 and prev_test_results[-3:] == [True, True, True]:
        # if the last 3 tests were successful, add to the regression tests
        test_report.metrics.is_regression = True
        SingletonReportManager().REGRESSION_MANAGER.add_test(
            test_name, test_report.model_dump(include={"difficulty", "data_path"})
        )


def make_empty_test_report(
    challenge_info: ChallengeInfo,
) -> Test:
    difficulty = challenge_info.difficulty
    if isinstance(difficulty, DifficultyLevel):
        difficulty = difficulty.value

    return Test(
        category=[c.value for c in challenge_info.category],
        difficulty=difficulty,
        data_path=challenge_info.source_uri,
        description=challenge_info.description or "",
        task=challenge_info.task,
        answer=challenge_info.reference_answer or "",
        metrics=TestMetrics(attempted=False, is_regression=False),
        results=[],
    )


def add_test_result_to_report(
    test_report: Test,
    item: pytest.Item,
    call: pytest.CallInfo,
    config: AgentBenchmarkConfig,
) -> None:
    user_properties: dict = dict(item.user_properties)
    test_name: str = user_properties.get("test_name", "")

    mock = os.getenv("IS_MOCK")  # Check if --mock is in sys.argv

    if call.excinfo:
        if not mock:
            SingletonReportManager().REGRESSION_MANAGER.remove_test(test_name)

        test_report.metrics.attempted = call.excinfo.typename != "Skipped"
    else:
        test_report.metrics.attempted = True

    try:
        test_report.results.append(
            TestResult(
                success=call.excinfo is None,
                run_time=f"{str(round(call.duration, 3))} seconds",
                fail_reason=(
                    str(call.excinfo.value) if call.excinfo is not None else None
                ),
                reached_cutoff=user_properties.get("timed_out", False),
                n_steps=user_properties.get("n_steps"),
                steps=user_properties.get("steps", []),
                cost=user_properties.get("agent_task_cost"),
            )
        )
        test_report.metrics.success_percentage = (
            sum(r.success or False for r in test_report.results)
            / len(test_report.results)
            * 100
        )
    except ValidationError:
        if call.excinfo:
            logger.error(
                "Validation failed on TestResult; "
                f"call.excinfo = {repr(call.excinfo)};\n{call.excinfo.getrepr()})"
            )
        raise

    prev_test_results: list[bool | None] = get_and_update_success_history(
        test_name, test_report.results[-1].success
    )

    update_regression_tests(prev_test_results, test_report, test_name)

    if test_report and test_name:
        # if "--mock" not in sys.argv and os.environ.get("HELICONE_API_KEY"):
        #     logger.debug("Getting cost from Helicone")
        #     test_report.metrics.cost = get_data_from_helicone(test_name)
        #     logger.debug(f"Cost: {cost}")

        if not mock:
            update_challenges_already_beaten(
                config.challenges_already_beaten_file, test_report, test_name
            )

        SingletonReportManager().INFO_MANAGER.add_test_report(test_name, test_report)


def update_challenges_already_beaten(
    challenges_already_beaten_file: Path, test_report: Test, test_name: str
) -> None:
    current_run_successful = any(r.success for r in test_report.results)
    try:
        with open(challenges_already_beaten_file, "r") as f:
            challenges_beaten_before = json.load(f)
    except FileNotFoundError:
        challenges_beaten_before = {}

    has_ever_been_beaten = challenges_beaten_before.get(test_name)
    challenges_beaten_before[test_name] = has_ever_been_beaten or current_run_successful

    with open(challenges_already_beaten_file, "w") as f:
        json.dump(challenges_beaten_before, f, indent=4)


def session_finish(agbenchmark_config: AgentBenchmarkConfig) -> None:
    SingletonReportManager().INFO_MANAGER.finalize_session_report(agbenchmark_config)
    SingletonReportManager().REGRESSION_MANAGER.save()
    SingletonReportManager().SUCCESS_RATE_TRACKER.save()
