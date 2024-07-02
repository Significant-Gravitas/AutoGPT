import copy
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.reports.processing.graphs import save_single_radar_chart
from agbenchmark.reports.processing.process_report import (
    get_highest_achieved_difficulty_per_category,
)
from agbenchmark.reports.processing.report_types import MetricsOverall, Report, Test
from agbenchmark.utils.utils import get_highest_success_difficulty

logger = logging.getLogger(__name__)


class SingletonReportManager:
    instance = None

    INFO_MANAGER: "SessionReportManager"
    REGRESSION_MANAGER: "RegressionTestsTracker"
    SUCCESS_RATE_TRACKER: "SuccessRatesTracker"

    def __new__(cls):
        if not cls.instance:
            cls.instance = super(SingletonReportManager, cls).__new__(cls)

            agent_benchmark_config = AgentBenchmarkConfig.load()
            benchmark_start_time_dt = datetime.now(
                timezone.utc
            )  # or any logic to fetch the datetime

            # Make the Managers class attributes
            cls.INFO_MANAGER = SessionReportManager(
                agent_benchmark_config.get_report_dir(benchmark_start_time_dt)
                / "report.json",
                benchmark_start_time_dt,
            )
            cls.REGRESSION_MANAGER = RegressionTestsTracker(
                agent_benchmark_config.regression_tests_file
            )
            cls.SUCCESS_RATE_TRACKER = SuccessRatesTracker(
                agent_benchmark_config.success_rate_file
            )

        return cls.instance

    @classmethod
    def clear_instance(cls):
        cls.instance = None
        del cls.INFO_MANAGER
        del cls.REGRESSION_MANAGER
        del cls.SUCCESS_RATE_TRACKER


class BaseReportManager:
    """Abstracts interaction with the regression tests file"""

    tests: dict[str, Any]

    def __init__(self, report_file: Path):
        self.report_file = report_file

        self.load()

    def load(self) -> None:
        if not self.report_file.exists():
            self.report_file.parent.mkdir(exist_ok=True)

        try:
            with self.report_file.open("r") as f:
                data = json.load(f)
                self.tests = {k: data[k] for k in sorted(data)}
        except FileNotFoundError:
            self.tests = {}
        except json.decoder.JSONDecodeError as e:
            logger.warning(f"Could not parse {self.report_file}: {e}")
            self.tests = {}

    def save(self) -> None:
        with self.report_file.open("w") as f:
            json.dump(self.tests, f, indent=4)

    def remove_test(self, test_name: str) -> None:
        if test_name in self.tests:
            del self.tests[test_name]
            self.save()

    def reset(self) -> None:
        self.tests = {}
        self.save()


class SessionReportManager(BaseReportManager):
    """Abstracts interaction with the regression tests file"""

    tests: dict[str, Test]
    report: Report | None = None

    def __init__(self, report_file: Path, benchmark_start_time: datetime):
        super().__init__(report_file)

        self.start_time = time.time()
        self.benchmark_start_time = benchmark_start_time

    def save(self) -> None:
        with self.report_file.open("w") as f:
            if self.report:
                f.write(self.report.model_dump_json(indent=4))
            else:
                json.dump(
                    {k: v.model_dump() for k, v in self.tests.items()}, f, indent=4
                )

    def load(self) -> None:
        super().load()

        if "tests" in self.tests:
            self.report = Report.model_validate(self.tests)
        else:
            self.tests = {n: Test.model_validate(d) for n, d in self.tests.items()}

    def add_test_report(self, test_name: str, test_report: Test) -> None:
        if self.report:
            raise RuntimeError("Session report already finalized")

        if test_name.startswith("Test"):
            test_name = test_name[4:]
        self.tests[test_name] = test_report

        self.save()

    def finalize_session_report(self, config: AgentBenchmarkConfig) -> None:
        command = " ".join(sys.argv)

        if self.report:
            raise RuntimeError("Session report already finalized")

        self.report = Report(
            command=command.split(os.sep)[-1],
            benchmark_git_commit_sha="---",
            agent_git_commit_sha="---",
            completion_time=datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            benchmark_start_time=self.benchmark_start_time.strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            metrics=MetricsOverall(
                run_time=str(round(time.time() - self.start_time, 2)) + " seconds",
                highest_difficulty=get_highest_success_difficulty(self.tests),
                total_cost=self.get_total_costs(),
            ),
            tests=copy.copy(self.tests),
            config=config.model_dump(exclude={"reports_folder"}, exclude_none=True),
        )

        agent_categories = get_highest_achieved_difficulty_per_category(self.report)
        if len(agent_categories) > 1:
            save_single_radar_chart(
                agent_categories,
                config.get_report_dir(self.benchmark_start_time) / "radar_chart.png",
            )

        self.save()

    def get_total_costs(self):
        if self.report:
            tests = self.report.tests
        else:
            tests = self.tests

        total_cost = 0
        all_costs_none = True
        for test_data in tests.values():
            cost = sum(r.cost or 0 for r in test_data.results)

            if cost is not None:  # check if cost is not None
                all_costs_none = False
                total_cost += cost  # add cost to total
        if all_costs_none:
            total_cost = None
        return total_cost


class RegressionTestsTracker(BaseReportManager):
    """Abstracts interaction with the regression tests file"""

    tests: dict[str, dict]

    def add_test(self, test_name: str, test_details: dict) -> None:
        if test_name.startswith("Test"):
            test_name = test_name[4:]

        self.tests[test_name] = test_details
        self.save()

    def has_regression_test(self, test_name: str) -> bool:
        return self.tests.get(test_name) is not None


class SuccessRatesTracker(BaseReportManager):
    """Abstracts interaction with the regression tests file"""

    tests: dict[str, list[bool | None]]

    def update(self, test_name: str, success_history: list[bool | None]) -> None:
        if test_name.startswith("Test"):
            test_name = test_name[4:]

        self.tests[test_name] = success_history
        self.save()
