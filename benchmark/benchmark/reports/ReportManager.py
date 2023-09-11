import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

from benchmark.reports.processing.graphs import save_single_radar_chart
from benchmark.reports.processing.process_report import get_agent_category
from benchmark.reports.processing.report_types import Report
from benchmark.utils.utils import get_highest_success_difficulty
from benchmark.utils.data_types import AgentBenchmarkConfig
from benchmark.__main__ import BENCHMARK_START_TIME

class ReportManager:
    """Abstracts interaction with the regression tests file"""

    def __init__(self, filename: str):
        self.filename = filename
        self.start_time = time.time()
        self.load()

    def load(self) -> None:
        if not os.path.exists(self.filename):
            os.makedirs(os.path.dirname(self.filename), exist_ok=True)
            with open(self.filename, 'w') as f:
                pass
        
        try:
            with open(self.filename, "r") as f:
                file_content = (
                    f.read().strip()
                )  # read the content and remove any leading/trailing whitespace
                if file_content:  # if file is not empty, load the json
                    data = json.loads(file_content)
                    self.tests = {k: data[k] for k in sorted(data)}
                else:  # if file is empty, assign an empty dictionary
                    self.tests = {}
        except FileNotFoundError:
            self.tests = {}
        except json.decoder.JSONDecodeError:  # If JSON is invalid
            self.tests = {}
        self.save()

    def save(self) -> None:
        with open(self.filename, "w") as f:
            json.dump(self.tests, f, indent=4)

    def add_test(self, test_name: str, test_details: dict | list) -> None:
        self.tests[test_name] = test_details

        self.save()

    def remove_test(self, test_name: str) -> None:
        if test_name in self.tests:
            del self.tests[test_name]
            self.save()

    def reset(self) -> None:
        self.tests = {}
        self.save()

    def end_info_report(self, config: AgentBenchmarkConfig) -> None:

        command = " ".join(sys.argv)

        self.tests = {
            "command": command.split(os.sep)[-1],
            "benchmark_git_commit_sha": '---',
            "agent_git_commit_sha": '---',
            "completion_time": datetime.now(timezone.utc).strftime(
                "%Y-%m-%dT%H:%M:%S+00:00"
            ),
            "benchmark_start_time": BENCHMARK_START_TIME,
            "metrics": {
                "run_time": str(round(time.time() - self.start_time, 2)) + " seconds",
                "highest_difficulty": get_highest_success_difficulty(self.tests),
                "total_cost": self.get_total_costs(),
            },
            "tests": self.tests,
            "config": {k: v for k, v in json.loads(config.json()).items() if v is not None},
        }

        converted_data = Report.parse_obj(self.tests)

        agent_categories = get_agent_category(converted_data)

        save_single_radar_chart(
            agent_categories,
            
            config.get_reports_path() / "radar_chart.png",
        )

        self.save()

    def get_total_costs(self):
        total_cost = 0
        all_costs_none = True
        for test_name, test_data in self.tests.items():
            cost = test_data["metrics"].get(
                "cost", 0
            )  # gets the cost or defaults to 0 if cost is missing
            if cost is not None:  # check if cost is not None
                all_costs_none = False
                total_cost += cost  # add cost to total
        if all_costs_none:
            total_cost = None
        return total_cost
