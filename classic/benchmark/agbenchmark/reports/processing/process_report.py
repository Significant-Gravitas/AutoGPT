import json
import logging
import os
from pathlib import Path
from typing import Any

from agbenchmark.reports.processing.get_files import (
    get_latest_report_from_agent_directories,
)
from agbenchmark.reports.processing.report_types import Report
from agbenchmark.utils.data_types import STRING_DIFFICULTY_MAP

logger = logging.getLogger(__name__)


def get_reports_data(report_path: str) -> dict[str, Any]:
    latest_files = get_latest_report_from_agent_directories(report_path)

    reports_data = {}

    if latest_files is None:
        raise Exception("No files found in the reports directory")

    # This will print the latest file in each s
    # ubdirectory and add to the files_data dictionary
    for subdir, file in latest_files:
        subdir_name = os.path.basename(os.path.normpath(subdir))
        with open(Path(subdir) / file, "r") as f:
            # Load the JSON data from the file
            json_data = json.load(f)
            converted_data = Report.model_validate(json_data)
            # get the last directory name in the path as key
            reports_data[subdir_name] = converted_data

    return reports_data


def get_highest_achieved_difficulty_per_category(report: Report) -> dict[str, Any]:
    categories: dict[str, Any] = {}

    for _, test_data in report.tests.items():
        for category in test_data.category:
            if category in ("interface", "iterate", "product_advisor"):
                continue
            categories.setdefault(category, 0)
            if (
                test_data.results
                and all(r.success for r in test_data.results)
                and test_data.difficulty
            ):
                num_dif = STRING_DIFFICULTY_MAP[test_data.difficulty]
                if num_dif > categories[category]:
                    categories[category] = num_dif

    return categories


def all_agent_categories(reports_data: dict[str, Any]) -> dict[str, Any]:
    all_categories: dict[str, Any] = {}

    for name, report in reports_data.items():
        categories = get_highest_achieved_difficulty_per_category(report)
        if categories:  # only add to all_categories if categories is not empty
            logger.debug(f"Adding {name}: {categories}")
            all_categories[name] = categories

    return all_categories
