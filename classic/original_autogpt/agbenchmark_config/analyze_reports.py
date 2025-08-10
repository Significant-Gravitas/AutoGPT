#!/usr/bin/env python3

import json
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path

from tabulate import tabulate

info = "-v" in sys.argv
debug = "-vv" in sys.argv
granular = "--granular" in sys.argv

logging.basicConfig(
    level=logging.DEBUG if debug else logging.INFO if info else logging.WARNING
)
logger = logging.getLogger(__name__)

# Get a list of all JSON files in the directory
report_files = [
    report_file
    for dir in (Path(__file__).parent / "reports").iterdir()
    if re.match(r"^\d{8}T\d{6}_", dir.name)
    and (report_file := dir / "report.json").is_file()
]

labels = list[str]()
runs_per_label = defaultdict[str, int](lambda: 0)
suite_names = list[str]()
test_names = list[str]()

# Create a dictionary to store grouped success values by suffix and test
grouped_success_values = defaultdict[str, list[str]](list[str])

# Loop through each JSON file to collect suffixes and success values
for report_file in sorted(report_files):
    with open(report_file) as f:
        logger.info(f"Loading {report_file}...")

        data = json.load(f)
        if "tests" in data:
            test_tree = data["tests"]
            label = data["agent_git_commit_sha"].rsplit("/", 1)[1][:7]  # commit hash
        else:
            # Benchmark run still in progress
            test_tree = data
            label = report_file.parent.name.split("_", 1)[1]
            logger.info(f"Run '{label}' seems to be in progress")

        runs_per_label[label] += 1

        def process_test(test_name: str, test_data: dict):
            result_group = grouped_success_values[f"{label}|{test_name}"]

            if "tests" in test_data:
                logger.debug(f"{test_name} is a test suite")

                # Test suite
                suite_attempted = any(
                    test["metrics"]["attempted"] for test in test_data["tests"].values()
                )
                logger.debug(f"suite_attempted: {suite_attempted}")
                if not suite_attempted:
                    return

                if test_name not in test_names:
                    test_names.append(test_name)

                if test_data["metrics"]["percentage"] == 0:
                    result_indicator = "❌"
                else:
                    highest_difficulty = test_data["metrics"]["highest_difficulty"]
                    result_indicator = {
                        "interface": "🔌",
                        "novice": "🌑",
                        "basic": "🌒",
                        "intermediate": "🌓",
                        "advanced": "🌔",
                        "hard": "🌕",
                    }[highest_difficulty]

                logger.debug(f"result group: {result_group}")
                logger.debug(f"runs_per_label: {runs_per_label[label]}")
                if len(result_group) + 1 < runs_per_label[label]:
                    result_group.extend(
                        ["❔"] * (runs_per_label[label] - len(result_group) - 1)
                    )
                result_group.append(result_indicator)
                logger.debug(f"result group (after): {result_group}")

                if granular:
                    for test_name, test in test_data["tests"].items():
                        process_test(test_name, test)
                return

            test_metrics = test_data["metrics"]
            result_indicator = "❔"

            if "attempted" not in test_metrics:
                return
            elif test_metrics["attempted"]:
                if test_name not in test_names:
                    test_names.append(test_name)

                success_value = test_metrics["success"]
                result_indicator = {True: "✅", False: "❌"}[success_value]

            if len(result_group) + 1 < runs_per_label[label]:
                result_group.extend(
                    ["  "] * (runs_per_label[label] - len(result_group) - 1)
                )
            result_group.append(result_indicator)

        for test_name, suite in test_tree.items():
            try:
                process_test(test_name, suite)
            except KeyError:
                print(f"{test_name}.metrics: {suite['metrics']}")
                raise

    if label not in labels:
        labels.append(label)

# Create headers
headers = ["Test Name"] + list(labels)

# Prepare data for tabulation
table_data = list[list[str]]()
for test_name in test_names:
    row = [test_name]
    for label in labels:
        results = grouped_success_values.get(f"{label}|{test_name}", ["❔"])
        if len(results) < runs_per_label[label]:
            results.extend(["❔"] * (runs_per_label[label] - len(results)))
        if len(results) > 1 and all(r == "❔" for r in results):
            results.clear()
        row.append(" ".join(results))
    table_data.append(row)

# Print tabulated data
print(tabulate(table_data, headers=headers, tablefmt="grid"))
