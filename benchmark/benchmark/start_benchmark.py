import glob
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import pytest
from helicone.lock import HeliconeLockManager
import sys
sys.path.append('/Users/swifty/dev/Auto-GPT/benchmark')

from agbenchmark.reports.ReportManager import ReportManager
from agbenchmark.utils.utils import (
    AGENT_NAME,
    calculate_dynamic_paths,
    # get_git_commit_sha,
)

CURRENT_DIRECTORY = Path(__file__).resolve().parent
BENCHMARK_START_TIME = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")
if os.environ.get("HELICONE_API_KEY"):
    HeliconeLockManager.write_custom_property(
        "benchmark_start_time", BENCHMARK_START_TIME
    )

(
    HOME_DIRECTORY,
    CONFIG_PATH,
    REGRESSION_TESTS_PATH,
    REPORTS_PATH,
    SUCCESS_RATE_PATH,
    CHALLENGES_PATH,
) = calculate_dynamic_paths()
BENCHMARK_GIT_COMMIT_SHA = "---" # get_git_commit_sha(HOME_DIRECTORY / ".." / "..")
AGENT_GIT_COMMIT_SHA = "---" # get_git_commit_sha(HOME_DIRECTORY)
# open a file in the challenges/optional_categories
with open(
    Path(__file__).resolve().parent / "challenges" / "optional_categories.json"
) as f:
    OPTIONAL_CATEGORIES = json.load(f)["optional_categories"]


def get_unique_categories() -> set[str]:
    """Find all data.json files in the directory relative to this file and its subdirectories,
    read the "category" field from each file, and return a set of unique categories."""
    categories = set()

    # Get the directory of this file
    this_dir = os.path.dirname(os.path.abspath(__file__))

    glob_path = os.path.join(this_dir, "./challenges/**/data.json")
    # Use it as the base for the glob pattern
    for data_file in glob.glob(glob_path, recursive=True):
        with open(data_file, "r") as f:
            try:
                data = json.load(f)
                categories.update(data.get("category", []))
            except json.JSONDecodeError:
                print(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                print(f"IOError: file could not be read: {data_file}")
                continue

    return categories


def get_report_managers() -> tuple[ReportManager, ReportManager, ReportManager]:
    # tests that consistently pass are considered regression tests
    REGRESSION_MANAGER = ReportManager(REGRESSION_TESTS_PATH)

    # print(f"Using {REPORTS_PATH} for reports")
    # user facing reporting information
    INFO_MANAGER = ReportManager(str(Path(REPORTS_PATH) / "report.json"))

    # internal db step in replacement track pass/fail rate
    INTERNAL_INFO_MANAGER = ReportManager(SUCCESS_RATE_PATH)

    return REGRESSION_MANAGER, INFO_MANAGER, INTERNAL_INFO_MANAGER


(REGRESSION_MANAGER, INFO_MANAGER, INTERNAL_INFO_MANAGER) = get_report_managers()


def run_benchmark(
    maintain: bool = False,
    improve: bool = False,
    explore: bool = False,
    mock: bool = False,
    no_dep: bool = False,
    nc: bool = False,
    category: Optional[list[str]] = None,
    skip_category: Optional[list[str]] = None,
    test: Optional[str] = None,
    suite: Optional[str] = None,
    cutoff: Optional[int] = None,
    api_mode: bool = False,
    host: Optional[str] = None,
) -> int:
    """Start the benchmark tests. If a category flag is provided, run the categories with that mark."""
    # Check if configuration file exists and is not empty

    if maintain and improve and explore:
        print(
            "Error: You can't use --maintain, --improve or --explore at the same time. Please choose one."
        )
        return 1

    if test and (category or skip_category or maintain or improve or suite or explore):
        print(
            "Error: If you're running a specific test make sure no other options are selected. Please just pass the --test."
        )
        return 1

    # TODO: test and ensure that this functionality works before removing
    # change elif suite below if removing
    if suite and (category or skip_category or maintain or improve or explore):
        print(
            "Error: If you're running a specific suite make sure no other options are selected. Please just pass the --suite."
        )
        return 1

    if os.path.join("Auto-GPT-Benchmarks") in str(HOME_DIRECTORY) and not AGENT_NAME:
        print(
            "If you are running from the Auto-GPT-Benchmarks repo, you must have AGENT_NAME defined."
        )
        return 1

    if os.path.exists(CONFIG_PATH) and os.stat(CONFIG_PATH).st_size:
        # If the configuration file exists and is not empty, load it
        with open(CONFIG_PATH, "r") as f:
            config = json.load(f)
    else:
        config = {}
    host = host or config.get("host")
    api_mode = api_mode or config.get("api_mode")
    if host:
        config["host"] = host
    if api_mode:
        config["api_mode"] = api_mode
    print("benchmark run path", CONFIG_PATH, HOME_DIRECTORY)
    if not config.get("workspace"):
        config["workspace"] = click.prompt(
            "Please enter a new workspace path",
            default=os.path.join("workspace"),
            show_default=True,
        )

    if api_mode and not host:
        config["host"] = click.prompt(
            "Please enter the Agent API host address",
            default="http://localhost:8000",
            show_default=True,
        )

    with open(CONFIG_PATH, "w") as f:
        json.dump(config, f)

    print("Current configuration:")
    for key, value in config.items():
        print(f"{key}: {value}")

    pytest_args = ["-vs"]
    if test:
        print("Running specific test:", test)
        pytest_args.extend(["-k", test, "--test"])
    elif suite:
        print("Running specific suite:", suite)
        pytest_args.extend(["--suite"])
    else:
        # Categories that are used in the challenges
        categories = get_unique_categories()
        if category:
            invalid_categories = set(category) - categories
            assert (
                not invalid_categories
            ), f"Invalid categories: {invalid_categories}. Valid categories are: {categories}"

        if category:
            categories_to_run = set(category)
            if skip_category:
                categories_to_run = categories_to_run.difference(set(skip_category))
                assert categories_to_run, "Error: You can't skip all categories"
            pytest_args.extend(["-m", " or ".join(categories_to_run), "--category"])
            print("Running tests of category:", categories_to_run)
        elif skip_category:
            categories_to_run = categories - set(skip_category)
            assert categories_to_run, "Error: You can't skip all categories"
            pytest_args.extend(["-m", " or ".join(categories_to_run), "--category"])
            print("Running tests of category:", categories_to_run)
        else:
            print("Running all categories")

        if maintain:
            print("Running only regression tests")
            pytest_args.append("--maintain")
        elif improve:
            print("Running only non-regression tests")
            pytest_args.append("--improve")
        elif explore:
            print("Only attempt challenges that have never been beaten")
            pytest_args.append("--explore")
    if host:
        pytest_args.append(f"--host={host}")
    if api_mode:
        pytest_args.append("--api_mode")
    if mock:
        pytest_args.append("--mock")

    if no_dep:
        pytest_args.append("--no_dep")

    if nc and cutoff:
        print(
            "Error: You can't use both --nc and --cutoff at the same time. Please choose one."
        )
        return 1

    if nc:
        pytest_args.append("--nc")
    if cutoff:
        pytest_args.append("--cutoff")
        print(f"Setting cuttoff override to {cutoff} seconds.")

    pytest_args.extend((str(CURRENT_DIRECTORY), "--cache-clear"))
    pytest_args.append("--disable-warnings")

    return pytest.main(pytest_args)


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--backend", is_flag=True, help="If it's being run from the cli")
@click.option("-c", "--category", multiple=True, help="Specific category to run")
@click.option(
    "-s",
    "--skip-category",
    multiple=True,
    help="Skips preventing the tests from this category from running",
)
@click.option("--test", help="Specific test to run")
@click.option("--maintain", is_flag=True, help="Runs only regression tests")
@click.option("--improve", is_flag=True, help="Run only non-regression tests")
@click.option(
    "--explore",
    is_flag=True,
    help="Only attempt challenges that have never been beaten",
)
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option("--suite", help="Run a suite of related tests")
@click.option(
    "--no_dep",
    is_flag=True,
    help="Run without dependencies (can be useful for a suite run)",
)
@click.option("--nc", is_flag=True, help="Run without cutoff")
@click.option("--cutoff", help="Set or override tests cutoff (seconds)")
@click.option("--api_mode", help="API mode")
@click.option("--host", help="Define API host")
def start(
    maintain: bool,
    improve: bool,
    explore: bool,
    mock: bool,
    no_dep: bool,
    nc: bool,
    category: Optional[list[str]] = None,
    skip_category: Optional[list[str]] = None,
    test: Optional[str] = None,
    suite: Optional[str] = None,
    cutoff: Optional[int] = None,
    backend: Optional[bool] = False,
    api_mode: bool = False,
    host: Optional[str] = None,
) -> Any:
    # Redirect stdout if backend is True
    original_stdout = sys.stdout  # Save the original standard output
    exit_code = None

    if backend:
        with open("backend/backend_stdout.txt", "w") as f:
            sys.stdout = f
            exit_code = run_benchmark(
                maintain=maintain,
                improve=improve,
                explore=explore,
                mock=mock,
                no_dep=no_dep,
                nc=nc,
                category=category,
                skip_category=skip_category,
                test=test,
                suite=suite,
                cutoff=cutoff,
                api_mode=api_mode,
                host=host,
            )

        sys.stdout = original_stdout

        with open(Path(REPORTS_PATH) / "report.json", "r") as file:
            latest_report = json.load(file)

        print(latest_report)

    else:
        exit_code = run_benchmark(
            maintain=maintain,
            improve=improve,
            explore=explore,
            mock=mock,
            no_dep=no_dep,
            nc=nc,
            category=category,
            skip_category=skip_category,
            test=test,
            suite=suite,
            cutoff=cutoff,
        )

        sys.exit(exit_code)


def get_regression_data() -> Any:
    with open(REGRESSION_TESTS_PATH, "r") as file:
        data = json.load(file)

    return data

@cli.command()
def version():
    """Print the version of the benchmark tool."""
    import toml
    version = toml.load(CURRENT_DIRECTORY / ".." / "pyproject.toml")["tool"]["poetry"]["version"]
    print(f"Benchmark Tool Version {version}")



# def run_from_backend(
#     maintain: bool = False,
#     improve: bool = False,
#     explore: bool = False,
#     mock: bool = False,
#     no_dep: bool = False,
#     nc: bool = False,
#     category: Optional[list[str]] = None,
#     skip_category: Optional[list[str]] = None,
#     test: Optional[str] = None,
#     suite: Optional[str] = None,
#     cutoff: Optional[int] = None,
# ) -> Any:
#     global HOME_DIRECTORY, CONFIG_PATH, REGRESSION_TESTS_PATH, REPORTS_PATH, SUCCESS_RATE_PATH, CHALLENGES_PATH
#     global REGRESSION_MANAGER, INFO_MANAGER, INTERNAL_INFO_MANAGER

#     if INFO_MANAGER.tests != {}:
#         (
#             HOME_DIRECTORY,
#             CONFIG_PATH,
#             REGRESSION_TESTS_PATH,
#             REPORTS_PATH,
#             SUCCESS_RATE_PATH,
#             CHALLENGES_PATH,
#         ) = calculate_dynamic_paths()

#         (
#             REGRESSION_MANAGER,
#             INFO_MANAGER,
#             INTERNAL_INFO_MANAGER,
#         ) = get_report_managers()

#     sys.argv = ["run_benchmark"]

#     if maintain:
#         sys.argv.append("--maintain")
#     if improve:
#         sys.argv.append("--improve")
#     if explore:
#         sys.argv.append("--explore")
#     if mock:
#         sys.argv.append("--mock")
#     if no_dep:
#         sys.argv.append("--no_dep")
#     if nc:
#         sys.argv.append("--nc")

#     if category:
#         for cat in category:
#             sys.argv.extend(["-c", cat])

#     if skip_category:
#         for skip_cat in skip_category:
#             sys.argv.extend(["-s", skip_cat])

#     if test:
#         sys.argv.extend(["--test", test])

#     if suite:
#         sys.argv.extend(["--suite", suite])

#     if cutoff is not None:
#         sys.argv.extend(["--cutoff", str(cutoff)])

#     exit_code = run_benchmark(
#         maintain=maintain,
#         improve=improve,
#         explore=explore,
#         mock=mock,
#         no_dep=no_dep,
#         nc=nc,
#         category=category,
#         skip_category=skip_category,
#         test=test,
#         suite=suite,
#         cutoff=cutoff,
#     )

#     if exit_code != 0:
#         return f"pytest failed with exit code: {exit_code}"

#     with open(Path(REPORTS_PATH) / "report.json", "r") as file:
#         latest_report = json.load(file)

#     return latest_report


<<<<<<< HEAD:benchmark/agbenchmark/start_benchmark.py
# if __name__ == "__main__":
# start()
=======
if __name__ == "__main__":
    cli()
>>>>>>> 61b4afcb (Fixing benchmarks):benchmark/benchmark/start_benchmark.py
