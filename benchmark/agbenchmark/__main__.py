import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import click
import pytest
import toml
from dotenv import load_dotenv

from agbenchmark.app import app
from agbenchmark.config import load_agbenchmark_config
from agbenchmark.reports.ReportManager import SingletonReportManager
from agbenchmark.utils.logging import configure_logging
from agbenchmark.utils.path_manager import PATH_MANAGER

load_dotenv()

try:
    if os.getenv("HELICONE_API_KEY"):
        import helicone

        helicone_enabled = True
    else:
        helicone_enabled = False
except ImportError:
    helicone_enabled = False


class InvalidInvocationError(ValueError):
    pass


logger = logging.getLogger(__name__)

BENCHMARK_START_TIME_DT = datetime.now(timezone.utc)
BENCHMARK_START_TIME = BENCHMARK_START_TIME_DT.strftime("%Y-%m-%dT%H:%M:%S+00:00")


if helicone_enabled:
    from helicone.lock import HeliconeLockManager

    HeliconeLockManager.write_custom_property(
        "benchmark_start_time", BENCHMARK_START_TIME
    )

with open(
    Path(__file__).resolve().parent / "challenges" / "optional_categories.json"
) as f:
    OPTIONAL_CATEGORIES = json.load(f)["optional_categories"]


def get_unique_categories() -> set[str]:
    """
    Find all data.json files in the directory relative to this file and its
    subdirectories, read the "category" field from each file, and return a set of unique
    categories.
    """
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
                logger.error(f"Error: {data_file} is not a valid JSON file.")
                continue
            except IOError:
                logger.error(f"IOError: file could not be read: {data_file}")
                continue

    return categories


def run_benchmark(
    maintain: bool = False,
    improve: bool = False,
    explore: bool = False,
    mock: bool = False,
    no_dep: bool = False,
    no_cutoff: bool = False,
    keep_answers: bool = False,
    categories: Optional[list[str]] = None,
    skip_categories: Optional[list[str]] = None,
    test: Optional[str] = None,
    cutoff: Optional[int] = None,
    server: bool = False,
) -> int:
    """
    Starts the benchmark. If a category flag is provided, only challenges with the
    corresponding mark will be run.
    """
    validate_args(
        maintain=maintain,
        improve=improve,
        explore=explore,
        test=test,
        categories=categories,
        skip_categories=skip_categories,
        no_cutoff=no_cutoff,
        cutoff=cutoff,
    )

    initialize_updates_file()
    SingletonReportManager()
    agent_benchmark_config = load_agbenchmark_config()

    assert agent_benchmark_config.host, "Error: host needs to be added to the config."

    for key, value in vars(agent_benchmark_config).items():
        logger.debug(f"config.{key} = {repr(value)}")

    pytest_args = ["-vs"]
    if keep_answers:
        pytest_args.append("--keep-answers")

    if test:
        logger.info("Running specific test:", test)
    else:
        # Categories that are used in the challenges
        all_categories = get_unique_categories()
        if categories:
            invalid_categories = set(categories) - all_categories
            assert not invalid_categories, (
                f"Invalid categories: {invalid_categories}. "
                f"Valid categories are: {all_categories}"
            )

        if categories:
            categories_to_run = set(categories)
            if skip_categories:
                categories_to_run = categories_to_run.difference(set(skip_categories))
                assert categories_to_run, "Error: You can't skip all categories"
            pytest_args.extend(["-m", " or ".join(categories_to_run), "--category"])
            logger.info("Running tests of category:", categories_to_run)
        elif skip_categories:
            categories_to_run = all_categories - set(skip_categories)
            assert categories_to_run, "Error: You can't skip all categories"
            pytest_args.extend(["-m", " or ".join(categories_to_run), "--category"])
            logger.info("Running tests of category:", categories_to_run)
        else:
            logger.info("Running all categories")

        if maintain:
            logger.info("Running only regression tests")
            pytest_args.append("--maintain")
        elif improve:
            logger.info("Running only non-regression tests")
            pytest_args.append("--improve")
        elif explore:
            logger.info("Only attempt challenges that have never been beaten")
            pytest_args.append("--explore")

    if mock:
        pytest_args.append("--mock")
        os.environ[
            "IS_MOCK"
        ] = "True"  # ugly hack to make the mock work when calling from API

    if no_dep:
        pytest_args.append("--no_dep")

    if no_cutoff:
        pytest_args.append("--nc")
    if cutoff:
        pytest_args.append("--cutoff")
        logger.debug(f"Setting cuttoff override to {cutoff} seconds.")
    current_dir = Path(__file__).resolve().parent
    pytest_args.append(str(current_dir))

    pytest_args.append("--cache-clear")
    exit_code = pytest.main(pytest_args)

    SingletonReportManager.clear_instance()
    return exit_code


def validate_args(
    maintain: bool = False,
    improve: bool = False,
    explore: bool = False,
    test: Optional[str] = None,
    categories: Optional[list[str]] = None,
    skip_categories: Optional[list[str]] = None,
    no_cutoff: bool = False,
    cutoff: Optional[int] = None,
) -> None:
    if (maintain + improve + explore) > 1:
        raise InvalidInvocationError(
            "You can't use --maintain, --improve or --explore at the same time. "
            "Please choose one."
        )

    if test and (categories or skip_categories or maintain or improve or explore):
        raise InvalidInvocationError(
            "If you're running a specific test make sure no other options are "
            "selected. Please just pass the --test."
        )

    if no_cutoff and cutoff:
        raise InvalidInvocationError(
            "You can't use both --nc and --cutoff at the same time. "
            "Please choose one."
        )


@click.group(invoke_without_command=True)
@click.option("--debug", is_flag=True, help="Enable debug output")
@click.option("--backend", is_flag=True, help="If it's being run from the cli")
@click.option("-c", "--category", multiple=True, help="Specific category to run")
@click.option(
    "-s",
    "--skip-category",
    multiple=True,
    help="Skips preventing the tests from this category from running",
)
@click.option("--test", multiple=True, help="Specific test to run")
@click.option("--maintain", is_flag=True, help="Runs only regression tests")
@click.option("--improve", is_flag=True, help="Run only non-regression tests")
@click.option(
    "--explore",
    is_flag=True,
    help="Only attempt challenges that have never been beaten",
)
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option(
    "--no_dep",
    is_flag=True,
    help="Run without dependencies",
)
@click.option("--nc", is_flag=True, help="Run without cutoff")
@click.option("--keep-answers", is_flag=True, help="Keep answers")
@click.option("--cutoff", help="Set or override tests cutoff (seconds)")
@click.argument("value", type=str, required=False)
def cli(
    debug: bool,
    maintain: bool,
    improve: bool,
    explore: bool,
    mock: bool,
    no_dep: bool,
    nc: bool,
    keep_answers: bool,
    category: Optional[list[str]] = None,
    skip_category: Optional[list[str]] = None,
    test: Optional[str] = None,
    cutoff: Optional[int] = None,
    backend: Optional[bool] = False,
    value: Optional[str] = None,
) -> Any:
    configure_logging(logging.DEBUG if debug else logging.INFO)

    # Redirect stdout if backend is True
    if value == "start":
        raise DeprecationWarning(
            "`agbenchmark start` is removed. Run `agbenchmark` instead."
        )
    if value == "serve":
        return serve()

    try:
        validate_args(
            maintain=maintain,
            improve=improve,
            explore=explore,
            test=test,
            categories=category,
            skip_categories=skip_category,
            no_cutoff=nc,
            cutoff=cutoff,
        )
    except InvalidInvocationError as e:
        logger.error("Error: " + "\n".join(e.args))
        sys.exit(1)

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
                no_cutoff=nc,
                keep_answers=keep_answers,
                categories=category,
                skip_categories=skip_category,
                test=test,
                cutoff=cutoff,
            )

        sys.stdout = original_stdout

    else:
        exit_code = run_benchmark(
            maintain=maintain,
            improve=improve,
            explore=explore,
            mock=mock,
            no_dep=no_dep,
            no_cutoff=nc,
            keep_answers=keep_answers,
            categories=category,
            skip_categories=skip_category,
            test=test,
            cutoff=cutoff,
        )

        sys.exit(exit_code)


@cli.command()
def version():
    """Print the version of the benchmark tool."""
    current_directory = Path(__file__).resolve().parent
    version = toml.load(current_directory / ".." / "pyproject.toml")["tool"]["poetry"][
        "version"
    ]
    click.echo(f"Benchmark Tool Version {version}")


def serve():
    import uvicorn

    # Run the FastAPI application using uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)


def initialize_updates_file():
    if os.path.exists(PATH_MANAGER.updates_json_file):
        # If the file already exists, overwrite it with an empty list
        with open(PATH_MANAGER.updates_json_file, "w") as file:
            json.dump([], file, indent=2)
        logger.debug("Initialized updates.json by overwriting with an empty array")
    else:
        # If the file doesn't exist, create it and write an empty list
        with open(PATH_MANAGER.updates_json_file, "w") as file:
            json.dump([], file, indent=2)
        logger.debug("Created updates.json and initialized it with an empty array")


if __name__ == "__main__":
    cli()
