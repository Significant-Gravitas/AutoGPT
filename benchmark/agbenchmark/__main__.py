import glob
import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence

import click
from click_default_group import DefaultGroup
from dotenv import load_dotenv

from agbenchmark.config import AgentBenchmarkConfig
from agbenchmark.utils.logging import configure_logging

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
    config: AgentBenchmarkConfig,
    maintain: bool = False,
    improve: bool = False,
    explore: bool = False,
    tests: tuple[str] = tuple(),
    categories: tuple[str] = tuple(),
    skip_categories: tuple[str] = tuple(),
    mock: bool = False,
    no_dep: bool = False,
    no_cutoff: bool = False,
    cutoff: Optional[int] = None,
    keep_answers: bool = False,
    server: bool = False,
) -> int:
    """
    Starts the benchmark. If a category flag is provided, only challenges with the
    corresponding mark will be run.
    """
    import pytest

    from agbenchmark.reports.ReportManager import SingletonReportManager

    validate_args(
        maintain=maintain,
        improve=improve,
        explore=explore,
        tests=tests,
        categories=categories,
        skip_categories=skip_categories,
        no_cutoff=no_cutoff,
        cutoff=cutoff,
    )

    initialize_updates_file(config)
    SingletonReportManager()

    assert config.host, "Error: host needs to be added to the config."

    for key, value in vars(config).items():
        logger.debug(f"config.{key} = {repr(value)}")

    pytest_args = ["-vs"]

    if tests:
        logger.info(f"Running specific test(s): {' '.join(tests)}")
        pytest_args += [f"--test={t}" for t in tests]
    else:
        # Categories that are used in the challenges
        all_categories = get_unique_categories()
        if categories:
            invalid_categories = set(categories) - all_categories
            assert not invalid_categories, (
                f"Invalid categories: {invalid_categories}. "
                f"Valid categories are: {all_categories}"
            )

        if categories or skip_categories:
            categories_to_run = set(categories) or all_categories
            if skip_categories:
                categories_to_run = categories_to_run.difference(set(skip_categories))
            assert categories_to_run, "Error: You can't skip all categories"
            pytest_args += [f"--category={c}" for c in categories_to_run]
            logger.info(f"Running tests of category: {categories_to_run}")
        else:
            logger.info("Running all categories")

        if maintain:
            logger.info("Running only regression tests")
        elif improve:
            logger.info("Running only non-regression tests")
        elif explore:
            logger.info("Only attempt challenges that have never been beaten")

    if mock:
        # TODO: unhack
        os.environ[
            "IS_MOCK"
        ] = "True"  # ugly hack to make the mock work when calling from API

    # Pass through flags
    for flag, active in {
        "--maintain": maintain,
        "--improve": improve,
        "--explore": explore,
        "--no-dep": no_dep,
        "--mock": mock,
        "--nc": no_cutoff,
        "--keep-answers": keep_answers,
    }.items():
        if active:
            pytest_args.append(flag)

    if cutoff:
        pytest_args.append(f"--cutoff={cutoff}")
        logger.debug(f"Setting cuttoff override to {cutoff} seconds.")

    current_dir = Path(__file__).resolve().parent
    pytest_args.append(str(current_dir / "generate_test.py"))

    pytest_args.append("--cache-clear")
    exit_code = pytest.main(pytest_args)

    SingletonReportManager.clear_instance()
    return exit_code


def validate_args(
    maintain: bool,
    improve: bool,
    explore: bool,
    tests: Sequence[str],
    categories: Sequence[str],
    skip_categories: Sequence[str],
    no_cutoff: bool,
    cutoff: Optional[int],
) -> None:
    if (maintain + improve + explore) > 1:
        raise InvalidInvocationError(
            "You can't use --maintain, --improve or --explore at the same time. "
            "Please choose one."
        )

    if tests and (categories or skip_categories or maintain or improve or explore):
        raise InvalidInvocationError(
            "If you're running a specific test make sure no other options are "
            "selected. Please just pass the --test."
        )

    if no_cutoff and cutoff:
        raise InvalidInvocationError(
            "You can't use both --nc and --cutoff at the same time. "
            "Please choose one."
        )


@click.group(cls=DefaultGroup, default_if_no_args=True)
@click.option("--debug", is_flag=True, help="Enable debug output")
def cli(
    debug: bool,
) -> Any:
    configure_logging(logging.DEBUG if debug else logging.INFO)


@cli.command(hidden=True)
def start():
    raise DeprecationWarning(
        "`agbenchmark start` is deprecated. Use `agbenchmark run` instead."
    )


@cli.command(default=True)
@click.option(
    "-c",
    "--category",
    multiple=True,
    help="(+) Select a category to run.",
)
@click.option(
    "-s",
    "--skip-category",
    multiple=True,
    help="(+) Exclude a category from running.",
)
@click.option("--test", multiple=True, help="(+) Select a test to run.")
@click.option("--maintain", is_flag=True, help="Run only regression tests.")
@click.option("--improve", is_flag=True, help="Run only non-regression tests.")
@click.option(
    "--explore",
    is_flag=True,
    help="Run only challenges that have never been beaten.",
)
@click.option(
    "--no-dep",
    is_flag=True,
    help="Run all (selected) challenges, regardless of dependency success/failure.",
)
@click.option("--cutoff", type=int, help="Override the challenge time limit (seconds).")
@click.option("--nc", is_flag=True, help="Disable the challenge time limit.")
@click.option("--mock", is_flag=True, help="Run with mock")
@click.option("--keep-answers", is_flag=True, help="Keep answers")
@click.option(
    "--backend",
    is_flag=True,
    help="Write log output to a file instead of the terminal.",
)
# @click.argument(
#     "agent_path", type=click.Path(exists=True, file_okay=False), required=False
# )
def run(
    maintain: bool,
    improve: bool,
    explore: bool,
    mock: bool,
    no_dep: bool,
    nc: bool,
    keep_answers: bool,
    test: tuple[str],
    category: tuple[str],
    skip_category: tuple[str],
    cutoff: Optional[int] = None,
    backend: Optional[bool] = False,
    # agent_path: Optional[Path] = None,
) -> None:
    """
    Run the benchmark on the agent in the current directory.

    Options marked with (+) can be specified multiple times, to select multiple items.
    """
    agbenchmark_config = AgentBenchmarkConfig.load()
    logger.debug(f"agbenchmark_config: {agbenchmark_config.agbenchmark_config_path}")
    try:
        validate_args(
            maintain=maintain,
            improve=improve,
            explore=explore,
            tests=test,
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
                config=agbenchmark_config,
                maintain=maintain,
                improve=improve,
                explore=explore,
                mock=mock,
                no_dep=no_dep,
                no_cutoff=nc,
                keep_answers=keep_answers,
                tests=test,
                categories=category,
                skip_categories=skip_category,
                cutoff=cutoff,
            )

        sys.stdout = original_stdout

    else:
        exit_code = run_benchmark(
            config=agbenchmark_config,
            maintain=maintain,
            improve=improve,
            explore=explore,
            mock=mock,
            no_dep=no_dep,
            no_cutoff=nc,
            keep_answers=keep_answers,
            tests=test,
            categories=category,
            skip_categories=skip_category,
            cutoff=cutoff,
        )

        sys.exit(exit_code)


@cli.command()
@click.option("--port", type=int, help="Port to run the API on.")
def serve(port: Optional[int] = None):
    """Serve the benchmark frontend and API on port 8080."""
    import uvicorn

    from agbenchmark.app import setup_fastapi_app

    config = AgentBenchmarkConfig.load()
    app = setup_fastapi_app(config)

    # Run the FastAPI application using uvicorn
    port = port or int(os.getenv("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)


@cli.command()
def config():
    """Displays info regarding the present AGBenchmark config."""
    try:
        config = AgentBenchmarkConfig.load()
    except FileNotFoundError as e:
        click.echo(e, err=True)
        return 1

    k_col_width = max(len(k) for k in config.dict().keys())
    for k, v in config.dict().items():
        click.echo(f"{k: <{k_col_width}} = {v}")


@cli.command()
def version():
    """Print version info for the AGBenchmark application."""
    import toml

    package_root = Path(__file__).resolve().parent.parent
    pyproject = toml.load(package_root / "pyproject.toml")
    version = pyproject["tool"]["poetry"]["version"]
    click.echo(f"AGBenchmark version {version}")


def initialize_updates_file(config: AgentBenchmarkConfig):
    if os.path.exists(config.updates_json_file):
        # If the file already exists, overwrite it with an empty list
        with open(config.updates_json_file, "w") as file:
            json.dump([], file, indent=2)
        logger.debug("Initialized updates.json by overwriting with an empty array")
    else:
        # If the file doesn't exist, create it and write an empty list
        with open(config.updates_json_file, "w") as file:
            json.dump([], file, indent=2)
        logger.debug("Created updates.json and initialized it with an empty array")


if __name__ == "__main__":
    cli()
