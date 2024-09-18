import logging
import os
from pathlib import Path
from typing import Optional, Sequence

from dotenv import load_dotenv

from agbenchmark.challenges import get_unique_categories
from agbenchmark.config import AgentBenchmarkConfig

load_dotenv()

logger = logging.getLogger(__name__)


def run_benchmark(
    config: AgentBenchmarkConfig,
    maintain: bool = False,
    improve: bool = False,
    explore: bool = False,
    tests: tuple[str, ...] = tuple(),
    categories: tuple[str, ...] = tuple(),
    skip_categories: tuple[str, ...] = tuple(),
    attempts_per_challenge: int = 1,
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

    SingletonReportManager()

    for key, value in vars(config).items():
        logger.debug(f"config.{key} = {repr(value)}")

    pytest_args = ["-vs"]

    if tests:
        logger.info(f"Running specific test(s): {' '.join(tests)}")
        pytest_args += [f"--test={t}" for t in tests]
    else:
        all_categories = get_unique_categories()

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

    if attempts_per_challenge > 1:
        pytest_args.append(f"--attempts={attempts_per_challenge}")

    if cutoff:
        pytest_args.append(f"--cutoff={cutoff}")
        logger.debug(f"Setting cuttoff override to {cutoff} seconds.")

    current_dir = Path(__file__).resolve().parent
    pytest_args.append(str(current_dir / "generate_test.py"))

    pytest_args.append("--cache-clear")
    logger.debug(f"Running Pytest with args: {pytest_args}")
    exit_code = pytest.main(pytest_args)

    SingletonReportManager.clear_instance()
    return exit_code


class InvalidInvocationError(ValueError):
    pass


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
    if categories:
        all_categories = get_unique_categories()
        invalid_categories = set(categories) - all_categories
        if invalid_categories:
            raise InvalidInvocationError(
                "One or more invalid categories were specified: "
                f"{', '.join(invalid_categories)}.\n"
                f"Valid categories are: {', '.join(all_categories)}."
            )

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
