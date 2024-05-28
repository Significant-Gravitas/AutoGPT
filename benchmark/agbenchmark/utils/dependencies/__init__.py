"""
A module that provides the pytest hooks for this plugin.

The logic itself is in main.py.
"""

import warnings
from typing import Any, Callable, Optional

import pytest
from _pytest.config.argparsing import OptionGroup, Parser
from _pytest.nodes import Item

from .main import DependencyManager

managers: list[DependencyManager] = []


DEPENDENCY_PROBLEM_ACTIONS: dict[str, Callable[[str], None] | None] = {
    "run": None,
    "skip": lambda m: pytest.skip(m),
    "fail": lambda m: pytest.fail(m, False),
    "warning": lambda m: warnings.warn(m),
}


def _add_ini_and_option(
    parser: Any,
    group: OptionGroup,
    name: str,
    help: str,
    default: str | bool | int,
    **kwargs: Any,
) -> None:
    """
    Add an option to both the ini file and the command line flags.
    Command line flags/options takes precedence over the ini config.
    """
    parser.addini(
        name,
        help + " This overrides the similarly named option from the config.",
        default=default,
    )
    group.addoption(f'--{name.replace("_", "-")}', help=help, default=None, **kwargs)


def _get_ini_or_option(
    config: Any, name: str, choices: Optional[list[str]]
) -> str | None:
    """
    Get an option from either the ini file or the command line flags,
    with the latter taking precedence.
    """
    value = config.getini(name)
    if value is not None and choices is not None and value not in choices:
        raise ValueError(
            f'Invalid ini value for {name}, choose from {", ".join(choices)}'
        )
    return config.getoption(name) or value


def pytest_addoption(parser: Parser) -> None:
    # get all current option strings
    current_options = []
    for action in parser._anonymous.options:
        current_options += action._short_opts + action._long_opts

    for group in parser._groups:
        for action in group.options:
            current_options += action._short_opts + action._long_opts

    group = parser.getgroup("depends")

    # Add a flag to list all names + the tests they resolve to
    if "--list-dependency-names" not in current_options:
        group.addoption(
            "--list-dependency-names",
            action="store_true",
            default=False,
            help=(
                "List all non-nodeid dependency names + the tests they resolve to. "
                "Will also list all nodeid dependency names in verbose mode."
            ),
        )

    # Add a flag to list all (resolved) dependencies for all tests + unresolvable names
    if "--list-processed-dependencies" not in current_options:
        group.addoption(
            "--list-processed-dependencies",
            action="store_true",
            default=False,
            help=(
                "List all dependencies of all tests as a list of nodeids "
                "+ the names that could not be resolved."
            ),
        )

    # Add an ini option + flag to choose the action to take for failed dependencies
    if "--failed-dependency-action" not in current_options:
        _add_ini_and_option(
            parser,
            group,
            name="failed_dependency_action",
            help=(
                "The action to take when a test has dependencies that failed. "
                'Use "run" to run the test anyway, "skip" to skip the test, '
                'and "fail" to fail the test.'
            ),
            default="skip",
            choices=DEPENDENCY_PROBLEM_ACTIONS.keys(),
        )

    # Add an ini option + flag to choose the action to take for unresolved dependencies
    if "--missing-dependency-action" not in current_options:
        _add_ini_and_option(
            parser,
            group,
            name="missing_dependency_action",
            help=(
                "The action to take when a test has dependencies that cannot be found "
                "within the current scope. "
                'Use "run" to run the test anyway, "skip" to skip the test, '
                'and "fail" to fail the test.'
            ),
            default="warning",
            choices=DEPENDENCY_PROBLEM_ACTIONS.keys(),
        )


def pytest_configure(config: Any) -> None:
    manager = DependencyManager()
    managers.append(manager)

    # Setup the handling of problems with dependencies
    manager.options["failed_dependency_action"] = _get_ini_or_option(
        config,
        "failed_dependency_action",
        list(DEPENDENCY_PROBLEM_ACTIONS.keys()),
    )
    manager.options["missing_dependency_action"] = _get_ini_or_option(
        config,
        "missing_dependency_action",
        list(DEPENDENCY_PROBLEM_ACTIONS.keys()),
    )

    # Register marker
    config.addinivalue_line(
        "markers",
        "depends(name='name', on=['other_name']): marks dependencies between tests.",
    )


@pytest.hookimpl(trylast=True)
def pytest_collection_modifyitems(config: Any, items: list[pytest.Function]) -> None:
    manager = managers[-1]

    # Register the founds tests on the manager
    manager.items = items

    # Show the extra information if requested
    if config.getoption("list_dependency_names"):
        verbose = config.getoption("verbose") > 1
        manager.print_name_map(verbose)
    if config.getoption("list_processed_dependencies"):
        color = config.getoption("color")
        manager.print_processed_dependencies(color)

    # Reorder the items so that tests run after their dependencies
    items[:] = manager.sorted_items


@pytest.hookimpl(tryfirst=True, hookwrapper=True)
def pytest_runtest_makereport(item: Item) -> Any:
    manager = managers[-1]

    # Run the step
    outcome = yield

    # Store the result on the manager
    manager.register_result(item, outcome.get_result())


def pytest_runtest_call(item: Item) -> None:
    manager = managers[-1]

    # Handle missing dependencies
    missing_dependency_action = DEPENDENCY_PROBLEM_ACTIONS[
        manager.options["missing_dependency_action"]
    ]
    missing = manager.get_missing(item)
    if missing_dependency_action and missing:
        missing_dependency_action(
            f'{item.nodeid} depends on {", ".join(missing)}, which was not found'
        )

    # Check whether all dependencies succeeded
    failed_dependency_action = DEPENDENCY_PROBLEM_ACTIONS[
        manager.options["failed_dependency_action"]
    ]
    failed = manager.get_failed(item)
    if failed_dependency_action and failed:
        failed_dependency_action(f'{item.nodeid} depends on {", ".join(failed)}')


def pytest_unconfigure() -> None:
    managers.pop()
