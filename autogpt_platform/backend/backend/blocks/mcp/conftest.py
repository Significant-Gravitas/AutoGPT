"""
Conftest for MCP block tests.

Registers the e2e marker and --run-e2e CLI option so MCP end-to-end tests
(which hit real external servers) can be gated behind a flag.
"""

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "e2e: end-to-end tests requiring network")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    """Skip e2e tests unless --run-e2e is passed."""
    if not config.getoption("--run-e2e", default=False):
        skip_e2e = pytest.mark.skip(reason="need --run-e2e option to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-e2e", action="store_true", default=False, help="run e2e tests"
    )
