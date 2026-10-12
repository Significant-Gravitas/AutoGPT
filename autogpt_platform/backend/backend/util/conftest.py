"""Override session-scoped fixtures from parent conftest.py so unit tests
in this directory can run without the full server stack.

Also the fixtures that capture what a service process writes to its logs and
would send to Sentry, for tests of what a request leaves behind."""

import io
import logging
import sys
from collections.abc import Iterator

import pytest
import sentry_sdk
from sentry_sdk.integrations.asyncio import AsyncioIntegration

from backend.util import logging as service_logging
from backend.util import metrics
from backend.util.logging_test_helpers import CapturingTransport

_FAKE_DSN = "https://key@o1.ingest.us.sentry.io/1"


@pytest.fixture(scope="session")
def server():
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


@pytest.fixture
def log_level() -> str:
    """Production's log level. Parametrize a test on ``log_level`` to change it."""
    return "INFO"


@pytest.fixture(params=[False, True], ids=["plain", "cloud"])
def service_output(
    request: pytest.FixtureRequest, log_level: str, monkeypatch: pytest.MonkeyPatch
) -> Iterator[io.StringIO]:
    """Everything a service process writes to stdout and stderr at
    ``log_level``. ``cloud`` is the structured JSON Cloud Logging collects."""
    monkeypatch.setenv("LOG_LEVEL", log_level)
    monkeypatch.setattr(
        service_logging, "is_structured_logging_enabled", lambda: request.param
    )
    root = logging.getLogger()
    level = root.level
    monkeypatch.setattr(root, "handlers", [])
    service_logging.configure_logging()
    output = io.StringIO()
    for handler in root.handlers:
        assert isinstance(handler, logging.StreamHandler)
        handler.setStream(output)
    yield output
    root.setLevel(level)


@pytest.fixture
def sentry(monkeypatch: pytest.MonkeyPatch) -> Iterator[CapturingTransport]:
    """What a live SDK client, built from the options ``sentry_init()`` passes,
    would send to Sentry while the test runs."""
    calls: list[dict] = []
    with monkeypatch.context() as during_init:
        during_init.setattr(metrics, "_sentry_init", lambda **kw: calls.append(kw))
        during_init.setattr(metrics.settings.secrets, "sentry_dsn", _FAKE_DSN)
        during_init.delitem(sys.modules, "pytest")
        during_init.delenv("PYTEST_CURRENT_TEST", raising=False)
        metrics.sentry_init()
    [options] = calls
    # The profiler starts a sampler thread, and the asyncio integration patches
    # the session's shared event loop; neither sees request data.
    options["profiles_sample_rate"] = 0.0
    options["integrations"] = [
        integration
        for integration in options["integrations"]
        if not isinstance(integration, AsyncioIntegration)
    ]
    transport = CapturingTransport()
    client = sentry_sdk.Client(**options, transport=transport)
    monkeypatch.setattr(sentry_sdk.get_global_scope(), "client", client)
    # Breadcrumbs left by earlier tests would otherwise ride along.
    with sentry_sdk.isolation_scope() as scope:
        scope.clear_breadcrumbs()
        yield transport
    client.close()
