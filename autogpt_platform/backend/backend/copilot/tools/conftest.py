"""Local conftest for copilot/tools tests.

Overrides the session-scoped `server` and `graph_cleanup` autouse fixtures from
backend/conftest.py so that integration tests in this directory do not trigger
the full SpinTestServer startup (which requires Postgres + RabbitMQ).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():  # type: ignore[override]
    """No-op server stub — tools tests don't need the full backend."""
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():  # type: ignore[override]
    """No-op graph cleanup stub."""
    yield


@pytest.fixture(autouse=True)
def stub_user_lookup(monkeypatch):
    """Stub ``user_db.get_user_by_id`` for every test in this dir.

    ``prepare_block_for_execution`` reads the user record to plumb
    ``user_timezone`` into ``ExecutionContext``. The existing tests don't
    need a real DB — return a minimal user with ``timezone='UTC'`` so the
    timezone resolves to UTC (matches `ExecutionContext`'s default).
    """
    user = MagicMock()
    user.timezone = "UTC"
    monkeypatch.setattr(
        "backend.copilot.tools.helpers.user_db.get_user_by_id",
        AsyncMock(return_value=user),
    )
