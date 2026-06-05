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
def stub_user_lookup_in_helpers(monkeypatch):
    """Stub ``user_db.get_user_by_id`` ONLY for the helpers.py-local binding.

    ``prepare_block_for_execution`` reads the user record to plumb
    ``user_timezone`` into ``ExecutionContext``. The existing tests don't
    need a real DB for that.

    ⚠️ We must patch the ``user_db`` name on the helpers module itself,
    NOT ``helpers.user_db.get_user_by_id`` — the latter resolves through
    the module reference and mutates ``backend.data.user.get_user_by_id``
    globally, which leaks into unrelated callers (e.g. ``rate_limit``'s
    ``user_db().get_user_by_id`` in ``run_agent_test``) and clobbers
    their real-DB test users with our MagicMock.
    """
    user = MagicMock()
    user.timezone = "UTC"
    stub = MagicMock()
    stub.get_user_by_id = AsyncMock(return_value=user)
    monkeypatch.setattr("backend.copilot.tools.helpers.user_db", stub)
