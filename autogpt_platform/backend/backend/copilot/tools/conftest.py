"""Local conftest for copilot/tools tests.

Overrides the session-scoped `server` and `graph_cleanup` autouse fixtures from
backend/conftest.py so that integration tests in this directory do not trigger
the full SpinTestServer startup (which requires Postgres + RabbitMQ).
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.copilot.delegation_settings import DelegationSettings


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
    """Stub the ``user_db()`` accessor ONLY for the helpers.py-local binding.

    ``execute_block`` reads the user record via ``user_db().get_user_by_id``
    to plumb ``user_timezone`` into ``ExecutionContext``. ``user_db`` is the
    connection-aware accessor from ``db_accessors`` (a callable returning the
    direct module or the DatabaseManager RPC client). The existing tests don't
    need a real DB for that.

    ⚠️ We must patch the ``user_db`` name on the helpers module itself,
    NOT ``helpers.user_db().get_user_by_id`` — patching through the returned
    client would mutate the shared accessor target globally, which leaks into
    unrelated callers (e.g. ``rate_limit``'s ``user_db().get_user_by_id`` in
    ``run_agent_test``) and clobbers their real-DB test users with our
    MagicMock.
    """
    user = MagicMock()
    user.timezone = "UTC"
    client = MagicMock()
    client.get_user_by_id = AsyncMock(return_value=user)
    stub = MagicMock(return_value=client)
    monkeypatch.setattr("backend.copilot.tools.helpers.user_db", stub)


@pytest.fixture(autouse=True)
def stub_sub_session_costs(monkeypatch):
    """Report no logged spend and default delegation settings unless a test
    says otherwise.

    The sub-session tools read cost through the ``delegation_db()`` accessor,
    which falls back to the DatabaseManager RPC client when Prisma is not
    connected; without a stub every spawn/poll test would wait on that RPC.
    """
    client = MagicMock()
    client.get_session_costs = AsyncMock(return_value={})
    client.get_delegation_settings = AsyncMock(return_value=DelegationSettings())
    client.get_delegation_spend_since = AsyncMock(return_value=0)
    client.get_expert_hired_at = AsyncMock(return_value=None)
    for module in (
        "backend.copilot.tools.sub_session_facts",
        "backend.copilot.tools.delegation_policy",
        "backend.copilot.gate.delegation_rules",
    ):
        monkeypatch.setattr(f"{module}.delegation_db", MagicMock(return_value=client))
    return client
