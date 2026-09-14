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
def stub_learning_registry(monkeypatch):
    """Route the skill registry's learning hooks to an in-memory store.

    ``store_user_skill`` appends a version and ``list_all_skills`` /
    ``read_skill`` consult the learning registry through the
    ``skill_versions_db()`` accessor. Without a connected Prisma client that
    accessor is the DatabaseManager RPC client, whose connection retries
    would make every registry test crawl. Tests that need the store can
    request ``fake_learning_store``.
    """
    from backend.copilot.learning._fake_store import FakeLearningStore

    store = FakeLearningStore()
    for target in (
        "backend.copilot.learning.retrieval.skill_versions_db",
        "backend.copilot.learning.retrieval.skill_use_db",
        "backend.copilot.learning.history.skill_versions_db",
        "backend.copilot.learning.history.skill_publication_db",
        "backend.copilot.tools.skills.skill_versions_db",
    ):
        monkeypatch.setattr(target, lambda: store)
    return store


@pytest.fixture
def fake_learning_store(stub_learning_registry):
    return stub_learning_registry
