"""Local conftest for copilot/tools tests.

Overrides the session-scoped `server` fixture from
backend/conftest.py so that integration tests in this directory do not trigger
the full SpinTestServer startup (which requires Postgres + RabbitMQ).
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from backend.copilot.delegation_settings import DelegationSettings


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():  # type: ignore[override]
    """No-op server stub — tools tests don't need the full backend."""
    return None


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
    seen: dict[str, str] = {}

    async def mark_seen(session_id, revision):
        seen[session_id] = revision

    monkeypatch.setattr(
        "backend.copilot.tools.skills.read_seen_index_revision",
        AsyncMock(side_effect=lambda session_id: seen.get(session_id)),
    )
    monkeypatch.setattr(
        "backend.copilot.tools.skills.mark_index_revision_seen", mark_seen
    )
    for target in (
        "backend.copilot.learning.retrieval.skill_versions_db",
        "backend.copilot.learning.retrieval.skill_use_db",
        "backend.copilot.learning.history.skill_versions_db",
        "backend.copilot.learning.history.skill_publication_db",
        "backend.copilot.tools.skills.skill_versions_db",
        "backend.copilot.tools.skills.skill_deletion_db",
    ):
        monkeypatch.setattr(target, lambda: store)
    return store


@pytest.fixture
def fake_learning_store(stub_learning_registry):
    return stub_learning_registry


@pytest.fixture(autouse=True)
def login_chain_unchanged(request):
    """Sandbox doubles hold no login files; tests of that check opt out."""
    if request.node.get_closest_marker("real_login_chain"):
        yield
        return
    unchanged = AsyncMock(return_value={})
    with (
        patch("backend.util.sandbox_login.changed_login_files", unchanged),
        patch("backend.copilot.tools.bash_exec.changed_login_files", unchanged),
        patch("backend.copilot.tools.e2b_sandbox.take_baseline", AsyncMock()),
    ):
        yield


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
    client.stop_delegation_at_cap = AsyncMock()
    for module in (
        "backend.copilot.tools.sub_session_facts",
        "backend.copilot.tools.delegation_policy",
        "backend.copilot.gate.delegation_rules",
        "backend.copilot.delegation_cap",
    ):
        monkeypatch.setattr(f"{module}.delegation_db", MagicMock(return_value=client))
    # The cap writes drop the cached session after their row write.
    monkeypatch.setattr(
        "backend.copilot.delegation_cap.invalidate_session_cache", AsyncMock()
    )
    # A thread's cap is read fresh from its row; with no row, what the caller
    # already knows stands.
    monkeypatch.setattr(
        "backend.copilot.tools.delegation_policy.get_chat_session_metadata",
        AsyncMock(return_value=None),
    )
    return client


@pytest.fixture(autouse=True)
def external_toggle_off(monkeypatch):
    """The gate as it was before ``ask_before_external``: the mode table alone
    decides an outward call. Tests of the toggle itself override this with
    ``external_toggle_on``, or live in ``gate/delegation_rules_test.py``.

    Patching the per-turn read (not the DB behind it) also keeps a settings
    cache left by one test from answering the next.
    """
    read = AsyncMock(return_value=DelegationSettings(ask_before_external=False))
    monkeypatch.setattr(
        "backend.copilot.gate.delegation_rules.turn_delegation_settings", read
    )
    return read


@pytest.fixture
def external_toggle_on(external_toggle_off):
    external_toggle_off.return_value = DelegationSettings(ask_before_external=True)
    return external_toggle_off
