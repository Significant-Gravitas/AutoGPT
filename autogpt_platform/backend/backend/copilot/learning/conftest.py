"""Local conftest for copilot/learning tests.

Overrides the session-scoped ``server`` and ``graph_cleanup`` autouse
fixtures from backend/conftest.py so the pure unit tests in this package do
not spin the full backend. The DB-backed integration tests for the learning
data layer live under ``backend/data`` and use the real server fixture.
"""

from unittest.mock import AsyncMock

import pytest
import pytest_asyncio

from backend.copilot import usage_activation_fixtures

# A learning run captures the account's usage attribution before its first
# review (#15208's ``attributed_usage``); every test here gets the ready,
# untrialed state, and one that needs a trial or a conversion overrides it.
usage_snapshot = usage_activation_fixtures.usage_snapshot


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():  # type: ignore[override]
    """No-op server stub — learning unit tests don't need the backend."""
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():  # type: ignore[override]
    """No-op graph cleanup stub."""
    yield


@pytest.fixture
def fake_store(monkeypatch):
    """In-memory learning store patched into every accessor seam."""
    from backend.copilot.learning._fake_store import FakeLearningStore

    store = FakeLearningStore()
    monkeypatch.setattr(
        "backend.copilot.learning.publish.read_user_skill_files",
        AsyncMock(return_value=[]),
    )
    monkeypatch.setattr(
        "backend.copilot.learning.nightly.load_reviewed_packages",
        AsyncMock(return_value={}),
    )
    monkeypatch.setattr(
        "backend.copilot.learning.publish.saved_skill_limit_reached",
        AsyncMock(return_value=False),
    )
    modules = (
        "retrieval",
        "history",
        "publish",
        "publication_state",
        "owner_actions",
        "revocation",
        "nightly",
        "dispositions",
        "review_preconditions",
        "content_approvals",
        "review_replay",
        "chat_source",
    )
    seams = (
        "skill_learning_db",
        "skill_reviews_db",
        "skill_versions_db",
        "skill_publication_db",
        "skill_use_db",
    )
    for module in modules:
        for seam in seams:
            monkeypatch.setattr(
                f"backend.copilot.learning.{module}.{seam}",
                lambda: store,
                raising=False,
            )
    return store
