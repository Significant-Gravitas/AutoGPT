"""The drift suite needs Redis and nothing else, so it opts out of SpinTestServer."""

from collections.abc import Iterator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from backend.copilot.baseline import service as baseline
from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatSession
from backend.copilot.model_router import ResolvedModel


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def server():  # type: ignore[override]
    return None


@pytest_asyncio.fixture(scope="session", loop_scope="session", autouse=True)
async def graph_cleanup():  # type: ignore[override]
    yield


@pytest.fixture
def baseline_io(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[ChatSession]]:
    """Every session the baseline engine persists, with its I/O stubbed."""
    persisted: list[ChatSession] = []

    async def save(session: ChatSession) -> ChatSession:
        persisted.append(session.model_copy(deep=True))
        return session

    monkeypatch.setattr(
        baseline,
        "config",
        baseline.config.model_copy(
            update={"use_e2b_sandbox": False, "use_local": False}
        ),
    )
    for name, value in {
        "drain_pending_safe": [],
        "resolve_model_route": ResolvedModel(
            model="anthropic/claude-sonnet-4-6", source="env"
        ),
        "_build_system_prompt": ("System prompt", None),
        "build_turn_budget_block": "",
        "is_enabled_for_user": False,
        "is_feature_enabled": False,
        "persist_and_record_usage": None,
        "download_transcript": None,
    }.items():
        monkeypatch.setattr(baseline, name, AsyncMock(return_value=value))
    monkeypatch.setattr(baseline, "_get_main_client", MagicMock())
    monkeypatch.setattr(baseline, "upsert_chat_session", AsyncMock(side_effect=save))
    try:
        yield persisted
    finally:
        set_execution_context(None, None)
