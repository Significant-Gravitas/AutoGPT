from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.explicit_learning import (
    capture_explicit_correction,
    explicit_correction,
)
from backend.copilot.tools.models import MemoryStoreResponse


def _session(*, expert_id: str | None = None):
    return SimpleNamespace(session_id="session-1", expert_id=expert_id)


def test_only_explicit_operating_corrections_are_detected():
    assert explicit_correction("Use your judgment from now on") is not None
    assert explicit_correction("Never ask me about reversible decisions") is not None
    assert explicit_correction("Please research this market") is None


@pytest.mark.asyncio
async def test_flag_off_does_not_retain_correction():
    with (
        patch(
            "backend.copilot.explicit_learning.is_feature_enabled",
            new=AsyncMock(return_value=False),
        ),
        patch(
            "backend.copilot.explicit_learning.MemoryStoreTool._execute",
            new=AsyncMock(),
        ) as store,
    ):
        retained = await capture_explicit_correction(
            user_id="user-1",
            session=_session(),
            message="From now on use your judgment",
        )

    assert retained is False
    store.assert_not_awaited()


@pytest.mark.asyncio
async def test_expert_correction_becomes_learned_note_without_graphiti():
    promote = AsyncMock()
    learned_db = SimpleNamespace(promote_learned_notes=promote)
    with (
        patch(
            "backend.copilot.explicit_learning.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch(
            "backend.copilot.explicit_learning.expert_learned_notes_db",
            return_value=learned_db,
        ),
        patch(
            "backend.copilot.explicit_learning.MemoryStoreTool._execute",
            new=AsyncMock(),
        ) as store,
    ):
        retained = await capture_explicit_correction(
            user_id="user-1",
            session=_session(expert_id="expert-1"),
            message="Remember that you should never ask me about reversible choices",
        )

    assert retained is True
    promote.assert_awaited_once()
    candidate = promote.await_args.args[2][0]
    assert candidate.text.startswith("Remember that")
    assert candidate.source_session_id == "session-1"
    store.assert_not_awaited()


@pytest.mark.asyncio
async def test_autopilot_correction_is_stored_as_a_rule():
    store = AsyncMock(
        return_value=MemoryStoreResponse(
            message="Stored",
            session_id="session-1",
            memory_name="correction",
        )
    )
    with (
        patch(
            "backend.copilot.explicit_learning.is_feature_enabled",
            new=AsyncMock(return_value=True),
        ),
        patch("backend.copilot.explicit_learning.MemoryStoreTool._execute", store),
    ):
        retained = await capture_explicit_correction(
            user_id="user-1",
            session=_session(),
            message="From now on, use your own judgment",
        )

    assert retained is True
    assert store.await_args.kwargs["memory_kind"] == "rule"
    assert store.await_args.kwargs["source_kind"] == "user_asserted"
