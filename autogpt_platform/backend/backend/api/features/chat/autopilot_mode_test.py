"""The mode on the turn request lands in the session, and only when it changed."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.api.features.chat.routes import _apply_autopilot_mode
from backend.copilot.model import ChatSessionMetadata

_ROUTES = "backend.api.features.chat.routes"


def _session(mode=None):
    session = MagicMock()
    session.session_id = "s1"
    session.metadata = ChatSessionMetadata(origin="interactive", autopilot_mode=mode)
    return session


@pytest.mark.parametrize(
    "stored, requested, flag, writes",
    [
        (None, "ask_first", True, True),
        ("ask_first", "unsupervised", True, True),
        ("ask_first", "ask_first", True, False),
        # A later turn that sends no mode keeps the one the chat has.
        ("ask_first", None, True, False),
        (None, "ask_first", False, False),
    ],
)
async def test_mode_persists_only_when_it_changes(stored, requested, flag, writes):
    session = _session(stored)
    update = AsyncMock(return_value=True)
    with (
        patch(f"{_ROUTES}.is_feature_enabled", AsyncMock(return_value=flag)),
        patch(f"{_ROUTES}.update_session_autopilot_mode", update),
    ):
        await _apply_autopilot_mode(session, "u1", requested)

    assert update.await_count == int(writes)
    assert session.metadata.autopilot_mode == (requested if writes else stored)
