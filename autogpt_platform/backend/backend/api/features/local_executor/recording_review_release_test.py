"""A recording review releases the session's shim, except from a running turn."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.local_executor import routes


def _session(chat_status: str) -> SimpleNamespace:
    return SimpleNamespace(
        chat_status=chat_status,
        metadata=SimpleNamespace(
            execution_target=SimpleNamespace(kind="local", machine_id="machine-1")
        ),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("chat_status", "released"), [("running", False), ("idle", True)]
)
async def test_a_review_leaves_the_shim_to_a_running_turn(
    chat_status: str, released: bool
) -> None:
    """The review detached the session and killed its shim, which a running
    turn shares, so the turn lost its files and computer mid-call. A running
    turn detaches and closes the shim itself when it ends."""
    shim = SimpleNamespace(kill=AsyncMock())
    detach = AsyncMock()

    with (
        patch.object(
            routes,
            "get_chat_session_metadata",
            AsyncMock(return_value=_session(chat_status)),
        ),
        patch.object(routes, "get_machine_presence", AsyncMock(return_value="seen")),
        patch.object(routes, "detach_machine_session", detach),
    ):
        await routes._release_reviewed_local_shim("session-1", "owner-1", shim)

    assert shim.kill.await_count == int(released)
    assert detach.await_count == int(released)
