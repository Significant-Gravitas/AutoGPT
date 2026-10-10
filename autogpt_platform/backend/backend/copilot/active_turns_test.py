"""Unit tests for active_turns: per-user concurrent Otto turn tracking.

Backed by ``ChatSession.chatStatus`` accessed through ``chat_db()``;
tests patch ``backend.copilot.active_turns.chat_db`` to return an
:class:`unittest.mock.AsyncMock`.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import active_turns
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    acquire_turn_slot,
    release_turn_slot,
)


def _mock_db(*, admit: str = "admitted", current_status: str = "idle") -> MagicMock:
    """Mock ``chat_db()`` return value.

    * ``admit`` — what ``admit_chat_session_turn`` decides under its lock:
      ``"admitted"``, ``"full"`` (at the cap) or ``"busy"`` (not idle), in
      which case ``current_status`` is consulted.
    * ``current_status`` — return value of ``get_chat_session_status``.
    """
    db = MagicMock()
    db.admit_chat_session_turn = AsyncMock(return_value=admit)
    db.update_chat_session_status = AsyncMock(return_value=True)
    db.count_chat_sessions_by_status = AsyncMock(return_value=0)
    db.list_chat_sessions_by_status = AsyncMock(return_value=[])
    db.get_chat_session_status = AsyncMock(return_value=current_status)
    return db


# ── release_turn_slot ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_release_flips_running_to_idle_with_userid_guard() -> None:
    """``release_turn_slot`` calls ``update_chat_session_status`` with
    a userId guard so a misrouted call can't release another user's row."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        await release_turn_slot("user-1", "session-a")
    db.update_chat_session_status.assert_awaited_once_with(
        session_id="session-a",
        expect_status="running",
        status="idle",
        user_id="user-1",
    )


@pytest.mark.asyncio
async def test_release_anonymous_user_is_noop() -> None:
    """``release_turn_slot`` with empty user_id makes no DB write."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        await release_turn_slot("", "session-a")
    db.update_chat_session_status.assert_not_awaited()


# ── acquire_turn_slot lifecycle ───────────────────────────────────────


@pytest.mark.asyncio
async def test_admitted_slot_releases_on_exit_without_keep() -> None:
    """Forgetting ``keep()`` on a clean exit releases the slot."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot("user-1", "session-a"):
            pass
    # The admit flipped under its lock; the exit releases (running → idle).
    assert db.update_chat_session_status.await_count == 1


@pytest.mark.asyncio
async def test_admitted_slot_releases_on_exception() -> None:
    """An exception inside the with-block also releases the slot."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(RuntimeError, match="downstream blew up"):
            async with acquire_turn_slot("user-1", "session-a"):
                raise RuntimeError("downstream blew up")
    assert db.update_chat_session_status.await_count == 1


@pytest.mark.asyncio
async def test_kept_slot_is_not_released_on_exit() -> None:
    """``keep()`` transfers ownership; the context manager leaves the
    slot held for ``mark_session_completed`` to clean up."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot("user-1", "session-a") as slot:
            slot.keep()
    # Nothing released; that is the caller's responsibility now.
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_full_cap_raises_without_flipping() -> None:
    """The cap is checked before the flip, under the same lock."""
    db = _mock_db(admit="full")
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(ConcurrentTurnLimitError):
            async with acquire_turn_slot("user-1", "session-a", capacity=5):
                pytest.fail("body must not run on rejection")  # pragma: no cover
    assert db.admit_chat_session_turn.await_args.kwargs["capacity"] == 5
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_queued_session_raises_so_caller_falls_through_to_queue() -> None:
    """CAS failure + current status == 'queued' means the user already
    has a pending task for this session.  Raise ConcurrentTurnLimitError
    so the route falls through to ``try_enqueue_turn`` instead of
    double-dispatching."""
    db = _mock_db(admit="busy", current_status="queued")
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(ConcurrentTurnLimitError):
            async with acquire_turn_slot("user-1", "session-a"):
                pytest.fail("body must not run on rejection")  # pragma: no cover


@pytest.mark.asyncio
async def test_refreshed_slot_is_not_released_on_clean_exit() -> None:
    """CAS failure + current status == 'running' is the SSE-retry
    refresh path: no admit, no release ownership, no error."""
    db = _mock_db(admit="busy", current_status="running")
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot("user-1", "session-a"):
            pass
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_refreshed_slot_is_not_released_on_exception() -> None:
    """Same-session retry's failure must NOT tear down the original turn."""
    db = _mock_db(admit="busy", current_status="running")
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(RuntimeError, match="boom"):
            async with acquire_turn_slot("user-1", "session-a"):
                raise RuntimeError("boom")
    db.update_chat_session_status.assert_not_awaited()


@pytest.mark.asyncio
async def test_anonymous_user_skips_gate() -> None:
    """``user_id`` falsy → no DB query, no exception."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot(None, "session-a"):
            pass
    db.admit_chat_session_turn.assert_not_awaited()


# ── default cap pinning ───────────────────────────────────────────────


def test_schema_default_concurrent_turn_limit_is_15() -> None:
    """Pin the schema default so a config drift can't silently relax the
    abuse cap. Reads the field default directly so a local ``.env``
    override (e.g. lower cap for development) doesn't break the test."""
    from backend.util.settings import Config

    assert Config.model_fields["max_inflight_copilot_turns_per_user"].default == 15
