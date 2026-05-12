"""Unit tests for active_turns: per-user concurrent AutoPilot turn tracking.

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


def _mock_db(
    *,
    admit_cas_ok: bool = True,
    running_count: int = 0,
    current_status: str = "idle",
) -> MagicMock:
    """Mock ``chat_db()`` return value.

    * ``admit_cas_ok`` — return value of ``update_chat_session_status``
      for the idle→running CAS in ``acquire_turn_slot``.  True simulates
      a successful fresh admit; False simulates the CAS failing (session
      was not idle), in which case ``current_status`` is consulted.
    * ``running_count`` — return value of
      ``count_chat_sessions_by_status(status='running')`` after the flip.
    * ``current_status`` — return value of ``get_chat_session_status``
      after a CAS failure (the disambiguation read).
    """
    db = MagicMock()
    db.update_chat_session_status = AsyncMock(return_value=admit_cas_ok)
    db.count_chat_sessions_by_status = AsyncMock(return_value=running_count)
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
    db = _mock_db(admit_cas_ok=True, running_count=1)
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot("user-1", "session-a"):
            pass
    # First flip: admit (idle → running). Second flip: release (running → idle).
    assert db.update_chat_session_status.await_count == 2


@pytest.mark.asyncio
async def test_admitted_slot_releases_on_exception() -> None:
    """An exception inside the with-block also releases the slot."""
    db = _mock_db(admit_cas_ok=True, running_count=1)
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(RuntimeError, match="downstream blew up"):
            async with acquire_turn_slot("user-1", "session-a"):
                raise RuntimeError("downstream blew up")
    assert db.update_chat_session_status.await_count == 2


@pytest.mark.asyncio
async def test_kept_slot_is_not_released_on_exit() -> None:
    """``keep()`` transfers ownership; the context manager leaves the
    slot held for ``mark_session_completed`` to clean up."""
    db = _mock_db(admit_cas_ok=True, running_count=1)
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot("user-1", "session-a") as slot:
            slot.keep()
    # Only the admit fires; release is the caller's responsibility now.
    assert db.update_chat_session_status.await_count == 1


@pytest.mark.asyncio
async def test_rejection_rolls_back_admit_when_over_cap() -> None:
    """Admit count-after-flip exceeds the cap → roll back to idle and
    raise ConcurrentTurnLimitError."""
    db = _mock_db(admit_cas_ok=True, running_count=6)  # 6 > capacity=5
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(ConcurrentTurnLimitError):
            async with acquire_turn_slot("user-1", "session-a", capacity=5):
                pytest.fail("body must not run on rejection")  # pragma: no cover
    # Two flips: admit then rollback.
    assert db.update_chat_session_status.await_count == 2


@pytest.mark.asyncio
async def test_queued_session_raises_so_caller_falls_through_to_queue() -> None:
    """CAS failure + current status == 'queued' means the user already
    has a pending task for this session.  Raise ConcurrentTurnLimitError
    so the route falls through to ``try_enqueue_turn`` instead of
    double-dispatching."""
    db = _mock_db(admit_cas_ok=False, current_status="queued")
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(ConcurrentTurnLimitError):
            async with acquire_turn_slot("user-1", "session-a"):
                pytest.fail("body must not run on rejection")  # pragma: no cover


@pytest.mark.asyncio
async def test_refreshed_slot_is_not_released_on_clean_exit() -> None:
    """CAS failure + current status == 'running' is the SSE-retry
    refresh path: no admit, no release ownership, no error."""
    db = _mock_db(admit_cas_ok=False, current_status="running")
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot("user-1", "session-a"):
            pass
    # Only the CAS attempt fires; nothing else.
    assert db.update_chat_session_status.await_count == 1


@pytest.mark.asyncio
async def test_refreshed_slot_is_not_released_on_exception() -> None:
    """Same-session retry's failure must NOT tear down the original turn."""
    db = _mock_db(admit_cas_ok=False, current_status="running")
    with patch.object(active_turns, "chat_db", return_value=db):
        with pytest.raises(RuntimeError, match="boom"):
            async with acquire_turn_slot("user-1", "session-a"):
                raise RuntimeError("boom")
    assert db.update_chat_session_status.await_count == 1


@pytest.mark.asyncio
async def test_anonymous_user_skips_gate() -> None:
    """``user_id`` falsy → no DB query, no exception."""
    db = _mock_db()
    with patch.object(active_turns, "chat_db", return_value=db):
        async with acquire_turn_slot(None, "session-a"):
            pass
    db.update_chat_session_status.assert_not_awaited()
    db.count_chat_sessions_by_status.assert_not_awaited()


# ── default cap pinning ───────────────────────────────────────────────


def test_schema_default_concurrent_turn_limit_is_15() -> None:
    """Pin the schema default so a config drift can't silently relax the
    abuse cap. Reads the field default directly so a local ``.env``
    override (e.g. lower cap for development) doesn't break the test."""
    from backend.util.settings import Config

    assert Config.model_fields["max_inflight_copilot_turns_per_user"].default == 15
