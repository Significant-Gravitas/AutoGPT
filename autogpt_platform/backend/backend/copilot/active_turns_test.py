"""Unit tests for active_turns: per-user concurrent AutoPilot turn tracking."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import active_turns as active_turns_module
from backend.copilot.active_turns import (
    ConcurrentTurnLimitError,
    acquire_turn_slot,
    release_turn_slot,
)


def _patch_redis(monkeypatch: pytest.MonkeyPatch, redis_mock: MagicMock) -> None:
    monkeypatch.setattr(
        active_turns_module, "get_redis_async", AsyncMock(return_value=redis_mock)
    )


def _redis_mock_eval(eval_return: int) -> MagicMock:
    """Build a mock Redis client whose ``eval`` returns the given Lua result.

    Lua return values match :class:`SlotAdmission`: 0=REJECTED, 1=ADMITTED, 2=REFRESHED.
    """
    redis_mock = MagicMock()
    redis_mock.eval = AsyncMock(return_value=eval_return)
    redis_mock.zrem = AsyncMock(return_value=1)
    return redis_mock


# ── release_turn_slot ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_release_zrems_session_from_user_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_mock = MagicMock()
    redis_mock.zrem = AsyncMock(return_value=1)
    _patch_redis(monkeypatch, redis_mock)

    await release_turn_slot("user-1", "session-a")
    redis_mock.zrem.assert_called_once()
    args, _ = redis_mock.zrem.call_args
    assert args[1] == "session-a"
    assert "user-1" in args[0]


@pytest.mark.asyncio
async def test_release_swallows_redis_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Redis errors during release are logged, not raised — the slot
    will be swept on the next acquisition's stale-cutoff."""
    monkeypatch.setattr(
        active_turns_module,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("down")),
    )
    await release_turn_slot("user-1", "session-a")  # must not raise


# ── acquire_turn_slot lifecycle ───────────────────────────────────────


@pytest.mark.asyncio
async def test_admitted_slot_releases_on_exit_without_keep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Forgetting ``keep()`` on a clean exit releases the slot."""
    redis_mock = _redis_mock_eval(1)  # ADMITTED
    _patch_redis(monkeypatch, redis_mock)

    async with acquire_turn_slot("user-1", "session-a"):
        pass

    redis_mock.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_admitted_slot_releases_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the with-block also releases the slot."""
    redis_mock = _redis_mock_eval(1)  # ADMITTED
    _patch_redis(monkeypatch, redis_mock)

    with pytest.raises(RuntimeError, match="downstream blew up"):
        async with acquire_turn_slot("user-1", "session-a"):
            raise RuntimeError("downstream blew up")

    redis_mock.zrem.assert_called_once()


@pytest.mark.asyncio
async def test_kept_slot_is_not_released_on_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``keep()`` transfers ownership; the context manager leaves the slot held."""
    redis_mock = _redis_mock_eval(1)  # ADMITTED
    _patch_redis(monkeypatch, redis_mock)

    async with acquire_turn_slot("user-1", "session-a") as slot:
        slot.keep()

    redis_mock.zrem.assert_not_called()


@pytest.mark.asyncio
async def test_rejection_raises_concurrent_turn_limit_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis_mock = _redis_mock_eval(0)  # REJECTED
    _patch_redis(monkeypatch, redis_mock)

    with pytest.raises(ConcurrentTurnLimitError):
        async with acquire_turn_slot("user-1", "session-a"):
            pytest.fail("body should not run when acquire fails")  # pragma: no cover

    # Reject path never admitted, so it must not release.
    redis_mock.zrem.assert_not_called()


@pytest.mark.asyncio
async def test_refreshed_slot_is_not_released_on_clean_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Refreshed (re-entry) → caller doesn't own the slot, no release on exit."""
    redis_mock = _redis_mock_eval(2)  # REFRESHED
    _patch_redis(monkeypatch, redis_mock)

    async with acquire_turn_slot("user-1", "session-a"):
        pass

    redis_mock.zrem.assert_not_called()


@pytest.mark.asyncio
async def test_refreshed_slot_is_not_released_on_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Same-session retry's failure must NOT tear down the original turn's slot."""
    redis_mock = _redis_mock_eval(2)  # REFRESHED
    _patch_redis(monkeypatch, redis_mock)

    with pytest.raises(RuntimeError, match="boom"):
        async with acquire_turn_slot("user-1", "session-a"):
            raise RuntimeError("boom")

    redis_mock.zrem.assert_not_called()


@pytest.mark.asyncio
async def test_anonymous_user_skips_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``user_id`` falsy → no slot acquired, no Redis touched, no exception."""
    redis_mock = _redis_mock_eval(0)  # would reject if hit
    _patch_redis(monkeypatch, redis_mock)

    async with acquire_turn_slot(None, "session-a"):
        pass

    redis_mock.eval.assert_not_called()
    redis_mock.zrem.assert_not_called()


@pytest.mark.asyncio
async def test_redis_brownout_fails_open_admitting_the_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cap is a safeguard, not a budget — a Redis brown-out should not
    429 every user. The context manager admits the turn and continues."""
    monkeypatch.setattr(
        active_turns_module,
        "get_redis_async",
        AsyncMock(side_effect=ConnectionError("down")),
    )

    async with acquire_turn_slot("user-1", "session-a") as slot:
        slot.keep()  # body runs as if admitted; pretend a turn ran


# ── default cap pinning ───────────────────────────────────────────────


def test_schema_default_concurrent_turn_limit_is_15() -> None:
    """Pin the schema default so a config drift can't silently relax the
    abuse cap. Reads the field default directly so a local ``.env``
    override (e.g. lower cap for development) doesn't break the test."""
    from backend.util.settings import Config

    assert Config.model_fields["max_concurrent_copilot_turns_per_user"].default == 15
