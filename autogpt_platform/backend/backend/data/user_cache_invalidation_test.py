"""Clearing the user-lookup cache doesn't stall the event loop.

``get_or_create_user`` can only be cleared whole, which SCANs every Redis
shard on the shared cache's synchronous client: seconds on prod's keyspace,
during which a clear run on the loop holds up every request the process
serves.
"""

import asyncio
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.data import user as user_db

_CLEAR_SECONDS = 0.3


@pytest.fixture
def slow_clear():
    """The whole-cache clear as prod's SCAN behaves: it blocks its thread.
    Yields the threads it ran on."""
    threads: list[threading.Thread] = []

    def clear() -> None:
        threads.append(threading.current_thread())
        time.sleep(_CLEAR_SECONDS)

    with (
        patch.object(user_db.get_or_create_user, "cache_clear", clear),
        patch.object(user_db.get_user_by_id, "cache_delete"),
        patch.object(user_db.get_user_by_email, "cache_delete"),
    ):
        yield threads


@pytest.fixture
def users():
    """``PrismaUser.prisma()`` with one account row."""
    row = MagicMock(
        id="user-1",
        email="a@example.com",
        marketingOptOutAt=None,
        termsAcceptedAt=None,
        termsVersion=None,
    )
    actions = MagicMock(
        find_unique=AsyncMock(return_value=row),
        update=AsyncMock(return_value=row),
        update_many=AsyncMock(return_value=1),
    )
    with (
        patch.object(user_db, "PrismaUser", MagicMock(prisma=lambda: actions)),
        patch.object(
            user_db.User, "from_db", MagicMock(return_value=SimpleNamespace())
        ),
    ):
        yield actions


async def _ticks_during(work) -> int:
    """How often the event loop got round to another task while *work* ran."""
    ticks = 0

    async def tick() -> None:
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    ticker = asyncio.create_task(tick())
    try:
        await work
    finally:
        ticker.cancel()
    return ticks


def _assert_cleared_off_the_loop(threads: list[threading.Thread], ticks: int):
    assert len(threads) == 1
    assert threads[0] is not threading.current_thread()
    # About 30 ticks of 10 ms fit in the clear; a blocked loop gets none.
    assert ticks >= 10


@pytest.mark.asyncio
async def test_clearing_the_user_lookup_cache_leaves_the_event_loop_running(
    slow_clear,
):
    ticks = await _ticks_during(
        user_db._invalidate_user_caches("user-1", "a@example.com")
    )
    _assert_cleared_off_the_loop(slow_clear, ticks)


@pytest.mark.asyncio
async def test_a_timezone_change_clears_the_cache_off_the_event_loop(slow_clear, users):
    scheduling = "backend.copilot.briefing.scheduling"
    with (
        patch("backend.copilot.dream.registry.reregister_user", new=AsyncMock()),
        patch(f"{scheduling}.clear_briefing_registration_marker", new=AsyncMock()),
        patch(f"{scheduling}.ensure_morning_briefing_scheduled", new=AsyncMock()),
    ):
        ticks = await _ticks_during(
            user_db.update_user_timezone("user-1", "Europe/Paris")
        )
        # The re-registrations it started in the background finish here.
        await asyncio.sleep(0)
    _assert_cleared_off_the_loop(slow_clear, ticks)


@pytest.mark.asyncio
async def test_a_default_chat_route_change_clears_the_cache_off_the_event_loop(
    slow_clear, users
):
    ticks = await _ticks_during(
        user_db.set_user_default_chat_route("user-1", "openai", "cred-1")
    )
    _assert_cleared_off_the_loop(slow_clear, ticks)


@pytest.mark.asyncio
async def test_a_marketing_opt_out_clears_the_cache_off_the_event_loop(
    slow_clear, users
):
    ticks = await _ticks_during(
        user_db.record_marketing_opt_out_by_email("a@example.com", "test")
    )
    _assert_cleared_off_the_loop(slow_clear, ticks)


@pytest.mark.asyncio
async def test_signup_consent_clears_the_cache_off_the_event_loop(slow_clear, users):
    with patch.object(
        user_db, "_write_signup_consent", AsyncMock(return_value=(True, False))
    ):
        ticks = await _ticks_during(
            user_db.record_signup_consent("user-1", "2026-10", False)
        )
    _assert_cleared_off_the_loop(slow_clear, ticks)
