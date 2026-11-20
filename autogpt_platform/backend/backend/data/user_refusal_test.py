"""The gate for background work: a banned or a closed account gets none."""

import asyncio
from typing import Any
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from pytest_mock import MockerFixture
from redis.exceptions import ConnectionError as RedisConnectionError

from backend.data import user as user_module
from backend.data.user import account_refusal, forget_refusal, is_user_refused
from backend.util import cache as cache_module


@pytest.fixture
def lookups(mocker: MockerFixture) -> MagicMock:
    parent = MagicMock()
    parent.attach_mock(
        mocker.patch.object(user_module, "is_user_banned", AsyncMock()), "banned"
    )
    parent.attach_mock(
        mocker.patch.object(user_module, "is_user_account_closed", AsyncMock()),
        "closed",
    )
    return parent


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "banned, closed, refused",
    [(True, False, True), (False, True, True), (False, False, False)],
)
async def test_a_banned_or_closed_account_is_refused(
    lookups: MagicMock, banned: bool, closed: bool, refused: bool
):
    lookups.banned.return_value = banned
    lookups.closed.return_value = closed

    assert await is_user_refused("user-1") is refused


@pytest.mark.asyncio
async def test_a_ban_answers_without_the_closed_lookup(lookups: MagicMock):
    lookups.banned.return_value = True

    assert await is_user_refused("user-1") is True
    lookups.closed.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_failed_lookup_propagates(lookups: MagicMock):
    """Each gate decides what an unknown means."""
    lookups.banned.return_value = False
    lookups.closed.side_effect = ConnectionError("db manager down")

    with pytest.raises(ConnectionError):
        await is_user_refused("user-1")


def test_forget_refusal_drops_the_cached_closed_answer(mocker: MockerFixture):
    drop = mocker.patch.object(user_module.is_user_account_closed, "cache_delete")

    forget_refusal("user-1")

    drop.assert_called_once_with("user-1")


def test_forget_refusal_never_raises(mocker: MockerFixture):
    mocker.patch.object(
        user_module.is_user_account_closed,
        "cache_delete",
        side_effect=ConnectionError("redis down"),
    )

    forget_refusal("user-1")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "banned, closed, reason",
    [(True, True, "banned"), (False, True, "closed"), (False, False, None)],
)
async def test_the_refusal_says_why_and_a_ban_wins(
    lookups: MagicMock, banned: bool, closed: bool, reason: str | None
):
    lookups.banned.return_value = banned
    lookups.closed.return_value = closed

    assert await account_refusal("user-1") == reason


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "lookup, db_method",
    [
        (user_module.is_user_banned, "is_auth_user_banned"),
        (user_module.is_user_account_closed, "is_account_closed"),
    ],
    ids=["ban", "closure"],
)
async def test_distinct_users_lookups_all_reach_the_database_at_once_without_redis(
    lookup: Any, db_method: str, monkeypatch: pytest.MonkeyPatch
):
    """Every message a WebSocket handles checks its account, and while Redis
    can't be reached every check misses. Distinct users' lookups all go to the
    database at once, so the checks take one lookup's time, not a queue's: a
    queue behind the shared cache's default cap on misses would run a busy
    process's checks past their 5 s bound, and close its sockets with 1013."""

    async def unreachable():
        raise RedisConnectionError("Redis is unreachable")

    monkeypatch.setattr(cache_module, "_get_async_redis", unreachable)
    users = [
        f"user-{uuid4().hex}" for _ in range(4 * cache_module.MAX_CONCURRENT_MISSES)
    ]
    everyone_waiting, release = asyncio.Event(), asyncio.Event()
    waiting = 0

    async def slow_lookup(user_id: str) -> bool:
        nonlocal waiting
        waiting += 1
        if waiting == len(users):
            everyone_waiting.set()
        await release.wait()
        return False

    db = MagicMock()
    setattr(db, db_method, slow_lookup)
    monkeypatch.setattr(user_module, "user_db", lambda: db)
    checks = [asyncio.create_task(lookup(user)) for user in users]
    try:
        async with asyncio.timeout(10):
            await everyone_waiting.wait()
            release.set()
            assert await asyncio.gather(*checks) == [False] * len(users)
    finally:
        release.set()
        async with asyncio.timeout(10):
            await asyncio.gather(*checks, return_exceptions=True)
