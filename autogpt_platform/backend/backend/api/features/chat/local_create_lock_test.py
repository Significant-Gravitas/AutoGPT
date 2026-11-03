import asyncio
from typing import Any

import pytest
from fastapi import HTTPException

from backend.api.features.chat import local_create_lock
from backend.api.features.chat.local_create_lock import client_named_local_create


class _LockRedis:
    """The three calls AsyncClusterLock makes: SET NX, GET and the release
    script, which deletes the key only for its owner."""

    def __init__(self) -> None:
        self.values: dict[str, str] = {}

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        if nx and key in self.values:
            return False
        self.values[key] = value
        return True

    async def get(self, key: str) -> str | None:
        return self.values.get(key)

    async def eval(self, _script: str, _numkeys: int, key: str, owner: str) -> int:
        if self.values.get(key) != owner:
            return 0
        del self.values[key]
        return 1


@pytest.fixture
def redis(monkeypatch: pytest.MonkeyPatch) -> _LockRedis:
    fake = _LockRedis()

    async def get_redis_async() -> Any:
        return fake

    monkeypatch.setattr(local_create_lock, "get_redis_async", get_redis_async)
    return fake


async def _hold(
    session_id: str | None, entered: asyncio.Event, release: asyncio.Event
) -> None:
    async with client_named_local_create("user-1", session_id):
        entered.set()
        await release.wait()


@pytest.mark.asyncio
async def test_a_resend_waits_for_the_first_create_to_finish(
    redis: _LockRedis,
) -> None:
    """The resend must not look for the session until the first attempt has
    kept or deleted it."""
    first_in, first_done = asyncio.Event(), asyncio.Event()
    second_in, second_done = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(_hold("session-1", first_in, first_done))
    await asyncio.wait_for(first_in.wait(), timeout=5)
    second = asyncio.create_task(_hold("session-1", second_in, second_done))

    # Several poll intervals pass without the resend getting in.
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(second_in.wait(), timeout=1)

    first_done.set()
    await asyncio.wait_for(second_in.wait(), timeout=5)
    second_done.set()
    await asyncio.wait_for(asyncio.gather(first, second), timeout=5)
    assert redis.values == {}


@pytest.mark.asyncio
async def test_different_sessions_do_not_wait_for_each_other(
    redis: _LockRedis,
) -> None:
    first_in, first_done = asyncio.Event(), asyncio.Event()
    other_in, other_done = asyncio.Event(), asyncio.Event()
    first = asyncio.create_task(_hold("session-1", first_in, first_done))
    other = asyncio.create_task(_hold("session-2", other_in, other_done))

    await asyncio.wait_for(asyncio.gather(first_in.wait(), other_in.wait()), 5)
    first_done.set()
    other_done.set()
    await asyncio.wait_for(asyncio.gather(first, other), timeout=5)


@pytest.mark.asyncio
async def test_a_server_named_session_takes_no_lock(redis: _LockRedis) -> None:
    async with client_named_local_create("user-1", None):
        assert redis.values == {}


@pytest.mark.asyncio
async def test_a_resend_gives_up_when_the_first_create_never_finishes(
    redis: _LockRedis, monkeypatch: pytest.MonkeyPatch
) -> None:
    redis.values["copilot:local-session-create:user-1:session-1"] = "stuck"
    clock = [0.0]

    async def advance(seconds: float) -> None:
        clock[0] += seconds

    monkeypatch.setattr(local_create_lock, "monotonic", lambda: clock[0])
    monkeypatch.setattr(local_create_lock, "sleep", advance)

    with pytest.raises(HTTPException) as exc_info:
        async with client_named_local_create("user-1", "session-1"):
            pytest.fail("entered a create another attempt still holds")

    assert exc_info.value.status_code == 503
    assert redis.values["copilot:local-session-create:user-1:session-1"] == "stuck"
