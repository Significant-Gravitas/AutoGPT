from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.credential_selection import (
    remember_selection,
    selected_credentials,
)


class _FakeRedis:
    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiries: dict[str, int] = {}

    async def hset(self, key, field, value):
        self.hashes.setdefault(key, {})[field] = value

    async def expire(self, key, ttl):
        self.expiries[key] = ttl

    async def hgetall(self, key):
        # Redis hands back bytes unless decoding is configured.
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}


@pytest.fixture
def redis():
    fake = _FakeRedis()
    with patch(
        "backend.copilot.credential_selection.get_redis_async",
        new=AsyncMock(return_value=fake),
    ):
        yield fake


@pytest.mark.asyncio
async def test_a_pick_is_kept_for_the_session(redis):
    await remember_selection("s-1", {"github": "cred-a"})
    assert await selected_credentials("s-1") == {"github": "cred-a"}
    assert await selected_credentials("s-2") == {}
    assert redis.expiries  # a session's picks do not live forever


@pytest.mark.asyncio
async def test_a_later_pick_replaces_the_earlier_one_for_that_provider(redis):
    await remember_selection("s-1", {"github": "cred-a", "linear": "cred-l"})
    await remember_selection("s-1", {"github": "cred-b"})
    assert await selected_credentials("s-1") == {"github": "cred-b", "linear": "cred-l"}


@pytest.mark.asyncio
async def test_an_unreadable_selection_means_nothing_was_picked():
    with patch(
        "backend.copilot.credential_selection.get_redis_async",
        new=AsyncMock(side_effect=ConnectionError("redis down")),
    ):
        assert await selected_credentials("s-1") == {}
    assert await selected_credentials(None) == {}
