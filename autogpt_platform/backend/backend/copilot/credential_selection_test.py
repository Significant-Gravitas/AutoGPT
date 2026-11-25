from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.credential_selection import (
    credential_matches_selection,
    host_selection_key,
    remember_selection,
    selected_credentials,
)
from backend.data.model import HostScopedCredentials


@pytest.mark.parametrize(
    "url, expected",
    [
        ("https://API.Linear.app:443/graphql", "http:api.linear.app"),
        ("http://api.linear.app:80/graphql", "http:api.linear.app"),
        ("https://api.linear.app:8443/graphql", "http:api.linear.app:8443"),
        ("api.linear.app/graphql", "http:api.linear.app"),
        ("https://[::1]:8443/path", "http:[::1]:8443"),
        ("https://[invalid", "http"),
        ("https://api.linear.app:invalid", "http"),
    ],
)
def test_host_selection_keys_match_browser_hosts(url, expected):
    assert host_selection_key("http", url) == expected


class _FakeRedis:
    """Enough of redis.asyncio for the store: a transactional pipeline that
    applies its queued commands only on execute."""

    def __init__(self):
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiries: dict[str, int] = {}

    def pipeline(self, transaction: bool = True):
        assert transaction, "picks must be written atomically"
        return _FakePipeline(self)

    async def hgetall(self, key):
        # Redis hands back bytes unless decoding is configured.
        return {k.encode(): v.encode() for k, v in self.hashes.get(key, {}).items()}


class _FakePipeline:
    def __init__(self, redis: _FakeRedis):
        self.redis = redis
        self.queued: list = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def hset(self, key, field, value):
        self.queued.append(("hset", key, field, value))

    def expire(self, key, ttl):
        self.queued.append(("expire", key, ttl))

    async def execute(self):
        for command in self.queued:
            if command[0] == "hset":
                _, key, field, value = command
                self.redis.hashes.setdefault(key, {})[field] = value
            else:
                _, key, ttl = command
                self.redis.expiries[key] = ttl
        self.queued = []


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


@pytest.mark.parametrize(
    "host, key, expected",
    [
        ("api.linear.app:443", "http:api.linear.app", True),
        ("api.linear.app:80", "http:api.linear.app", True),
        ("api.linear.app:8443", "http:api.linear.app:8443", True),
        ("api.linear.app:8443", "http:api.linear.app", False),
        ("*.linear.app", "http:api.linear.app", True),
        ("api.github.com", "http:api.linear.app", False),
        ("api.linear.app", "github:api.linear.app", False),
        ("api.linear.app", "http:api.linear.app/path", False),
        ("api.linear.app", "http:api.linear.app:bad", False),
    ],
)
def test_host_selection_validates_the_credential_scope(host, key, expected):
    credential = HostScopedCredentials(id="key", provider="http", host=host)
    assert credential_matches_selection(credential, key) is expected
