"""Shared fixtures for the Exa block tests."""

from unittest.mock import patch

import pytest

from backend.blocks.exa import agent_billing


class _FakeRedis:
    """Just enough of Redis for the Agent billing claim: ``SET key NX``."""

    def __init__(self) -> None:
        self.keys: set[str] = set()

    async def set(self, key: str, value: str, *, nx: bool, ex: int) -> bool:
        if nx and key in self.keys:
            return False
        self.keys.add(key)
        return True


@pytest.fixture(autouse=True)
def billing_claims():
    """Keep the once-per-run billing claim in memory instead of Redis."""
    fake = _FakeRedis()

    async def get_fake_redis() -> _FakeRedis:
        return fake

    with patch.object(agent_billing, "get_redis_async", get_fake_redis):
        yield fake
