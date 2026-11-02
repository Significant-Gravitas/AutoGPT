"""Tests for the Slack pending-install marker."""

from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.bot.adapters.slack import pending

_P = "backend.copilot.bot.adapters.slack.pending"


class _FakeRedis:
    """Enough of the Redis surface to round-trip one key."""

    def __init__(self):
        self.store: dict[str, str] = {}

    async def set(self, key: str, value: str, ex: int | None = None):
        self.store[key] = value
        self.ex = ex

    async def get(self, key: str):
        return self.store.get(key)

    async def delete(self, key: str):
        self.store.pop(key, None)


def _redis(client):
    return patch(f"{_P}.get_redis_async", AsyncMock(return_value=client))


@pytest.mark.asyncio
async def test_marker_round_trips_through_redis():
    client = _FakeRedis()
    install = pending.PendingSlackInstall(team_id="T1", team_name="Acme", app_id="A1")
    with _redis(client):
        await pending.mark_pending("user-1", install)
        loaded = await pending.get_pending("user-1")

    assert loaded == install
    # The key is what the settings route reads back; a rename would silently
    # strand every in-flight install.
    assert "copilot:bot:slack:pending-install:user-1" in client.store
    assert client.ex == pending._TTL_SECONDS


@pytest.mark.asyncio
async def test_marker_is_scoped_to_one_user():
    client = _FakeRedis()
    with _redis(client):
        await pending.mark_pending("user-1", pending.PendingSlackInstall(team_id="T1"))
        assert await pending.get_pending("user-2") is None


@pytest.mark.asyncio
async def test_clearing_removes_the_marker():
    client = _FakeRedis()
    with _redis(client):
        await pending.mark_pending("user-1", pending.PendingSlackInstall(team_id="T1"))
        await pending.clear_pending("user-1")
        assert await pending.get_pending("user-1") is None


@pytest.mark.asyncio
async def test_missing_marker_reads_as_none():
    with _redis(_FakeRedis()):
        assert await pending.get_pending("nobody") is None


@pytest.mark.asyncio
async def test_a_redis_outage_never_breaks_the_install_flow():
    # The marker is a UX hint; Redis being down must not fail the OAuth
    # callback or the settings page.
    down = AsyncMock(side_effect=RuntimeError("redis is down"))
    with patch(f"{_P}.get_redis_async", down):
        await pending.mark_pending("user-1", pending.PendingSlackInstall(team_id="T1"))
        await pending.clear_pending("user-1")
        assert await pending.get_pending("user-1") is None


@pytest.mark.asyncio
async def test_a_read_failure_reads_as_none():
    client = _FakeRedis()
    client.get = AsyncMock(side_effect=RuntimeError("connection reset"))
    with _redis(client):
        assert await pending.get_pending("user-1") is None


@pytest.mark.asyncio
async def test_unreadable_payloads_read_as_none():
    client = _FakeRedis()
    with _redis(client):
        client.store[pending._key("user-1")] = "not json"
        assert await pending.get_pending("user-1") is None
        # Valid JSON, wrong shape: team_id is required.
        client.store[pending._key("user-1")] = '{"team_name": "Acme"}'
        assert await pending.get_pending("user-1") is None


def test_bot_dm_url_points_at_the_app_in_that_workspace():
    assert (
        pending.bot_dm_url("A1", "T1")
        == "https://slack.com/app_redirect?app=A1&team=T1"
    )
