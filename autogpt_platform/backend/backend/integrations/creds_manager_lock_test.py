"""Rotating Copilot refreshes must not wait out a long-lived credential lease."""

import asyncio
import socket
import time
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from autogpt_libs.utils.synchronize import AsyncRedisKeyedMutex
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.lock import Lock

from backend.copilot import integration_creds
from backend.data import redis_client
from backend.data.model import OAuth2Credentials
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth.github import GitHubOAuthHandler

# Each wait below only stops a hung test. A CI shard runs its tests one after
# another on one event loop against a real Redis cluster, so work that is
# instant on its own can take over a second there.
_HANG_SECONDS = 10


@pytest.fixture
async def scenario(mocker):
    try:
        socket.create_connection((redis_client.HOST, redis_client.PORT), 1).close()
    except OSError:
        pytest.skip("local redis cluster not reachable")
    redis = RedisCluster(
        host=redis_client.HOST,
        port=redis_client.PORT,
        password=redis_client.PASSWORD,
        address_remap=redis_client._address_remap,
        socket_connect_timeout=1,
        socket_timeout=1,
    )
    test = _RefreshScenario(mocker, redis)
    try:
        yield test
    finally:
        await test.mutex.release_all_locks()
        await redis.delete(*test.lock_keys)
        await redis.aclose()


@pytest.mark.parametrize("sweep_waiting", [False, True])
async def test_copilot_times_out_without_releasing_the_lease_holder(
    scenario, sweep_waiting
):
    lease = await scenario.manager.acquire_lease(scenario.user, scenario.credentials.id)
    scenario.expire()
    sweep = None
    try:
        if sweep_waiting:
            scenario.read_started.clear()
            sweep = asyncio.create_task(
                scenario.manager.refresh_if_needed(scenario.user, scenario.credentials)
            )
            await asyncio.wait_for(scenario.read_started.wait(), timeout=_HANG_SECONDS)
        token = await asyncio.wait_for(scenario.get_token(), timeout=_HANG_SECONDS)
        assert token is None
        await lease.validate()
        scenario.refresh.assert_not_awaited()
        scenario.write.assert_not_awaited()
    finally:
        if sweep is not None:
            sweep.cancel()
            await asyncio.gather(sweep, return_exceptions=True)
        await lease.release()

    assert await scenario.redis.exists(*scenario.lock_keys) == 0
    assert (
        await asyncio.wait_for(scenario.get_token(), timeout=_HANG_SECONDS)
        == "new-access"
    )


async def test_default_refresh_still_waits_for_the_lease(scenario):
    lease = await scenario.manager.acquire_lease(scenario.user, scenario.credentials.id)
    scenario.expire()
    refresh = asyncio.create_task(
        scenario.manager.refresh_if_needed(scenario.user, scenario.credentials)
    )
    try:
        await asyncio.sleep(0.1)
        assert not refresh.done()
        await lease.validate()
        await lease.release()
        fresh = await asyncio.wait_for(refresh, timeout=_HANG_SECONDS)
        assert fresh.access_token.get_secret_value() == "new-access"
    finally:
        refresh.cancel()
        await asyncio.gather(refresh, return_exceptions=True)
        await lease.release()
    assert await scenario.redis.exists(*scenario.lock_keys) == 0


async def test_lock_wait_timeout_does_not_cancel_the_provider_refresh(scenario):
    scenario.expire()
    started, finish = asyncio.Event(), asyncio.Event()

    async def refresh(_credentials):
        started.set()
        await finish.wait()
        return scenario.fresh()

    scenario.refresh.side_effect = refresh
    lookup = asyncio.create_task(scenario.get_token())
    try:
        await asyncio.wait_for(started.wait(), timeout=_HANG_SECONDS)
        await asyncio.sleep(0.1)
        assert not lookup.done()
        finish.set()
        assert await asyncio.wait_for(lookup, timeout=_HANG_SECONDS) == "new-access"
    finally:
        finish.set()
        lookup.cancel()
        await asyncio.gather(lookup, return_exceptions=True)
    assert await scenario.redis.exists(*scenario.lock_keys) == 0


@pytest.mark.parametrize("credential_lock_acquired", [False, True])
async def test_cancelled_copilot_lookup_releases_only_its_own_locks(
    scenario, credential_lock_acquired, mocker
):
    mocker.patch("backend.integrations.creds_manager._REFRESH_LOCK_WAIT_SECONDS", 5.0)
    lease = None
    if not credential_lock_acquired:
        lease = await scenario.manager.acquire_lease(
            scenario.user, scenario.credentials.id
        )
    scenario.expire()
    blocked = _pause_lookup(scenario, mocker, credential_lock_acquired)
    lookup = asyncio.create_task(scenario.get_token())
    try:
        await asyncio.wait_for(blocked.wait(), timeout=_HANG_SECONDS)
        lookup.cancel()
        with pytest.raises(asyncio.CancelledError):
            await lookup
        assert await scenario.redis.exists(scenario.lock_keys[1]) == 0
        if lease is not None:
            await lease.validate()
        else:
            assert await scenario.redis.exists(scenario.lock_keys[0]) == 0
        scenario.refresh.assert_not_awaited()
    finally:
        lookup.cancel()
        await asyncio.gather(lookup, return_exceptions=True)
        if lease is not None:
            await lease.release()


def _pause_lookup(scenario, mocker, after_acquisition: bool) -> asyncio.Event:
    blocked = asyncio.Event()
    reads = 0
    acquire = Lock.do_acquire

    async def read(*_args):
        nonlocal reads
        reads += 1
        if after_acquisition and reads == 2:
            blocked.set()
            await asyncio.Event().wait()
        return scenario.credentials

    async def acquire_then_pause(lock: Lock, token):
        acquired = await acquire(lock, token)
        if not after_acquisition and not acquired:
            blocked.set()
            await asyncio.Event().wait()
        return acquired

    scenario.manager.store.get_creds_by_id.side_effect = read
    mocker.patch.object(Lock, "do_acquire", acquire_then_pause)
    return blocked


class _RefreshScenario:
    def __init__(self, mocker, redis: RedisCluster):
        self.user = f"refresh-lock-test-{uuid4()}"
        self.redis = redis
        self.mutex = AsyncRedisKeyedMutex(redis)
        self.manager = IntegrationCredentialsManager()
        self.credentials = OAuth2Credentials(
            id="credential-id",
            provider="github",
            access_token="old-access",
            refresh_token="old-refresh",
            access_token_expires_at=int(time.time()) + 3600,
            scopes=["read"],
        )
        self.read_started = asyncio.Event()
        self.lock_keys = [
            str(
                self.manager._credentials_lock_key(
                    self.user, self.credentials.id, *args
                )
            )
            for args in [(), ("refresh",), ("!time_sensitive",)]
        ]
        self._mock_storage(mocker)
        self._mock_runtime(mocker)
        self.clock = mocker.patch("time.time", return_value=time.time())

    def _mock_storage(self, mocker):
        mocker.patch.object(self.manager, "locks", AsyncMock(return_value=self.mutex))
        mocker.patch.object(
            self.manager.store, "get_creds_by_id", AsyncMock(side_effect=self.read)
        )
        mocker.patch.object(
            self.manager.store,
            "get_creds_by_provider",
            AsyncMock(side_effect=lambda *_args: [self.credentials]),
        )
        self.write = mocker.patch.object(
            self.manager.store, "update_creds", AsyncMock()
        )

    def _mock_runtime(self, mocker):
        handler = GitHubOAuthHandler("client", "secret", "https://example.com")
        self.refresh = mocker.patch.object(
            handler,
            "refresh_tokens",
            AsyncMock(side_effect=lambda _creds: self.fresh()),
        )
        mocker.patch.object(
            self.manager, "_get_oauth_handler", AsyncMock(return_value=handler)
        )
        mocker.patch(
            "backend.integrations.creds_manager._REFRESH_LOCK_WAIT_SECONDS",
            0.02,
        )
        mocker.patch("backend.integrations.creds_manager._invoke_creds_changed_hook")
        mocker.patch.object(integration_creds, "_manager", self.manager)
        mocker.patch.object(integration_creds, "_ensure_cache_invalidation_listener")
        mocker.patch.object(integration_creds, "_token_cache", {})
        mocker.patch.object(integration_creds, "_null_cache", {})

    async def read(self, user_id, credential_id):
        assert (user_id, credential_id) == (self.user, self.credentials.id)
        self.read_started.set()
        return self.credentials

    def expire(self):
        self.clock.return_value = self.credentials.access_token_expires_at + 1

    def fresh(self):
        return OAuth2Credentials(
            id=self.credentials.id,
            provider="github",
            access_token="new-access",
            refresh_token="new-refresh",
            access_token_expires_at=int(time.time()) + 3600,
            scopes=["read"],
        )

    async def get_token(self):
        return await integration_creds.get_provider_token(self.user, "github")
