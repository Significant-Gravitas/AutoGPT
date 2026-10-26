"""Copilot and the idle sweep must share a rotating credential's refresh."""

import asyncio
import time
from unittest.mock import AsyncMock, MagicMock, PropertyMock

import pytest
from pydantic import SecretStr
from pytest_mock import MockerFixture
from redis.asyncio.lock import Lock

from backend.blocks.airtable._oauth import AirtableOAuthHandler
from backend.blocks.linear._oauth import LinearOAuthHandler
from backend.copilot import integration_creds
from backend.data.model import OAuth2Credentials
from backend.integrations.credential_refresh import REFRESH_PENDING, REFRESH_REVISION
from backend.integrations.creds_manager import IntegrationCredentialsManager
from backend.integrations.oauth.base import BaseOAuthHandler
from backend.integrations.oauth.github import GitHubOAuthHandler

_USER = "refresh-race-owner"


@pytest.fixture(
    params=[
        pytest.param(GitHubOAuthHandler, id="github"),
        pytest.param(AirtableOAuthHandler, id="airtable"),
        pytest.param(LinearOAuthHandler, id="linear"),
    ]
)
def scenario(request: pytest.FixtureRequest, mocker: MockerFixture):
    handler = request.param("fake-client", "fake-secret", "https://example.com")
    return _RefreshScenario(mocker, handler)


async def test_copilot_reloads_a_snapshot_taken_before_the_idle_sweep(scenario):
    await asyncio.wait_for(scenario.manager.refresh_idle_credentials(), timeout=2)

    token = await asyncio.wait_for(
        integration_creds.get_provider_token(_USER, scenario.stored.provider), timeout=2
    )

    scenario.assert_one_refresh(token)


async def test_copilot_waits_for_the_idle_sweep_to_finish_rotating(scenario):
    scenario.finish_refresh.clear()
    sweep = asyncio.create_task(scenario.manager.refresh_idle_credentials())
    tasks = [sweep]
    try:
        await asyncio.wait_for(scenario.refresh_started.wait(), timeout=2)
        copilot = asyncio.create_task(
            integration_creds.get_provider_token(_USER, scenario.stored.provider)
        )
        tasks.append(copilot)
        await asyncio.wait_for(scenario.snapshot_read.wait(), timeout=2)
        scenario.finish_refresh.set()
        _, token = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)

        scenario.assert_one_refresh(token)
    finally:
        scenario.finish_refresh.set()
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class _RefreshScenario:
    def __init__(self, mocker: MockerFixture, handler: BaseOAuthHandler):
        self.stored = OAuth2Credentials(
            id="refresh-race-credential",
            provider=handler.PROVIDER_NAME,
            access_token=SecretStr("access-old"),
            refresh_token=SecretStr("refresh-old"),
            access_token_expires_at=int(time.time()) - 15 * 86400,
            scopes=["read"],
        )
        self.snapshot = self.stored.model_copy(deep=True)
        self.attempts: list[str] = []
        self.provider_refresh_token = "refresh-old"
        self.locks: dict[tuple[str, ...], asyncio.Lock] = {}
        self.owners: dict[tuple[str, ...], object] = {}
        self.checkpoints: dict[str, OAuth2Credentials] = {}
        self.refresh_started = asyncio.Event()
        self.finish_refresh = asyncio.Event()
        self.finish_refresh.set()
        self.snapshot_read = asyncio.Event()
        self.manager = IntegrationCredentialsManager()
        self._mock_storage(mocker)
        self._mock_runtime(mocker, handler)

    def _mock_storage(self, mocker: MockerFixture):
        mocker.patch.object(
            self.manager.store, "get_creds_by_id", AsyncMock(side_effect=self.read)
        )
        mocker.patch.object(
            self.manager.store,
            "get_creds_by_provider",
            AsyncMock(side_effect=self.read_snapshot),
        )
        mocker.patch.object(
            self.manager.store,
            "_get_persisted_user_creds_unlocked",
            AsyncMock(side_effect=self.read_persisted),
        )
        self.prepares = mocker.patch.object(
            self.manager.store, "begin_refresh", AsyncMock(side_effect=self.prepare)
        )
        self.writes = mocker.patch.object(
            self.manager.store, "complete_refresh", AsyncMock(side_effect=self.write)
        )
        mocker.patch.object(
            type(self.manager.store),
            "db_manager",
            new_callable=PropertyMock,
            return_value=MagicMock(
                get_near_expiry_oauth_credentials=AsyncMock(
                    return_value=[(_USER, self.snapshot)]
                )
            ),
        )

    def _mock_runtime(self, mocker: MockerFixture, handler: BaseOAuthHandler):
        mocker.patch.object(
            self.manager, "_get_oauth_handler", AsyncMock(return_value=handler)
        )
        mocker.patch.object(
            self.manager,
            "_fresh_refresh_lock",
            AsyncMock(side_effect=self.acquire),
        )
        mocker.patch(
            "backend.integrations.creds_manager._invoke_creds_changed_hook", AsyncMock()
        )
        mocker.patch(
            "backend.util.request.Requests.post", AsyncMock(side_effect=self.exchange)
        )
        mocker.patch.object(
            GitHubOAuthHandler, "_request_username", AsyncMock(return_value="fake-user")
        )
        mocker.patch.object(integration_creds, "_manager", self.manager)
        mocker.patch.object(integration_creds, "_ensure_cache_invalidation_listener")
        mocker.patch.object(integration_creds, "_token_cache", {})
        mocker.patch.object(integration_creds, "_credential_id_cache", {})
        mocker.patch.object(integration_creds, "_null_cache", {})

    async def read(self, user_id: str, credential_id: str):
        assert (user_id, credential_id) == (_USER, self.stored.id)
        return self.stored.model_copy(deep=True)

    async def read_snapshot(self, user_id: str, provider: str):
        assert (user_id, provider) == (_USER, self.stored.provider)
        self.snapshot_read.set()
        return [self.snapshot.model_copy(deep=True)]

    async def read_persisted(self, user_id: str):
        assert user_id == _USER
        return [self.stored.model_copy(deep=True)]

    async def prepare(
        self, user_id: str, expected: OAuth2Credentials, nonce: str, deadline: float
    ):
        assert (user_id, expected.id) == (_USER, self.stored.id)
        if (
            time.time() >= deadline
            or self.stored != expected
            or self.stored.metadata.get(REFRESH_PENDING)
        ):
            return None
        self.stored = expected.model_copy(
            deep=True,
            update={
                "access_token": SecretStr(""),
                "refresh_token": None,
                "provider_state": None,
                "access_token_expires_at": 1,
                "metadata": {
                    **expected.metadata,
                    REFRESH_PENDING: nonce,
                    REFRESH_REVISION: nonce + ":pending",
                },
            },
        )
        checkpoint = "checkpoint:" + nonce
        self.checkpoints[checkpoint] = self.stored.model_copy(deep=True)
        return checkpoint

    async def write(
        self,
        user_id: str,
        credentials: OAuth2Credentials,
        checkpoint: str,
        deadline: float,
    ):
        assert (user_id, credentials.id) == (_USER, self.stored.id)
        assert not credentials.metadata.get(REFRESH_PENDING)
        if time.time() >= deadline or self.checkpoints.get(checkpoint) != self.stored:
            return False
        self.stored = credentials.model_copy(deep=True)
        return True

    async def acquire(self, key: tuple[str, ...], *, blocking_timeout=None):
        mutex = self.locks.setdefault(key, asyncio.Lock())
        await asyncio.wait_for(mutex.acquire(), timeout=blocking_timeout)
        owner = object()
        self.owners[key] = owner

        def release():
            assert self.owners.get(key) is owner
            del self.owners[key]
            mutex.release()

        lock = AsyncMock(spec=Lock)
        lock.name = str(key)
        lock.locked.side_effect = mutex.locked
        lock.owned.side_effect = lambda: self.owners.get(key) is owner
        lock.extend = AsyncMock(
            side_effect=lambda *_args, **_kwargs: self.owners.get(key) is owner
        )
        lock.release.side_effect = release
        return lock

    async def exchange(self, _url: str, *, data: dict[str, str], **_kwargs):
        self.attempts.append(data["refresh_token"])
        assert self.stored.metadata.get(REFRESH_PENDING)
        assert self.stored.access_token.get_secret_value() == ""
        assert self.stored.refresh_token is None
        assert len(self.owners) == 3
        self.refresh_started.set()
        await self.finish_refresh.wait()
        if data["refresh_token"] != self.provider_refresh_token:
            raise ValueError("invalid_grant: refresh token already consumed")
        self.provider_refresh_token = "refresh-new"
        return MagicMock(
            ok=True,
            json=MagicMock(
                return_value={
                    "access_token": "access-new",
                    "refresh_token": "refresh-new",
                    "expires_in": 3600,
                    "refresh_expires_in": 60 * 86400,
                    "refresh_token_expires_in": 180 * 86400,
                    "token_type": "Bearer",
                    "scope": "read",
                }
            ),
        )

    def assert_one_refresh(self, token: str | None):
        assert token == "access-new"
        assert self.attempts == ["refresh-old"]
        self.prepares.assert_awaited_once()
        self.writes.assert_awaited_once()
        nonce = self.prepares.await_args.args[2]
        assert self.stored.metadata[REFRESH_REVISION] == nonce + ":complete"
        assert REFRESH_PENDING not in self.stored.metadata
        assert self.stored.refresh_token is not None
        assert self.stored.refresh_token.get_secret_value() == "refresh-new"
        assert self.stored.access_token_expires_at is not None
        assert self.stored.access_token_expires_at > time.time()
        assert not self.owners
        assert all(not lock.locked() for lock in self.locks.values())
