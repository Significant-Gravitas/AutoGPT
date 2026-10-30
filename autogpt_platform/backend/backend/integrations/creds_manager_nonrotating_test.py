from contextlib import asynccontextmanager, nullcontext
from copy import copy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pydantic import SecretStr

from backend.data import db as db_module
from backend.data import user as user_module
from backend.data.model import OAuth2Credentials
from backend.integrations import creds_manager as manager_module
from backend.integrations.credential_refresh import (
    REFRESH_PENDING,
    REFRESH_REVISION,
    CredentialRefreshRequired,
)
from backend.integrations.credentials_store import IntegrationCredentialsStore
from backend.integrations.oauth.google import GoogleOAuthHandler
from backend.util.encryption import JSONCryptor


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", [True, False], ids=["locked", "lock-free"])
@pytest.mark.parametrize("upgraded", [True, False], ids=["scope-upgrade", "legacy"])
async def test_refresh_persists_after_handler_drops_revision(rig, lock, upgraded):
    if upgraded:
        await rig.reconnect()
    before = rig.database.credential
    writes_before = rig.database.writes

    refreshed = await rig.manager.get("user-a", before.id, lock=lock)

    assert refreshed == rig.database.credential
    assert refreshed.access_token == SecretStr("synthetic-refreshed-access")
    assert refreshed.refresh_token == before.refresh_token
    assert refreshed.scopes == before.scopes
    assert refreshed.metadata == {
        key: value for key, value in before.metadata.items() if key == REFRESH_REVISION
    }
    assert rig.database.writes == writes_before + 1
    assert await rig.manager.get("user-a", before.id, lock=lock) == refreshed
    rig.exchange.assert_awaited_once()
    rig.notify.assert_awaited_once_with("user-a", "google")


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", [True, False])
@pytest.mark.parametrize("upgraded", [True, False])
async def test_revision_is_captured_before_handler_mutates_input(rig, lock, upgraded):
    if upgraded:
        await rig.reconnect()
    expected_revision = rig.database.credential.metadata.get(REFRESH_REVISION)

    async def exchange(credentials):
        credentials.metadata[REFRESH_REVISION] = "handler-revision"
        fresh = rig.fresh(credentials)
        fresh.metadata = {REFRESH_REVISION: "handler-revision", "provider_value": "new"}
        return fresh

    rig.exchange.side_effect = exchange
    refreshed = await rig.manager.get("user-a", "credential-a", lock=lock)

    assert refreshed == rig.database.credential
    expected_metadata = {"provider_value": "new"}
    if expected_revision is not None:
        expected_metadata[REFRESH_REVISION] = expected_revision
    assert refreshed.metadata == expected_metadata


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", [True, False])
@pytest.mark.parametrize("upgraded", [True, False])
@pytest.mark.parametrize("rewrite_revision", [True, False])
async def test_reconnect_during_exchange_rejects_stale_refresh(
    rig, lock, upgraded, rewrite_revision
):
    if upgraded:
        await rig.reconnect()

    async def exchange(credentials):
        winner = await rig.reconnect(upgrade=False)
        fresh = rig.fresh(credentials)
        if rewrite_revision:
            revision = winner.metadata[REFRESH_REVISION]
            credentials.metadata[REFRESH_REVISION] = revision
            fresh.metadata[REFRESH_REVISION] = revision
        return fresh

    rig.exchange.side_effect = exchange
    writes_before = rig.database.writes

    with pytest.raises(ValueError, match="Credential changed before it could be saved"):
        await rig.manager.get("user-a", "credential-a", lock=lock)

    assert rig.database.credential.access_token == SecretStr("synthetic-reconnected")
    assert rig.database.writes == writes_before + 1
    rig.notify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("lock_boundary", ["refresh", "credential"])
async def test_locked_refresh_uses_revision_reloaded_after_either_lock(
    rig, lock_boundary, monkeypatch
):
    original_acquire = rig.manager._acquire_lock

    async def acquire(user_id, credential_id, *args, **kwargs):
        if ("refresh" if args else "credential") == lock_boundary:
            await rig.reconnect()
        return await original_acquire(user_id, credential_id, *args, **kwargs)

    monkeypatch.setattr(rig.manager, "_acquire_lock", acquire)
    refreshed = await rig.manager.get("user-a", "credential-a")

    assert refreshed == rig.database.credential
    assert refreshed.metadata[REFRESH_REVISION]
    assert refreshed.scopes == ["read", "write"]
    assert rig.database.writes == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", [True, False])
@pytest.mark.parametrize("boundary", ["owner", "payload", "deadline"])
async def test_failed_database_cas_cannot_publish_refreshed_credentials(
    rig, lock, boundary
):
    await rig.reconnect()
    rig.database.reject_cas = boundary
    ciphertext = rig.database.row.encryptedPayload
    writes_before = rig.database.writes

    with pytest.raises(ValueError, match="Credential changed before it could be saved"):
        await rig.manager.get("user-a", "credential-a", lock=lock)

    assert rig.database.writes == writes_before
    if boundary == "payload":
        assert rig.database.row.encryptedPayload != ciphertext
    else:
        assert rig.database.row.encryptedPayload == ciphertext
    rig.notify.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("lock", [True, False])
@pytest.mark.parametrize("boundary", ["pending", "scopes"])
async def test_refresh_preserves_store_pending_and_scope_guards(rig, lock, boundary):
    await rig.reconnect()

    async def exchange(credentials):
        fresh = rig.fresh(credentials)
        if boundary == "scopes":
            fresh.scopes = ["read"]
        else:
            pending = rig.database.credential
            pending.metadata[REFRESH_PENDING] = "other-refresh"
            rig.database.save(pending)
        return fresh

    rig.exchange.side_effect = exchange
    writes_before = rig.database.writes
    error = CredentialRefreshRequired if boundary == "pending" else ValueError
    with pytest.raises(error):
        await rig.manager.get("user-a", "credential-a", lock=lock)
    assert rig.database.writes == writes_before
    rig.notify.assert_not_awaited()


@pytest.fixture
def rig(monkeypatch):
    return RefreshRig(monkeypatch)


class RefreshRig:
    def __init__(self, monkeypatch):
        self.database = CredentialDatabase()
        monkeypatch.setattr(user_module, "prisma", self.database)
        monkeypatch.setattr(db_module, "prisma", self.database)
        monkeypatch.setattr(
            IntegrationCredentialsStore, "db_manager", property(lambda _: user_module)
        )
        self.manager = manager_module.IntegrationCredentialsManager()
        monkeypatch.setattr(
            self.manager.store,
            "locked_user_integrations",
            AsyncMock(side_effect=lambda _: nullcontext()),
        )
        held_lock = SimpleNamespace(
            locked=AsyncMock(return_value=True),
            owned=AsyncMock(return_value=True),
            release=AsyncMock(),
        )
        monkeypatch.setattr(
            self.manager, "_acquire_lock", AsyncMock(return_value=held_lock)
        )
        handler = GoogleOAuthHandler("synthetic-client", "synthetic-secret", "")
        self.exchange = AsyncMock(side_effect=self.fresh)
        monkeypatch.setattr(handler, "_refresh_tokens", self.exchange)
        monkeypatch.setattr(
            self.manager, "_get_oauth_handler", AsyncMock(return_value=handler)
        )
        self.notify = AsyncMock()
        monkeypatch.setattr(manager_module, "_invoke_creds_changed_hook", self.notify)

    async def reconnect(self, *, upgrade=True):
        replacement = self.database.credential.model_copy(
            update={
                "access_token": SecretStr("synthetic-reconnected"),
                "scopes": (
                    ["read", "write"] if upgrade else self.database.credential.scopes
                ),
                "metadata": {"granted_scopes": ["read", "write"]},
            }
        )
        return await self.manager.store.upsert_single_provider_creds(
            "user-a", replacement
        )

    @staticmethod
    def fresh(credentials):
        return OAuth2Credentials(
            id=credentials.id,
            provider=credentials.provider,
            access_token=SecretStr("synthetic-refreshed-access"),
            refresh_token=credentials.refresh_token,
            access_token_expires_at=4_000_000_000,
            scopes=credentials.scopes,
        )


class CredentialDatabase:
    def __init__(self):
        self.cryptor = JSONCryptor()
        self.row = SimpleNamespace(
            id="credential-a", ownerType="USER", ownerId="user-a", status="active"
        )
        self.save(
            OAuth2Credentials(
                id=self.row.id,
                provider="google",
                access_token=SecretStr("synthetic-original-access"),
                refresh_token=SecretStr("synthetic-reusable-refresh"),
                access_token_expires_at=1,
                scopes=["read"],
            )
        )
        self.integrationcredential = self
        self.writes = 0
        self.reject_cas = None

    @property
    def credential(self):
        return OAuth2Credentials.model_validate(
            self.cryptor.decrypt(self.row.encryptedPayload)
        )

    def save(self, credential):
        self.row.encryptedPayload = self.cryptor.encrypt(credential.model_dump())

    @asynccontextmanager
    async def tx(self, **_kwargs):
        yield self

    async def find_many(self, *, where, **_kwargs):
        if all(vars(self.row)[key] == value for key, value in where.items()):
            return [copy(self.row)]
        return []

    async def query_raw(
        self, sql, credential_id, owner_id, expected, status, **_kwargs
    ):
        assert '"ownerType" = \'USER\' AND "ownerId" = $2' in sql
        assert '"status" = $4 AND "encryptedPayload" = $3' in sql
        assert "FOR UPDATE" in sql
        if self.reject_cas == "owner":
            self.row.ownerId = "other-user"
        elif self.reject_cas == "payload":
            self.save(self.credential)
        matches = (
            self.row.id == credential_id
            and self.row.ownerType == "USER"
            and self.row.ownerId == owner_id
            and self.row.encryptedPayload == expected
            and self.row.status == status
        )
        return [{"id": credential_id}] if matches else []

    async def execute_raw(self, sql, *args):
        if sql.startswith("SET LOCAL") or "pg_advisory_xact_lock" in sql:
            return 1
        updated, credential_id, owner_id, expected, deadline, status, _title = args
        assert '"ownerType" = \'USER\' AND "ownerId" = $3' in sql
        assert '"status" = $6 AND "encryptedPayload" = $4' in sql
        assert "clock_timestamp() < to_timestamp($5::double precision)" in sql
        assert deadline > 0
        if not (
            self.row.id == credential_id
            and self.row.ownerId == owner_id
            and self.row.encryptedPayload == expected
            and self.row.status == status
            and self.reject_cas != "deadline"
        ):
            return 0
        self.row.encryptedPayload = updated
        self.writes += 1
        return 1
