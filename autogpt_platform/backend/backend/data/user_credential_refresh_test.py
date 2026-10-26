"""Per-owner, deadline-fenced credential persistence with a mocked database."""

from __future__ import annotations

import asyncio
import copy
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel, Field, SecretStr

from backend.data import user as user_module
from backend.integrations.credential_refresh import REFRESH_PENDING, REFRESH_REVISION


class Credential(BaseModel):
    id: str = "credential-a"
    provider: str = "codex"
    type: str = "oauth2"
    access_token: SecretStr = SecretStr("synthetic-access")
    refresh_token: SecretStr | None = SecretStr("synthetic-refresh")
    provider_state: SecretStr | None = SecretStr("synthetic-state")
    access_token_expires_at: int = 1
    metadata: dict = Field(default_factory=dict)
    title: str | None = None


@pytest.fixture
def seam(monkeypatch):
    payloads = {}

    class Cryptor:
        def encrypt(self, payload):
            ciphertext = "ciphertext-" + str(len(payloads))
            payloads[ciphertext] = copy.deepcopy(payload)
            return ciphertext

        def decrypt(self, ciphertext):
            return copy.deepcopy(payloads[ciphertext])

    credential = Credential()
    ciphertext = Cryptor().encrypt(credential.model_dump())
    row = SimpleNamespace(
        id=credential.id, status="active", encryptedPayload=ciphertext
    )
    prisma = SimpleNamespace(
        integrationcredential=SimpleNamespace(
            find_first=AsyncMock(return_value=row),
            find_many=AsyncMock(return_value=[row]),
        ),
        execute_raw=AsyncMock(return_value=1),
        query_raw=AsyncMock(return_value=[{"id": credential.id}]),
    )
    transaction = MagicMock()
    transaction.__aenter__ = AsyncMock(return_value=prisma)
    transaction.__aexit__ = AsyncMock(return_value=False)
    prisma.tx = MagicMock(return_value=transaction)

    async def query_with_schema(sql, *args, client):
        assert client is prisma
        assert '{schema_prefix}"IntegrationCredential"' in sql
        return await client.query_raw(sql, *args)

    async def execute_with_schema(sql, *args, client):
        assert client is prisma
        assert '{schema_prefix}"IntegrationCredential"' in sql
        return await client.execute_raw(sql, *args)

    monkeypatch.setattr(user_module, "prisma", prisma)
    monkeypatch.setattr(user_module, "transaction", prisma.tx)
    monkeypatch.setattr(user_module, "JSONCryptor", Cryptor)
    monkeypatch.setattr(
        user_module,
        "CREDENTIALS_ADAPTER",
        SimpleNamespace(validate_python=Credential.model_validate),
    )
    monkeypatch.setattr(user_module, "query_raw_with_schema", query_with_schema)
    monkeypatch.setattr(user_module, "execute_raw_with_schema", execute_with_schema)
    return SimpleNamespace(
        begin=user_module.begin_user_credential_refresh,
        complete=user_module.complete_user_credential_refresh,
        save_list=user_module.set_user_credentials,
        prisma=prisma,
        row=row,
        credential=credential,
        payloads=payloads,
    )


@pytest.mark.asyncio
async def test_prepare_uses_owner_and_payload_cas_and_strips_every_token(seam):
    deadline = time.time() + 20
    checkpoint = await seam.begin("user-a", seam.credential, "test-nonce", deadline)
    seam.prisma.integrationcredential.find_first.assert_awaited_once_with(
        where={
            "id": "credential-a",
            "ownerType": "USER",
            "ownerId": "user-a",
            "status": "active",
        }
    )
    pending = seam.payloads[checkpoint]
    assert pending["id"] == seam.credential.id and pending["provider"] == "codex"
    assert pending["access_token"].get_secret_value() == ""
    assert pending["refresh_token"] is None and pending["provider_state"] is None
    assert pending["metadata"][REFRESH_PENDING] == "test-nonce"
    assert pending["metadata"][REFRESH_REVISION] == "test-nonce:pending"
    assert pending["access_token_expires_at"] == 1
    sql, *args = seam.prisma.execute_raw.await_args.args
    assert args == [
        checkpoint,
        "credential-a",
        "user-a",
        seam.row.encryptedPayload,
        deadline,
        "active",
        None,
    ]
    for predicate in (
        "\"ownerType\" = 'USER'",
        '"ownerId" = $3',
        '"status" = $6',
        '"encryptedPayload" = $4',
        "clock_timestamp() < to_timestamp($5::double precision)",
    ):
        assert predicate in sql


@pytest.mark.asyncio
@pytest.mark.parametrize("change", ["missing", "replacement", "pending"])
async def test_prepare_refuses_missing_changed_or_already_pending_rows(seam, change):
    if change == "missing":
        seam.prisma.integrationcredential.find_first.return_value = None
    elif change == "replacement":
        seam.payloads[seam.row.encryptedPayload]["access_token"] = SecretStr(
            "different"
        )
    else:
        seam.payloads[seam.row.encryptedPayload]["metadata"] = {
            REFRESH_PENDING: "other-nonce"
        }
    assert (
        await seam.begin("user-a", seam.credential, "nonce", time.time() + 20) is None
    )
    seam.prisma.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_prepare_reports_failed_cas_without_starting_a_refresh(seam):
    seam.prisma.execute_raw.return_value = 0
    assert (
        await seam.begin("user-a", seam.credential, "nonce", time.time() + 20) is None
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [0, 1])
async def test_checkpoint_is_single_row_cas_with_database_deadline(seam, count):
    seam.prisma.execute_raw.return_value = count
    deadline = time.time() + 20
    assert await seam.complete(
        "user-a", seam.credential, "pending-ciphertext", deadline
    ) is bool(count)
    sql, *args = seam.prisma.execute_raw.await_args.args
    assert args[1:] == [
        "credential-a",
        "user-a",
        "pending-ciphertext",
        deadline,
        "active",
        None,
    ]
    assert "\"ownerType\" = 'USER'" in sql and '"ownerId" = $3' in sql
    assert '"encryptedPayload" = $4' in sql
    assert "clock_timestamp() < to_timestamp($5::double precision)" in sql
    assert seam.payloads[args[0]] == seam.credential.model_dump()


@pytest.mark.asyncio
async def test_checkpoint_cannot_publish_pending_credentials(seam):
    pending = seam.credential.model_copy(
        update={"metadata": {REFRESH_PENDING: "nonce"}}
    )
    with pytest.raises(ValueError, match="pending"):
        await seam.complete("user-a", pending, "checkpoint", time.time() + 20)
    seam.prisma.execute_raw.assert_not_awaited()


@pytest.mark.asyncio
async def test_stale_full_list_write_cannot_restore_pre_rotation_tokens(seam):
    seam.payloads[seam.row.encryptedPayload]["metadata"] = {
        REFRESH_REVISION: "new:complete"
    }
    with pytest.raises(ValueError, match="changed"):
        await seam.save_list("user-a", [seam.credential])
    assert not any(
        "UPDATE" in call.args[0] for call in seam.prisma.execute_raw.await_args_list
    )


@pytest.mark.asyncio
async def test_full_list_write_losing_cas_does_not_overwrite_a_refresh(seam):
    seam.prisma.query_raw.return_value = []
    with pytest.raises(ValueError, match="changed"):
        await seam.save_list("user-a", [seam.credential])
    assert not any(
        "UPDATE" in call.args[0] for call in seam.prisma.execute_raw.await_args_list
    )


@pytest.mark.asyncio
async def test_explicit_reconnect_can_replace_pending_revision(seam):
    seam.payloads[seam.row.encryptedPayload]["metadata"] = {
        REFRESH_REVISION: "old:pending",
        REFRESH_PENDING: "old",
    }
    replacement = seam.credential.model_copy(
        update={"metadata": {REFRESH_REVISION: "new"}}
    )
    await seam.save_list(
        "user-a", [replacement], reconnect_revisions={replacement.id: "old:pending"}
    )
    assert (
        seam.payloads[seam.prisma.execute_raw.await_args.args[1]]
        == replacement.model_dump()
    )


@pytest.mark.asyncio
async def test_delayed_reconnect_cannot_replace_a_newer_reconnect(seam):
    seam.payloads[seam.row.encryptedPayload]["metadata"] = {REFRESH_REVISION: "winner"}
    replacement = seam.credential.model_copy(
        update={"metadata": {REFRESH_REVISION: "late"}}
    )
    with pytest.raises(ValueError, match="changed"):
        await seam.save_list(
            "user-a", [replacement], reconnect_revisions={replacement.id: "old:pending"}
        )
    assert not any(
        "UPDATE" in call.args[0] for call in seam.prisma.execute_raw.await_args_list
    )


@pytest.mark.asyncio
async def test_ordinary_oauth_save_preserves_rename_and_reactivation(seam):
    seam.row.status = "revoked"
    renamed = seam.credential.model_copy(update={"title": "Renamed connection"})
    await seam.save_list("user-a", [renamed])
    sql, *args = seam.prisma.execute_raw.await_args.args
    assert args[-2:] == ["revoked", "Renamed connection"]
    assert (
        "\"status\" = 'active'" in sql
        and '"displayName" = COALESCE($7::text, "displayName")' in sql
    )


@pytest.mark.asyncio
async def test_deadline_update_is_issued_only_after_row_lock_returns(seam):
    locked = asyncio.Event()
    release = asyncio.Event()

    async def lock_row(*_args):
        locked.set()
        await release.wait()
        return [{"id": seam.credential.id}]

    seam.prisma.query_raw.side_effect = lock_row
    task = asyncio.create_task(
        seam.complete("user-a", seam.credential, "checkpoint", time.time() + 20)
    )
    await locked.wait()
    assert not any(
        "UPDATE" in call.args[0] for call in seam.prisma.execute_raw.await_args_list
    )
    release.set()
    assert await task


@pytest.mark.asyncio
async def test_oauth_list_replace_uses_one_transaction_and_owner_lock(seam):
    await seam.save_list("user-a", [seam.credential])
    seam.prisma.tx.assert_called_once_with()
    first = seam.prisma.execute_raw.await_args_list[0]
    assert "pg_advisory_xact_lock" in first.args[0]
    assert first.args[1] == "integration-credentials:user-a"
    seam.prisma.integrationcredential.find_many.assert_awaited_once_with(
        where={"ownerType": "USER", "ownerId": "user-a"}
    )
