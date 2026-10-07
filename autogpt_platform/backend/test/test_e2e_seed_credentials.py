from test import e2e_test_data
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import pytest
from cryptography.fernet import Fernet

from backend.util.encryption import JSONCryptor


@pytest.fixture
def seeded_credentials(monkeypatch):
    old_cryptor = JSONCryptor(Fernet.generate_key().decode())
    current_cryptor = JSONCryptor(Fernet.generate_key().decode())
    users = [SimpleNamespace(id=str(uuid4())) for _ in e2e_test_data.SEEDED_TEST_EMAILS]
    rows = []
    for user in users:
        credential_id = str(uuid4())
        rows.append(
            SimpleNamespace(
                id=credential_id,
                ownerId=user.id,
                encryptedPayload=old_cryptor.encrypt(
                    e2e_test_data.make_seed_credential(credential_id).model_dump()
                ),
            )
        )

    async def update(*, where, data):
        row = next(row for row in rows if row.id == where["id"])
        row.encryptedPayload = data["encryptedPayload"]
        return row

    db = SimpleNamespace(
        user=SimpleNamespace(find_many=AsyncMock(return_value=users)),
        integrationcredential=SimpleNamespace(
            find_many=AsyncMock(return_value=rows), update=AsyncMock(side_effect=update)
        ),
    )
    monkeypatch.setattr(e2e_test_data, "prisma", db)
    monkeypatch.setattr(e2e_test_data, "JSONCryptor", lambda: current_cryptor)
    return db, rows, current_cryptor


async def test_cached_credentials_remain_readable_with_a_fresh_key(seeded_credentials):
    db, rows, cryptor = seeded_credentials
    assert all(cryptor.decrypt(row.encryptedPayload) == {} for row in rows)
    await e2e_test_data.refresh_seeded_credentials()
    assert db.integrationcredential.update.await_count == len(rows)
    for row in rows:
        assert (
            cryptor.decrypt(row.encryptedPayload)
            == e2e_test_data.make_seed_credential(row.id).model_dump()
        )
    db.user.find_many.assert_awaited_once_with(
        where={"email": {"in": e2e_test_data.SEEDED_TEST_EMAILS}}
    )
    db.integrationcredential.find_many.assert_awaited_once_with(
        where={
            "ownerType": e2e_test_data.prisma_enums.CredentialOwnerType.USER,
            "ownerId": {"in": sorted(row.ownerId for row in rows)},
            "provider": "github",
            "displayName": "Kitchen-sink GitHub",
        }
    )


async def test_missing_seeded_user_fails_before_updates(seeded_credentials):
    db, _, _ = seeded_credentials
    db.user.find_many.return_value = db.user.find_many.return_value[:-1]
    with pytest.raises(ValueError, match="owners are missing"):
        await e2e_test_data.refresh_seeded_credentials()
    db.integrationcredential.update.assert_not_awaited()


@pytest.mark.parametrize(
    "invalid_rows", ["missing", "duplicate_owner", "foreign_owner"]
)
async def test_incomplete_or_wrong_fixture_set_fails_closed(
    seeded_credentials, invalid_rows
):
    db, rows, _ = seeded_credentials
    if invalid_rows == "missing":
        db.integrationcredential.find_many.return_value = rows[:-1]
    elif invalid_rows == "duplicate_owner":
        rows[-1].ownerId = rows[0].ownerId
    else:
        rows[-1].ownerId = str(uuid4())
    with pytest.raises(ValueError, match="one marked credential per seeded user"):
        await e2e_test_data.refresh_seeded_credentials()
    db.integrationcredential.update.assert_not_awaited()


async def test_database_update_failure_is_not_swallowed(seeded_credentials):
    db, _, _ = seeded_credentials
    db.integrationcredential.update.side_effect = RuntimeError("update failed")
    with pytest.raises(RuntimeError, match="update failed"):
        await e2e_test_data.refresh_seeded_credentials()


@pytest.mark.parametrize("returned_row", ["missing", "stale"])
async def test_failed_encryption_verification_is_not_swallowed(
    seeded_credentials, returned_row
):
    db, rows, _ = seeded_credentials
    db.integrationcredential.update.side_effect = None
    db.integrationcredential.update.return_value = (
        None if returned_row == "missing" else rows[0]
    )
    with pytest.raises(ValueError, match="encryption verification"):
        await e2e_test_data.refresh_seeded_credentials()


@pytest.mark.parametrize("refresh_only", [False, True])
async def test_command_routes_without_reseeding_cached_data(monkeypatch, refresh_only):
    db = SimpleNamespace(connect=AsyncMock(), disconnect=AsyncMock())
    creator = SimpleNamespace(create_all_test_data=AsyncMock())
    factory = Mock(return_value=creator)
    refresh = AsyncMock()
    monkeypatch.setattr(e2e_test_data, "prisma", db)
    monkeypatch.setattr(e2e_test_data, "TestDataCreator", factory)
    monkeypatch.setattr(e2e_test_data, "refresh_seeded_credentials", refresh)
    args = ["e2e_test_data.py"]
    if refresh_only:
        args.append("--refresh-credentials-only")
    monkeypatch.setattr("sys.argv", args)
    await e2e_test_data.main()
    db.connect.assert_awaited_once()
    db.disconnect.assert_awaited_once()
    if refresh_only:
        refresh.assert_awaited_once()
        factory.assert_not_called()
    else:
        creator.create_all_test_data.assert_awaited_once()
        refresh.assert_not_awaited()


async def test_refresh_command_disconnects_and_propagates_failure(monkeypatch):
    db = SimpleNamespace(connect=AsyncMock(), disconnect=AsyncMock())
    monkeypatch.setattr(e2e_test_data, "prisma", db)
    monkeypatch.setattr(
        e2e_test_data,
        "refresh_seeded_credentials",
        AsyncMock(side_effect=ValueError("invalid seeded credential")),
    )
    monkeypatch.setattr("sys.argv", ["e2e_test_data.py", "--refresh-credentials-only"])
    with pytest.raises(ValueError, match="invalid seeded credential"):
        await e2e_test_data.main()
    db.disconnect.assert_awaited_once()
