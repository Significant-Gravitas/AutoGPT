"""Real-DB regression tests for #15296: REVOKED is a terminal API key status.

These run against the CI test database through the real data-layer functions,
so the conditional UPDATE is exercised by Postgres rather than by a mock.
"""

from datetime import datetime, timezone
from uuid import uuid4

import pytest
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey as PrismaAPIKey
from prisma.models import User

from backend.data.auth import api_key as api_key_db
from backend.util.test import SpinTestServer


async def _create_user() -> str:
    user_id = f"api-key-revoked-{uuid4()}"
    await User.prisma().create(
        data={"id": user_id, "email": f"{user_id}@example.com", "name": "Key Test"}
    )
    return user_id


async def _delete_user(user_id: str) -> None:
    await PrismaAPIKey.prisma().delete_many(where={"userId": user_id})
    await User.prisma().delete_many(where={"id": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_revoked_key_stays_revoked(server: SpinTestServer):
    user_id = await _create_user()
    try:
        key, _ = await api_key_db.create_api_key(
            "k", user_id, [APIKeyPermission.EXECUTE_GRAPH]
        )
        revoked = await api_key_db.revoke_api_key(key.id, user_id)
        assert revoked.revoked_at is not None

        with pytest.raises(ValueError, match="revoked"):
            await api_key_db.suspend_api_key(key.id, user_id)
        with pytest.raises(ValueError, match="revoked"):
            await api_key_db.update_api_key_permissions(
                key.id, user_id, [APIKeyPermission.READ_GRAPH]
            )
        again = await api_key_db.revoke_api_key(key.id, user_id)

        row = await PrismaAPIKey.prisma().find_unique(where={"id": key.id})
        assert row is not None
        assert row.status == APIKeyStatus.REVOKED
        assert row.revokedAt == revoked.revoked_at
        assert again.revoked_at == revoked.revoked_at
        assert row.permissions == [APIKeyPermission.EXECUTE_GRAPH]
    finally:
        await _delete_user(user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_revoke_between_read_and_suspend_is_not_overwritten(
    server: SpinTestServer, mocker
):
    """Suspend reads the key while it is ACTIVE, a revoke commits, then the
    suspend writes. The write must not turn REVOKED back into SUSPENDED."""
    user_id = await _create_user()
    try:
        key, _ = await api_key_db.create_api_key(
            "k", user_id, [APIKeyPermission.EXECUTE_GRAPH]
        )
        real_read = api_key_db._get_owned_api_key

        async def read_then_concurrent_revoke(*args, **kwargs):
            row = await real_read(*args, **kwargs)
            assert row.status == APIKeyStatus.ACTIVE
            # Another request revokes the key before this one writes.
            await PrismaAPIKey.prisma().update(
                where={"id": key.id},
                data={
                    "status": APIKeyStatus.REVOKED,
                    "revokedAt": datetime.now(timezone.utc),
                },
            )
            return row

        mocker.patch.object(
            api_key_db, "_get_owned_api_key", side_effect=read_then_concurrent_revoke
        )

        with pytest.raises(ValueError, match="revoked"):
            await api_key_db.suspend_api_key(key.id, user_id)

        row = await PrismaAPIKey.prisma().find_unique(where={"id": key.id})
        assert row is not None
        assert row.status == APIKeyStatus.REVOKED
    finally:
        await _delete_user(user_id)
