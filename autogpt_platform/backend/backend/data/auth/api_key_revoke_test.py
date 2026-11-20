"""Revoking every API key a user holds at once, against the real database."""

from typing import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import APIKeyPermission, APIKeyStatus
from prisma.models import APIKey, User

from backend.data.auth.api_key import (
    create_api_key,
    revoke_api_key,
    revoke_user_api_keys,
    suspend_api_key,
    validate_api_key,
)
from backend.util.test import SpinTestServer


async def _settle_the_pool() -> None:
    """An earlier test can leave the Prisma pool bound to a closed event loop;
    only the first query on the session loop after that fails."""
    try:
        await User.prisma().count(where={"id": "pool-check"})
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise


@pytest_asyncio.fixture(loop_scope="session")
async def users(server: SpinTestServer) -> AsyncIterator[tuple[str, str]]:
    """The user whose keys are revoked, and a bystander who keeps theirs."""
    user_id, bystander_id = str(uuid4()), str(uuid4())
    await _settle_the_pool()
    try:
        for uid in (user_id, bystander_id):
            await User.prisma().create(
                data={"id": uid, "email": f"key-revoke-{uid}@example.com"}
            )
        yield user_id, bystander_id
    finally:
        await User.prisma().delete_many(where={"id": {"in": [user_id, bystander_id]}})


async def _new_key(user_id: str, **tenancy: str) -> tuple[str, str]:
    info, plaintext = await create_api_key(
        name="test key",
        user_id=user_id,
        permissions=[APIKeyPermission.IDENTITY],
        **tenancy,
    )
    return info.id, plaintext


@pytest.mark.asyncio(loop_scope="session")
async def test_every_unrevoked_key_of_the_user_is_revoked(users: tuple[str, str]):
    user_id, bystander_id = users
    active_id, active_plaintext = await _new_key(user_id)
    suspended_id, _ = await _new_key(user_id)
    await suspend_api_key(suspended_id, user_id)
    # An org-owned key still acts as the member who created it.
    org_key_id, org_plaintext = await _new_key(
        user_id, organization_id=str(uuid4()), owner_type="ORG"
    )
    bystander_key_id, bystander_plaintext = await _new_key(bystander_id)

    assert await revoke_user_api_keys(user_id) == 3

    for key_id in (active_id, suspended_id, org_key_id):
        key = await APIKey.prisma().find_unique_or_raise(where={"id": key_id})
        assert key.status == APIKeyStatus.REVOKED
        assert key.revokedAt is not None
    assert await validate_api_key(active_plaintext) is None
    assert await validate_api_key(org_plaintext) is None

    bystander_key = await APIKey.prisma().find_unique_or_raise(
        where={"id": bystander_key_id}
    )
    assert bystander_key.status == APIKeyStatus.ACTIVE
    assert bystander_key.revokedAt is None
    validated = await validate_api_key(bystander_plaintext)
    assert validated is not None and validated.user_id == bystander_id


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_run_revokes_nothing_and_keeps_the_revocation_time(
    users: tuple[str, str],
):
    user_id, _ = users
    revoked_earlier_id, _ = await _new_key(user_id)
    revoked_earlier = await revoke_api_key(revoked_earlier_id, user_id)
    active_id, _ = await _new_key(user_id)

    assert await revoke_user_api_keys(user_id) == 1
    first = await APIKey.prisma().find_unique_or_raise(where={"id": active_id})

    assert await revoke_user_api_keys(user_id) == 0

    second = await APIKey.prisma().find_unique_or_raise(where={"id": active_id})
    assert second.revokedAt == first.revokedAt
    untouched = await APIKey.prisma().find_unique_or_raise(
        where={"id": revoked_earlier_id}
    )
    assert untouched.revokedAt == revoked_earlier.revoked_at


@pytest.mark.asyncio(loop_scope="session")
async def test_a_user_without_keys_revokes_nothing(users: tuple[str, str]):
    user_id, _ = users

    assert await revoke_user_api_keys(user_id) == 0
