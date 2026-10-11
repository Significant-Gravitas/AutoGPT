"""Closing an account against the real database.

The request row, the Better Auth ban and the session cleanup must land
together or not at all, and a second close must change nothing.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import AccountDeletionSource
from prisma.models import AccountDeletionRequest, AuthSession, AuthUser, User

from backend.data.account_deletion import (
    ACCOUNT_DELETED_BAN_REASON,
    close_account,
    get_deletion_request,
    is_account_closed,
)
from backend.util.test import SpinTestServer


async def _add_session(user_id: str) -> None:
    await AuthSession.prisma().create(
        data={
            "id": str(uuid4()),
            "token": str(uuid4()),
            "userId": user_id,
            "expiresAt": datetime.now(timezone.utc) + timedelta(days=1),
        }
    )


async def _create_account(user_id: str) -> None:
    email = f"deletion-{user_id}@example.com"
    await User.prisma().create(data={"id": user_id, "email": email})
    await AuthUser.prisma().create(
        data={"id": user_id, "name": "Deletion test", "email": email}
    )
    await _add_session(user_id)
    await _add_session(user_id)


async def _remove_account(user_id: str) -> None:
    await AccountDeletionRequest.prisma().delete_many(where={"userId": user_id})
    await AuthUser.prisma().delete_many(where={"id": user_id})
    await User.prisma().delete_many(where={"id": user_id})


async def _settle_the_pool() -> None:
    """The transient conftest's ``_create_user_with_loop_retry`` absorbs: an
    earlier test can leave the Prisma pool bound to a closed event loop, and
    only the first session-loop query after that fails."""
    try:
        await User.prisma().count(where={"id": "pool-check"})
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise


@pytest_asyncio.fixture(loop_scope="session")
async def accounts(server: SpinTestServer) -> AsyncIterator[tuple[str, str]]:
    """The account under test, and a bystander that must stay untouched."""
    user_id, bystander_id = str(uuid4()), str(uuid4())
    await _settle_the_pool()
    try:
        await _create_account(user_id)
        await _create_account(bystander_id)
        yield user_id, bystander_id
    finally:
        await _remove_account(user_id)
        await _remove_account(bystander_id)


async def _session_count(user_id: str) -> int:
    return await AuthSession.prisma().count(where={"userId": user_id})


@pytest.mark.asyncio(loop_scope="session")
async def test_close_account_records_the_request_bans_and_signs_out(
    accounts: tuple[str, str],
):
    user_id, bystander_id = accounts
    assert await get_deletion_request(user_id) is None
    assert await is_account_closed(user_id) is False

    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is True

    request = await get_deletion_request(user_id)
    assert request is not None
    assert request.userId == user_id
    assert request.source == AccountDeletionSource.SETTINGS
    assert request.completedAt is None
    assert await is_account_closed(user_id) is True

    auth_user = await AuthUser.prisma().find_unique_or_raise(where={"id": user_id})
    assert auth_user.banned is True
    assert auth_user.banReason == ACCOUNT_DELETED_BAN_REASON == "account_deleted"
    assert auth_user.banExpires is None
    assert await _session_count(user_id) == 0

    bystander = await AuthUser.prisma().find_unique_or_raise(where={"id": bystander_id})
    assert not bystander.banned
    assert await _session_count(bystander_id) == 2
    assert await is_account_closed(bystander_id) is False


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_close_returns_false_and_changes_nothing(
    accounts: tuple[str, str],
):
    user_id, _ = accounts
    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is True
    first = await get_deletion_request(user_id)
    # Undo the ban and sign in again, so a second close that still ran its
    # other statements would show.
    await AuthUser.prisma().update(
        where={"id": user_id}, data={"banned": False, "banReason": None}
    )
    await _add_session(user_id)

    assert await close_account(user_id, AccountDeletionSource.PAYWALL) is False

    assert await get_deletion_request(user_id) == first
    auth_user = await AuthUser.prisma().find_unique_or_raise(where={"id": user_id})
    assert auth_user.banned is False
    assert auth_user.banReason is None
    assert await _session_count(user_id) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_concurrent_closes_create_exactly_one_request(
    accounts: tuple[str, str],
):
    user_id, _ = accounts

    results = await asyncio.gather(
        close_account(user_id, AccountDeletionSource.SETTINGS),
        close_account(user_id, AccountDeletionSource.PAYWALL),
    )

    assert sorted(results) == [False, True]
    assert await AccountDeletionRequest.prisma().count(where={"userId": user_id}) == 1
