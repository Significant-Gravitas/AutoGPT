"""Closing and reopening an account against the real database.

The request row, the session cleanup, the Marketplace listings taken down, the
share links turned off and the chat platforms unlinked must land together or
not at all, and a second close must change nothing. Closing bans nothing, and
leaves a ban for another reason as it was. The request row has no foreign key
to User, so it outlives the erasure that removes the User. Reopening deletes
the row only while the grace period lasts.
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import AsyncIterator
from uuid import uuid4

import pytest
import pytest_asyncio
from prisma.enums import AccountDeletionSource, PlatformType
from prisma.models import (
    AccountDeletionRequest,
    AuthSession,
    AuthUser,
    ChatSession,
    PlatformLink,
    PlatformUserLink,
    Profile,
    SkillListing,
    StoreListing,
    User,
)

import backend.data.account_deletion as account_deletion
from backend.data.account_deletion import (
    GRACE_PERIOD,
    REOPENED_DAILY_EMAIL_LIMIT,
    ReopenOutcome,
    close_account,
    get_deletion_request,
    is_account_closed,
    mark_confirmation_sent,
    reopen_account,
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


def _tomorrow() -> datetime:
    return datetime.now(timezone.utc) + timedelta(days=1)


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
    # Before the User: StoreListing's owner foreign key restricts its delete.
    await StoreListing.prisma().delete_many(where={"owningUserId": user_id})
    await SkillListing.prisma().delete_many(where={"owningUserId": user_id})
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
async def test_close_account_records_the_request_and_signs_out_without_a_ban(
    accounts: tuple[str, str],
):
    """Signing in again is how the owner keeps the account, so it is not
    banned; backend auth rejects it while the request exists."""
    user_id, bystander_id = accounts
    assert await get_deletion_request(user_id) is None
    assert await is_account_closed(user_id) is False

    created = await close_account(user_id, AccountDeletionSource.SETTINGS)

    request = await get_deletion_request(user_id)
    assert request is not None and request == created
    assert request.userId == user_id
    assert request.source == AccountDeletionSource.SETTINGS
    assert request.completedAt is None
    assert request.eraseAfter == request.requestedAt + GRACE_PERIOD
    assert GRACE_PERIOD == timedelta(days=14)
    assert await is_account_closed(user_id) is True

    auth_user = await AuthUser.prisma().find_unique_or_raise(where={"id": user_id})
    assert not auth_user.banned
    assert auth_user.banReason is None
    assert await _session_count(user_id) == 0

    bystander = await AuthUser.prisma().find_unique_or_raise(where={"id": bystander_id})
    assert not bystander.banned
    assert await _session_count(bystander_id) == 2
    assert await is_account_closed(bystander_id) is False


@pytest.mark.asyncio(loop_scope="session")
async def test_a_ban_for_another_reason_is_kept_as_it_was(
    accounts: tuple[str, str],
):
    """A banned person may delete their account, and the ban stays, so they
    can never sign in to reopen it."""
    user_id, _ = accounts
    expires = _tomorrow().replace(microsecond=0)
    await AuthUser.prisma().update(
        where={"id": user_id},
        data={"banned": True, "banReason": "spam", "banExpires": expires},
    )

    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is not None

    auth_user = await AuthUser.prisma().find_unique_or_raise(where={"id": user_id})
    assert auth_user.banned is True
    assert auth_user.banReason == "spam"
    assert auth_user.banExpires == expires


@pytest.mark.asyncio(loop_scope="session")
async def test_a_second_close_returns_none_and_changes_nothing(
    accounts: tuple[str, str],
):
    user_id, _ = accounts
    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is not None
    first = await get_deletion_request(user_id)
    # Sign in again, so a second close that still ran its other statements
    # would show.
    await _add_session(user_id)

    assert await close_account(user_id, AccountDeletionSource.PAYWALL) is None

    assert await get_deletion_request(user_id) == first
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

    assert sorted(result is not None for result in results) == [False, True]
    assert await AccountDeletionRequest.prisma().count(where={"userId": user_id}) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_a_failure_after_the_request_insert_changes_nothing(
    accounts: tuple[str, str], monkeypatch: pytest.MonkeyPatch
):
    """A half-committed close would leave the request row, so every retry
    returns early while the person can still sign in."""
    user_id, _ = accounts
    real_transaction = account_deletion.transaction

    @asynccontextmanager
    async def failing_transaction():
        async with real_transaction() as tx:
            monkeypatch.setattr(tx, "authsession", _FailingSessionCleanup())
            yield tx

    monkeypatch.setattr(account_deletion, "transaction", failing_transaction)

    with pytest.raises(RuntimeError, match="session cleanup failed"):
        await close_account(user_id, AccountDeletionSource.SETTINGS)

    assert await get_deletion_request(user_id) is None
    assert await is_account_closed(user_id) is False
    auth_user = await AuthUser.prisma().find_unique_or_raise(where={"id": user_id})
    assert not auth_user.banned
    assert auth_user.banReason is None
    assert await _session_count(user_id) == 2


@pytest.mark.asyncio(loop_scope="session")
async def test_the_sent_confirmation_is_recorded_on_the_request_only(
    accounts: tuple[str, str],
):
    """A null confirmationSentAt is what the erasure routine resends from."""
    user_id, bystander_id = accounts
    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is not None
    assert await close_account(bystander_id, AccountDeletionSource.PAYWALL) is not None
    request = await get_deletion_request(user_id)
    assert request is not None and request.confirmationSentAt is None

    before = datetime.now(timezone.utc)
    await mark_confirmation_sent(user_id)

    request = await get_deletion_request(user_id)
    assert request is not None
    assert request.confirmationSentAt is not None
    assert request.confirmationSentAt >= before - timedelta(seconds=5)
    bystander = await get_deletion_request(bystander_id)
    assert bystander is not None and bystander.confirmationSentAt is None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_request_outlives_the_erased_user(accounts: tuple[str, str]):
    """The row is the hand-off to the erasure, and what keeps rejecting the
    account's tokens: deleting the User must not take it along."""
    user_id, _ = accounts
    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is not None

    await AuthUser.prisma().delete(where={"id": user_id})
    await User.prisma().delete(where={"id": user_id})

    request = await get_deletion_request(user_id)
    assert request is not None
    assert request.completedAt is None
    assert await is_account_closed(user_id) is True


@pytest.mark.asyncio(loop_scope="session")
async def test_the_close_takes_every_marketplace_listing_down(
    accounts: tuple[str, str],
):
    """Approved ones leave the Marketplace now; pending ones can no longer be
    approved. Another creator's listing stays up."""
    user_id, bystander_id = accounts
    approved = await _list_agent(user_id, approved=True)
    pending = await _list_agent(user_id, approved=False)
    skill = await SkillListing.prisma().create(
        data={
            "slug": f"skill-{uuid4()}",
            "owningUserId": user_id,
            "hasApprovedVersion": True,
        }
    )
    bystanders = await _list_agent(bystander_id, approved=True)

    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is not None

    for listing_id in (approved, pending):
        listing = await StoreListing.prisma().find_unique_or_raise(
            where={"id": listing_id}
        )
        assert listing.isDeleted is True
    skill = await SkillListing.prisma().find_unique_or_raise(where={"id": skill.id})
    assert skill.isDeleted is True
    untouched = await StoreListing.prisma().find_unique_or_raise(
        where={"id": bystanders}
    )
    assert untouched.isDeleted is False


@pytest.mark.asyncio(loop_scope="session")
async def test_the_close_turns_off_share_links_and_unlinks_chat_platforms(
    accounts: tuple[str, str],
):
    """A shared chat stops being served (its token is gone), and the bot stops
    answering in the linked servers and DMs. The bystander's stay."""
    user_id, bystander_id = accounts
    shared = await _shared_chat(user_id)
    bystanders_chat = await _shared_chat(bystander_id)
    for owner in (user_id, bystander_id):
        await _link_chat_platforms(owner)

    assert await close_account(user_id, AccountDeletionSource.SETTINGS) is not None

    chat = await ChatSession.prisma().find_unique_or_raise(where={"id": shared})
    assert (chat.isShared, chat.shareToken, chat.sharedAt) == (False, None, None)
    untouched = await ChatSession.prisma().find_unique_or_raise(
        where={"id": bystanders_chat}
    )
    assert untouched.isShared is True and untouched.shareToken is not None
    for model in (PlatformLink, PlatformUserLink):
        assert await model.prisma().count(where={"userId": user_id}) == 0
        assert await model.prisma().count(where={"userId": bystander_id}) == 1


@pytest.mark.asyncio(loop_scope="session")
async def test_reopening_within_the_grace_period_cancels_the_request(
    accounts: tuple[str, str],
):
    user_id, bystander_id = accounts
    await close_account(user_id, AccountDeletionSource.SETTINGS)
    await close_account(bystander_id, AccountDeletionSource.SETTINGS)
    await User.prisma().update(where={"id": user_id}, data={"maxEmailsPerDay": 0})

    assert await reopen_account(user_id) is ReopenOutcome.REOPENED

    assert await is_account_closed(user_id) is False
    user = await User.prisma().find_unique_or_raise(where={"id": user_id})
    assert user.maxEmailsPerDay == REOPENED_DAILY_EMAIL_LIMIT
    assert await is_account_closed(bystander_id) is True
    assert await reopen_account(user_id) is ReopenOutcome.NOT_CLOSED
    # A new request starts a new grace period.
    assert await close_account(user_id, AccountDeletionSource.PAYWALL) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_reopening_after_the_grace_period_changes_nothing(
    accounts: tuple[str, str],
):
    """By then the erasure may be running: the request must stay."""
    user_id, _ = accounts
    request = await close_account(user_id, AccountDeletionSource.SETTINGS)
    assert request is not None

    outcome = await reopen_account(user_id, now=request.eraseAfter)

    assert outcome is ReopenOutcome.TOO_LATE
    assert await get_deletion_request(user_id) == request


@pytest.mark.asyncio(loop_scope="session")
async def test_an_erased_account_cannot_be_reopened(accounts: tuple[str, str]):
    user_id, _ = accounts
    await close_account(user_id, AccountDeletionSource.SETTINGS)
    await AccountDeletionRequest.prisma().update(
        where={"userId": user_id},
        data={"completedAt": datetime.now(timezone.utc)},
    )

    assert await reopen_account(user_id) is ReopenOutcome.TOO_LATE
    assert await is_account_closed(user_id) is True


async def _shared_chat(user_id: str) -> str:
    chat = await ChatSession.prisma().create(
        data={
            "userId": user_id,
            "isShared": True,
            "shareToken": str(uuid4()),
            "sharedAt": datetime.now(timezone.utc),
        }
    )
    return chat.id


async def _link_chat_platforms(user_id: str) -> None:
    await PlatformLink.prisma().create(
        data={
            "userId": user_id,
            "platform": PlatformType.DISCORD,
            "platformServerId": str(uuid4()),
            "ownerPlatformUserId": str(uuid4()),
        }
    )
    await PlatformUserLink.prisma().create(
        data={
            "userId": user_id,
            "platform": PlatformType.TELEGRAM,
            "platformUserId": str(uuid4()),
        }
    )


class _FailingSessionCleanup:
    """Stands in for the transaction's AuthSession actions: by the session
    cleanup, the request insert has run."""

    async def delete_many(self, **_: object) -> int:
        raise RuntimeError("session cleanup failed")


async def _list_agent(user_id: str, *, approved: bool) -> str:
    """A Marketplace agent listing of *user_id*'s; its creator profile first."""
    await Profile.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {
                "userId": user_id,
                "name": "Deletion test",
                "username": f"deletion-{user_id}",
                "description": "",
                "links": [],
            },
            "update": {},
        },
    )
    listing = await StoreListing.prisma().create(
        data={
            "slug": f"agent-{uuid4()}",
            "agentGraphId": str(uuid4()),
            "owningUserId": user_id,
            "hasApprovedVersion": approved,
        }
    )
    return listing.id
