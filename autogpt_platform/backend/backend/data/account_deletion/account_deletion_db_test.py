"""Against the real schema: an account with a marketplace listing, a review of
another account's listing (both ON DELETE RESTRICT), a credit ledger (NO
ACTION), a library entry on its own agent in a folder, a personal org, an
auth identity with sessions and sign-in accounts is deleted completely, the
other account is untouched, a rerun is a no-op, and the orphan-healing job
finds nothing to bring the account back from."""

import uuid
from datetime import datetime, timedelta, timezone

import prisma.enums
import prisma.models
import pytest

from backend.data.account_deletion.database import delete_rows
from backend.data.account_deletion.marketplace import delete_marketplace
from backend.data.account_deletion.preflight import load_subject
from backend.data.user import find_orphaned_auth_identities, get_or_create_user
from backend.util.test import SpinTestServer


async def _account() -> str:
    user_id = str(uuid.uuid4())
    email = f"deletion-{user_id[:8]}@example.com"
    # Provisions the marketplace Profile and the personal org, as sign-up does.
    await get_or_create_user({"sub": user_id, "email": email})
    await prisma.models.AuthUser.prisma().create(
        data={"id": user_id, "name": "Sam", "email": email, "emailVerified": True}
    )
    await prisma.models.AuthSession.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "token": str(uuid.uuid4()),
            "userId": user_id,
            "expiresAt": datetime.now(timezone.utc) + timedelta(days=1),
        }
    )
    await prisma.models.AuthAccount.prisma().create(
        data={
            "id": str(uuid.uuid4()),
            "accountId": user_id,
            "providerId": "credential",
            "userId": user_id,
        }
    )
    return user_id


async def _listing(owner_id: str) -> str:
    """A graph of `owner_id`'s with an approved listing; the version ID."""
    graph = await prisma.models.AgentGraph.prisma().create(
        data={"userId": owner_id, "name": "Deletion test agent"}
    )
    listing = await prisma.models.StoreListing.prisma().create(
        data={
            "slug": f"deletion-{graph.id}",
            "agentGraphId": graph.id,
            "owningUserId": owner_id,
            "hasApprovedVersion": True,
        }
    )
    version = await prisma.models.StoreListingVersion.prisma().create(
        data={
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "name": "Deletion test agent",
            "subHeading": "Test",
            "description": "Test",
            "storeListingId": listing.id,
            "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        }
    )
    return version.id


async def _review(version_id: str, reviewer_id: str) -> None:
    await prisma.models.StoreListingReview.prisma().create(
        data={
            "storeListingVersionId": version_id,
            "reviewByUserId": reviewer_id,
            "score": 5,
        }
    )


async def _holdings(user_id: str) -> None:
    """A credit ledger row and a library entry of its own agent in a folder."""
    await prisma.models.CreditTransaction.prisma().create(
        data={
            "userId": user_id,
            "amount": 100,
            "type": prisma.enums.CreditTransactionType.GRANT,
        }
    )
    graph = await prisma.models.AgentGraph.prisma().create(
        data={"userId": user_id, "name": "Own agent"}
    )
    folder = await prisma.models.LibraryFolder.prisma().create(
        data={"userId": user_id, "name": "Mine"}
    )
    await prisma.models.LibraryAgent.prisma().create(
        data={
            "userId": user_id,
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "folderId": folder.id,
        }
    )


async def _delete(user_id: str) -> None:
    subject = await load_subject(user_id)
    await delete_marketplace(subject, dry_run=False)
    await delete_rows(subject, dry_run=False)


async def _left_of(user_id: str) -> dict[str, int]:
    return {
        "user": await prisma.models.User.prisma().count(where={"id": user_id}),
        "identity": await prisma.models.AuthUser.prisma().count(where={"id": user_id}),
        "sessions": await prisma.models.AuthSession.prisma().count(
            where={"userId": user_id}
        ),
        "sign_in_accounts": await prisma.models.AuthAccount.prisma().count(
            where={"userId": user_id}
        ),
        "credit_transactions": await prisma.models.CreditTransaction.prisma().count(
            where={"userId": user_id}
        ),
        "personal_orgs": await prisma.models.Organization.prisma().count(
            where={"bootstrapUserId": user_id}
        ),
        "graphs": await prisma.models.AgentGraph.prisma().count(
            where={"userId": user_id}
        ),
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_an_account_is_deleted_completely_and_stays_deleted(
    server: SpinTestServer,
):
    user_id = await _account()
    other_id = await _account()
    own_version = await _listing(user_id)
    other_version = await _listing(other_id)
    await _review(other_version, user_id)
    await _review(own_version, other_id)
    await _holdings(user_id)

    dry = await delete_rows(await load_subject(user_id), dry_run=True)
    assert dry.counts["user"] == 1
    assert (await _left_of(user_id))["user"] == 1

    await _delete(user_id)

    assert set((await _left_of(user_id)).values()) == {0}
    # The other account keeps its listing; only the review the deleted
    # account wrote on it is gone.
    assert (await _left_of(other_id))["user"] == 1
    assert await prisma.models.StoreListingVersion.prisma().count(
        where={"id": other_version}
    )
    assert not await prisma.models.StoreListingReview.prisma().count(
        where={"reviewByUserId": user_id}
    )

    # A rerun finds nothing and changes nothing.
    again = await delete_rows(await load_subject(user_id), dry_run=False)
    assert set(again.counts.values()) == {0}

    # The healing job finds no orphaned identity to re-create the account
    # from. (Its query, read-only: healing would provision other tests' rows.)
    orphans = await find_orphaned_auth_identities(
        datetime.now(timezone.utc) + timedelta(minutes=1), limit=10_000
    )
    assert user_id not in {orphan.id for orphan in orphans}

    await _delete(other_id)
