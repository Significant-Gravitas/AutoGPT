"""Against the real schema.

An account with a published agent (and the sub-agent it runs) that another
account has in its library and has run, a private agent in a folder, a draft
listing, reviews both ways, a credit ledger and a personal org is deleted:
- its published agent, sub-agent and listing go to the "deleted user"
  account, and the other account can still open and run it;
- everything else of the account goes, including its auth identity, and the
  orphan-healing job finds nothing to bring it back from;
- the ledger stays, stripped of the block inputs and personal Stripe details;
- a rerun is a no-op.

An account whose agents live in an org another member is active in: the
org-home and team agents (which every member, or the team, can open, though
nobody has added or run them) go to the placeholder with their org and team,
while an org whose only other member is suspended goes with the account.

And the lock and unlock: a lock bans sign-in and records the date, an unlock
lifts only what the lock did, and an account banned before the lock stays
banned, for its own reason, after the unlock.
"""

import uuid
from datetime import datetime, timedelta, timezone

import prisma.enums
import prisma.models
import pytest

from backend.blocks.agent import AgentExecutorBlock
from backend.data.account_deletion.activity import disable_sign_in
from backend.data.account_deletion.database import delete_rows
from backend.data.account_deletion.deleted_user import (
    DELETED_USER_ID,
    DELETED_USER_NAME,
)
from backend.data.account_deletion.lock import lift_lock, record_lock
from backend.data.account_deletion.marketplace import delete_marketplace
from backend.data.account_deletion.models import Subject
from backend.data.account_deletion.preflight import load_subject
from backend.data.account_deletion.shared_agents import (
    plan_transfer,
    transfer_shared_agents,
)
from backend.data.account_deletion.storage import kept_media
from backend.data.db import execute_raw_with_schema
from backend.data.graph import get_graph
from backend.data.user import find_orphaned_auth_identities, get_or_create_user
from backend.util.json import SafeJson
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


async def _graph(owner_id: str, name: str, sub_graph_id: str | None = None):
    nodes = (
        {
            "Nodes": {
                "create": [
                    {
                        "agentBlockId": AgentExecutorBlock().id,
                        "constantInput": SafeJson(
                            {"graph_id": sub_graph_id, "graph_version": 1}
                        ),
                    }
                ]
            }
        }
        if sub_graph_id
        else {}
    )
    return await prisma.models.AgentGraph.prisma().create(
        data={"userId": owner_id, "name": name, **nodes}
    )


async def _listing(
    owner_id: str,
    graph,
    *,
    status=prisma.enums.SubmissionStatus.APPROVED,
    image_urls: list[str] | None = None,
) -> str:
    """A listing of `graph`; the version ID."""
    listing = await prisma.models.StoreListing.prisma().create(
        data={
            "slug": f"deletion-{graph.id}",
            "agentGraphId": graph.id,
            "owningUserId": owner_id,
            "hasApprovedVersion": status == prisma.enums.SubmissionStatus.APPROVED,
        }
    )
    version = await prisma.models.StoreListingVersion.prisma().create(
        data={
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "name": "Deletion test agent",
            "subHeading": "Test",
            "description": "Test",
            "imageUrls": image_urls or [],
            "categories": [],
            "storeListingId": listing.id,
            "submissionStatus": status,
        }
    )
    return version.id


async def _list_under_personal_org(user_id: str, version_id: str) -> None:
    """As the org cutover did: the listing belongs to the personal org, which
    goes with the account while the listing stays."""
    org = await prisma.models.Organization.prisma().find_first_or_raise(
        where={"bootstrapUserId": user_id, "isPersonal": True}
    )
    await execute_raw_with_schema(
        'UPDATE {schema_prefix}"StoreListing" SET "owningOrgId" = $1 '
        'WHERE "id" = (SELECT "storeListingId" FROM '
        '{schema_prefix}"StoreListingVersion" WHERE "id" = $2)',
        org.id,
        version_id,
    )


async def _review(version_id: str, reviewer_id: str) -> None:
    await prisma.models.StoreListingReview.prisma().create(
        data={
            "storeListingVersionId": version_id,
            "reviewByUserId": reviewer_id,
            "score": 5,
        }
    )


async def _uses(other_id: str, graph, creator_id: str) -> None:
    """`other_id` adds `graph` from the marketplace and runs it."""
    creator = await prisma.models.Profile.prisma().find_unique_or_raise(
        where={"userId": creator_id}
    )
    await prisma.models.LibraryAgent.prisma().create(
        data={
            "userId": other_id,
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
            "creatorId": creator.id,
        }
    )
    await prisma.models.AgentGraphExecution.prisma().create(
        data={
            "userId": other_id,
            "agentGraphId": graph.id,
            "agentGraphVersion": graph.version,
        }
    )


async def _holdings(user_id: str) -> None:
    """Ledger rows and a library entry of its own agent in a folder."""
    for metadata in (
        {
            "graph_exec_id": "exec-1",
            "block_id": "block-1",
            "input": {"prompt": "my secret plans"},
            "reason": "Ran a block",
        },
        {
            "id": "cs_test_1",
            "payment_intent": {"id": "pi_test_1", "receipt_email": "sam@x.com"},
            "customer_details": {"email": "sam@x.com", "name": "Sam"},
            "amount_total": 500,
        },
    ):
        await prisma.models.CreditTransaction.prisma().create(
            data={
                "userId": user_id,
                "amount": 100,
                "type": prisma.enums.CreditTransactionType.GRANT,
                "metadata": SafeJson(metadata),
            }
        )
    graph = await _graph(user_id, "Own agent")
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


async def _verifications(user_id: str, email: str) -> list[str]:
    """A reset link (value = the user ID) and an email cooldown (identifier
    names the address, in any case) of the account; and a cooldown of an
    address that only ends like its own, which stays. The ID of the one that
    stays."""
    expires = datetime.now(timezone.utc) + timedelta(minutes=10)
    rows = {
        f"{user_id}-reset": (f"reset-password:{uuid.uuid4()}", user_id),
        f"{user_id}-cooldown": (f"verify-email:{email.upper()}", "sent"),
        f"{user_id}-other": (f"verify-email:x.{email}", "sent"),
    }
    for row_id, (identifier, value) in rows.items():
        await prisma.models.AuthVerification.prisma().create(
            data={
                "id": row_id,
                "identifier": identifier,
                "value": value,
                "expiresAt": expires,
            }
        )
    return [f"{user_id}-other"]


async def _delete(user_id: str) -> Subject:
    subject = await load_subject(user_id)
    plan = await plan_transfer(user_id)
    subject.shared_graph_ids = plan.graph_ids
    subject.kept_listing_ids = plan.listing_ids
    await transfer_shared_agents(subject, dry_run=False)
    await delete_marketplace(subject, dry_run=False)
    await delete_rows(subject, dry_run=False)
    return subject


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
        "personal_orgs": await prisma.models.Organization.prisma().count(
            where={"bootstrapUserId": user_id}
        ),
        "graphs": await prisma.models.AgentGraph.prisma().count(
            where={"userId": user_id}
        ),
        "listings": await prisma.models.StoreListing.prisma().count(
            where={"owningUserId": user_id}
        ),
    }


@pytest.mark.asyncio(loop_scope="session")
async def test_an_account_is_deleted_and_its_shared_agents_keep_working(
    server: SpinTestServer,
):
    user_id = await _account()
    other_id = await _account()
    sub_agent = await _graph(user_id, "Sub-agent")
    published = await _graph(user_id, "Published agent", sub_graph_id=sub_agent.id)
    image = f"https://storage.googleapis.com/media/users/{user_id}/images/a.png"
    own_version = await _listing(user_id, published, image_urls=[image])
    await _list_under_personal_org(user_id, own_version)
    draft = await _graph(user_id, "Draft agent")
    await _listing(user_id, draft, status=prisma.enums.SubmissionStatus.DRAFT)
    other_version = await _listing(other_id, await _graph(other_id, "Theirs"))
    await _review(other_version, user_id)
    await _review(own_version, other_id)
    await _uses(other_id, published, creator_id=user_id)
    await _holdings(user_id)
    email = f"deletion-{user_id[:8]}@example.com"
    kept_verification = await _verifications(user_id, email)

    plan = await plan_transfer(user_id)
    assert sorted(plan.graph_ids) == sorted([published.id, sub_agent.id])
    assert len(plan.listing_ids) == 1
    dry = await delete_rows(await load_subject(user_id), dry_run=True)
    assert dry.counts["user"] == 1
    assert dry.counts["credit_transactions_scrubbed"] == 2

    subject = await _delete(user_id)

    assert set((await _left_of(user_id)).values()) == {0}
    verifications = await prisma.models.AuthVerification.prisma().find_many(
        where={"id": {"startswith": user_id}}
    )
    assert [row.id for row in verifications] == kept_verification
    # The published agent, its sub-agent and its listing now belong to the
    # placeholder, and the listing stays live with the other's review on it.
    for graph in (published, sub_agent):
        kept = await prisma.models.AgentGraph.prisma().find_first_or_raise(
            where={"id": graph.id}
        )
        assert kept.userId == DELETED_USER_ID
    assert not await prisma.models.AgentGraph.prisma().count(where={"id": draft.id})
    version = await prisma.models.StoreListingVersion.prisma().find_unique_or_raise(
        where={"id": own_version}, include={"StoreListing": True}
    )
    assert version.StoreListing and version.StoreListing.owningUserId == (
        DELETED_USER_ID
    )
    assert version.submissionStatus == prisma.enums.SubmissionStatus.APPROVED
    assert await prisma.models.StoreListingReview.prisma().count(
        where={"storeListingVersionId": own_version}
    )
    placeholder = await prisma.models.Profile.prisma().find_unique_or_raise(
        where={"userId": DELETED_USER_ID}
    )
    assert placeholder.name == DELETED_USER_NAME

    # The other account keeps its library entry (now by the placeholder),
    # its run, and can still open the agent with its sub-agent.
    entry = await prisma.models.LibraryAgent.prisma().find_first_or_raise(
        where={"userId": other_id, "agentGraphId": published.id}
    )
    assert entry.creatorId == placeholder.id
    assert await prisma.models.AgentGraphExecution.prisma().count(
        where={"userId": other_id, "agentGraphId": published.id}
    )
    opened = await get_graph(
        published.id, published.version, other_id, include_subgraphs=True
    )
    assert opened is not None
    assert [g.id for g in opened.sub_graphs] == [sub_agent.id]
    # The listing's image is kept; nothing else of the account's is.
    assert await kept_media(subject) == {f"users/{user_id}/images/a.png"}

    # The ledger stays, without the block input or the Stripe customer.
    ledger = await prisma.models.CreditTransaction.prisma().find_many(
        where={"userId": user_id}
    )
    metadata = [row.metadata for row in ledger]
    assert len(metadata) == 2
    assert {"graph_exec_id": "exec-1", "block_id": "block-1"} in metadata
    assert {
        "id": "cs_test_1",
        "payment_intent": "pi_test_1",
        "amount_total": 500,
    } in metadata
    assert [row.amount for row in ledger] == [100, 100]

    # The other account is untouched; only the review the deleted account
    # wrote on its listing is gone.
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
    assert await plan_transfer(user_id) == type(plan)()

    # The healing job finds no orphaned identity to re-create the account
    # from. (Its query, read-only: healing would provision other tests' rows.)
    orphans = await find_orphaned_auth_identities(
        datetime.now(timezone.utc) + timedelta(minutes=1), limit=10_000
    )
    assert user_id not in {orphan.id for orphan in orphans}

    await _delete(other_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_lock_bans_sign_in_and_an_unlock_lifts_only_that(
    server: SpinTestServer,
):
    user_id = await _account()
    subject = await load_subject(user_id)

    await disable_sign_in(subject, dry_run=False)
    await record_lock(subject, dry_run=False)

    identity = await prisma.models.AuthUser.prisma().find_unique_or_raise(
        where={"id": user_id}
    )
    user = await prisma.models.User.prisma().find_unique_or_raise(where={"id": user_id})
    assert identity.banned
    assert user.deletionLockedAt is not None
    assert user.marketingOptOutAt is not None
    assert not await prisma.models.AuthSession.prisma().count(where={"userId": user_id})

    await lift_lock(await load_subject(user_id), dry_run=False)

    identity = await prisma.models.AuthUser.prisma().find_unique_or_raise(
        where={"id": user_id}
    )
    user = await prisma.models.User.prisma().find_unique_or_raise(where={"id": user_id})
    assert not identity.banned
    assert user.deletionLockedAt is None
    assert user.marketingOptOutAt is None

    await _delete(user_id)


async def _org(*members: tuple[str, str]) -> str:
    """An org with these (user ID, status) members, all owners."""
    org = await prisma.models.Organization.prisma().create(
        data={"name": "Deletion test org", "slug": f"deletion-{uuid.uuid4()}"}
    )
    for user_id, status in members:
        await prisma.models.OrgMember.prisma().create(
            data={
                "orgId": org.id,
                "userId": user_id,
                "isOwner": True,
                "status": prisma.enums.OrgMemberStatus(status),
            }
        )
    return org.id


@pytest.mark.asyncio(loop_scope="session")
async def test_agents_another_org_member_can_open_go_to_the_placeholder(
    server: SpinTestServer,
):
    user_id = await _account()
    colleague = await _account()
    shared_org = await _org((user_id, "ACTIVE"), (colleague, "ACTIVE"))
    team = await prisma.models.Team.prisma().create(
        data={"name": "Deletion test team", "orgId": shared_org}
    )
    org_home = await prisma.models.AgentGraph.prisma().create(
        data={"userId": user_id, "name": "Org agent", "organizationId": shared_org}
    )
    team_agent = await prisma.models.AgentGraph.prisma().create(
        data={
            "userId": user_id,
            "name": "Team agent",
            "organizationId": shared_org,
            "teamId": team.id,
        }
    )
    # Only suspended members besides the account: nobody else can open it.
    lapsed_org = await _org((user_id, "ACTIVE"), (colleague, "SUSPENDED"))
    lapsed = await prisma.models.AgentGraph.prisma().create(
        data={"userId": user_id, "name": "Lapsed", "organizationId": lapsed_org}
    )
    personal = await prisma.models.AgentGraph.prisma().create(
        data={"userId": user_id, "name": "Personal agent"}
    )
    # Before: the colleague sees the org agent without any row of its own.
    assert await get_graph(org_home.id, None, colleague, organization_id=shared_org)

    subject = await load_subject(user_id)
    assert shared_org not in subject.sole_org_ids
    assert lapsed_org in subject.sole_org_ids
    assert sorted((await plan_transfer(user_id)).graph_ids) == sorted(
        [org_home.id, team_agent.id]
    )

    await _delete(user_id)

    for graph, team_id in ((org_home, None), (team_agent, team.id)):
        kept = await prisma.models.AgentGraph.prisma().find_first_or_raise(
            where={"id": graph.id}
        )
        assert kept.userId == DELETED_USER_ID
        assert kept.organizationId == shared_org
        assert kept.teamId == team_id
    opened = await get_graph(org_home.id, None, colleague, organization_id=shared_org)
    assert opened is not None
    for gone in (lapsed, personal):
        assert not await prisma.models.AgentGraph.prisma().count(where={"id": gone.id})
    assert await prisma.models.Organization.prisma().count(where={"id": shared_org})
    assert not await prisma.models.Organization.prisma().count(where={"id": lapsed_org})
    assert await prisma.models.OrgMember.prisma().count(
        where={"orgId": shared_org, "userId": colleague}
    )

    await _delete(colleague)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_ban_in_force_before_the_lock_survives_the_unlock(
    server: SpinTestServer,
):
    user_id = await _account()
    await prisma.models.AuthUser.prisma().update(
        where={"id": user_id},
        data={"banned": True, "banReason": "Abuse", "banExpires": None},
    )
    subject = await load_subject(user_id)

    await disable_sign_in(subject, dry_run=False)
    await record_lock(subject, dry_run=False)

    identity = await prisma.models.AuthUser.prisma().find_unique_or_raise(
        where={"id": user_id}
    )
    assert (identity.banned, identity.banReason) == (True, "Abuse")
    assert not await prisma.models.AuthSession.prisma().count(where={"userId": user_id})

    unlocked = await lift_lock(await load_subject(user_id), dry_run=False)

    identity = await prisma.models.AuthUser.prisma().find_unique_or_raise(
        where={"id": user_id}
    )
    assert (identity.banned, identity.banReason) == (True, "Abuse")
    assert unlocked.counts["ban_lifted"] == 0
    assert unlocked.counts["lock_cleared"] == 1

    await _delete(user_id)
