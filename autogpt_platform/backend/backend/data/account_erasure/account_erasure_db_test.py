"""Against the real schema.

An account with a published agent (and the sub-agent it runs) that another
account has in its library and has run, a private agent in a folder, a draft
listing, a published skill with a file, reviews both ways, a credit ledger and
a personal org is closed the way Settings closes it (`close_account`, SECRT-2877), and its deletion
request is then processed:
- it is pending once past the grace, and the confirmation the close did not
  send goes out first, to its address;
- its published agent, sub-agent and listing go to the "deleted user"
  account, the listing still taken down, and the other account can still
  open and run the agent;
- everything else of the account goes, including its auth identity and its
  skill listing with the listing's version and file (not left ownerless), and
  the orphan-healing job finds nothing to bring it back from;
- the ledger stays, stripped of the block inputs and personal Stripe details;
- the request stays, marked done, and is no longer pending;
- a rerun is a no-op.

An account whose agents live in an org another member is active in, closed by
a request made by email: the org-home and team agents (which every member, or
the team, can open, though nobody has added or run them) go to the
placeholder with their org and team, while an org whose only other member is
suspended goes with the account.

An account that registered an OAuth app another account still has a live
grant to is not erased until the operator moves or disables the app.
"""

import uuid
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock

import prisma
import prisma.enums
import prisma.models
import pytest

from backend.blocks.agent import AgentExecutorBlock
from backend.data.account_deletion import close_account
from backend.data.account_erasure import request, routine
from backend.data.account_erasure.database import delete_rows
from backend.data.account_erasure.deleted_user import DELETED_USER_ID, DELETED_USER_NAME
from backend.data.account_erasure.marketplace import delete_marketplace
from backend.data.account_erasure.models import StepReport, StepStatus, Subject
from backend.data.account_erasure.preflight import find_blockers, load_subject
from backend.data.account_erasure.shared_agents import (
    plan_transfer,
    transfer_shared_agents,
)
from backend.data.account_erasure.storage import kept_media
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
            "expiresAt": datetime.now(UTC) + timedelta(days=1),
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


async def _skill_listing(owner_id: str) -> tuple[str, str, str]:
    """A published skill listing with one file: its, its version's and its
    file's IDs."""
    listing = await prisma.models.SkillListing.prisma().create(
        data={
            "slug": f"deletion-skill-{owner_id}",
            "owningUserId": owner_id,
            "hasApprovedVersion": True,
        }
    )
    version = await prisma.models.SkillListingVersion.prisma().create(
        data={
            "skillListingId": listing.id,
            "name": "Deletion test skill",
            "description": "Test",
            "body": "Test",
            "categories": [],
            "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        }
    )
    await prisma.models.SkillListing.prisma().update(
        where={"id": listing.id}, data={"activeVersionId": version.id}
    )
    file = await prisma.models.SkillListingFile.prisma().create(
        data={
            "skillListingVersionId": version.id,
            "relativePath": "scripts/run.py",
            "sizeBytes": 5,
            "sha256": "x",
            "content": prisma.Base64.encode(b"print"),
        }
    )
    return listing.id, version.id, file.id


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
    expires = datetime.now(UTC) + timedelta(minutes=10)
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


async def _erase_rows(user_id: str) -> Subject:
    """The steps that change rows, without the routine's outside services."""
    subject = await load_subject(user_id)
    plan = await plan_transfer(user_id)
    subject.shared_graph_ids = plan.graph_ids
    subject.kept_listing_ids = plan.listing_ids
    await transfer_shared_agents(subject, dry_run=False)
    await delete_marketplace(subject, dry_run=False)
    await delete_rows(subject, dry_run=False)
    return subject


@pytest.fixture
def outside_services(monkeypatch):
    """The erasure's steps that reach other services, stubbed: the work stop
    (scheduler, queues), files, memory, PostHog and MailerLite. The
    confirmation email is captured."""
    for name in (
        "stop_work",
        "delete_files",
        "delete_memory",
        "delete_posthog_person",
        "forget_mailerlite_subscriber",
    ):
        step = name.removeprefix("delete_").removeprefix("forget_")

        async def done(subject: Subject, *, dry_run: bool, _step=step):
            return StepReport(name=_step, status=StepStatus.DONE)

        monkeypatch.setattr(routine, name, done)
    sent = AsyncMock()
    monkeypatch.setattr(request, "send_account_deleted_email", sent)
    return sent


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
async def test_a_closed_account_is_erased_and_its_shared_agents_keep_working(
    server: SpinTestServer, outside_services: AsyncMock
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
    skill, skill_version, skill_file = await _skill_listing(user_id)
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

    # Closed from Settings, with the confirmation email not sent.
    assert await close_account(user_id, prisma.enums.AccountDeletionSource.SETTINGS)
    soon = datetime.now(UTC) + timedelta(minutes=1)
    assert user_id not in await request.pending_user_ids(now=soon)
    assert user_id in await request.pending_user_ids(grace=timedelta(0), now=soon)
    subject = await load_subject(user_id)
    subject.kept_listing_ids = (await plan_transfer(user_id)).listing_ids

    report = await routine.erase_account(user_id, dry_run=False)

    assert report.complete, report
    outside_services.assert_awaited_once_with(
        email, prisma.enums.AccountDeletionSource.SETTINGS
    )
    assert set((await _left_of(user_id)).values()) == {0}
    # Gone, not left behind with no owner (which reads as a platform skill).
    assert not await prisma.models.SkillListing.prisma().count(where={"id": skill})
    assert not await prisma.models.SkillListingVersion.prisma().count(
        where={"id": skill_version}
    )
    assert not await prisma.models.SkillListingFile.prisma().count(
        where={"id": skill_file}
    )
    done = await prisma.models.AccountDeletionRequest.prisma().find_unique_or_raise(
        where={"userId": user_id}
    )
    assert done.confirmationSentAt is not None
    assert done.completedAt is not None
    assert user_id not in await request.pending_user_ids(grace=timedelta(0), now=soon)
    verifications = await prisma.models.AuthVerification.prisma().find_many(
        where={"id": {"startswith": user_id}}
    )
    assert [row.id for row in verifications] == kept_verification
    # The published agent, its sub-agent and its listing now belong to the
    # placeholder, with the other's review on it. The close took the listing
    # down, and the erasure does not put it back up.
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
    assert version.StoreListing.isDeleted
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

    # A rerun finds nothing, changes nothing and sends nothing.
    again = await delete_rows(await load_subject(user_id), dry_run=False)
    assert set(again.counts.values()) == {0}
    assert await plan_transfer(user_id) == type(plan)()
    rerun = await routine.erase_account(user_id, dry_run=False)
    assert rerun.complete
    outside_services.assert_awaited_once()
    assert (
        await prisma.models.AccountDeletionRequest.prisma().find_unique_or_raise(
            where={"userId": user_id}
        )
    ).completedAt == done.completedAt

    # The healing job finds no orphaned identity to re-create the account
    # from. (Its query, read-only: healing would provision other tests' rows.)
    orphans = await find_orphaned_auth_identities(
        datetime.now(UTC) + timedelta(minutes=1), limit=10_000
    )
    assert user_id not in {orphan.id for orphan in orphans}

    await _erase_rows(other_id)


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

    # Closed for a request made by email.
    assert await close_account(user_id, prisma.enums.AccountDeletionSource.EMAIL)
    subject = await load_subject(user_id)
    assert subject.request_source == prisma.enums.AccountDeletionSource.EMAIL
    assert shared_org not in subject.sole_org_ids
    assert lapsed_org in subject.sole_org_ids
    assert sorted((await plan_transfer(user_id)).graph_ids) == sorted(
        [org_home.id, team_agent.id]
    )

    await _erase_rows(user_id)

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

    await _erase_rows(colleague)


async def _oauth_app(owner_id: str) -> str:
    app = await prisma.models.OAuthApplication.prisma().create(
        data={
            "name": "Deletion test app",
            "clientId": f"deletion-{uuid.uuid4()}",
            "clientSecret": "hash",
            "clientSecretSalt": "salt",
            "redirectUris": ["https://example.com/callback"],
            "scopes": [prisma.enums.APIKeyPermission.EXECUTE_GRAPH],
            "ownerId": owner_id,
        }
    )
    return app.id


async def _grant(app_id: str, user_id: str, *, revoked: bool = False) -> None:
    now = datetime.now(UTC)
    await prisma.models.OAuthRefreshToken.prisma().create(
        data={
            "token": f"deletion-{uuid.uuid4()}",
            "expiresAt": now + timedelta(days=30),
            "applicationId": app_id,
            "userId": user_id,
            "scopes": [prisma.enums.APIKeyPermission.EXECUTE_GRAPH],
            "revokedAt": now if revoked else None,
        }
    )


async def _blocker_names(user_id: str) -> list[str]:
    return [b.name for b in await find_blockers(await load_subject(user_id))]


@pytest.mark.asyncio(loop_scope="session")
async def test_an_oauth_app_other_accounts_use_waits_for_the_operator(
    server: SpinTestServer,
):
    developer = await _account()
    user = await _account()
    used = await _oauth_app(developer)
    await _grant(used, user)
    # Only a revoked grant: nobody uses it any more, so it goes.
    unused = await _oauth_app(developer)
    await _grant(unused, user, revoked=True)
    assert await close_account(developer, prisma.enums.AccountDeletionSource.SETTINGS)

    assert await _blocker_names(developer) == ["oauth_apps_in_use"]

    # Disabled, it already works for nobody: the erasure may delete it.
    await prisma.models.OAuthApplication.prisma().update(
        where={"id": used}, data={"isActive": False}
    )
    assert await _blocker_names(developer) == []

    await _erase_rows(developer)
    assert not await prisma.models.OAuthApplication.prisma().count(
        where={"id": {"in": [used, unused]}}
    )
    await _erase_rows(user)
