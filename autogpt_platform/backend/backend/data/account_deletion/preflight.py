"""Read-only: what the account holds, and what would stop its deletion.

Runs before anything is deleted, so a blocked deletion leaves the account
exactly as it was.
"""

from prisma.models import (
    AuthUser,
    Expert,
    OAuthApplication,
    Organization,
    OrganizationSubscription,
    OrgMember,
    User,
    UserWorkspace,
)
from pydantic import BaseModel

from backend.data.account_deletion.models import Blocker, Subject
from backend.data.db import query_raw_with_schema

# Other accounts' rows that point at this account's agents. The database
# refuses to delete an agent graph while a library entry, preset or store
# version of another account still references it.
_SHARED_AGENT_COUNTS = """
    SELECT
      (SELECT COUNT(*) FROM {schema_prefix}"LibraryAgent" la
         JOIN {schema_prefix}"AgentGraph" g
           ON g."id" = la."agentGraphId" AND g."version" = la."agentGraphVersion"
        WHERE g."userId" = $1 AND la."userId" <> $1)::int AS library_entries,
      (SELECT COUNT(*) FROM {schema_prefix}"AgentPreset" p
         JOIN {schema_prefix}"AgentGraph" g
           ON g."id" = p."agentGraphId" AND g."version" = p."agentGraphVersion"
        WHERE g."userId" = $1 AND p."userId" <> $1)::int AS presets,
      (SELECT COUNT(*) FROM {schema_prefix}"StoreListingVersion" v
         JOIN {schema_prefix}"StoreListing" l ON l."id" = v."storeListingId"
         JOIN {schema_prefix}"AgentGraph" g
           ON g."id" = v."agentGraphId" AND g."version" = v."agentGraphVersion"
        WHERE g."userId" = $1 AND l."owningUserId" <> $1)::int AS foreign_listings
"""


class SharedAgentCounts(BaseModel):
    library_entries: int = 0
    presets: int = 0
    foreign_listings: int = 0


async def load_subject(user_id: str, email: str | None = None) -> Subject:
    user = await User.prisma().find_unique(where={"id": user_id})
    identity = await AuthUser.prisma().find_unique(where={"id": user_id})
    workspace = await UserWorkspace.prisma().find_unique(where={"userId": user_id})
    apps = await OAuthApplication.prisma().find_many(where={"ownerId": user_id})
    experts = await Expert.prisma().find_many(where={"ownerUserId": user_id})
    sole_orgs, _ = await _orgs(user_id)
    return Subject(
        user_id=user_id,
        email=(user and user.email) or (identity and identity.email) or email,
        has_user_row=user is not None,
        has_auth_identity=identity is not None,
        workspace_id=workspace.id if workspace else None,
        oauth_app_ids=[app.id for app in apps],
        sole_org_ids=[org.id for org in sole_orgs],
        stripe_customer_ids=await _stripe_customers(user, sole_orgs),
        expert_ids=[expert.id for expert in experts],
    )


async def find_blockers(
    subject: Subject, shared: SharedAgentCounts, *, remove_shared_agents: bool
) -> list[Blocker]:
    _, solely_owned_shared_orgs = await _orgs(subject.user_id)
    blockers: list[Blocker] = []
    if shared.library_entries + shared.presets and not remove_shared_agents:
        blockers.append(
            Blocker(
                name="agents_used_by_other_accounts",
                count=shared.library_entries + shared.presets,
                resolution="Other accounts have this account's agents in their "
                "library or presets. Re-run with --remove-shared-agents to take "
                "them out (their runs of those agents go too), once that is "
                "agreed.",
            )
        )
    if shared.foreign_listings:
        blockers.append(
            Blocker(
                name="agents_listed_by_other_accounts",
                count=shared.foreign_listings,
                resolution="Another account lists one of this account's agents "
                "in the marketplace. Remove that listing by hand first.",
            )
        )
    if solely_owned_shared_orgs:
        blockers.append(
            Blocker(
                name="shared_orgs_without_another_owner",
                count=len(solely_owned_shared_orgs),
                resolution="This account is the only owner of an organization "
                "with other members. Make another member an owner first.",
            )
        )
    return blockers


async def shared_agent_counts(user_id: str) -> SharedAgentCounts:
    rows = await query_raw_with_schema(
        _SHARED_AGENT_COUNTS, user_id, model=SharedAgentCounts
    )
    return rows[0] if rows else SharedAgentCounts()


async def _orgs(user_id: str) -> tuple[list[Organization], list[Organization]]:
    """The orgs only this account is in, and the shared orgs it alone owns.
    A personal org whose membership row is gone still counts as its own."""
    memberships = await OrgMember.prisma().find_many(
        where={"userId": user_id}, include={"Org": True}
    )
    owns = {m.orgId for m in memberships if m.isOwner}
    orgs = {m.orgId: m.Org for m in memberships if m.Org}
    for org in await Organization.prisma().find_many(
        where={"isPersonal": True, "bootstrapUserId": user_id}
    ):
        orgs.setdefault(org.id, org)
    sole: list[Organization] = []
    solely_owned_shared: list[Organization] = []
    for org in orgs.values():
        others = await OrgMember.prisma().find_many(
            where={"orgId": org.id, "userId": {"not": user_id}}
        )
        if not others:
            sole.append(org)
        elif org.id in owns and not any(m.isOwner for m in others):
            solely_owned_shared.append(org)
    return sole, solely_owned_shared


async def _stripe_customers(user: User | None, orgs: list[Organization]) -> list[str]:
    subscriptions = await OrganizationSubscription.prisma().find_many(
        where={"organizationId": {"in": [org.id for org in orgs]}}
    )
    ids = [user.stripeCustomerId if user else None]
    ids += [org.stripeCustomerId for org in orgs]
    ids += [sub.stripeCustomerId for sub in subscriptions]
    return list(dict.fromkeys(i for i in ids if i))
