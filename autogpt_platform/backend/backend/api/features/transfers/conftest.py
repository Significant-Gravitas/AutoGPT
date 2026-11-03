from uuid import uuid4

import pytest_asyncio
from prisma.models import (
    AgentGraph,
    AuditLog,
    LibraryAgent,
    Organization,
    OrgMember,
    Profile,
    StoreListing,
    Team,
    TeamMember,
    TransferRequest,
    User,
)


@pytest_asyncio.fixture(loop_scope="session", params=["AgentGraph", "StoreListing"])
async def transfer_world(server, request):
    """Disposable resource, three organizations, and two users in the real DB:
    the source user administers org A, the target user orgs B and C."""
    source_user, target_user = str(uuid4()), str(uuid4())
    for user_id in (source_user, target_user):
        await User.prisma().create(
            data={"id": user_id, "email": f"transfer-{user_id}@example.com"}
        )
    orgs = [
        (
            await Organization.prisma().create(
                data={"name": f"Transfer {name}", "slug": f"transfer-{name}-{uuid4()}"}
            )
        ).id
        for name in ("a", "b", "c")
    ]
    org_a, org_b, org_c = orgs
    # A transfer is created, approved and executed by an admin of each side.
    for org_id, user_id in (
        (org_a, source_user),
        (org_b, target_user),
        (org_c, target_user),
    ):
        await OrgMember.prisma().create(
            data={
                "orgId": org_id,
                "userId": user_id,
                "isOwner": True,
                "isAdmin": True,
                "status": "ACTIVE",
            }
        )
    # A listed workflow moves to the target approver, who needs a profile.
    await Profile.prisma().create(
        data={
            "userId": target_user,
            "name": "Transfer recipient",
            "username": f"transfer-{uuid4()}",
            "description": "",
            "links": [],
        }
    )
    team = await Team.prisma().create(
        data={"orgId": org_a, "name": "Transfer source", "slug": str(uuid4())}
    )
    # A team-scoped workflow's owner must be on that team.
    await TeamMember.prisma().create(
        data={
            "teamId": team.id,
            "userId": source_user,
            "isAdmin": True,
            "status": "ACTIVE",
        }
    )
    graph_id = str(uuid4())
    for version in (1, 2):
        await AgentGraph.prisma().create(
            data={
                "id": graph_id,
                "version": version,
                "userId": source_user,
                "organizationId": org_a,
                "teamId": team.id,
                "isActive": version == 2,
            }
        )
    resource_id = graph_id
    if request.param == "StoreListing":
        await Profile.prisma().create(
            data={
                "userId": source_user,
                "name": "Transfer creator",
                "username": f"transfer-{uuid4()}",
                "description": "",
                "links": [],
            }
        )
        listing = await StoreListing.prisma().create(
            data={
                "agentGraphId": graph_id,
                "owningUserId": source_user,
                "owningOrgId": org_a,
                "slug": f"transfer-{uuid4()}",
            }
        )
        resource_id = listing.id

    yield {
        "source_user": source_user,
        "target_user": target_user,
        "org_a": org_a,
        "org_b": org_b,
        "org_c": org_c,
        "graph_id": graph_id,
        "resource_id": resource_id,
        "resource_type": request.param,
    }

    await AuditLog.prisma().delete_many(where={"organizationId": {"in": orgs}})
    await TransferRequest.prisma().delete_many(where={"resourceId": resource_id})
    await StoreListing.prisma().delete_many(where={"agentGraphId": graph_id})
    await LibraryAgent.prisma().delete_many(where={"agentGraphId": graph_id})
    await AgentGraph.prisma().delete_many(where={"id": graph_id})
    await Team.prisma().delete(where={"id": team.id})
    await Organization.prisma().delete_many(where={"id": {"in": orgs}})
    await User.prisma().delete_many(where={"id": {"in": [source_user, target_user]}})
