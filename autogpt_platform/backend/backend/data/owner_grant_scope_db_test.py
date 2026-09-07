from types import SimpleNamespace
from uuid import uuid4

import pytest
import pytest_asyncio

from backend.api.features.orgs.grant_db import revoke_grant
from backend.data import grants, graph
from backend.data.db import prisma
from backend.data.org_migration import create_personal_org

pytestmark = pytest.mark.integration


@pytest_asyncio.fixture
async def scoped_owner_grant(server):
    owner_id, consumer_id, graph_id = (str(uuid4()) for _ in range(3))
    org_ids = []
    try:
        for user_id in [owner_id, consumer_id]:
            await prisma.user.create(
                data={"id": user_id, "email": f"{user_id}@test.invalid"}
            )
            org = await create_personal_org(
                user_id, f"scope-grant-{user_id}", "Grant test"
            )
            org_ids.append(org.id)
        source_org, consumer_org = org_ids
        team = await prisma.team.find_first(
            where={"orgId": source_org, "isDefault": True}
        )
        assert team is not None
        await prisma.orgmember.create(data={"orgId": source_org, "userId": consumer_id})
        await prisma.teammember.create(data={"teamId": team.id, "userId": consumer_id})
        await prisma.agentgraph.create(
            data={
                "id": graph_id,
                "version": 1,
                "name": "Public workflow",
                "userId": owner_id,
                "organizationId": source_org,
                "teamId": team.id,
            }
        )
        await prisma.profile.create(
            data={
                "userId": owner_id,
                "username": owner_id,
                "name": "Owner",
                "description": "Test",
                "links": [],
            }
        )
        listing = await prisma.storelisting.create(
            data={
                "agentGraphId": graph_id,
                "owningUserId": owner_id,
                "owningOrgId": source_org,
                "slug": graph_id,
            }
        )
        await prisma.storelistingversion.create(
            data={
                "storeListingId": listing.id,
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "name": "Public workflow",
                "subHeading": "Test",
                "description": "Test",
                "imageUrls": [],
                "categories": [],
                "submissionStatus": "APPROVED",
                "organizationId": source_org,
                "teamId": team.id,
            }
        )
        for org_id in org_ids:
            await prisma.libraryagent.create(
                data={
                    "userId": consumer_id,
                    "agentGraphId": graph_id,
                    "agentGraphVersion": 1,
                    "organizationId": org_id,
                    "teamId": None,
                }
            )
        grant = await prisma.agentgraphgrant.create(
            data={
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "organizationId": source_org,
                "principalType": "TEAM",
                "principalId": team.id,
                "createdByUserId": owner_id,
                "capability": "EXECUTE",
                "credentialMode": "OWNER",
            }
        )
        yield SimpleNamespace(
            owner=owner_id,
            consumer=consumer_id,
            graph=graph_id,
            grant=grant.id,
            source_org=source_org,
            consumer_org=consumer_org,
            team=team.id,
        )
    finally:
        await prisma.agentgraphgrant.delete_many(where={"agentGraphId": graph_id})
        await prisma.libraryagent.delete_many(where={"agentGraphId": graph_id})
        await prisma.storelisting.delete_many(where={"agentGraphId": graph_id})
        await prisma.agentgraph.delete_many(where={"id": graph_id})
        await prisma.organization.delete_many(where={"id": {"in": org_ids}})
        await prisma.user.delete_many(where={"id": {"in": [owner_id, consumer_id]}})


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "scope_case",
    [
        "grant_team",
        "org_home",
        "foreign_org_home",
        "foreign_org_team",
        "wrong_team",
        "missing_scope",
    ],
)
async def test_owner_credential_authorization_is_pinned_to_execution_scope(
    scoped_owner_grant, scope_case
):
    data = scoped_owner_grant
    organization_id, team_id = data.source_org, data.team
    if scope_case in {"org_home", "foreign_org_home"}:
        team_id = None
    if scope_case.startswith("foreign_org"):
        organization_id = data.consumer_org
    if scope_case == "wrong_team":
        team_id = str(uuid4())
    if scope_case == "missing_scope":
        organization_id, team_id = None, None
    if scope_case in {"org_home", "foreign_org_home"}:
        await graph.validate_graph_execution_permissions(
            data.consumer,
            data.graph,
            1,
            organization_id=organization_id,
            team_id_restriction=None,
        )
    resolved = await grants.resolve_execution_credentials_owner(
        data.consumer,
        data.graph,
        1,
        organization_id=organization_id,
        team_id_restriction=team_id,
    )
    valid = await grants.validate_execution_credentials_owner(
        data.consumer,
        data.graph,
        1,
        data.owner,
        data.grant,
        organization_id=organization_id,
        team_id_restriction=team_id,
    )
    assert resolved == (
        (data.owner, data.grant) if scope_case == "grant_team" else None
    )
    assert valid is (scope_case == "grant_team")


@pytest.mark.asyncio
@pytest.mark.parametrize("revocation", ["grant", "membership"])
async def test_scoped_owner_grant_revalidation_rejects_revocation(
    scoped_owner_grant, revocation
):
    data = scoped_owner_grant
    kwargs = {"organization_id": data.source_org, "team_id_restriction": data.team}
    assert await grants.validate_execution_credentials_owner(
        data.consumer,
        data.graph,
        1,
        data.owner,
        data.grant,
        **kwargs,
    )
    if revocation == "grant":
        await revoke_grant(
            data.source_org,
            data.graph,
            data.grant,
            revoked_by_user_id=data.owner,
            revoker_is_org_admin=True,
        )
    else:
        await prisma.teammember.update(
            where={"teamId_userId": {"teamId": data.team, "userId": data.consumer}},
            data={"status": "REMOVED"},
        )
    assert (
        await grants.resolve_execution_credentials_owner(
            data.consumer,
            data.graph,
            1,
            **kwargs,
        )
        is None
    )
    assert not await grants.validate_execution_credentials_owner(
        data.consumer,
        data.graph,
        1,
        data.owner,
        data.grant,
        **kwargs,
    )
