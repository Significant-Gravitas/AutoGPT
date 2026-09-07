from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from backend.data import db, org_migration


@pytest.mark.asyncio
async def test_legacy_store_listings_migrate_after_graphs_and_remain_idempotent(mocker):
    user_id = str(uuid4())
    graph_id = str(uuid4())
    listing_id = str(uuid4())
    try:
        await db.prisma.user.create(
            data={"id": user_id, "email": f"{user_id}@store-migration.example"}
        )
        await db.prisma.profile.create(
            data={
                "userId": user_id,
                "username": f"store-{user_id}",
                "name": "Store",
                "description": "Migration test",
                "links": [],
            }
        )
        organization = await org_migration.create_personal_org(
            user_id, f"store-{user_id}", "Store"
        )
        team = await db.prisma.team.find_first_or_raise(
            where={"orgId": organization.id, "isDefault": True}
        )
        for version in (1, 2):
            await db.prisma.agentgraph.create(
                data={"id": graph_id, "version": version, "userId": user_id}
            )
        await db.prisma.storelisting.create(
            data={
                "id": listing_id,
                "agentGraphId": graph_id,
                "owningUserId": user_id,
                "slug": "legacy-agent",
            }
        )
        for version in (1, 2):
            await db.prisma.storelistingversion.create(
                data={
                    "agentGraphId": graph_id,
                    "agentGraphVersion": version,
                    "storeListingId": listing_id,
                    "version": version,
                    "name": "Legacy agent",
                    "subHeading": "Legacy",
                    "description": "Before organization rollout",
                    "imageUrls": [],
                    "categories": [],
                }
            )
        redis = AsyncMock()
        redis.set.return_value = True
        redis.execute_command.return_value = 1
        mocker.patch("backend.data.redis_client.get_redis_async", return_value=redis)
        for step in (
            "create_orgs_for_existing_users",
            "migrate_org_balances",
            "migrate_credit_transactions",
            "create_store_listing_aliases",
            "migrate_credentials_to_table",
        ):
            mocker.patch.object(org_migration, step, new=AsyncMock(return_value=0))

        await org_migration.run_migration()

        graphs = await db.prisma.agentgraph.find_many(where={"id": graph_id})
        assert len(graphs) == 2
        assert {(graph.organizationId, graph.teamId) for graph in graphs} == {
            (organization.id, team.id)
        }
        listing = await db.prisma.storelisting.find_unique_or_raise(
            where={"id": listing_id}
        )
        assert listing.owningOrgId == organization.id
        versions = await db.prisma.storelistingversion.find_many(
            where={"storeListingId": listing_id}
        )
        assert len(versions) == 2
        assert {(version.organizationId, version.teamId) for version in versions} == {
            (organization.id, team.id)
        }
        counts = await org_migration.assign_resources_to_teams()
        assert counts["AgentGraph"] == 0
        assert counts["StoreListing"] == 0
        assert counts["StoreListingVersion"] == 0
    finally:
        await db.prisma.storelisting.delete_many(where={"id": listing_id})
        await db.prisma.agentgraph.delete_many(where={"id": graph_id})
        await db.prisma.organization.delete_many(where={"bootstrapUserId": user_id})
        await db.prisma.user.delete_many(where={"id": user_id})


@pytest.mark.asyncio
async def test_store_backfill_does_not_claim_another_owners_graph():
    users = [str(uuid4()), str(uuid4())]
    listing_id = str(uuid4())
    graph_id = str(uuid4())
    try:
        for user_id in users:
            await db.prisma.user.create(
                data={"id": user_id, "email": f"{user_id}@store-owner.example"}
            )
            await db.prisma.profile.create(
                data={
                    "userId": user_id,
                    "username": f"store-{user_id}",
                    "name": "Store",
                    "description": "Migration test",
                    "links": [],
                }
            )
            await org_migration.create_personal_org(
                user_id, f"store-{user_id}", "Store"
            )
        await db.prisma.agentgraph.create(data={"id": graph_id, "userId": users[1]})
        await db.prisma.storelisting.create(
            data={
                "id": listing_id,
                "agentGraphId": graph_id,
                "owningUserId": users[0],
                "slug": "legacy-owner-mismatch",
            }
        )
        version = await db.prisma.storelistingversion.create(
            data={
                "agentGraphId": graph_id,
                "agentGraphVersion": 1,
                "storeListingId": listing_id,
                "name": "Legacy agent",
                "subHeading": "Legacy",
                "description": "Mismatched legacy owner",
                "imageUrls": [],
                "categories": [],
            }
        )

        await org_migration.assign_resources_to_teams()

        listing = await db.prisma.storelisting.find_unique_or_raise(
            where={"id": listing_id}
        )
        assert listing.owningUserId == users[0]
        assert listing.owningOrgId is None
        persisted = await db.prisma.storelistingversion.find_unique_or_raise(
            where={"id": version.id}
        )
        assert (persisted.organizationId, persisted.teamId) == (None, None)
        graph = await db.prisma.agentgraph.find_first_or_raise(
            where={"id": graph_id, "version": 1}
        )
        assert graph.organizationId is not None
        organization = await db.prisma.organization.find_unique_or_raise(
            where={"id": graph.organizationId}
        )
        assert organization.bootstrapUserId == users[1]
    finally:
        await db.prisma.storelisting.delete_many(where={"id": listing_id})
        await db.prisma.agentgraph.delete_many(where={"id": graph_id})
        await db.prisma.organization.delete_many(
            where={"bootstrapUserId": {"in": users}}
        )
        await db.prisma.user.delete_many(where={"id": {"in": users}})
