"""mv_agent_run_counts against a real Postgres.

The view used to join AgentGraph on id only, so each run was counted once per
saved version of its graph. ``db_test.py`` mocks Prisma, so this needs the
database (``server`` starts the test stack).
"""

from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.data.db import execute_raw_with_schema, query_raw_with_schema


async def _refresh() -> None:
    await execute_raw_with_schema(
        "SELECT {schema_prefix}refresh_store_materialized_views()"
    )


async def _seed_listing(tag: str, owner: str, graph_versions: int) -> str:
    graph_id = f"graph-{graph_versions}v-{tag}"
    for version in range(1, graph_versions + 1):
        await prisma.models.AgentGraph.prisma().create(
            data={
                "id": graph_id,
                "version": version,
                "name": graph_id,
                "description": "test",
                "userId": owner,
                "isActive": version == graph_versions,
            }
        )
    listing = await prisma.models.StoreListing.prisma().create(
        data={
            "slug": graph_id,
            "agentGraphId": graph_id,
            "owningUserId": owner,
            "hasApprovedVersion": True,
        }
    )
    version = await prisma.models.StoreListingVersion.prisma().create(
        data={
            "storeListingId": listing.id,
            "version": 1,
            "agentGraphId": graph_id,
            "agentGraphVersion": graph_versions,
            "name": graph_id,
            "subHeading": "",
            "description": "",
            "imageUrls": [],
            "categories": [],
            "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        }
    )
    await prisma.models.StoreListing.prisma().update(
        where={"id": listing.id}, data={"activeVersionId": version.id}
    )
    return graph_id


async def _run_counts(graph_ids: list[str]) -> dict[str, tuple[int, int]]:
    rows = await query_raw_with_schema(
        """
        SELECT sl."agentGraphId" AS graph_id,
               COALESCE(mv.run_count, 0)::int AS mv_runs,
               sa.runs::int AS store_agent_runs
        FROM {schema_prefix}"StoreListing" sl
        LEFT JOIN {schema_prefix}mv_agent_run_counts mv
               ON mv.graph_id = sl."agentGraphId"
        JOIN {schema_prefix}"StoreAgent" sa ON sa.listing_id = sl.id
        WHERE sl."agentGraphId" = ANY($1::text[])
        """,
        graph_ids,
    )
    return {r["graph_id"]: (r["mv_runs"], r["store_agent_runs"]) for r in rows}


@pytest.mark.asyncio(loop_scope="session")
async def test_runs_are_not_multiplied_by_saved_graph_versions(server):
    tag = uuid4().hex[:12]
    owner, runner = f"owner-{tag}", f"runner-{tag}"
    for user_id in (owner, runner):
        await prisma.models.User.prisma().create(
            data={"id": user_id, "email": f"{user_id}@example.invalid"}
        )
    await prisma.models.Profile.prisma().create(
        data={
            "userId": owner,
            "name": owner,
            "username": owner,
            "avatarUrl": "",
            "description": "",
            "links": [],
        }
    )
    graph_ids: list[str] = []
    try:
        three = await _seed_listing(tag, owner, graph_versions=3)
        one = await _seed_listing(tag, owner, graph_versions=1)
        graph_ids = [three, one]
        # Three runs by someone other than the owner on the latest version of
        # each graph, plus owner runs, which are excluded from the count.
        for graph_id, version in ((three, 3), (one, 1)):
            for user_id, count in ((runner, 3), (owner, 2)):
                for _ in range(count):
                    await prisma.models.AgentGraphExecution.prisma().create(
                        data={
                            "agentGraphId": graph_id,
                            "agentGraphVersion": version,
                            "userId": user_id,
                        }
                    )
        await _refresh()

        counts = await _run_counts(graph_ids)

        assert counts[three] == (3, 3)
        assert counts[one] == (3, 3)
    finally:
        await prisma.models.AgentGraphExecution.prisma().delete_many(
            where={"agentGraphId": {"in": graph_ids}}
        )
        # Deleting a listing cascades to its versions.
        await prisma.models.StoreListing.prisma().delete_many(
            where={"agentGraphId": {"in": graph_ids}}
        )
        await prisma.models.AgentGraph.prisma().delete_many(
            where={"id": {"in": graph_ids}}
        )
        await prisma.models.User.prisma().delete_many(
            where={"id": {"in": [owner, runner]}}
        )
        await _refresh()
