"""Creator stats from the StoreCreator view against a real Postgres (#15760).

The view used to fan out to one row per listing x approved version x review x
category, which multiplied agent_runs and averaged a 0 into agent_rating for
every unreviewed approved version. ``db_test.py`` mocks Prisma, so this needs
the database (``server`` starts the test stack).
"""

from uuid import uuid4

import prisma.enums
import prisma.models
import pytest

from backend.api.features.store import db as store_db
from backend.api.features.store.db import StoreCreatorsSortOptions
from backend.data.db import execute_raw_with_schema


async def _seed(tag: str) -> dict[str, str]:
    ids = {
        "a": f"creator-a-{tag}",
        "b": f"creator-b-{tag}",
        "reviewer": f"reviewer-{tag}",
        "graph_a": f"graph-a-{tag}",
        "graph_b": f"graph-b-{tag}",
        "listing_a": f"listing-a-{tag}",
        "listing_b": f"listing-b-{tag}",
    }
    for key in ("a", "b", "reviewer"):
        await prisma.models.User.prisma().create(
            data={"id": ids[key], "email": f"{ids[key]}@example.invalid"}
        )
    for key in ("a", "b"):
        await prisma.models.Profile.prisma().create(
            data={
                "userId": ids[key],
                "name": ids[key],
                "username": ids[key],
                "avatarUrl": "",
                "description": "",
                "links": [],
            }
        )
    for owner, graph in (("a", "graph_a"), ("b", "graph_b")):
        await prisma.models.AgentGraph.prisma().create(
            data={
                "id": ids[graph],
                "version": 1,
                "name": ids[graph],
                "description": "test",
                "userId": ids[owner],
                "isActive": True,
            }
        )

    # Creator A: one listing, three approved versions, two categories.
    await prisma.models.StoreListing.prisma().create(
        data={
            "id": ids["listing_a"],
            "slug": ids["listing_a"],
            "agentGraphId": ids["graph_a"],
            "owningUserId": ids["a"],
            "hasApprovedVersion": True,
        }
    )
    version_ids = []
    for n in (1, 2, 3):
        version = await prisma.models.StoreListingVersion.prisma().create(
            data={
                "storeListingId": ids["listing_a"],
                "version": n,
                "agentGraphId": ids["graph_a"],
                "agentGraphVersion": 1,
                "name": "A",
                "subHeading": "",
                "description": "",
                "imageUrls": [],
                "categories": ["research", "writing"],
                "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
            }
        )
        version_ids.append(version.id)
    # Two reviews, both on the last version only.
    for reviewer, score in (("b", 5), ("reviewer", 3)):
        await prisma.models.StoreListingReview.prisma().create(
            data={
                "storeListingVersionId": version_ids[-1],
                "reviewByUserId": ids[reviewer],
                "score": score,
            }
        )

    # Creator B: one listing, one approved version, no reviews.
    await prisma.models.StoreListing.prisma().create(
        data={
            "id": ids["listing_b"],
            "slug": ids["listing_b"],
            "agentGraphId": ids["graph_b"],
            "owningUserId": ids["b"],
            "hasApprovedVersion": True,
        }
    )
    await prisma.models.StoreListingVersion.prisma().create(
        data={
            "storeListingId": ids["listing_b"],
            "version": 1,
            "agentGraphId": ids["graph_b"],
            "agentGraphVersion": 1,
            "name": "B",
            "subHeading": "",
            "description": "",
            "imageUrls": [],
            "categories": ["data"],
            "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
        }
    )

    # Runs by someone other than the graph's owner: 3 on A's agent, 5 on B's.
    for graph, runner, count in (("graph_a", "b", 3), ("graph_b", "a", 5)):
        for _ in range(count):
            await prisma.models.AgentGraphExecution.prisma().create(
                data={
                    "agentGraphId": ids[graph],
                    "agentGraphVersion": 1,
                    "userId": ids[runner],
                }
            )

    await execute_raw_with_schema(
        "SELECT {schema_prefix}refresh_store_materialized_views()"
    )
    return ids


async def _cleanup(ids: dict[str, str]) -> None:
    graph_ids = [ids["graph_a"], ids["graph_b"]]
    await prisma.models.AgentGraphExecution.prisma().delete_many(
        where={"agentGraphId": {"in": graph_ids}}
    )
    # Deleting a listing cascades to its versions and their reviews.
    await prisma.models.StoreListing.prisma().delete_many(
        where={"id": {"in": [ids["listing_a"], ids["listing_b"]]}}
    )
    await prisma.models.AgentGraph.prisma().delete_many(where={"id": {"in": graph_ids}})
    await prisma.models.User.prisma().delete_many(
        where={"id": {"in": [ids["a"], ids["b"], ids["reviewer"]]}}
    )
    await execute_raw_with_schema(
        "SELECT {schema_prefix}refresh_store_materialized_views()"
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_creator_stats_count_each_listing_and_review_once(server):
    tag = uuid4().hex[:12]
    ids = await _seed(tag)
    try:
        a = await store_db.get_store_creator(ids["a"])
        b = await store_db.get_store_creator(ids["b"])

        assert a.num_agents == 1
        assert a.agent_runs == 3
        assert a.agent_rating == pytest.approx(4.0)
        assert a.top_categories == ["research", "writing"]

        assert b.num_agents == 1
        assert b.agent_runs == 5
        assert b.agent_rating == 0.0
        assert b.top_categories == ["data"]

        by_runs = await store_db.get_store_creators(
            search_query=tag, sorted_by=StoreCreatorsSortOptions.AGENT_RUNS
        )
        assert [c.username for c in by_runs.creators] == [ids["b"], ids["a"]]
    finally:
        await _cleanup(ids)
