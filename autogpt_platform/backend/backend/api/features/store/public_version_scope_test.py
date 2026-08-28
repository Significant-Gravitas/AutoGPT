from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from prisma.enums import SubmissionStatus

from backend.api.features.store import db, routes
from backend.util.exceptions import NotFoundError


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    yield


def _graph_with_refs(
    graph_id: str,
    version: int,
    refs: list[tuple[str, int]],
) -> MagicMock:
    nodes = [
        MagicMock(
            AgentBlock=MagicMock(id=db._AGENT_EXECUTOR_BLOCK_ID),
            constantInput={"graph_id": ref_id, "graph_version": ref_version},
        )
        for ref_id, ref_version in refs
    ]
    return MagicMock(
        id=graph_id,
        version=version,
        userId="creator-1",
        organizationId="org-1",
        teamId="team-1",
        Nodes=nodes,
    )


@asynccontextmanager
async def _attachment_barriers(_graph_ids):
    yield


@asynccontextmanager
async def _allowed_graph_barrier(*_args):
    yield True


@pytest.mark.asyncio
async def test_store_subgraphs_accept_complete_same_tenant_composition(mocker) -> None:
    parent = _graph_with_refs("parent", 1, [("child", 2)])
    child = _graph_with_refs("child", 2, [])
    mocker.patch.object(db, "get_sub_graphs", AsyncMock(return_value=[child]))

    assert await db._get_exact_store_subgraphs(parent) == [child]


@pytest.mark.asyncio
async def test_store_subgraphs_reject_mixed_or_missing_tenant_reference(mocker) -> None:
    parent = _graph_with_refs("parent", 1, [("child", 2)])
    child = _graph_with_refs("child", 2, [])
    child.teamId = "team-2"
    mocker.patch.object(db, "get_sub_graphs", AsyncMock(return_value=[child]))

    with pytest.raises(NotFoundError, match="inaccessible subgraph"):
        await db._get_exact_store_subgraphs(parent)


@pytest.mark.asyncio
async def test_store_subgraphs_reject_nested_mixed_tenant_reference(mocker) -> None:
    parent = _graph_with_refs("parent", 1, [("child", 2)])
    child = _graph_with_refs("child", 2, [("foreign", 3)])
    mocker.patch.object(db, "get_sub_graphs", AsyncMock(return_value=[child]))

    with pytest.raises(NotFoundError, match="inaccessible subgraph"):
        await db._get_exact_store_subgraphs(parent)


@pytest.mark.asyncio
async def test_submission_rejects_mixed_tenant_subgraph_before_write(mocker) -> None:
    graph = _graph_with_refs("parent", 1, [("child", 2)])
    graph.User = MagicMock(Profile=MagicMock())
    graph_client = MagicMock(find_first=AsyncMock(return_value=graph))
    version_client = MagicMock(create=AsyncMock())
    member_client = MagicMock(find_first=AsyncMock(return_value=MagicMock()))
    mocker.patch("prisma.models.AgentGraph.prisma", return_value=graph_client)
    mocker.patch(
        "prisma.models.StoreListingVersion.prisma", return_value=version_client
    )
    mocker.patch("prisma.models.OrgMember.prisma", return_value=member_client)
    mocker.patch.object(
        db,
        "_get_exact_store_subgraphs",
        AsyncMock(side_effect=NotFoundError("inaccessible subgraph")),
    )

    with pytest.raises(NotFoundError, match="inaccessible subgraph"):
        await db.create_store_submission(
            user_id="creator-1",
            graph_id="parent",
            graph_version=1,
            slug="agent",
            name="Agent",
            organization_id="org-1",
            team_id_restriction="team-1",
        )

    graph_client.find_first.assert_awaited_once()
    graph_where = graph_client.find_first.await_args.kwargs["where"]
    assert graph_where["organizationId"] == "org-1"
    assert graph_where["teamId"] == "team-1"
    version_client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_submission_rechecks_composition_under_graph_barriers(mocker) -> None:
    initial = _graph_with_refs("parent", 1, [])
    changed = _graph_with_refs("parent", 1, [("foreign", 2)])
    profile = MagicMock()
    initial.User = MagicMock(Profile=profile)
    changed.User = MagicMock(Profile=profile)
    graph_client = MagicMock(find_first=AsyncMock(side_effect=[initial, changed]))
    version_client = MagicMock(create=AsyncMock())
    member_client = MagicMock(find_first=AsyncMock(return_value=MagicMock()))
    mocker.patch("prisma.models.AgentGraph.prisma", return_value=graph_client)
    mocker.patch(
        "prisma.models.StoreListingVersion.prisma", return_value=version_client
    )
    mocker.patch("prisma.models.OrgMember.prisma", return_value=member_client)
    exact_subgraphs = mocker.patch.object(
        db,
        "_get_exact_store_subgraphs",
        AsyncMock(
            side_effect=[
                [],
                NotFoundError("inaccessible subgraph after barrier acquisition"),
            ]
        ),
    )
    mocker.patch.object(db, "agent_graph_attachment_barriers", _attachment_barriers)
    mocker.patch.object(db, "live_agent_graph_access_barrier", _allowed_graph_barrier)

    with pytest.raises(NotFoundError, match="after barrier acquisition"):
        await db.create_store_submission(
            user_id="creator-1",
            graph_id="parent",
            graph_version=1,
            slug="agent",
            name="Agent",
            organization_id="org-1",
            team_id_restriction="team-1",
        )

    assert exact_subgraphs.await_count == 2
    version_client.create.assert_not_awaited()


@pytest.mark.asyncio
async def test_graph_metadata_requires_approved_live_listing(mocker) -> None:
    client = MagicMock(find_first=AsyncMock(return_value=None))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=client)

    with pytest.raises(NotFoundError):
        await db.get_available_graph("pending-version")

    client.find_first.assert_awaited_once_with(
        where={
            "id": "pending-version",
            "isDeleted": False,
            "StoreListing": {"is": {"isDeleted": False}},
            "OR": [
                {
                    "isAvailable": True,
                    "submissionStatus": SubmissionStatus.APPROVED,
                }
            ],
        },
        include={
            "AgentGraph": {"include": db.AGENT_GRAPH_INCLUDE},
            "StoreListing": True,
        },
    )


@pytest.mark.asyncio
async def test_graph_metadata_allows_exact_creator_with_live_scope(mocker) -> None:
    graph = _graph_with_refs("graph-1", 4, [])
    listing = MagicMock(owningUserId="creator-1")
    version = MagicMock(
        AgentGraph=graph,
        StoreListing=listing,
        isAvailable=True,
        submissionStatus=SubmissionStatus.PENDING,
    )
    client = MagicMock(find_first=AsyncMock(side_effect=[version, version]))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=client)

    barrier_calls: list[tuple] = []

    @asynccontextmanager
    async def allowed(*args):
        barrier_calls.append(args)
        yield True

    mocker.patch.object(db, "live_agent_graph_access_barrier", allowed)
    graph_model = MagicMock()
    from_db = mocker.patch.object(db.GraphModelWithoutNodes, "from_db")
    from_db.return_value = graph_model

    result = await db.get_available_graph("pending-version", user_id="creator-1")

    assert result is graph_model
    assert barrier_calls == [("creator-1", "org-1", "team-1", "view", "graph-1", 4)]
    assert client.find_first.await_args_list[1].kwargs["where"] == {
        "id": "pending-version",
        "isDeleted": False,
        "AgentGraph": {
            "is": {
                "id": "graph-1",
                "version": 4,
                "userId": "creator-1",
                "organizationId": "org-1",
                "teamId": "team-1",
            }
        },
        "StoreListing": {"is": {"isDeleted": False, "owningUserId": "creator-1"}},
    }


@pytest.mark.asyncio
async def test_graph_metadata_rejects_revoked_creator_scope(mocker) -> None:
    graph = _graph_with_refs("graph-1", 4, [])
    version = MagicMock(
        AgentGraph=graph,
        StoreListing=MagicMock(owningUserId="creator-1"),
        isAvailable=True,
        submissionStatus=SubmissionStatus.PENDING,
    )
    client = MagicMock(find_first=AsyncMock(return_value=version))
    mocker.patch("prisma.models.StoreListingVersion.prisma", return_value=client)

    @asynccontextmanager
    async def denied(*_args):
        yield False

    mocker.patch.object(db, "live_agent_graph_access_barrier", denied)

    with pytest.raises(NotFoundError):
        await db.get_available_graph("pending-version", user_id="creator-1")

    client.find_first.assert_awaited_once()


@pytest.mark.asyncio
async def test_graph_metadata_route_passes_authenticated_creator(mocker) -> None:
    graph = MagicMock()
    get_available_graph = mocker.patch.object(
        routes.store_db,
        "get_available_graph",
        AsyncMock(return_value=graph),
    )

    result = await routes.get_graph_meta_by_store_listing_version_id(
        "version-1", user_id="creator-1"
    )

    assert result is graph
    get_available_graph.assert_awaited_once_with("version-1", user_id="creator-1")


@pytest.mark.asyncio
async def test_review_requires_path_to_match_active_public_version(mocker) -> None:
    candidate = MagicMock(storeListingId="listing-1")
    listing = MagicMock(owningUserId="creator-1")
    profile = MagicMock(id="profile-1")
    version_client = MagicMock(
        find_unique=AsyncMock(return_value=candidate),
        find_first=AsyncMock(return_value=candidate),
    )
    review_client = MagicMock(
        upsert=AsyncMock(return_value=MagicMock(score=5, comments="Great"))
    )
    tx = MagicMock()

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    mocker.patch.object(db, "transaction", fake_transaction)
    mocker.patch(
        "prisma.models.StoreListingVersion.prisma", return_value=version_client
    )
    listing_client = MagicMock(find_unique=AsyncMock(return_value=listing))
    profile_client = MagicMock(find_unique=AsyncMock(return_value=profile))
    mocker.patch("prisma.models.StoreListing.prisma", return_value=listing_client)
    mocker.patch("prisma.models.Profile.prisma", return_value=profile_client)
    mocker.patch("prisma.models.StoreListingReview.prisma", return_value=review_client)
    locks = mocker.patch.object(db, "_lock_store_row", AsyncMock(return_value=True))

    result = await db.create_store_review(
        user_id="reviewer-1",
        username="creator",
        agent_name="agent-slug",
        store_listing_version_id="version-1",
        score=5,
        comments="Great",
    )

    public_where = {
        "id": "version-1",
        "submissionStatus": SubmissionStatus.APPROVED,
        "isAvailable": True,
        "isDeleted": False,
        "StoreListing": {
            "is": {
                "slug": "agent-slug",
                "isDeleted": False,
                "hasApprovedVersion": True,
                "activeVersionId": "version-1",
                "CreatorProfile": {"is": {"username": "creator"}},
            }
        },
    }
    version_client.find_unique.assert_awaited_once_with(where={"id": "version-1"})
    version_client.find_first.assert_awaited_once_with(where=public_where)
    assert locks.await_args_list[0].args == (tx, "StoreListingVersion", "version-1")
    assert locks.await_args_list[1].args == (tx, "StoreListing", "listing-1")
    assert locks.await_args_list[2].args == (tx, "Profile", "profile-1")
    review_client.upsert.assert_awaited_once()
    assert result.score == 5


@pytest.mark.asyncio
async def test_review_rejects_path_version_mismatch_before_write(mocker) -> None:
    candidate = MagicMock(storeListingId="listing-1")
    listing = MagicMock(owningUserId="creator-1")
    profile = MagicMock(id="profile-1")
    version_client = MagicMock(
        find_unique=AsyncMock(return_value=candidate),
        find_first=AsyncMock(return_value=None),
    )
    review_client = MagicMock(upsert=AsyncMock())
    tx = MagicMock()

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    mocker.patch.object(db, "transaction", fake_transaction)
    mocker.patch(
        "prisma.models.StoreListingVersion.prisma", return_value=version_client
    )
    mocker.patch(
        "prisma.models.StoreListing.prisma",
        return_value=MagicMock(find_unique=AsyncMock(return_value=listing)),
    )
    mocker.patch(
        "prisma.models.Profile.prisma",
        return_value=MagicMock(find_unique=AsyncMock(return_value=profile)),
    )
    mocker.patch("prisma.models.StoreListingReview.prisma", return_value=review_client)
    mocker.patch.object(db, "_lock_store_row", AsyncMock(return_value=True))

    with pytest.raises(NotFoundError):
        await db.create_store_review(
            user_id="reviewer-1",
            username="creator",
            agent_name="different-agent",
            store_listing_version_id="pending-version",
            score=1,
        )

    review_client.upsert.assert_not_awaited()


@pytest.mark.asyncio
async def test_review_rechecks_public_version_after_lock_before_write(mocker) -> None:
    candidate = MagicMock(storeListingId="listing-1")
    listing = MagicMock(owningUserId="creator-1")
    profile = MagicMock(id="profile-1")
    version_client = MagicMock(
        find_unique=AsyncMock(return_value=candidate),
        find_first=AsyncMock(return_value=None),
    )
    review_client = MagicMock(upsert=AsyncMock())
    tx = MagicMock()

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    mocker.patch.object(db, "transaction", fake_transaction)
    mocker.patch(
        "prisma.models.StoreListingVersion.prisma", return_value=version_client
    )
    mocker.patch(
        "prisma.models.StoreListing.prisma",
        return_value=MagicMock(find_unique=AsyncMock(return_value=listing)),
    )
    mocker.patch(
        "prisma.models.Profile.prisma",
        return_value=MagicMock(find_unique=AsyncMock(return_value=profile)),
    )
    mocker.patch("prisma.models.StoreListingReview.prisma", return_value=review_client)
    mocker.patch.object(db, "_lock_store_row", AsyncMock(return_value=True))

    with pytest.raises(NotFoundError):
        await db.create_store_review(
            user_id="reviewer-1",
            username="creator",
            agent_name="agent-slug",
            store_listing_version_id="version-1",
            score=5,
        )

    review_client.upsert.assert_not_awaited()
