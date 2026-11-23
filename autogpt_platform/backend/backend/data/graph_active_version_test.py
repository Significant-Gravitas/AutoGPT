from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs.auth.models import RequestContext
from prisma.models import AgentGraph
from pytest_mock import MockerFixture

from backend.api.features.schedules import routes as schedules_routes
from backend.api.features.schedules.model import ScheduleCreationRequest
from backend.api.model import CreateGraph
from backend.data.graph import (
    Graph,
    create_graph,
    get_graph,
    get_graph_as_admin,
    get_graph_metadata,
    list_graphs_paginated,
    set_graph_active_version,
)
from backend.data.user import DEFAULT_USER_ID
from backend.util.test import SpinTestServer

_ACTIVE_ELSE_NEWEST = [{"isActive": "desc"}, {"version": "desc"}]
_UNVERSIONED_LOOKUPS = ["get_graph", "get_graph_metadata", "get_graph_as_admin"]


@pytest.mark.parametrize("lookup", _UNVERSIONED_LOOKUPS)
@pytest.mark.parametrize(
    ("versions", "expected"),
    [
        # Rolled back: v2 is newest but v1 is the live one.
        ([(1, True), (2, False)], 1),
        ([(1, False), (2, True), (3, False)], 2),
        # Normal saves: newest is active.
        ([(1, False), (2, True)], 2),
        # Nothing active (e.g. created with is_active=False): newest, not 404.
        ([(1, False), (2, False)], 2),
    ],
)
@pytest.mark.asyncio
async def test_unversioned_lookup_resolves_active_else_newest(
    lookup: str, versions: list[tuple[int, bool]], expected: int
) -> None:
    prisma_double = _fake_versions_prisma(versions)
    with (
        patch("backend.data.graph.AgentGraph.prisma", prisma_double),
        patch(
            "backend.data.graph.GraphModel.from_db",
            side_effect=lambda graph, *_, **__: graph,
        ),
    ):
        assert await _lookup_version(lookup, None) == expected
        # A pinned version is still exactly that version.
        assert await _lookup_version(lookup, 1) == 1

    orders = [
        call.kwargs["order"]
        for call in prisma_double.return_value.find_first.await_args_list
    ]
    assert orders == [_ACTIVE_ELSE_NEWEST, {"version": "desc"}]


@pytest.mark.asyncio(loop_scope="session")
async def test_unversioned_lookups_follow_a_rollback(
    server: SpinTestServer, mocker: MockerFixture
) -> None:
    """Repro of #15663: save v2, roll back to v1. Every unversioned lookup and
    a schedule created without `graph_version` must land on v1, matching
    what the graph list already shows."""
    created = await server.agent_server.test_create_graph(
        CreateGraph(graph=Graph(name="Rollback", description="")), DEFAULT_USER_ID
    )
    assert (created.version, created.is_active) == (1, True)
    await create_graph(
        Graph(id=created.id, version=2, name="Rollback", description=""),
        DEFAULT_USER_ID,
    )
    await set_graph_active_version(created.id, 2, DEFAULT_USER_ID)
    # Roll back, as the builder chat "Revert" does.
    await set_graph_active_version(created.id, 1, DEFAULT_USER_ID)

    unversioned = await get_graph(created.id, None, user_id=DEFAULT_USER_ID)
    assert unversioned and unversioned.version == 1
    metadata = await get_graph_metadata(created.id)
    assert metadata and metadata.version == 1
    as_admin = await get_graph_as_admin(created.id)
    assert as_admin and as_admin.version == 1
    pinned = await get_graph(created.id, 2, user_id=DEFAULT_USER_ID)
    assert pinned and pinned.version == 2

    listed, _ = await list_graphs_paginated(DEFAULT_USER_ID, page_size=10_000)
    assert [g.version for g in listed if g.id == created.id] == [1]

    add_schedule = AsyncMock(return_value=MagicMock(next_run_time=None))
    mocker.patch.object(
        schedules_routes,
        "get_scheduler_client",
        return_value=MagicMock(add_execution_schedule=add_schedule),
    )
    mocker.patch.object(
        schedules_routes.experts_db,
        "resolve_expert_for_graph",
        AsyncMock(return_value=None),
    )
    mocker.patch.object(schedules_routes, "complete_onboarding_step", AsyncMock())
    await schedules_routes.create_graph_execution_schedule(
        user_id=DEFAULT_USER_ID,
        ctx=_owner_context(DEFAULT_USER_ID),
        graph_id=created.id,
        schedule_params=ScheduleCreationRequest(
            name="nightly", cron="0 0 * * *", inputs={}, timezone="UTC"
        ),
    )
    assert add_schedule.await_args.kwargs["graph_version"] == 1

    # With no version active at all, fall back to the newest instead of 404.
    await AgentGraph.prisma().update_many(
        where={"id": created.id}, data={"isActive": False}
    )
    fallback = await get_graph(created.id, None, user_id=DEFAULT_USER_ID)
    assert fallback and fallback.version == 2


def _fake_versions_prisma(versions: list[tuple[int, bool]]) -> MagicMock:
    """An `AgentGraph.prisma()` double over one graph's `(version, isActive)`
    rows whose `find_first` honours `where.version` and the `order` clause."""
    rows = [_version_row(version, is_active) for version, is_active in versions]

    async def find_first(**kwargs: Any) -> MagicMock | None:
        where = kwargs["where"]
        matches = [
            row
            for row in rows
            if where.get("id") == row.id
            and ("version" not in where or where["version"] == row.version)
        ]
        order = kwargs["order"]
        # Stable sorts applied least-significant key first.
        for key in reversed(order if isinstance(order, list) else [order]):
            ((field, direction),) = key.items()
            assert direction in ("asc", "desc"), f"unsupported order {key!r}"
            matches.sort(key=lambda r: getattr(r, field), reverse=direction == "desc")
        return matches[0] if matches else None

    prisma_double = MagicMock()
    prisma_double.return_value.find_first = AsyncMock(side_effect=find_first)
    return prisma_double


def _version_row(version: int, is_active: bool) -> MagicMock:
    row = MagicMock(
        id="graph-id",
        userId="owner-user-id",
        version=version,
        isActive=is_active,
        description="",
        Nodes=[],
    )
    row.name = f"v{version}"  # `MagicMock(name=...)` would name the mock itself
    return row


async def _lookup_version(lookup: str, version: int | None) -> int | None:
    if lookup == "get_graph":
        graph = await get_graph("graph-id", version, user_id="owner-user-id")
    elif lookup == "get_graph_metadata":
        graph = await get_graph_metadata("graph-id", version)
    else:
        graph = await get_graph_as_admin("graph-id", version, user_id="admin-id")
    return graph.version if graph else None


def _owner_context(user_id: str) -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id=f"test-org-{user_id}",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )
