from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth import RequestContext

from backend.api.features import v1


def _context() -> RequestContext:
    return RequestContext(
        user_id="user",
        org_id="org",
        team_id="team",
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=True,
        is_team_admin=True,
        is_team_billing_manager=True,
        seat_status="ACTIVE",
    )


def _graph(graph_id: str, version: int, *children: object) -> object:
    return SimpleNamespace(id=graph_id, version=version, sub_graphs=list(children))


@pytest.mark.asyncio
async def test_stable_graph_view_reloads_under_all_attachment_locks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    child = _graph("child", 1)
    candidate = _graph("root", 2, child)

    @asynccontextmanager
    async def barriers(graph_ids):
        nonlocal active
        assert set(graph_ids) == {"root", "child"}
        active = True
        try:
            yield
        finally:
            active = False

    async def get_graph(*_args, **_kwargs):
        if get_graph.calls:
            assert active is True
        get_graph.calls += 1
        return candidate

    get_graph.calls = 0
    monkeypatch.setattr(v1, "agent_graph_attachment_barriers", barriers)
    monkeypatch.setattr(v1.graph_db, "get_graph", AsyncMock(side_effect=get_graph))

    async with v1._stable_graph_view("root", 2, "user", False, _context()) as graph:
        assert graph is candidate
        assert active is True

    assert active is False


@pytest.mark.asyncio
async def test_stable_graph_view_drops_graph_moved_before_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = _graph("root", 2)

    @asynccontextmanager
    async def barriers(_graph_ids):
        yield

    monkeypatch.setattr(v1, "agent_graph_attachment_barriers", barriers)
    monkeypatch.setattr(
        v1.graph_db,
        "get_graph",
        AsyncMock(side_effect=[candidate, None]),
    )

    async with v1._stable_graph_view("root", 2, "user", False, _context()) as graph:
        assert graph is None
