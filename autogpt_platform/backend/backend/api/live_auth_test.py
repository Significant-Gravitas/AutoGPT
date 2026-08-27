from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated, cast

import pytest
from autogpt_libs.auth import OrgAction, RequestContext, TeamAction
from fastapi import FastAPI, HTTPException
from fastapi.params import Depends as DependsParameter
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient

from backend.api import live_auth


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


def _dependency(marker: DependsParameter):
    assert marker.scope == "function"
    assert marker.dependency is not None
    return marker.dependency


def test_live_dependency_finalizes_before_stream_body_is_consumed() -> None:
    events: list[str] = []

    async def dependency() -> AsyncIterator[None]:
        events.append("enter")
        try:
            yield
        finally:
            events.append("exit")

    app = FastAPI()

    @app.get("/")
    async def stream(
        _guard: Annotated[None, live_auth.live_dependency(dependency)],
    ) -> StreamingResponse:
        events.append("endpoint")

        async def body() -> AsyncIterator[bytes]:
            events.append("body")
            yield b"ok"

        return StreamingResponse(body())

    with TestClient(app) as client:
        assert client.get("/").content == b"ok"

    assert events == ["enter", "endpoint", "exit", "body"]


@pytest.mark.asyncio
async def test_live_resource_dependency_holds_barrier_through_consumer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False

    @asynccontextmanager
    async def barrier(*_args):
        nonlocal active
        active = True
        try:
            yield True
        finally:
            active = False

    monkeypatch.setattr(live_auth, "live_resource_permission_barrier", barrier)
    dependency = _dependency(
        cast(
            DependsParameter,
            live_auth.requires_live_resource_permission(
                OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS
            ),
        )
    )
    iterator = dependency(user_id="user", ctx=_context())

    assert await anext(iterator) == _context()
    assert active is True
    await iterator.aclose()
    assert active is False


@pytest.mark.asyncio
async def test_live_resource_dependency_rejects_revoked_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @asynccontextmanager
    async def barrier(*_args):
        yield False

    monkeypatch.setattr(live_auth, "live_resource_permission_barrier", barrier)
    dependency = _dependency(
        cast(
            DependsParameter,
            live_auth.requires_live_resource_permission(
                OrgAction.VIEW_RESOURCES, TeamAction.VIEW_AGENTS
            ),
        )
    )

    with pytest.raises(HTTPException, match="Resource scope is inactive"):
        await anext(dependency(user_id="user", ctx=_context()))


@pytest.mark.asyncio
async def test_live_org_dependency_holds_requested_permission(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = None

    @asynccontextmanager
    async def barrier(user_id: str, org_id: str, action: OrgAction):
        nonlocal seen
        seen = (user_id, org_id, action)
        yield True

    monkeypatch.setattr(live_auth, "live_org_permission_barrier", barrier)
    dependency = _dependency(
        cast(
            DependsParameter,
            live_auth.requires_live_org_permission(OrgAction.MANAGE_BILLING),
        )
    )
    iterator = dependency(ctx=_context())

    assert await anext(iterator) == _context()
    assert seen == ("user", "org", OrgAction.MANAGE_BILLING)
    await iterator.aclose()


@pytest.mark.asyncio
async def test_live_actor_org_dependency_uses_membership_only_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen = None

    @asynccontextmanager
    async def barrier(user_id: str, org_id: str, action: OrgAction):
        nonlocal seen
        seen = (user_id, org_id, action)
        yield True

    monkeypatch.setattr(live_auth, "live_actor_org_permission_barrier", barrier)
    dependency = _dependency(
        cast(
            DependsParameter,
            live_auth.requires_live_actor_org_permission(OrgAction.MANAGE_MEMBERS),
        )
    )
    iterator = dependency(ctx=_context())

    assert await anext(iterator) == _context()
    assert seen == ("user", "org", OrgAction.MANAGE_MEMBERS)
    await iterator.aclose()
