from contextlib import asynccontextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth import RequestContext

from backend.api.features.chat import routes


def _context(team_id: str | None = "team") -> RequestContext:
    return RequestContext(
        user_id="user",
        org_id="org",
        team_id=team_id,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.mark.asyncio
async def test_list_sessions_holds_live_scope_through_redis_and_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    active = False
    lease_calls = 0
    now = datetime.now(UTC)
    session = SimpleNamespace(
        session_id="session",
        started_at=now,
        updated_at=now,
        title="title",
        chat_status="idle",
        metadata=SimpleNamespace(source_platform="web"),
        is_pinned=False,
        expert_id=None,
        organization_id="org",
        team_id="team-a",
    )

    @asynccontextmanager
    async def lease(_user_id, scopes, _access):
        nonlocal active, lease_calls
        lease_calls += 1
        assert scopes == [("org", None), ("org", "team-a"), ("org", "team-b")]
        active = True
        try:
            yield scopes
        finally:
            active = False

    async def get_sessions(*_args, **_kwargs):
        assert active is True
        return [session], 1

    class Pipeline:
        def hget(self, *_args) -> None:
            assert active is True

        async def execute(self) -> list[str]:
            assert active is True
            return ["running"]

    redis = SimpleNamespace(pipeline=lambda **_kwargs: Pipeline())
    monkeypatch.setattr(routes, "live_resource_scopes_lease", lease)
    monkeypatch.setattr(
        routes, "get_user_team_ids", AsyncMock(return_value=["team-b", "team-a"])
    )
    monkeypatch.setattr(routes, "get_user_sessions", get_sessions)
    monkeypatch.setattr(routes, "get_redis_async", AsyncMock(return_value=redis))

    response = await routes.list_sessions(
        user_id="user",
        ctx=_context(team_id=None),
        limit=50,
        offset=0,
        expert_id=None,
        pinned_first=True,
    )

    assert response.total == 1
    assert response.sessions[0].is_processing is True
    assert lease_calls == 1
    assert active is False
