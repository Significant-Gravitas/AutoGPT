"""
Tests for v2 tenancy: which organization and team a request acts in.

Two things must hold. Structurally, every v2 route resolves its tenant through
`tenancy.require_auth` — a handler that skipped it would silently fall back to
"the user, across every org they belong to". Behaviourally, the tenant reaching
the database is the one bound to the credential: a key minted in org A never
passes org B, and a key with no org passes the caller's personal org.
"""

from datetime import datetime, timezone
from unittest.mock import AsyncMock, Mock

import pytest
import pytest_mock
from fastapi.routing import APIRoute
from prisma.enums import APIKeyPermission
from starlette.requests import Request

from backend.data.auth.base import APIAuthorizationInfo
from backend.data.execution import ExecutionStatus, GraphExecutionMeta
from backend.util.exceptions import NotAuthorizedError

from . import tenancy
from .pagination import PageRequest
from .routes import v2_router
from .tenancy import TenantContext, resolve_tenant

USER_ID = "user-1"
ORG_A = "org-a"
ORG_B = "org-b"
TEAM_A = "team-a"
PERSONAL_ORG = "personal-org"


# ============================================================================
# Structural: no handler can bypass tenant resolution
# ============================================================================


def test_every_v2_route_resolves_the_tenant() -> None:
    """Every route's dependency tree contains the one tenancy dependency."""
    bypassing = [
        f"{sorted(route.methods)[0]} {route.path}"
        for route in v2_router.routes
        if isinstance(route, APIRoute)
        and not _depends_on_tenancy(route.dependant.dependencies)
    ]
    assert not bypassing, f"routes that never resolve a tenant: {bypassing}"


def test_v2_router_has_routes_to_check() -> None:
    """Guards the test above against passing on an empty router."""
    assert len([r for r in v2_router.routes if isinstance(r, APIRoute)]) > 50


# ============================================================================
# Behavioural: the tenant reaching the database is the credential's
# ============================================================================


@pytest.mark.asyncio
async def test_library_listing_is_scoped_to_the_key_org(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A key minted in org A lists org A's library, never org B's."""
    from .library.agents import list_library_agents

    list_db = mocker.patch(
        "backend.api.features.library.db.list_library_agents",
        new_callable=AsyncMock,
        return_value=Mock(agents=[], pagination=Mock(total_items=0)),
    )

    await list_library_agents(
        published=None, favorite=None, page=_page(), auth=_key_for(ORG_A)
    )

    assert list_db.await_args.kwargs["organization_id"] == ORG_A
    assert list_db.await_args.kwargs["organization_id"] != ORG_B


@pytest.mark.asyncio
async def test_run_listing_is_scoped_to_the_key_org(
    mocker: pytest_mock.MockFixture,
) -> None:
    from .runs import list_runs

    list_db = mocker.patch(
        "backend.data.execution.get_graph_executions_paginated",
        new_callable=AsyncMock,
        return_value=Mock(executions=[], pagination=Mock(total_items=0)),
    )

    await list_runs(graph_id=None, page=_page(), auth=_key_for(ORG_B))

    assert list_db.await_args.kwargs["organization_id"] == ORG_B


@pytest.mark.asyncio
async def test_personal_key_passes_the_personal_org_not_none(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A key minted before orgs existed acts in the caller's personal org.

    Passing None instead would list the user's rows across every org they
    belong to — the pre-tenancy behaviour this replaces.
    """
    from .library.agents import list_library_agents

    list_db = mocker.patch(
        "backend.api.features.library.db.list_library_agents",
        new_callable=AsyncMock,
        return_value=Mock(agents=[], pagination=Mock(total_items=0)),
    )

    await list_library_agents(
        published=None, favorite=None, page=_page(), auth=_key_for(PERSONAL_ORG)
    )

    assert list_db.await_args.kwargs["organization_id"] == PERSONAL_ORG


@pytest.mark.asyncio
async def test_run_creation_carries_org_and_team(
    mocker: pytest_mock.MockFixture,
) -> None:
    """A team-pinned key tenants the runs it starts to that team."""
    from .library.agents import execute_agent
    from .models import AgentRunRequest

    mocker.patch(
        "backend.api.external.v2.library.agents.get_credit_model",
        new_callable=AsyncMock,
        return_value=Mock(get_credits=AsyncMock(return_value=100)),
    )
    mocker.patch(
        "backend.api.features.library.db.get_library_agent",
        new_callable=AsyncMock,
        return_value=Mock(graph_id="graph-1", graph_version=1),
    )
    add_execution = mocker.patch(
        "backend.executor.utils.add_graph_execution",
        new_callable=AsyncMock,
        return_value=_run(),
    )

    await execute_agent(
        request=AgentRunRequest(inputs={}),
        agent_id="agent-1",
        auth=_key_for(ORG_A, team_id=TEAM_A),
    )

    assert add_execution.await_args.kwargs["organization_id"] == ORG_A
    assert add_execution.await_args.kwargs["team_id"] == TEAM_A


@pytest.mark.asyncio
async def test_billing_uses_the_org_credit_model(
    mocker: pytest_mock.MockFixture,
) -> None:
    """An org key bills the org ledger, not the owner's personal wallet."""
    from .credits import get_balance

    credit_model = mocker.patch(
        "backend.api.external.v2.credits.get_credit_model",
        new_callable=AsyncMock,
        return_value=Mock(get_credits=AsyncMock(return_value=42)),
    )

    balance = await get_balance(auth=_key_for(ORG_A))

    assert credit_model.await_args.args == (USER_ID, ORG_A)
    assert balance.balance == 42


@pytest.mark.asyncio
async def test_marketplace_submissions_are_scoped_to_the_key_org(
    mocker: pytest_mock.MockFixture,
) -> None:
    from .marketplace import list_submissions

    list_db = mocker.patch(
        "backend.api.features.store.db.get_store_submissions",
        new_callable=AsyncMock,
        return_value=Mock(submissions=[], pagination=Mock(total_items=0)),
    )

    await list_submissions(page=_page(), auth=_key_for(ORG_A))

    assert list_db.await_args.kwargs["organization_id"] == ORG_A


# ============================================================================
# Resolution: what the credential and the X-Team-Id header are allowed to say
# ============================================================================


@pytest.mark.asyncio
async def test_org_key_resolves_to_its_own_org(
    mocker: pytest_mock.MockFixture,
) -> None:
    _mock_membership(mocker, org_id=ORG_A)

    tenant = await resolve_tenant(_request(), _credential(organization_id=ORG_A))

    assert (tenant.organization_id, tenant.team_id) == (ORG_A, None)


@pytest.mark.asyncio
async def test_team_pinned_key_resolves_to_its_team(
    mocker: pytest_mock.MockFixture,
) -> None:
    _mock_membership(mocker, org_id=ORG_A)

    tenant = await resolve_tenant(
        _request(), _credential(organization_id=ORG_A, team_id_restriction=TEAM_A)
    )

    assert (tenant.organization_id, tenant.team_id) == (ORG_A, TEAM_A)


@pytest.mark.asyncio
async def test_key_without_org_falls_back_to_the_personal_org(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        new_callable=AsyncMock,
        return_value=(PERSONAL_ORG, "personal-team"),
    )

    tenant = await resolve_tenant(_request(), _credential())

    assert (tenant.organization_id, tenant.team_id) == (PERSONAL_ORG, "personal-team")


@pytest.mark.asyncio
async def test_key_is_rejected_once_membership_ends(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Leaving an org must disable the keys minted in it."""
    _mock_membership(mocker, org_id=ORG_A, status="INACTIVE")

    with pytest.raises(NotAuthorizedError):
        await resolve_tenant(_request(), _credential(organization_id=ORG_A))


@pytest.mark.asyncio
async def test_key_is_rejected_when_the_org_is_deleted(
    mocker: pytest_mock.MockFixture,
) -> None:
    _mock_membership(mocker, org_id=ORG_A, deleted_at=datetime.now(tz=timezone.utc))

    with pytest.raises(NotAuthorizedError):
        await resolve_tenant(_request(), _credential(organization_id=ORG_A))


@pytest.mark.asyncio
async def test_team_header_selects_a_team_the_caller_belongs_to(
    mocker: pytest_mock.MockFixture,
) -> None:
    prisma = _mock_membership(mocker, org_id=ORG_A)
    prisma.teammember.find_unique = AsyncMock(
        return_value=Mock(status="ACTIVE", Team=Mock(orgId=ORG_A))
    )

    tenant = await resolve_tenant(
        _request(team_header=TEAM_A), _credential(organization_id=ORG_A)
    )

    assert tenant.team_id == TEAM_A


@pytest.mark.asyncio
async def test_team_header_cannot_override_a_pinned_key(
    mocker: pytest_mock.MockFixture,
) -> None:
    _mock_membership(mocker, org_id=ORG_A)

    with pytest.raises(NotAuthorizedError):
        await resolve_tenant(
            _request(team_header="other-team"),
            _credential(organization_id=ORG_A, team_id_restriction=TEAM_A),
        )


@pytest.mark.asyncio
async def test_team_header_cannot_reach_another_orgs_team(
    mocker: pytest_mock.MockFixture,
) -> None:
    prisma = _mock_membership(mocker, org_id=ORG_A)
    prisma.teammember.find_unique = AsyncMock(
        return_value=Mock(status="ACTIVE", Team=Mock(orgId=ORG_B))
    )

    with pytest.raises(NotAuthorizedError):
        await resolve_tenant(
            _request(team_header="team-b"), _credential(organization_id=ORG_A)
        )


@pytest.mark.asyncio
async def test_account_without_a_personal_org_is_rejected(
    mocker: pytest_mock.MockFixture,
) -> None:
    mocker.patch(
        "backend.api.features.orgs.db.get_user_default_team",
        new_callable=AsyncMock,
        return_value=(None, None),
    )

    with pytest.raises(NotAuthorizedError):
        await resolve_tenant(_request(), _credential())


# ============================================================================
# Helpers
# ============================================================================


def _depends_on_tenancy(dependencies) -> bool:
    return any(
        dep.call is tenancy.require_auth or _depends_on_tenancy(dep.dependencies)
        for dep in dependencies
    )


def _key_for(organization_id: str, team_id: str | None = None) -> TenantContext:
    return TenantContext(
        user_id=USER_ID,
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id=organization_id,
        team_id=team_id,
    )


def _credential(
    organization_id: str | None = None, team_id_restriction: str | None = None
) -> APIAuthorizationInfo:
    return APIAuthorizationInfo(
        user_id=USER_ID,
        scopes=list(APIKeyPermission),
        type="api_key",
        created_at=datetime.now(tz=timezone.utc),
        organization_id=organization_id,
        team_id_restriction=team_id_restriction,
    )


def _request(team_header: str | None = None) -> Request:
    headers = [(b"x-team-id", team_header.encode())] if team_header else []
    return Request({"type": "http", "headers": headers})


def _mock_membership(
    mocker: pytest_mock.MockFixture,
    org_id: str,
    status: str = "ACTIVE",
    deleted_at: datetime | None = None,
) -> Mock:
    prisma = mocker.patch("backend.api.external.v2.tenancy.prisma")
    prisma.orgmember.find_unique = AsyncMock(
        return_value=Mock(status=status, Org=Mock(id=org_id, deletedAt=deleted_at))
    )
    return prisma


def _page() -> PageRequest:
    return PageRequest(limit=20)


def _run() -> GraphExecutionMeta:
    return GraphExecutionMeta.model_construct(
        id="run-1",
        user_id=USER_ID,
        graph_id="graph-1",
        graph_version=1,
        preset_id=None,
        status=ExecutionStatus.QUEUED,
        started_at=None,
        ended_at=None,
        inputs={},
        credential_inputs=None,
        is_shared=False,
        share_token=None,
        stats=None,
        organization_id=ORG_A,
        team_id=TEAM_A,
    )
