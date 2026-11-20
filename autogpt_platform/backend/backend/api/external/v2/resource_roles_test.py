"""The caller's role in the organization, and in the team, on v2.

A membership whose only role is billing manager covers spend, not the
organization's workflows, runs, library and schedules. The web app refuses such
a member those; a credential acting in the organization is refused them too,
and keeps reading the organization's credits.
"""

import inspect
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from fastapi import HTTPException, Response
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from prisma.enums import APIKeyPermission

from backend.data.tenancy import BILLING_ONLY_ORG_ROLE, BILLING_ONLY_TEAM_ROLE

from .app import v2_app
from .models import SearchContentType
from .pagination import PageRequest
from .search import search
from .tenancy import (
    TenantContext,
    require_paid_resource_permission,
    require_resource_permission,
)

_TENANT = TenantContext(
    user_id="user-1",
    scopes=list(APIKeyPermission),
    type="api_key",
    organization_id="org-1",
    team_id="team-1",
)

# Every variant's check is a closure over this one code object.
ROLE_CHECK = require_resource_permission().__code__

# Every route under these acts on the organization's workflows, runs, library
# or schedules.
RESOURCE_PATHS = (
    "/graphs",
    "/runs",
    "/library",
    "/schedules",
    "/marketplace/submissions",
)

# ...and so does this one, elsewhere.
ALSO_ON_RESOURCES = {"POST /marketplace/agents/{username}/{agent_name}/add-to-library"}

# Routes under those paths that any role may call, with the reason. Like the
# web app's own routes for them.
OPEN_TO_ANY_ROLE = {
    "GET /graphs/{graph_id}/marketplace-listing": "a public marketplace listing",
    "DELETE /runs/{run_id}/share": "it only withdraws a share",
    "POST /marketplace/submissions/media": "uploads the creator's own media",
}


@pytest.mark.parametrize(
    "check", [require_resource_permission, require_paid_resource_permission]
)
@pytest.mark.parametrize(
    "refusal",
    [BILLING_ONLY_ORG_ROLE, BILLING_ONLY_TEAM_ROLE.format(team_id="team-1")],
    ids=["organization", "team"],
)
async def test_a_role_that_covers_only_billing_gets_a_403_that_says_so(
    mocker: pytest_mock.MockFixture, check, refusal: str
):
    lookup = mocker.patch(
        "backend.api.external.v2.tenancy.resource_role_refusal",
        new_callable=AsyncMock,
        return_value=refusal,
    )

    with pytest.raises(HTTPException) as raised:
        await check(APIKeyPermission.READ_RUN)(tenant=_TENANT)

    assert raised.value.status_code == 403
    assert raised.value.detail == refusal
    lookup.assert_awaited_once_with("user-1", "org-1", "team-1")


async def test_a_role_that_covers_workflows_passes(roles_cover_resources: AsyncMock):
    check = require_resource_permission(APIKeyPermission.READ_RUN)

    assert await check(tenant=_TENANT) is _TENANT
    roles_cover_resources.assert_awaited_once_with("user-1", "org-1", "team-1")


def test_every_route_on_workflows_runs_library_or_schedules_checks_the_role():
    """And no other route does: a billing manager's credential keeps reading
    the organization's credits, its own identity and the public catalog."""
    unchecked, checked_elsewhere = [], []
    for key, route in _routes():
        on_resources = key in ALSO_ON_RESOURCES or (
            route.path.startswith(RESOURCE_PATHS) and key not in OPEN_TO_ANY_ROLE
        )
        checked = _checks_the_role(route.dependant)
        if on_resources and not checked:
            unchecked.append(key)
        elif checked and not on_resources:
            checked_elsewhere.append(key)

    assert not unchecked, (
        "Authenticate these with require_resource_permission, or list them in "
        f"OPEN_TO_ANY_ROLE with the reason any role may call them: {sorted(unchecked)}"
    )
    assert (
        not checked_elsewhere
    ), f"These reach no workflow, run, library or schedule: {sorted(checked_elsewhere)}"


def test_the_lists_name_only_routes_that_exist():
    keys = {key for key, _ in _routes()}

    assert not (set(OPEN_TO_ANY_ROLE) | ALSO_ON_RESOURCES) - keys
    assert {"GET /credits", "GET /credits/transactions", "GET /me"} <= keys


async def test_searching_the_library_is_held_to_the_role(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch("backend.api.external.v2.search.enforce", new_callable=AsyncMock)
    searched = mocker.patch(
        "backend.api.external.v2.search.unified_hybrid_search", new_callable=AsyncMock
    )
    mocker.patch(
        "backend.api.external.v2.tenancy.resource_role_refusal",
        new_callable=AsyncMock,
        return_value=BILLING_ONLY_ORG_ROLE,
    )

    with pytest.raises(HTTPException) as raised:
        await _search([SearchContentType.LIBRARY_AGENT, SearchContentType.BLOCK])

    assert (raised.value.status_code, raised.value.detail) == (
        403,
        BILLING_ONLY_ORG_ROLE,
    )
    searched.assert_not_awaited()


async def test_searching_public_content_reads_no_role(mocker: pytest_mock.MockFixture):
    mocker.patch("backend.api.external.v2.search.enforce", new_callable=AsyncMock)
    mocker.patch(
        "backend.api.external.v2.search.unified_hybrid_search",
        new_callable=AsyncMock,
        return_value=([], 0),
    )
    lookup = mocker.patch(
        "backend.api.external.v2.tenancy.resource_role_refusal",
        new_callable=AsyncMock,
        return_value=BILLING_ONLY_ORG_ROLE,
    )

    await _search(None)

    lookup.assert_not_awaited()


async def _search(content_types: list[SearchContentType] | None):
    return await search(
        response=Response(),
        query="report",
        content_types=content_types,
        category=None,
        page=PageRequest(limit=25),
        auth=_TENANT,
    )


def _routes():
    for route in v2_app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods:
                yield f"{method} {route.path}", route


def _checks_the_role(dependant: Dependant) -> bool:
    return any(
        (inspect.isfunction(dep.call) and dep.call.__code__ is ROLE_CHECK)
        or _checks_the_role(dep)
        for dep in dependant.dependencies
    )
