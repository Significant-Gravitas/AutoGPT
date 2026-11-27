"""The caller's role in the organization, and in the team, on v1.

A membership whose only role is billing manager covers spend, not workflows and
runs. A credential acting in the organization is refused v1's workflow and run
routes, as the web app refuses them, and keeps the rest.
"""

import inspect
from datetime import UTC, datetime
from unittest.mock import AsyncMock

import pytest
import pytest_mock
from fastapi import HTTPException
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_resource_permission
from backend.api.external.v1.app import v1_app
from backend.data.auth.base import APIAuthorizationInfo
from backend.data.tenancy import BILLING_ONLY_ORG_ROLE, BILLING_ONLY_TEAM_ROLE

ROLE_CHECK = require_resource_permission().__code__

ON_WORKFLOWS_AND_RUNS = {
    "POST /graphs",
    "POST /graphs/{graph_id}/execute/{graph_version}",
    "GET /graphs/{graph_id}/executions/{graph_exec_id}/results",
    "POST /tools/run-agent",
}


def _auth(organization_id: str | None, team_id: str | None = None):
    return APIAuthorizationInfo(
        user_id="user-1",
        scopes=list(APIKeyPermission),
        type="api_key",
        created_at=datetime.now(UTC),
        organization_id=organization_id,
        team_id_restriction=team_id,
    )


def test_the_workflow_and_run_routes_check_the_role_and_no_other_does():
    """Identity, blocks, the marketplace and credentials stay open to a billing
    manager's credential."""
    checked = {key for key, route in _routes() if _checks_the_role(route.dependant)}

    assert checked == ON_WORKFLOWS_AND_RUNS


@pytest.mark.parametrize(
    "team_id, refusal",
    [
        (None, BILLING_ONLY_ORG_ROLE),
        ("team-1", BILLING_ONLY_TEAM_ROLE.format(team_id="team-1")),
    ],
    ids=["organization", "team"],
)
async def test_a_role_that_covers_only_billing_gets_a_403_that_says_so(
    mocker: pytest_mock.MockFixture, team_id: str | None, refusal: str
):
    lookup = mocker.patch(
        "backend.api.external.middleware.resource_role_refusal",
        new_callable=AsyncMock,
        return_value=refusal,
    )
    check = require_resource_permission(APIKeyPermission.WRITE_GRAPH)

    with pytest.raises(HTTPException) as raised:
        await check(auth=_auth("org-1", team_id))

    assert (raised.value.status_code, raised.value.detail) == (403, refusal)
    lookup.assert_awaited_once_with("user-1", "org-1", team_id)


async def test_a_role_that_covers_workflows_passes(roles_cover_workflows: AsyncMock):
    auth = _auth("org-1", "team-1")

    assert await require_resource_permission()(auth=auth) is auth
    roles_cover_workflows.assert_awaited_once_with("user-1", "org-1", "team-1")


async def test_a_credential_without_an_organization_acts_in_the_personal_one(
    mocker: pytest_mock.MockFixture,
):
    """Its owner owns that organization, so no role is read."""
    lookup = mocker.patch(
        "backend.api.external.middleware.resource_role_refusal",
        new_callable=AsyncMock,
        return_value=BILLING_ONLY_ORG_ROLE,
    )
    mocker.patch(
        "backend.api.external.middleware.get_user_default_team",
        new_callable=AsyncMock,
        return_value=("personal-org", "personal-team"),
    )

    bound = await require_resource_permission()(auth=_auth(None))

    assert (bound.organization_id, bound.team_id_restriction) == ("personal-org", None)
    lookup.assert_not_awaited()


async def test_a_credential_without_an_organization_is_refused_without_a_personal_one(
    mocker: pytest_mock.MockFixture,
):
    mocker.patch(
        "backend.api.external.middleware.get_user_default_team",
        new_callable=AsyncMock,
        return_value=(None, None),
    )

    with pytest.raises(HTTPException) as raised:
        await require_resource_permission()(auth=_auth(None))

    assert raised.value.status_code == 403


def _routes():
    for route in v1_app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods:
                yield f"{method} {route.path}", route


def _checks_the_role(dependant: Dependant) -> bool:
    return any(
        (inspect.isfunction(dep.call) and dep.call.__code__ is ROLE_CHECK)
        or _checks_the_role(dep)
        for dep in dependant.dependencies
    )
