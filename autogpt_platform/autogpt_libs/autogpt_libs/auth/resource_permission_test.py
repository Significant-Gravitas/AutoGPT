"""``requires_resource_permission``: the org action in org-home context, and the
team action as well in a team context, checked through a real route."""

import pytest
from fastapi import FastAPI, Security
from fastapi.testclient import TestClient

from autogpt_libs.auth.dependencies import (
    get_request_context,
    requires_resource_permission,
)
from autogpt_libs.auth.models import RequestContext
from autogpt_libs.auth.permissions import OrgAction, TeamAction

app = FastAPI()


@app.get("/runs")
async def list_runs(
    ctx: RequestContext = Security(
        requires_resource_permission(
            OrgAction.VIEW_RESOURCES, TeamAction.VIEW_EXECUTIONS
        )
    ),
) -> dict:
    return {"org_id": ctx.org_id, "team_id": ctx.team_id}


def _ctx(
    *,
    team_id: str | None = None,
    org_owner: bool = False,
    org_admin: bool = False,
    org_billing: bool = False,
    team_admin: bool = False,
    team_billing: bool = False,
) -> RequestContext:
    return RequestContext(
        user_id="user-1",
        org_id="org-1",
        team_id=team_id,
        is_org_owner=org_owner,
        is_org_admin=org_admin,
        is_org_billing_manager=org_billing,
        is_team_admin=team_admin,
        is_team_billing_manager=team_billing,
        seat_status="ACTIVE",
    )


def _get_runs_as(ctx: RequestContext):
    async def resolved() -> RequestContext:
        return ctx

    app.dependency_overrides[get_request_context] = resolved
    try:
        return TestClient(app).get("/runs")
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize(
    "ctx",
    [
        _ctx(),
        _ctx(org_owner=True),
        _ctx(org_admin=True),
        _ctx(org_owner=True, org_billing=True),
        _ctx(org_admin=True, org_billing=True),
        _ctx(team_id="team-1"),
        _ctx(team_id="team-1", team_admin=True),
        _ctx(team_id="team-1", team_admin=True, team_billing=True),
        _ctx(team_id="team-1", org_admin=True, team_admin=True),
    ],
    ids=[
        "org-member",
        "org-owner",
        "org-admin",
        "org-owner-and-billing",
        "org-admin-and-billing",
        "team-member",
        "team-admin",
        "team-admin-and-billing",
        "org-admin-as-team-admin",
    ],
)
def test_members_admins_and_owners_get_through(ctx: RequestContext):
    response = _get_runs_as(ctx)

    assert response.status_code == 200
    assert response.json() == {"org_id": "org-1", "team_id": ctx.team_id}


def test_an_org_billing_manager_is_refused_in_org_home():
    response = _get_runs_as(_ctx(org_billing=True))

    assert response.status_code == 403
    assert "organization" in response.json()["detail"]


def test_an_org_billing_manager_is_refused_in_a_team_they_are_a_member_of():
    """The org role is checked in a team context too."""
    response = _get_runs_as(_ctx(team_id="team-1", org_billing=True))

    assert response.status_code == 403
    assert "organization" in response.json()["detail"]


def test_a_team_billing_manager_is_refused_in_that_team():
    response = _get_runs_as(_ctx(team_id="team-1", team_billing=True))

    assert response.status_code == 403
    assert "workspace" in response.json()["detail"]


def test_an_org_owner_who_is_only_billing_manager_of_the_team_is_refused_there():
    response = _get_runs_as(_ctx(team_id="team-1", org_owner=True, team_billing=True))

    assert response.status_code == 403
    assert "workspace" in response.json()["detail"]
