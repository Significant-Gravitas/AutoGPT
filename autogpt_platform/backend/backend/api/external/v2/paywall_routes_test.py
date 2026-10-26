"""Every v2 route either needs an active plan or is listed here as open.

The host app's ``backend/api/paywall_routes_test.py`` sees only its own routes,
and this app is mounted inside it, so it keeps its own list. The split follows
the internal API's: anything that runs compute, calls an outside service or
stores new files or rows needs a plan.
"""

import inspect

import fastapi
import fastapi.testclient
import pytest
from fastapi.dependencies.models import Dependant
from fastapi.routing import APIRoute
from prisma.enums import APIKeyPermission

from backend.api.external.v2.app import v2_app
from backend.api.external.v2.errors import add_v2_exception_handlers
from backend.api.external.v2.routes import v2_router
from backend.api.external.v2.tenancy import (
    TenantContext,
    require_auth,
    require_paid_permission,
)

# Every route's check is a closure over this one code object.
PLAN_CHECK = require_paid_permission().__code__

OPEN_WHILE_PAYWALLED = {
    # Reading the account's own data, or the catalog.
    "GET /blocks",
    "GET /files",
    "GET /files/{file_id}",
    "GET /files/{file_id}/download",
    "GET /graphs",
    "GET /graphs/{graph_id}",
    "GET /graphs/{graph_id}/blocks",
    "GET /graphs/{graph_id}/credentials",
    "GET /graphs/{graph_id}/library-agent",
    "GET /graphs/{graph_id}/marketplace-listing",
    "GET /graphs/{graph_id}/versions",
    "GET /integrations/credentials",
    "GET /library/agents",
    "GET /library/agents/{agent_id}",
    "GET /library/agents/{agent_id}/credentials",
    "GET /library/folders",
    "GET /library/folders/tree",
    "GET /library/folders/{folder_id}",
    "GET /library/presets",
    "GET /library/presets/{preset_id}",
    "GET /marketplace/agents",
    "GET /marketplace/agents/by-version/{version_id}",
    "GET /marketplace/agents/{username}/{agent_name}",
    "GET /marketplace/creators",
    "GET /marketplace/creators/{username}",
    "GET /marketplace/profile",
    "GET /marketplace/submissions",
    "GET /me",
    "GET /runs",
    "GET /runs/reviews",
    "GET /runs/{run_id}",
    "GET /schedules",
    # Seeing what the account is billed.
    "GET /credits",
    "GET /credits/cost-summary",
    "GET /credits/invoices",
    "GET /credits/subscription",
    "GET /credits/transactions",
    # Stopping, deleting or revoking.
    "DELETE /files/{file_id}",
    "DELETE /integrations/credentials/{credential_id}",
    "DELETE /library/agents/{agent_id}",
    "DELETE /library/folders/{folder_id}",
    "DELETE /library/presets/{preset_id}",
    "DELETE /marketplace/submissions/{version_id}",
    "DELETE /runs/{run_id}",
    "DELETE /runs/{run_id}/share",
    "POST /runs/{run_id}/stop",
    "DELETE /schedules/{schedule_id}",
    # Organising what the account already has: names, places, settings, sharing.
    "PATCH /graphs/{graph_id}/settings",
    "PATCH /library/agents/{agent_id}",
    "POST /library/folders",
    "PATCH /library/folders/{folder_id}",
    "POST /library/folders/{folder_id}/move",
    "PATCH /marketplace/profile",
    "POST /runs/{run_id}/share",
}

# One closed route per router, as a client calls it.
CLOSED_SAMPLES = [
    "POST /files/upload",
    "POST /graphs",
    "POST /integrations/credentials",
    "POST /library/agents/a-1/runs",
    "POST /library/presets/p-1/execute",
    "POST /marketplace/submissions",
    "POST /runs/r-1/reviews",
    "POST /schedules",
    "GET /search?query=weather",
]


def test_every_route_needs_a_plan_or_is_listed_open():
    unlisted, checked_but_listed = [], []
    for key, route in _routes():
        checked = _depends_on_plan_check(route.dependant)
        listed = key in OPEN_WHILE_PAYWALLED
        if checked and listed:
            checked_but_listed.append(key)
        elif not checked and not listed:
            unlisted.append(key)

    assert not unlisted, (
        "Authenticate these with require_paid_permission, or list them in "
        f"OPEN_WHILE_PAYWALLED with the reason they stay open: {sorted(unlisted)}"
    )
    assert not checked_but_listed, (
        "These routes need a plan, so drop them from the open list: "
        f"{sorted(checked_but_listed)}"
    )


def test_the_open_list_names_only_routes_that_exist():
    assert not OPEN_WHILE_PAYWALLED - {key for key, _ in _routes()}


@pytest.mark.parametrize("call", CLOSED_SAMPLES)
def test_an_account_without_a_plan_gets_402_from_a_closed_route(
    call: str, paywalled_user: str
):
    app = fastapi.FastAPI()
    app.include_router(v2_router)
    add_v2_exception_handlers(app)
    app.dependency_overrides[require_auth] = lambda: TenantContext(
        user_id=paywalled_user,
        scopes=list(APIKeyPermission),
        type="api_key",
        organization_id="org-1",
    )
    method, path = call.split(" ", 1)

    response = fastapi.testclient.TestClient(app).request(method, path)

    assert response.status_code == 402, response.text
    assert response.json()["error"]["code"] == "payment_required"


def _routes():
    for route in v2_app.routes:
        if isinstance(route, APIRoute):
            for method in route.methods:
                yield f"{method} {route.path}", route


def _depends_on_plan_check(dependant: Dependant) -> bool:
    return any(
        (inspect.isfunction(dep.call) and dep.call.__code__ is PLAN_CHECK)
        or _depends_on_plan_check(dep)
        for dep in dependant.dependencies
    )
