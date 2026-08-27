"""Override session-scoped fixtures for org tests.

Org tests mock at the Prisma boundary and don't need the full test server
or its graph cleanup hook.
"""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock

import pytest
from autogpt_libs.auth.models import RequestContext


@pytest.fixture(scope="session")
def server():
    """No-op — org tests don't need the full backend server."""
    yield None


@pytest.fixture(scope="session", autouse=True)
def graph_cleanup():
    """No-op — org tests don't create real graphs."""
    yield


@pytest.fixture(autouse=True)
def mock_live_tenancy_boundaries(mocker, request):
    live_ctx = RequestContext(
        user_id="user-owner-1",
        org_id="org-aaa",
        team_id=None,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=True,
        is_team_admin=True,
        is_team_billing_manager=True,
        seat_status="ACTIVE",
    )

    @asynccontextmanager
    async def allowed(*_args, **_kwargs):
        yield True

    @asynccontextmanager
    async def context(*_args, **_kwargs):
        yield live_ctx

    mocker.patch("backend.api.live_auth.live_org_permission_barrier", allowed)
    mocker.patch(
        "backend.api.features.orgs.routes.live_org_permission_barrier", allowed
    )
    mocker.patch(
        "backend.api.features.orgs.grant_routes.live_org_context_barrier", context
    )
    mocker.patch(
        "backend.api.features.orgs.grant_db.agent_graph_attachment_mutation_barrier",
        allowed,
    )
    mocker.patch("backend.data.graph.agent_graph_attachment_barriers", allowed)
    mocker.patch(
        "backend.data.graph.validate_graph_execution_permissions",
        new=AsyncMock(),
    )
    mocker.patch("backend.data.execution.live_resource_access_barrier", allowed)
    mocker.patch("backend.data.execution.agent_graph_attachment_barrier", allowed)
    mocker.patch(
        "backend.api.features.library.db.live_resource_access_barrier", allowed
    )
    mocker.patch(
        "backend.api.features.library.db.agent_graph_attachment_barrier", allowed
    )
    mocker.patch(
        "backend.api.features.store.db.agent_graph_attachment_barriers", allowed
    )
    mocker.patch(
        "backend.api.features.store.db.live_agent_graph_access_barrier", allowed
    )

    cls_name = request.node.cls.__name__ if request.node.cls is not None else ""
    live_team_classes = {
        "TestTeamManagementByTeamId",
        "TestTeamListVisibility",
        "TestTeamMembersVisibility",
    }
    if cls_name not in live_team_classes:
        mocker.patch(
            "backend.api.features.orgs.team_routes.live_org_context_barrier",
            context,
        )
        mocker.patch(
            "backend.api.features.orgs.team_db.lock_live_org_permission_scope",
            new=AsyncMock(return_value=live_ctx),
        )
        mocker.patch(
            "backend.api.features.orgs.team_db.lock_live_org_or_team_permission_scope",
            new=AsyncMock(return_value=live_ctx),
        )

    mocker.patch(
        "backend.api.features.orgs.db.lock_live_org_permission_scope",
        new=AsyncMock(return_value=live_ctx),
    )
    mocker.patch(
        "backend.api.features.orgs.db.lock_live_org_scope",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.features.orgs.invitation_routes.lock_live_org_permission_scope",
        new=AsyncMock(return_value=live_ctx),
    )
