from contextlib import asynccontextmanager
from datetime import datetime
from unittest.mock import AsyncMock, Mock

import fastapi
import fastapi.testclient
import pytest
import pytest_mock
from autogpt_libs.auth.dependencies import get_request_context
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException
from prisma.enums import APIKeyStatus

from backend.api.live_auth import resolve_write_team_id as _resolve_write_team_id
from backend.api.utils.exceptions import add_exception_handlers
from backend.data.auth.api_key import APIKeyInfo
from backend.data.graph import GraphModel
from backend.executor.scheduler import GraphExecutionJobInfo

from .api_keys import routes as api_keys_module
from .blocks import routes as blocks_module
from .graph_executions import routes as graph_executions_module
from .graphs import routes as graphs_module
from .schedules import routes as schedules_module

app = fastapi.FastAPI()
app.include_router(blocks_module.router)
app.include_router(graph_executions_module.router)
app.include_router(graphs_module.router)
app.include_router(schedules_module.router)
app.include_router(api_keys_module.router, prefix="/api-keys")
add_exception_handlers(app)
client = fastapi.testclient.TestClient(app)


TEAM_ID = "team-abc"


_GRAPH_BODY = {"name": "Test Graph", "description": "Test", "nodes": [], "links": []}


_SCHEDULE_BODY = {
    "name": "s",
    "cron": "0 0 * * *",
    "inputs": {},
    "timezone": "UTC",
}


def _test_ctx(user_id: str, team_id: str | None = "test-workspace") -> RequestContext:
    return RequestContext(
        user_id=user_id,
        org_id="test-org",
        team_id=team_id,
        is_org_owner=True,
        is_org_admin=True,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )


@pytest.fixture
def mock_shared_cache(monkeypatch):
    redis = Mock()
    redis.get.return_value = None
    redis.getex.return_value = None
    monkeypatch.setattr("backend.util.cache._get_redis", lambda: redis)


@pytest.fixture(autouse=True)
def setup_app_auth(mock_jwt_user, mock_shared_cache, test_user_id, monkeypatch):
    """Setup auth overrides for all tests in this module"""
    # Existing org/team action tests opt in; dedicated rollout tests cover off.
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "true")
    monkeypatch.setattr(graphs_module, "clear_unowned_auto_credentials", AsyncMock())
    from autogpt_libs.auth.dependencies import get_request_context
    from autogpt_libs.auth.jwt_utils import get_jwt_payload
    from autogpt_libs.auth.models import RequestContext

    app.dependency_overrides[get_jwt_payload] = mock_jwt_user["get_jwt_payload"]

    # Override get_request_context too — the real one queries Prisma to
    # resolve the user's personal org when no X-Org-Id header is set,
    # which closes/leaks the test event loop across sync TestClient calls.
    async def _fake_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id=None,
            is_org_owner=True,
            is_org_admin=True,
            is_org_billing_manager=False,
            is_team_admin=True,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_request_context] = _fake_request_context

    @asynccontextmanager
    async def allow_graph_mutation(*_args, **_kwargs):
        yield

    @asynccontextmanager
    async def allow_resource_action(*_args, **_kwargs):
        yield True

    for module in (
        blocks_module,
        graph_executions_module,
        graphs_module,
        schedules_module,
        api_keys_module,
    ):
        for name in (
            "live_resource_permission_barrier",
            "live_agent_graph_access_barrier",
        ):
            if hasattr(module, name):
                monkeypatch.setattr(module, name, allow_resource_action)
    monkeypatch.setattr(graphs_module, "_live_graph_mutation", allow_graph_mutation)
    monkeypatch.setattr(
        "backend.api.live_auth.live_resource_permission_barrier",
        allow_resource_action,
    )
    monkeypatch.setattr(
        "backend.api.live_auth.live_org_permission_barrier",
        allow_resource_action,
    )
    monkeypatch.setattr(
        "backend.api.live_auth.live_actor_org_permission_barrier",
        allow_resource_action,
    )
    yield
    app.dependency_overrides.clear()


def test_execute_graph_block_rechecks_live_permission_before_charge(
    mocker: pytest_mock.MockFixture,
) -> None:
    @asynccontextmanager
    async def denied(*_args, **_kwargs):
        yield False

    block = Mock(disabled=False)
    block.execute = AsyncMock()
    mocker.patch("backend.api.features.blocks.routes.get_block", return_value=block)
    mocker.patch(
        "backend.api.features.blocks.routes.get_user_by_id",
        return_value=Mock(timezone="UTC"),
    )
    charge = mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.charge_for_direct_block_execution",
        new_callable=AsyncMock,
    )
    mocker.patch(
        "backend.api.features.blocks.routes.live_resource_permission_barrier", denied
    )

    response = client.post("/blocks/test-block/execute", json={})

    assert response.status_code == 403
    charge.assert_not_awaited()
    block.execute.assert_not_called()


def test_execute_graph_block_holds_live_permission_through_provider_iteration(
    mocker: pytest_mock.MockFixture,
) -> None:
    active = False

    @asynccontextmanager
    async def barrier(*_args, **_kwargs):
        nonlocal active
        active = True
        try:
            yield True
        finally:
            active = False

    async def execute(*_args, **_kwargs):
        assert active is True
        yield "output", "done"

    async def charge(**_kwargs):
        assert active is True

    block = Mock(disabled=False, execution_timeout_seconds=5)
    block.execute = execute
    mocker.patch("backend.api.features.blocks.routes.get_block", return_value=block)
    mocker.patch(
        "backend.api.features.blocks.routes.get_user_by_id",
        return_value=Mock(timezone="UTC"),
    )
    mocker.patch(
        "backend.api.features.blocks.routes.execution_utils.charge_for_direct_block_execution",
        side_effect=charge,
    )
    mocker.patch(
        "backend.api.features.blocks.routes.live_resource_permission_barrier", barrier
    )

    response = client.post("/blocks/test-block/execute", json={})

    assert response.status_code == 200
    assert response.json() == {"output": ["done"]}
    assert active is False


@pytest.mark.asyncio
async def test_enable_execution_sharing_rejects_revoked_target_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Mock(graph_id="graph-1", team_id="team-revoked", outputs={})
    monkeypatch.setattr(
        graph_executions_module.execution_db,
        "get_graph_execution",
        AsyncMock(return_value=candidate),
    )

    @asynccontextmanager
    async def denied(user_id, organization_id, team_id, *_actions):
        assert (user_id, organization_id, team_id) == (
            "user-1",
            "test-org",
            "team-revoked",
        )
        yield False

    replace = AsyncMock()
    monkeypatch.setattr(
        graph_executions_module, "live_resource_permission_barrier", denied
    )
    monkeypatch.setattr(
        graph_executions_module.execution_db, "replace_execution_share", replace
    )

    with pytest.raises(HTTPException) as exc:
        await graph_executions_module.enable_execution_sharing(
            "graph-1",
            "exec-1",
            "user-1",
            _test_ctx("user-1", team_id=None),
            graph_executions_module.ExecutionShareRequest(),
        )

    assert exc.value.status_code == 403
    replace.assert_not_awaited()


@pytest.mark.asyncio
async def test_enable_execution_sharing_holds_exact_target_team_barrier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Mock(graph_id="graph-1", team_id="team-1", outputs={"out": []})
    monkeypatch.setattr(
        graph_executions_module.execution_db,
        "get_graph_execution",
        AsyncMock(return_value=candidate),
    )
    monkeypatch.setattr(
        graph_executions_module.execution_db,
        "get_graph_execution_exact_scope",
        AsyncMock(return_value=candidate),
    )
    active = False

    @asynccontextmanager
    async def barrier(user_id, organization_id, team_id, *_actions):
        nonlocal active
        assert (user_id, organization_id, team_id) == (
            "user-1",
            "test-org",
            "team-1",
        )
        active = True
        try:
            yield True
        finally:
            active = False

    async def mutate(**_kwargs):
        assert active is True

    monkeypatch.setattr(
        graph_executions_module, "live_resource_permission_barrier", barrier
    )
    replace = AsyncMock(side_effect=mutate)
    monkeypatch.setattr(
        graph_executions_module.execution_db, "replace_execution_share", replace
    )

    response = await graph_executions_module.enable_execution_sharing(
        "graph-1",
        "exec-1",
        "user-1",
        _test_ctx("user-1", team_id=None),
        graph_executions_module.ExecutionShareRequest(),
    )

    assert response.share_token
    assert replace.await_args.kwargs["team_id"] == "team-1"
    assert replace.await_args.kwargs["organization_id"] == "test-org"
    assert active is False


@pytest.mark.asyncio
async def test_delete_execution_rejects_revoked_target_team(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = Mock(team_id="team-revoked")
    monkeypatch.setattr(
        graph_executions_module.execution_db,
        "get_graph_execution",
        AsyncMock(return_value=candidate),
    )

    @asynccontextmanager
    async def denied(*_args, **_kwargs):
        yield False

    delete = AsyncMock()
    monkeypatch.setattr(
        graph_executions_module, "live_resource_permission_barrier", denied
    )
    monkeypatch.setattr(
        graph_executions_module.execution_db, "delete_graph_execution", delete
    )

    with pytest.raises(HTTPException) as exc:
        await graph_executions_module.delete_graph_execution(
            "exec-1", "user-1", _test_ctx("user-1", team_id=None)
        )

    assert exc.value.status_code == 403
    delete.assert_not_awaited()


def _saved_graph_model(user_id: str) -> GraphModel:
    return GraphModel(
        id="graph-123",
        version=1,
        is_active=True,
        name="Test Graph",
        description="A test graph",
        user_id=user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
    )


def _mock_graph_create_pipeline(mocker: pytest_mock.MockFixture, saved: GraphModel):
    """Stub the create pipeline so only team stamping is under test."""
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.make_graph_model",
        return_value=Mock(),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.before_graph_activate",
        new=AsyncMock(return_value=saved),
    )
    create_graph_mock = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.create_graph",
        new=AsyncMock(return_value=saved),
    )
    create_lib_agent_mock = mocker.patch(
        "backend.api.features.graphs.routes.library_db.create_library_agent",
        new=AsyncMock(),
    )
    return create_graph_mock, create_lib_agent_mock


def test_create_new_graph_stamps_explicit_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A valid team_id in the body lands the agent (and its library entry)
    under that team."""
    create_graph_mock, create_lib_agent_mock = _mock_graph_create_pipeline(
        mocker, _saved_graph_model(test_user_id)
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[TEAM_ID]),
    )

    response = client.post("/graphs", json={"graph": _GRAPH_BODY, "team_id": TEAM_ID})

    assert response.status_code == 200
    assert create_graph_mock.await_args.kwargs["team_id"] == TEAM_ID
    assert create_lib_agent_mock.await_args.kwargs["team_id"] == TEAM_ID


def test_create_new_graph_omitted_team_lands_org_home(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """No team_id -> org-home (teamId NULL), and no membership lookup happens."""
    create_graph_mock, create_lib_agent_mock = _mock_graph_create_pipeline(
        mocker, _saved_graph_model(test_user_id)
    )
    get_teams = mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[TEAM_ID]),
    )

    response = client.post("/graphs", json={"graph": _GRAPH_BODY})

    assert response.status_code == 200
    assert create_graph_mock.await_args.kwargs["team_id"] is None
    assert create_lib_agent_mock.await_args.kwargs["team_id"] is None
    get_teams.assert_not_awaited()


def test_create_new_graph_rejects_non_member_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A team the caller isn't an active member of -> 400, nothing persisted."""
    create_graph_mock, _ = _mock_graph_create_pipeline(
        mocker, _saved_graph_model(test_user_id)
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[]),
    )

    response = client.post("/graphs", json={"graph": _GRAPH_BODY, "team_id": TEAM_ID})

    assert response.status_code == 400
    assert "active member" in response.json()["detail"]
    create_graph_mock.assert_not_awaited()


def test_create_new_graph_rejects_cross_org_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A team in another org never appears in the caller's team list -> 400.

    get_user_team_ids only returns teams within the caller's org, so a
    cross-org team is indistinguishable from a non-member team here — both
    reject. We assert the org-scoped list can't be bypassed by a foreign id."""
    create_graph_mock, _ = _mock_graph_create_pipeline(
        mocker, _saved_graph_model(test_user_id)
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["team-in-my-org"]),
    )

    response = client.post(
        "/graphs", json={"graph": _GRAPH_BODY, "team_id": "team-in-another-org"}
    )

    assert response.status_code == 400
    create_graph_mock.assert_not_awaited()


def _existing_version(
    team_id: str | None = "team-existing", organization_id: str | None = "test-org"
) -> Mock:
    return Mock(
        version=1, is_active=True, team_id=team_id, organization_id=organization_id
    )


def _mock_update_graph_pipeline(
    mocker: pytest_mock.MockFixture,
    existing: Mock,
    saved: GraphModel,
    new_version_active: bool = False,
):
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph_all_versions",
        new=AsyncMock(return_value=[existing]),
    )
    # The caller owns the graph ID, so the save adds a version, not a fork.
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.graph_owner_id",
        new=AsyncMock(return_value=saved.user_id),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.make_graph_model",
        return_value=Mock(is_active=new_version_active),
    )
    if new_version_active:
        mocker.patch(
            "backend.api.features.graphs.routes.before_graph_activate",
            new=AsyncMock(side_effect=lambda graph, user_id: graph),
        )
        mocker.patch(
            "backend.api.features.graphs.routes.graph_db.set_graph_active_version",
            new=AsyncMock(),
        )
        mocker.patch(
            "backend.api.features.graphs.routes.on_graph_deactivate", new=AsyncMock()
        )
    create_graph_mock = mocker.patch(
        "backend.api.features.graphs.routes.graph_db.create_graph",
        new=AsyncMock(
            return_value=Mock(
                version=2, is_active=new_version_active, webhook_input_node=None
            )
        ),
    )
    mocker.patch(
        "backend.api.features.graphs.routes.graph_db.get_graph",
        new=AsyncMock(return_value=saved),
    )
    return create_graph_mock


def test_update_graph_inherits_existing_team_when_omitted(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """Regression (SECRT-2462): re-saving a team agent must keep its team, not
    reset it to org-home now that the active-team header is always null."""
    create_graph_mock = _mock_update_graph_pipeline(
        mocker, _existing_version(), _saved_graph_model(test_user_id)
    )

    response = client.put("/graphs/graph-123", json={"id": "graph-123", **_GRAPH_BODY})

    assert response.status_code == 200
    assert create_graph_mock.await_args.kwargs["team_id"] == "team-existing"


def test_update_graph_moves_to_explicit_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """An explicit team_id query param saves the new version into that team."""
    create_graph_mock = _mock_update_graph_pipeline(
        mocker, _existing_version(), _saved_graph_model(test_user_id)
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["team-new"]),
    )

    response = client.put(
        "/graphs/graph-123?team_id=team-new",
        json={"id": "graph-123", **_GRAPH_BODY},
    )

    assert response.status_code == 200
    assert create_graph_mock.await_args.kwargs["team_id"] == "team-new"


def test_update_graph_rejects_non_member_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """An explicit ?team_id= the caller isn't an active member of -> 400, and
    no new version is persisted."""
    create_graph_mock = _mock_update_graph_pipeline(
        mocker, _existing_version(), _saved_graph_model(test_user_id)
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["some-other-team"]),
    )

    response = client.put(
        "/graphs/graph-123?team_id=team-not-mine",
        json={"id": "graph-123", **_GRAPH_BODY},
    )

    assert response.status_code == 400
    assert "active member" in response.json()["detail"]
    create_graph_mock.assert_not_awaited()


def test_update_graph_does_not_inherit_team_from_another_org(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A version owned by a different org must not have its team id stamped
    onto a row tagged with the request's org — fall back to org-home."""
    create_graph_mock = _mock_update_graph_pipeline(
        mocker,
        _existing_version(team_id="team-in-other-org", organization_id="other-org"),
        _saved_graph_model(test_user_id),
    )

    response = client.put("/graphs/graph-123", json={"id": "graph-123", **_GRAPH_BODY})

    assert response.status_code == 200
    assert create_graph_mock.await_args.kwargs["team_id"] is None


def test_update_graph_syncs_library_agent_tenancy(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """Moving a version to another team must re-tag the library entry too —
    list badges and team filters read org/team off the LibraryAgent row."""
    create_graph_mock = _mock_update_graph_pipeline(
        mocker,
        _existing_version(),
        _saved_graph_model(test_user_id),
        new_version_active=True,
    )
    sync_mock = mocker.patch(
        "backend.api.features.graphs.routes.library_db.update_agent_version_in_library",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["team-new"]),
    )

    response = client.put(
        "/graphs/graph-123?team_id=team-new",
        json={"id": "graph-123", **_GRAPH_BODY},
    )

    assert response.status_code == 200
    assert create_graph_mock.await_args.kwargs["team_id"] == "team-new"
    assert sync_mock.await_args.kwargs["team_id"] == "team-new"
    assert sync_mock.await_args.kwargs["organization_id"] == "test-org"


@pytest.mark.asyncio
async def test_resolve_write_team_id_requires_org_context() -> None:
    """Requesting a team with no org context is a 400, not a silent stamp."""
    assert await _resolve_write_team_id("user-1", None, None) is None

    with pytest.raises(HTTPException) as exc_info:
        await _resolve_write_team_id("user-1", None, TEAM_ID)

    assert exc_info.value.status_code == 400
    assert "organization context" in exc_info.value.detail


def _mock_schedule_create_pipeline(
    mocker: pytest_mock.MockFixture, graph: Mock, user_id: str
):
    mocker.patch(
        "backend.api.features.schedules.routes.graph_db.get_graph",
        new=AsyncMock(return_value=graph),
    )
    mocker.patch(
        "backend.api.features.schedules.routes.complete_onboarding_step",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.features.schedules.routes.experts_db.resolve_expert_for_graph",
        new=AsyncMock(return_value=None),
    )
    mocker.patch.object(
        schedules_module.library_db,
        "get_library_agent_by_graph_id",
        AsyncMock(return_value=Mock(id="installed")),
    )
    add_schedule_mock = AsyncMock(
        return_value=GraphExecutionJobInfo(
            id="schedule-1",
            name="s",
            next_run_time="2025-09-04T13:37:00+00:00",
            user_id=user_id,
            graph_id="graph-123",
            graph_version=1,
            cron="0 0 * * *",
            input_data={},
        )
    )
    mocker.patch(
        "backend.api.features.schedules.routes.get_scheduler_client",
        return_value=Mock(add_execution_schedule=add_schedule_mock),
    )
    return add_schedule_mock


def test_create_schedule_stamps_explicit_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A valid team_id in the body scopes the schedule to that team."""
    add_schedule_mock = _mock_schedule_create_pipeline(
        mocker, _existing_version(), test_user_id
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[TEAM_ID]),
    )

    response = client.post(
        "/graphs/graph-123/schedules", json={**_SCHEDULE_BODY, "team_id": TEAM_ID}
    )

    assert response.status_code == 200
    assert add_schedule_mock.await_args.kwargs["team_id"] == TEAM_ID


def test_create_schedule_inherits_graph_team_when_omitted(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """Omitting team_id keeps the schedule with the agent it runs, instead of
    silently dropping it to org-home."""
    add_schedule_mock = _mock_schedule_create_pipeline(
        mocker, _existing_version(team_id="team-existing"), test_user_id
    )
    get_teams = mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[TEAM_ID]),
    )

    response = client.post("/graphs/graph-123/schedules", json=_SCHEDULE_BODY)

    assert response.status_code == 200
    assert add_schedule_mock.await_args.kwargs["team_id"] == "team-existing"
    get_teams.assert_not_awaited()


def test_create_schedule_does_not_inherit_team_from_another_org(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A graph owned by another org must not leak its team onto a schedule
    tagged with this request's org."""
    add_schedule_mock = _mock_schedule_create_pipeline(
        mocker,
        _existing_version(team_id="team-in-other-org", organization_id="other-org"),
        test_user_id,
    )

    response = client.post("/graphs/graph-123/schedules", json=_SCHEDULE_BODY)

    assert response.status_code == 200
    assert add_schedule_mock.await_args.kwargs["team_id"] is None


def test_create_schedule_rejects_non_member_team(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A team the caller isn't an active member of -> 400, no schedule created."""
    add_schedule_mock = _mock_schedule_create_pipeline(
        mocker, _existing_version(), test_user_id
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[]),
    )

    response = client.post(
        "/graphs/graph-123/schedules", json={**_SCHEDULE_BODY, "team_id": TEAM_ID}
    )

    assert response.status_code == 400
    assert "active member" in response.json()["detail"]
    add_schedule_mock.assert_not_awaited()


def test_schedule_creator_can_delete_from_member_workspace(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    async def _member_request_context() -> RequestContext:
        return RequestContext(
            user_id=test_user_id,
            org_id="test-org",
            team_id=TEAM_ID,
            is_org_owner=False,
            is_org_admin=False,
            is_org_billing_manager=False,
            is_team_admin=False,
            is_team_billing_manager=False,
            seat_status="ACTIVE",
        )

    app.dependency_overrides[get_request_context] = _member_request_context
    schedule = GraphExecutionJobInfo(
        id="schedule-1",
        name="s",
        next_run_time="2025-09-04T13:37:00+00:00",
        user_id=test_user_id,
        graph_id="graph-123",
        graph_version=1,
        cron="0 0 * * *",
        input_data={},
        organization_id="test-org",
        team_id=TEAM_ID,
    )
    get_schedule_mock = AsyncMock(return_value=schedule)
    delete_schedule_mock = AsyncMock(return_value=schedule)
    mocker.patch(
        "backend.api.features.schedules.routes.get_scheduler_client",
        return_value=Mock(
            get_execution_schedule=get_schedule_mock,
            delete_schedule=delete_schedule_mock,
        ),
    )
    barrier_calls: list[tuple[str, str | None, str | None, str]] = []

    @asynccontextmanager
    async def allow_creator_delete(
        user_id: str,
        organization_id: str | None,
        team_id: str | None,
        access: str,
    ):
        barrier_calls.append((user_id, organization_id, team_id, access))
        yield True

    mocker.patch(
        "backend.api.features.schedules.routes.live_resource_access_barrier",
        allow_creator_delete,
    )

    response = client.delete("/schedules/schedule-1")

    assert response.status_code == 200
    assert response.json() == {"id": "schedule-1"}
    assert barrier_calls == [(test_user_id, "test-org", TEAM_ID, "create")]
    assert get_schedule_mock.await_count == 2
    get_schedule_mock.assert_awaited_with(
        "schedule-1",
        user_id=test_user_id,
        organization_id="test-org",
        team_ids=[TEAM_ID],
    )
    delete_schedule_mock.assert_awaited_once_with(
        "schedule-1",
        user_id=test_user_id,
        organization_id="test-org",
        team_ids=[TEAM_ID],
    )


def _api_key_info(user_id: str):
    return APIKeyInfo(
        id="key-1",
        name="k",
        head="agpt_xxxx",
        tail="yyyy",
        status=APIKeyStatus.ACTIVE,
        scopes=[],
        user_id=user_id,
        created_at=datetime(2025, 9, 4, 13, 37),
    )


def test_create_api_key_stamps_team_restriction(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """A valid team_id pins the new key to that team (teamIdRestriction)."""
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(return_value=(_api_key_info(test_user_id), "agpt_plaintext")),
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[TEAM_ID]),
    )

    response = client.post(
        "/api-keys", json={"name": "k", "permissions": [], "team_id": TEAM_ID}
    )

    assert response.status_code == 200
    assert create_mock.await_args.kwargs["team_id_restriction"] == TEAM_ID


def test_create_api_key_omitted_team_is_unrestricted(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """No team_id -> an org-wide (unrestricted) key, today's behavior."""
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(return_value=(_api_key_info(test_user_id), "agpt_plaintext")),
    )

    response = client.post("/api-keys", json={"name": "k", "permissions": []})

    assert response.status_code == 200
    assert create_mock.await_args.kwargs["team_id_restriction"] is None


def test_create_api_key_rejects_non_member_team(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Scoping a key to a team the caller isn't in -> 400, no key minted."""
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[]),
    )

    response = client.post(
        "/api-keys", json={"name": "k", "permissions": [], "team_id": TEAM_ID}
    )

    assert response.status_code == 400
    create_mock.assert_not_awaited()


def test_create_api_key_rejects_empty_team_id(
    mocker: pytest_mock.MockFixture,
) -> None:
    """An empty-string team_id is an invalid team, not "unset" — it must be
    rejected rather than falling through to the request-context team."""
    _override_ctx_team("test-team")
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["test-team"]),
    )

    response = client.post(
        "/api-keys", json={"name": "k", "permissions": [], "team_id": ""}
    )

    assert response.status_code == 400
    create_mock.assert_not_awaited()


@pytest.mark.asyncio
async def test_stop_graph_run_rejects_team_visible_peer_before_cancel(mocker):
    from backend.api.features.graph_executions.routes import _stop_graph_run

    candidate = Mock(
        id="execution-1",
        user_id="other-user",
        graph_id="graph-1",
        status=graph_executions_module.execution_db.ExecutionStatus.RUNNING,
    )
    mocker.patch.object(
        graph_executions_module.execution_db,
        "get_graph_execution",
        new=AsyncMock(return_value=candidate),
    )
    stop = mocker.patch.object(
        graph_executions_module.execution_utils,
        "stop_graph_execution",
        new=AsyncMock(),
    )

    result = await _stop_graph_run(
        user_id="caller",
        graph_id="graph-1",
        graph_exec_id="execution-1",
        organization_id="org-1",
        team_id_restriction="team-1",
    )

    assert result == []
    stop.assert_not_awaited()


def _override_ctx_team(team_id: str | None) -> None:
    """Repoint get_request_context at a ctx with the given active team.

    The autouse setup_app_auth fixture already installs a ctx with
    team_id=None; this lets a single test exercise the header-set case. The
    fixture's teardown clears the override afterwards.
    """
    ctx = _test_ctx("test-user-id", team_id=team_id)
    app.dependency_overrides[get_request_context] = lambda: ctx


def test_create_api_key_stamps_ctx_team_restriction(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """An active-team ctx (X-Team-Id) pins the new key to that team."""
    _override_ctx_team("test-team")
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(return_value=(_api_key_info(test_user_id), "agpt_plaintext")),
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["test-team"]),
    )

    response = client.post("/api-keys", json={"name": "k", "permissions": []})

    assert response.status_code == 200
    assert create_mock.await_args.kwargs["team_id_restriction"] == "test-team"


def test_create_api_key_no_team_context_is_unrestricted(
    mocker: pytest_mock.MockFixture, test_user_id: str
) -> None:
    """Org-home ctx (team_id=None) -> an org-wide key; no membership lookup."""
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(return_value=(_api_key_info(test_user_id), "agpt_plaintext")),
    )
    get_teams = mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=["test-team"]),
    )

    response = client.post("/api-keys", json={"name": "k", "permissions": []})

    assert response.status_code == 200
    assert create_mock.await_args.kwargs["team_id_restriction"] is None
    get_teams.assert_not_awaited()


def test_create_api_key_rejects_ctx_team_user_not_member(
    mocker: pytest_mock.MockFixture,
) -> None:
    """Defense-in-depth: a ctx team the caller isn't an active member of is
    re-validated and rejected with 400 — no key minted."""
    _override_ctx_team("test-team")
    create_mock = mocker.patch(
        "backend.api.features.api_keys.routes.api_key_db.create_api_key",
        new=AsyncMock(),
    )
    mocker.patch(
        "backend.api.live_auth.get_user_team_ids",
        new=AsyncMock(return_value=[]),
    )

    response = client.post("/api-keys", json={"name": "k", "permissions": []})

    assert response.status_code == 400
    assert "active member" in response.json()["detail"]
    create_mock.assert_not_awaited()
