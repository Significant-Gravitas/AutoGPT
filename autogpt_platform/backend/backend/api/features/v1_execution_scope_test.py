from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock

import pytest
from autogpt_libs.auth.models import RequestContext
from fastapi import HTTPException

from backend.api.features import v1


@pytest.mark.asyncio
async def test_org_home_execute_rejects_team_only_graph_before_create(mocker) -> None:
    credit_model = MagicMock(get_credits=AsyncMock(return_value=1))
    mocker.patch.object(v1, "get_credit_model", AsyncMock(return_value=credit_model))
    exact_lookup = AsyncMock(return_value=None)
    resolve_grant = AsyncMock()
    add_execution = AsyncMock()
    mocker.patch.object(v1.library_db, "get_library_agent_by_graph_id", exact_lookup)
    mocker.patch.object(v1, "resolve_graph_grant", resolve_grant)
    mocker.patch.object(v1.execution_utils, "add_graph_execution", add_execution)
    ctx = RequestContext(
        user_id="user-1",
        org_id="org-1",
        team_id=None,
        is_org_owner=True,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )

    with pytest.raises(HTTPException) as exc_info:
        await v1.execute_graph(
            graph_id="graph-1",
            user_id="user-1",
            ctx=ctx,
            inputs={},
            credentials_inputs={},
            graph_version=1,
            dry_run=False,
        )

    assert exc_info.value.status_code == 404
    exact_lookup.assert_awaited_once_with(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        organization_id="org-1",
        team_id_restriction=None,
        exact_scope=True,
    )
    resolve_grant.assert_not_awaited()
    add_execution.assert_not_awaited()


@pytest.mark.asyncio
async def test_execute_rejects_foreign_preset_before_graph_lookup(mocker) -> None:
    credit_model = MagicMock(get_credits=AsyncMock(return_value=1))
    mocker.patch.object(v1, "get_credit_model", AsyncMock(return_value=credit_model))
    get_preset = AsyncMock(return_value=None)
    exact_lookup = AsyncMock()
    add_execution = AsyncMock()
    mocker.patch.object(v1.library_db, "get_preset", get_preset)
    mocker.patch.object(v1.library_db, "get_library_agent_by_graph_id", exact_lookup)
    mocker.patch.object(v1.execution_utils, "add_graph_execution", add_execution)
    ctx = RequestContext(
        user_id="user-1",
        org_id="org-1",
        team_id="team-a",
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )

    with pytest.raises(HTTPException) as exc_info:
        await v1.execute_graph(
            graph_id="graph-1",
            user_id="user-1",
            ctx=ctx,
            inputs={},
            credentials_inputs={},
            graph_version=3,
            preset_id="foreign-preset",
            dry_run=False,
        )

    assert exc_info.value.status_code == 404
    get_preset.assert_awaited_once_with(
        "user-1",
        "foreign-preset",
        organization_id="org-1",
        team_id="team-a",
        enforce_team_scope=True,
    )
    exact_lookup.assert_not_awaited()
    add_execution.assert_not_awaited()


@pytest.mark.parametrize("authorization", ["wrapper", "grant"])
@pytest.mark.asyncio
async def test_team_schedule_accepts_exact_consumer_authorization(
    mocker, authorization: str
) -> None:
    graph = MagicMock(
        id="graph-1",
        version=3,
        organization_id="publisher-org",
        team_id="publisher-team",
    )
    mocker.patch.object(v1.graph_db, "get_graph", AsyncMock(return_value=graph))
    mocker.patch.object(
        v1.experts_db, "resolve_expert_for_graph", AsyncMock(return_value=None)
    )
    exact_lookup = AsyncMock(
        return_value=MagicMock(id="consumer-install")
        if authorization == "wrapper"
        else None
    )
    grant = AsyncMock(
        return_value=MagicMock(principalId="team-a")
        if authorization == "grant"
        else None
    )
    mocker.patch.object(v1.library_db, "get_library_agent_by_graph_id", exact_lookup)
    mocker.patch.object(v1, "resolve_graph_grant", grant)
    scheduled = MagicMock(next_run_time=None)
    add_schedule = AsyncMock(return_value=scheduled)
    mocker.patch.object(
        v1,
        "get_scheduler_client",
        return_value=MagicMock(add_execution_schedule=add_schedule),
    )
    mocker.patch.object(v1, "complete_onboarding_step", AsyncMock())

    @asynccontextmanager
    async def allow_scope(*_args, **_kwargs):
        yield True

    mocker.patch.object(v1, "live_resource_permission_barrier", allow_scope)
    ctx = RequestContext(
        user_id="user-1",
        org_id="consumer-org",
        team_id="team-a",
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=True,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )

    result = await v1.create_graph_execution_schedule(
        user_id="user-1",
        ctx=ctx,
        graph_id="graph-1",
        schedule_params=v1.ScheduleCreationRequest(
            graph_version=3,
            name="Consumer schedule",
            cron="0 * * * *",
            inputs={},
            timezone="UTC",
        ),
    )

    assert result is scheduled
    exact_lookup.assert_awaited_once_with(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=3,
        organization_id="consumer-org",
        team_id_restriction="team-a",
        exact_scope=True,
    )
    assert add_schedule.await_args.kwargs["organization_id"] == "consumer-org"
    assert add_schedule.await_args.kwargs["team_id"] == "team-a"
    v1.experts_db.resolve_expert_for_graph.assert_awaited_once_with(
        "user-1",
        "graph-1",
        organization_id="consumer-org",
        team_id="team-a",
        enforce_scope=True,
    )
    if authorization == "grant":
        grant.assert_awaited_once()
    else:
        grant.assert_not_awaited()


@pytest.mark.asyncio
async def test_api_key_body_team_checks_target_manage_permission(mocker) -> None:
    mocker.patch.object(v1, "_resolve_write_team_id", AsyncMock(return_value="team-a"))
    create_key = AsyncMock()
    mocker.patch.object(v1.api_key_db, "create_api_key", create_key)
    checked_scope: list[tuple] = []

    @asynccontextmanager
    async def deny_target(*args, **_kwargs):
        checked_scope.append(args)
        yield False

    mocker.patch.object(v1, "live_resource_permission_barrier", deny_target)
    ctx = RequestContext(
        user_id="user-1",
        org_id="org-1",
        team_id=None,
        is_org_owner=False,
        is_org_admin=False,
        is_org_billing_manager=False,
        is_team_admin=False,
        is_team_billing_manager=False,
        seat_status="ACTIVE",
    )

    with pytest.raises(HTTPException) as exc_info:
        await v1.create_api_key(
            request=v1.CreateAPIKeyRequest(
                name="team key",
                permissions=[],
                team_id="team-a",
            ),
            user_id="user-1",
            ctx=ctx,
        )

    assert exc_info.value.status_code == 403
    assert checked_scope == [
        (
            "user-1",
            "org-1",
            "team-a",
            v1.OrgAction.MANAGE_CREDENTIALS,
            v1.TeamAction.MANAGE_CREDENTIALS,
        )
    ]
    create_key.assert_not_awaited()
