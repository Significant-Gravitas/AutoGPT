"""The collaboration rollout gates new subscriptions, not persisted execution work."""

import json
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import HTTPException, WebSocket, WebSocketDisconnect

from backend.api import conn_manager, org_rollout, ws_api
from backend.api.conn_manager import ConnectionManager
from backend.api.model import WSMessage, WSMethod


@pytest.fixture
def subscriptions(monkeypatch, mocker):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    default = mocker.patch.object(
        org_rollout,
        "get_user_default_team",
        AsyncMock(return_value=("personal", "default-team")),
    )
    manager = ConnectionManager()
    live = mocker.patch.object(
        manager, "_execution_scope_is_live", AsyncMock(return_value=True)
    )
    opened = mocker.patch.object(manager, "_open_subscription", AsyncMock())
    websocket = AsyncMock(spec=WebSocket)
    meta = mocker.patch.object(conn_manager, "get_graph_execution_meta", AsyncMock())
    return manager, websocket, meta, opened, live, default


async def subscribe(subscriptions, kind, org_id, team_id):
    manager, websocket, meta, *_ = subscriptions
    if kind == "execution":
        meta.return_value = SimpleNamespace(
            graph_id="graph", organization_id=org_id, team_id=team_id
        )
        return await manager.subscribe_graph_exec(
            user_id="owner", graph_exec_id="execution", websocket=websocket
        )
    return await manager.subscribe_graph_execs(
        user_id="owner",
        graph_id="graph",
        organization_id=org_id,
        team_id=team_id,
        websocket=websocket,
    )


@pytest.mark.parametrize("kind", ["execution", "graph"])
@pytest.mark.parametrize(
    "org_id,team_id",
    [
        ("shared", "shared-team"),
        ("shared", None),
        ("somebody-elses-personal", "their-default"),
        ("personal", "another-team"),
        (None, "shared-team"),
    ],
)
async def test_disabled_rejects_shared_requested_or_resolved_scope(
    subscriptions,
    kind,
    org_id,
    team_id,
):
    _, _, meta, opened, live, default = subscriptions
    with pytest.raises(HTTPException) as error:
        await subscribe(subscriptions, kind, org_id, team_id)
    assert error.value.status_code == 403
    assert "not enabled" in error.value.detail
    opened.assert_not_awaited()
    live.assert_awaited_once()
    default.assert_awaited_once_with("owner")
    if kind == "execution":
        # This request has only an execution ID: the resource's saved scope
        # must be checked even though no org/team was supplied by the caller.
        meta.assert_awaited_once_with("owner", "execution")


@pytest.mark.parametrize("kind", ["execution", "graph"])
@pytest.mark.parametrize(
    "org_id,team_id",
    [
        ("personal", "default-team"),
        ("personal", None),
        (None, None),
    ],
)
async def test_disabled_preserves_personal_and_legacy_subscription(
    subscriptions,
    kind,
    org_id,
    team_id,
):
    _, _, _, opened, live, default = subscriptions
    await subscribe(subscriptions, kind, org_id, team_id)
    opened.assert_awaited_once()
    live.assert_awaited_once()
    scope = opened.await_args.args[-1]
    assert (scope.organization_id, scope.team_id) == (org_id, team_id)
    if org_id is None:
        default.assert_not_awaited()


@pytest.mark.parametrize("kind", ["execution", "graph"])
async def test_enabled_preserves_shared_subscription(subscriptions, monkeypatch, kind):
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "true")
    _, _, _, opened, live, default = subscriptions
    await subscribe(subscriptions, kind, "shared", "shared-team")
    opened.assert_awaited_once()
    live.assert_awaited_once()
    default.assert_not_awaited()


@pytest.mark.parametrize("kind", ["execution", "graph"])
async def test_unavailable_flag_rejects_shared_subscription(
    subscriptions,
    monkeypatch,
    mocker,
    kind,
):
    monkeypatch.delenv("FORCE_FLAG_SHOW_ORG_SETTINGS")
    flag = mocker.patch(
        "backend.api.features.orgs.rollout.is_feature_enabled",
        AsyncMock(return_value=False),
    )
    with pytest.raises(HTTPException):
        await subscribe(subscriptions, kind, "shared", "shared-team")
    assert flag.await_args.kwargs["default"] is False
    subscriptions[3].assert_not_awaited()


@pytest.mark.parametrize("kind", ["execution", "graph"])
async def test_rollout_never_bypasses_subscription_authorization(subscriptions, kind):
    _, _, _, opened, live, default = subscriptions
    live.return_value = False
    with pytest.raises(ValueError, match="Access denied"):
        await subscribe(subscriptions, kind, "personal", "default-team")
    opened.assert_not_awaited()
    default.assert_not_awaited()


@pytest.mark.parametrize("kind", ["execution", "graph"])
async def test_disabled_does_not_block_unsubscribe(subscriptions, mocker, kind):
    manager, websocket, _, _, _, default = subscriptions
    close = mocker.patch.object(manager, "_close_subscription", AsyncMock())
    if kind == "execution":
        await manager.unsubscribe_graph_exec(
            user_id="owner", graph_exec_id="execution", websocket=websocket
        )
    else:
        await manager.unsubscribe_graph_execs(
            user_id="owner",
            graph_id="graph",
            organization_id="shared",
            team_id="shared-team",
            websocket=websocket,
        )
    close.assert_awaited_once()
    default.assert_not_awaited()


async def test_existing_shared_subscription_keeps_live_auth_after_flag_turns_off(
    subscriptions,
    monkeypatch,
    mocker,
):
    manager, websocket, _, _, live, default = subscriptions
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "true")
    # Restore the actual opener to exercise the existing delivery callback.
    mocker.patch.object(
        manager,
        "_open_subscription",
        ConnectionManager._open_subscription.__get__(manager),
    )
    subscription = MagicMock()
    subscription.start = AsyncMock()
    mocker.patch.object(conn_manager, "_Subscription", return_value=subscription)

    @asynccontextmanager
    async def allowed(_scope):
        yield True

    mocker.patch.object(manager, "_execution_scope_barrier", allowed)
    forward = mocker.patch.object(manager, "_forward_exec_event", AsyncMock())
    await subscribe(subscriptions, "graph", "shared", "shared-team")
    on_message = subscription.start.await_args.args[0]
    monkeypatch.setenv("FORCE_FLAG_SHOW_ORG_SETTINGS", "false")
    live.reset_mock()

    await on_message("persisted execution event")

    forward.assert_awaited_once()
    live.assert_awaited_once()
    default.assert_not_awaited()
    websocket.close.assert_not_awaited()


async def test_router_reports_rollout_rejection_and_still_accepts_heartbeat(mocker):
    websocket = AsyncMock(spec=WebSocket)
    manager = AsyncMock(spec=ConnectionManager)
    mocker.patch.object(
        ws_api, "authenticate_websocket", AsyncMock(return_value="owner")
    )
    manager.subscribe_graph_exec.side_effect = HTTPException(403, "private detail")
    websocket.receive_text.side_effect = [
        WSMessage(
            method=WSMethod.SUBSCRIBE_GRAPH_EXEC, data={"graph_exec_id": "execution"}
        ).model_dump_json(),
        WSMessage(method=WSMethod.HEARTBEAT).model_dump_json(),
        WebSocketDisconnect(),
    ]

    await ws_api.websocket_router(websocket, manager)

    response = json.loads(websocket.send_text.await_args.args[0])
    assert response["method"] == WSMethod.ERROR.value
    assert response["success"] is False
    assert response["error"] == "Subscription not available for this account"
    websocket.send_json.assert_awaited_once_with(
        {
            "method": WSMethod.HEARTBEAT.value,
            "data": "pong",
            "success": True,
        }
    )
    manager.disconnect_socket.assert_awaited_once_with(websocket, user_id="owner")
