from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs import auth
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.api.features.local_executor.routes import _require_owned_shim, router
from backend.api.features.local_executor.state import RecordingState
from backend.copilot.model import (
    CHAT_STATUS_RUNNING,
    ChatSessionMetadata,
    LocalExecutionTargetMetadata,
)
from backend.copilot.tools.local_pc_machine import (
    MachineConnectionStaleError,
    MachineNotConnectedError,
)
from backend.copilot.tools.local_pc_relay_protocol import RelayPresence
from backend.copilot.tools.local_pc_shim import (
    ComputerUseDriver,
    ShimHello,
    ShimRecordingError,
)
from backend.copilot.tools.recording_models import (
    RecordingReviewApplied,
    RecordingSummary,
    WorkflowRecording,
)

_DRIVER = ComputerUseDriver(name="cua-driver", version="0.38.0", runtime="direct")


@pytest.fixture(autouse=True)
def _enabled_local_executor_features():
    with (
        patch(
            "backend.api.features.local_executor.routes.get_machine_presence",
            AsyncMock(
                side_effect=MachineNotConnectedError("MACHINE_NOT_CONNECTED", "offline")
            ),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_local_executor_enabled",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_workflow_recording_enabled",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_computer_use_enabled",
            AsyncMock(return_value=True),
        ),
    ):
        yield


def _make_client(user_id: str = "owner-1") -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[auth.get_user_id] = lambda: user_id
    return TestClient(app)


def _owned_session() -> SimpleNamespace:
    return _local_owned_session()


def _cloud_owned_session() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        metadata=ChatSessionMetadata(),
    )


def _local_owned_session() -> SimpleNamespace:
    return SimpleNamespace(
        session_id="session-1",
        metadata=ChatSessionMetadata(
            execution_target=LocalExecutionTargetMetadata(
                machine_id="machine-1",
                directory_ref="directory-ref",
                allowed_root="C:\\Users\\Ada\\Projects",
                root_fingerprint="a" * 64,
                root_grant="root-grant",
                revision=1,
            )
        ),
    )


def test_cloud_session_ignores_stale_local_executor_presence():
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(machine_id="machine-1", allowed_root="/workspace")
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_cloud_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            AsyncMock(return_value="pending"),
        ),
    ):
        response = _make_client().get("/api/copilot/sessions/session-1/executor")
    assert response.status_code == 200
    assert response.json()["kind"] == "none"
    manager.get_hello_async.assert_not_awaited()


@pytest.mark.asyncio
async def test_cloud_session_cannot_use_cached_local_executor():
    manager = MagicMock()
    manager.get_or_create_shim_for_session = AsyncMock(return_value=object())
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_cloud_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        with pytest.raises(HTTPException) as error:
            await _require_owned_shim("session-1", "owner-1", recording=True)
    assert error.value.status_code == 409
    manager.get_or_create_shim_for_session.assert_not_awaited()


def test_list_executors_is_strictly_owner_scoped() -> None:
    presence = RelayPresence(
        session_id="machine-scope",
        connection_id="connection-1",
        user_id="owner-1",
        client_id="autogpt-local-executor",
        hello={
            "connection_kind": "machine",
            "machine_id": "machine-1",
            "display_name": "Workstation",
            "platform": "windows",
            "arch": "x86_64",
            "shim_version": "0.2.0",
            "capabilities": ["files", "shell"],
        },
        expires_at=9_999_999_999,
    )
    list_presences = AsyncMock(return_value=[presence])

    with patch(
        "backend.api.features.local_executor.routes.list_machine_presences",
        list_presences,
    ):
        response = _make_client().get("/api/copilot/executors")

    assert response.status_code == 200
    assert response.json() == {
        "executors": [
            {
                "machine_id": "machine-1",
                "connection_id": "connection-1",
                "display_name": "Workstation",
                "platform": "windows",
                "arch": "x86_64",
                "shim_version": "0.2.0",
                "capabilities": ["files", "shell"],
            }
        ]
    }
    assert list_presences.await_args.args[0] == "owner-1"


def test_directory_browse_rejects_stale_machine_generation() -> None:
    stale = AsyncMock(
        side_effect=MachineConnectionStaleError(
            "MACHINE_CONNECTION_STALE",
            "The Local PC executor reconnected",
        )
    )
    with patch(
        "backend.api.features.local_executor.routes.get_machine_presence",
        stale,
    ):
        response = _make_client().post(
            "/api/copilot/executors/machine-1/directories",
            json={"expected_connection_id": "old-connection"},
        )

    assert response.status_code == 409
    stale.assert_awaited_once()


def test_directory_browse_returns_frontend_contract() -> None:
    presence = RelayPresence(
        session_id="machine-scope",
        connection_id="connection-1",
        user_id="owner-1",
        client_id="autogpt-local-executor",
        hello={"connection_kind": "machine", "machine_id": "machine-1"},
        expires_at=9_999_999_999,
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_machine_presence",
            AsyncMock(return_value=presence),
        ),
        patch(
            "backend.api.features.local_executor.routes.machine_rpc",
            AsyncMock(
                return_value={
                    "type": "DIRECTORY_LIST_RESPONSE",
                    "payload": {
                        "browse_id": "browse-1",
                        "current": None,
                        "parent_ref": None,
                        "entries": [
                            {
                                "directory_ref": "dir-1",
                                "name": "Projects",
                                "path": "C:\\Projects",
                            }
                        ],
                        "truncated": False,
                        "expires_at": 9_999_999_999,
                    },
                }
            ),
        ),
    ):
        response = _make_client().post(
            "/api/copilot/executors/machine-1/directories",
            json={"expected_connection_id": "connection-1"},
        )

    assert response.status_code == 200
    assert response.json()["connection_id"] == "connection-1"
    assert response.json()["entries"][0]["path"] == "C:\\Projects"


def test_directory_browse_never_echoes_an_invalid_listing() -> None:
    """A listing that fails validation is the companion's data, which can
    name anything on the user's machine; the 502 must not repeat it."""
    presence = RelayPresence(
        session_id="machine-scope",
        connection_id="connection-1",
        user_id="owner-1",
        client_id="autogpt-local-executor",
        hello={"connection_kind": "machine", "machine_id": "machine-1"},
        expires_at=9_999_999_999,
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_machine_presence",
            AsyncMock(return_value=presence),
        ),
        patch(
            "backend.api.features.local_executor.routes.machine_rpc",
            AsyncMock(
                return_value={
                    "type": "DIRECTORY_LIST_RESPONSE",
                    "payload": {
                        "browse_id": "browse-1",
                        "entries": [{"name": "tax-return-2026.pdf"}],
                    },
                }
            ),
        ),
    ):
        response = _make_client().post(
            "/api/copilot/executors/machine-1/directories",
            json={"expected_connection_id": "connection-1"},
        )

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The Local PC executor returned an invalid folder listing"
    )
    assert "tax-return" not in response.text


def test_executor_status_requires_session_owner() -> None:
    client = _make_client()
    get_session = AsyncMock(return_value=None)

    with patch(
        "backend.api.features.local_executor.routes.get_chat_session_metadata",
        get_session,
    ):
        response = client.get("/api/copilot/sessions/other-session/executor")

    assert response.status_code == 404
    get_session.assert_awaited_once_with("other-session", "owner-1")


def test_executor_status_rejects_unbounded_session_id() -> None:
    response = _make_client().get(f"/api/copilot/sessions/{'a' * 129}/executor")

    assert response.status_code == 422


def test_executor_status_remains_readable_after_feature_is_disabled() -> None:
    client = _make_client()
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(return_value=None)

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_local_executor_enabled",
            AsyncMock(return_value=False),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            AsyncMock(return_value="pending"),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.get("/api/copilot/sessions/session-1/executor")

    assert response.status_code == 200
    assert response.json()["kind"] == "none"
    manager.get_hello_async.assert_awaited_once_with("session-1")


def test_executor_status_lists_exactly_the_tools_consent_would_grant() -> None:
    """The consent dialog describes these, so it never claims an action that
    approving would not grant, such as clipboard access on a machine that
    doesn't offer it."""
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(
            machine_id="machine-1",
            allowed_root="C:\\Users\\Ada\\Projects",
            capabilities=["computer_use"],
            computer_use_driver=_DRIVER,
            computer_use_tools=["type_text", "click", "get_window_state"],
        )
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            AsyncMock(return_value="pending"),
        ),
    ):
        response = _make_client().get("/api/copilot/sessions/session-1/executor")

    assert response.status_code == 200
    assert response.json()["computer_use_driver"] == "cua-driver 0.38.0"
    assert response.json()["computer_use_tools"] == [
        "click",
        "get_window_state",
        "type_text",
    ]


def test_executor_status_shows_no_computer_use_outside_its_rollout() -> None:
    """The consent dialog opens for a machine that offers computer use, so a
    user outside the computer-use flag was asked to allow what nothing then
    let Otto do."""
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(
            machine_id="machine-1",
            allowed_root="C:\\Users\\Ada\\Projects",
            capabilities=["files", "computer_use"],
            computer_use_driver=_DRIVER,
            computer_use_tools=["click", "type_text"],
        )
    )
    consent = AsyncMock(return_value="pending")
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            consent,
        ),
        patch(
            "backend.api.features.local_executor.routes.is_computer_use_enabled",
            AsyncMock(return_value=False),
        ),
    ):
        response = _make_client().get("/api/copilot/sessions/session-1/executor")

    assert response.status_code == 200
    body = response.json()
    assert body["kind"] == "shim"
    assert body["computer_use_driver"] is None
    assert body["computer_use_tools"] is None
    assert consent.await_args.kwargs["driver"] is None
    assert consent.await_args.kwargs["tools"] is None


@pytest.mark.parametrize("in_rollout", [False, True])
def test_computer_use_approval_refused_outside_its_rollout(in_rollout: bool) -> None:
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(
            machine_id="machine-1",
            allowed_root="C:\\Users\\Ada\\Projects",
            capabilities=["computer_use"],
            computer_use_driver=_DRIVER,
            computer_use_tools=["click"],
        )
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_local_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
        patch(
            "backend.api.features.local_executor.routes.is_computer_use_enabled",
            AsyncMock(return_value=in_rollout),
        ),
        patch(
            "backend.api.features.local_executor.routes.set_computer_use_consent",
            AsyncMock(return_value="approved"),
        ) as stored,
    ):
        response = _make_client().post(
            "/api/copilot/sessions/session-1/executor/consent",
            json={
                "approved": True,
                "expected_machine_id": "machine-1",
                "expected_driver": "cua-driver 0.38.0",
                "expected_tools": ["click"],
            },
        )

    if in_rollout:
        assert response.status_code == 200
        stored.assert_awaited_once()
    else:
        assert response.status_code == 404
        stored.assert_not_awaited()


def test_executor_status_offers_no_tools_without_a_driver() -> None:
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(
            machine_id="machine-1",
            allowed_root="C:\\Users\\Ada\\Projects",
            capabilities=["computer_use"],
            computer_use_tools=["click"],
        )
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            AsyncMock(return_value="pending"),
        ),
    ):
        response = _make_client().get("/api/copilot/sessions/session-1/executor")

    assert response.json()["computer_use_driver"] is None
    assert response.json()["computer_use_tools"] is None


def test_executor_status_includes_server_side_consent() -> None:
    client = _make_client()
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(return_value=None)

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            AsyncMock(return_value="denied"),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.get("/api/copilot/sessions/session-1/executor")

    assert response.status_code == 200
    assert response.json() == {
        "kind": "none",
        "computer_use_consent": "denied",
        "platform": None,
        "arch": None,
        "allowed_root": None,
        "machine_id": None,
        "shim_version": None,
        "capabilities": None,
        "computer_use_driver": None,
        "computer_use_tools": None,
        "recording_channels": None,
        "recording_routes": None,
    }


def test_executor_status_uses_persistent_machine_when_child_is_detached() -> None:
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(return_value=None)
    presence = RelayPresence(
        session_id="machine-scope",
        connection_id="connection-1",
        user_id="owner-1",
        client_id="autogpt-local-executor",
        hello={
            "connection_kind": "machine",
            "machine_id": "machine-1",
            "display_name": "Workstation",
            "platform": "windows",
            "arch": "x86_64",
            "shim_version": "0.2.0",
            "capabilities": ["files", "computer_use", "directory_browse"],
            "computer_use_driver": {
                "name": "cua-driver",
                "version": "0.38.0",
                "runtime": "direct",
            },
            "computer_use_tools": ["click"],
            "protocol_version": "2.0",
            "allowed_root": None,
        },
        expires_at=9_999_999_999,
    )
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_local_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_machine_presence",
            AsyncMock(return_value=presence),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_computer_use_consent",
            AsyncMock(return_value="pending"),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = _make_client().get("/api/copilot/sessions/session-1/executor")

    assert response.status_code == 200
    assert response.json()["kind"] == "shim"
    assert response.json()["allowed_root"] == "C:\\Users\\Ada\\Projects"
    assert response.json()["machine_id"] == "machine-1"
    assert response.json()["computer_use_tools"] == ["click"]


@pytest.mark.parametrize(("approved", "state"), [(True, "approved"), (False, "denied")])
def test_computer_use_consent_is_owner_scoped(approved: bool, state: str) -> None:
    client = _make_client()
    set_consent = AsyncMock(return_value=state)
    hello = ShimHello(
        machine_id="machine-1",
        allowed_root="C:\\Users\\Ada\\Projects",
        capabilities=["computer_use"],
        computer_use_driver=_DRIVER,
        computer_use_tools=["click", "get_window_state"],
    )
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(return_value=hello)

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.set_computer_use_consent",
            set_consent,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        request = {"approved": approved}
        if approved:
            request.update(
                {
                    "expected_machine_id": "machine-1",
                    "expected_driver": "cua-driver 0.38.0",
                    "expected_tools": ["get_window_state", "click"],
                }
            )
        response = client.post(
            "/api/copilot/sessions/session-1/executor/consent",
            json=request,
        )

    assert response.status_code == 200
    assert response.json() == {"computer_use_consent": state}
    set_consent.assert_awaited_once_with(
        "session-1",
        "owner-1",
        approved=approved,
        machine_id="machine-1",
        driver="cua-driver 0.38.0",
        tools=["click", "get_window_state"],
    )


def test_computer_use_approval_requires_connected_capable_shim() -> None:
    client = _make_client()
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(return_value=None)

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/consent",
            json={
                "approved": True,
                "expected_machine_id": "machine-1",
                "expected_driver": "cua-driver 0.38.0",
                "expected_tools": ["click"],
            },
        )

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("expected_machine_id", "expected_driver", "expected_tools"),
    [
        ("machine-2", "cua-driver 0.38.0", ["click", "type_text"]),
        ("machine-1", "cua-driver 0.37.0", ["click", "type_text"]),
        ("machine-1", "cua-driver 0.38.0", ["click"]),
        ("machine-1", "cua-driver 0.38.0", ["click", "launch_app", "type_text"]),
    ],
    ids=["machine", "driver", "fewer-tools", "more-tools"],
)
def test_computer_use_approval_rejects_changed_executor_scope(
    expected_machine_id: str,
    expected_driver: str,
    expected_tools: list[str],
) -> None:
    client = _make_client()
    set_consent = AsyncMock()
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(
            machine_id="machine-1",
            allowed_root="C:\\Users\\Ada\\Projects",
            capabilities=["computer_use"],
            computer_use_driver=_DRIVER,
            computer_use_tools=["click", "type_text"],
        )
    )

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.set_computer_use_consent",
            set_consent,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/consent",
            json={
                "approved": True,
                "expected_machine_id": expected_machine_id,
                "expected_driver": expected_driver,
                "expected_tools": expected_tools,
            },
        )

    assert response.status_code == 409
    set_consent.assert_not_awaited()


def test_recording_start_uses_native_consent_and_registers_for_copilot() -> None:
    client = _make_client()
    recording = SimpleNamespace(
        start_with_consent=AsyncMock(return_value="rec-1"),
        effective_route_for=MagicMock(return_value="extract_then_cloud"),
    )
    shim = SimpleNamespace(
        sandbox_id="session-1", capabilities=["recording"], recording=recording
    )
    register_started = MagicMock()
    mark_started = AsyncMock()

    with (
        patch(
            "backend.api.features.local_executor.routes._require_owned_shim",
            AsyncMock(return_value=shim),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.register_recording_started",
            register_started,
        ),
        patch(
            "backend.copilot.sdk.recording_tools.mark_recording_started",
            mark_started,
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start",
            json={
                "mode": "copilot",
                "interpretation_route": "extract_then_cloud",
                "channels": ["floor", "browser"],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"recording_id": "rec-1"}
    recording.start_with_consent.assert_awaited_once_with(
        mode="copilot",
        interpretation_route="extract_then_cloud",
        channels=["floor", "browser"],
    )
    register_started.assert_called_once_with(
        shim,
        "rec-1",
        mode="copilot",
        interpretation_route="extract_then_cloud",
        channels=["floor", "browser"],
    )
    mark_started.assert_awaited_once_with(
        "session-1",
        "rec-1",
        mode="copilot",
        interpretation_route="extract_then_cloud",
        channels=["floor", "browser"],
    )


def test_recording_start_respects_recording_feature_flag() -> None:
    client = _make_client()
    manager = MagicMock()

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_workflow_recording_enabled",
            AsyncMock(return_value=False),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start", json={}
        )

    assert response.status_code == 404
    manager.get_or_create_shim_for_session.assert_not_called()


def test_recording_start_returns_conflict_when_shim_is_not_connected() -> None:
    client = _make_client()
    manager = MagicMock()
    manager.get_or_create_shim_for_session = AsyncMock(
        side_effect=TimeoutError("not connected")
    )

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start", json={}
        )

    assert response.status_code == 409
    assert response.json()["detail"] == (
        "No Local PC executor is connected for this session"
    )


def test_recording_start_returns_structured_native_denial() -> None:
    client = _make_client()
    recording = SimpleNamespace(
        start_with_consent=AsyncMock(
            side_effect=ShimRecordingError(
                "CONSENT_DENIED", "The user declined workflow recording."
            )
        )
    )
    shim = SimpleNamespace(capabilities=["recording"], recording=recording)

    with patch(
        "backend.api.features.local_executor.routes._require_owned_shim",
        AsyncMock(return_value=shim),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start",
            json={},
        )

    assert response.status_code == 403
    assert response.json()["detail"]["code"] == "CONSENT_DENIED"


def test_recording_start_stops_capture_when_shared_state_fails() -> None:
    client = _make_client()
    recording = SimpleNamespace(
        start_with_consent=AsyncMock(return_value="rec-orphan"),
        effective_route_for=MagicMock(return_value="extract_then_cloud"),
        stop=AsyncMock(return_value=RecordingSummary(recording_id="rec-orphan")),
    )
    shim = SimpleNamespace(
        capabilities=["recording"],
        recording=recording,
        close_recording=MagicMock(),
    )

    with (
        patch(
            "backend.api.features.local_executor.routes._require_owned_shim",
            AsyncMock(return_value=shim),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.mark_recording_started",
            AsyncMock(side_effect=ConnectionError("redis unavailable")),
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start", json={}
        )

    assert response.status_code == 503
    assert response.json()["detail"]["code"] == "RECORDING_STATE_UNAVAILABLE"
    recording.stop.assert_awaited_once_with("rec-orphan")
    shim.close_recording.assert_called_once_with("rec-orphan")


def test_recording_start_says_when_the_rollback_stop_is_unconfirmed() -> None:
    """The response must not claim the recording stopped when the stop
    failed, and carries its id, the only handle left on it."""
    client = _make_client()
    recording = SimpleNamespace(
        start_with_consent=AsyncMock(return_value="rec-orphan"),
        effective_route_for=MagicMock(return_value="extract_then_cloud"),
        stop=AsyncMock(side_effect=TimeoutError("shim did not answer")),
    )
    shim = SimpleNamespace(
        capabilities=["recording"],
        recording=recording,
        close_recording=MagicMock(),
    )

    with (
        patch(
            "backend.api.features.local_executor.routes._require_owned_shim",
            AsyncMock(return_value=shim),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.mark_recording_started",
            AsyncMock(side_effect=ConnectionError("redis unavailable")),
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start", json={}
        )

    assert response.status_code == 503
    detail = response.json()["detail"]
    assert detail["code"] == "RECORDING_STOP_UNCONFIRMED"
    assert "was stopped" not in detail["message"]
    assert detail["details"] == {"recording_id": "rec-orphan"}
    shim.close_recording.assert_called_once_with("rec-orphan")


def test_recording_stop_fetches_review_data_and_registers_state() -> None:
    client = _make_client()
    summary = RecordingSummary(recording_id="rec-1", step_count=2)
    workflow = WorkflowRecording(recording_id="rec-1", steps=[])
    recording = SimpleNamespace(
        stop=AsyncMock(return_value=summary),
        fetch=AsyncMock(return_value=workflow),
    )
    shim = SimpleNamespace(
        capabilities=["recording"],
        recording=recording,
        close_recording=MagicMock(),
    )
    register_stopped = MagicMock()
    mark_stopped = AsyncMock()

    with (
        patch(
            "backend.api.features.local_executor.routes._require_owned_shim",
            AsyncMock(return_value=shim),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.register_recording_stopped",
            register_stopped,
        ),
        patch(
            "backend.copilot.sdk.recording_tools.mark_recording_stopped",
            mark_stopped,
        ),
        patch(
            "backend.copilot.sdk.recording_tools.get_recording_state",
            AsyncMock(return_value=None),
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/stop",
            json={"recording_id": "rec-1"},
        )

    assert response.status_code == 200
    assert response.json()["summary"]["step_count"] == 2
    assert response.json()["recording"]["recording_id"] == "rec-1"
    recording.stop.assert_awaited_once_with("rec-1")
    recording.fetch.assert_awaited_once_with("rec-1")
    register_stopped.assert_called_once_with(shim, "rec-1", summary)
    mark_stopped.assert_awaited_once_with(
        "session-1", "rec-1", summary=summary.to_dict()
    )
    shim.close_recording.assert_called_once_with("rec-1")


def test_recording_stop_retries_fetch_without_sending_duplicate_stop() -> None:
    client = _make_client()
    summary = RecordingSummary(recording_id="rec-1", step_count=2)
    workflow = WorkflowRecording(recording_id="rec-1", steps=[])
    recording = SimpleNamespace(
        stop=AsyncMock(return_value=summary),
        fetch=AsyncMock(
            side_effect=[
                ShimRecordingError("INTERNAL_ERROR", "fetch failed"),
                workflow,
            ]
        ),
    )
    shim = SimpleNamespace(
        capabilities=["recording"],
        recording=recording,
        close_recording=MagicMock(),
    )
    persisted = RecordingState(
        recording_id="rec-1",
        status="stopped",
        summary=summary.to_dict(),
    )
    get_state = AsyncMock(side_effect=[None, persisted])
    mark_stopped = AsyncMock()
    register_stopped = MagicMock()

    with (
        patch(
            "backend.api.features.local_executor.routes._require_owned_shim",
            AsyncMock(return_value=shim),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.get_recording_state",
            get_state,
        ),
        patch(
            "backend.copilot.sdk.recording_tools.mark_recording_stopped",
            mark_stopped,
        ),
        patch(
            "backend.copilot.sdk.recording_tools.register_recording_stopped",
            register_stopped,
        ),
    ):
        first = client.post(
            "/api/copilot/sessions/session-1/executor/recording/stop",
            json={"recording_id": "rec-1"},
        )
        second = client.post(
            "/api/copilot/sessions/session-1/executor/recording/stop",
            json={"recording_id": "rec-1"},
        )

    assert first.status_code == 502
    assert second.status_code == 200
    assert second.json()["recording"]["recording_id"] == "rec-1"
    recording.stop.assert_awaited_once_with("rec-1")
    assert recording.fetch.await_count == 2
    mark_stopped.assert_awaited_once_with(
        "session-1", "rec-1", summary=summary.to_dict()
    )
    register_stopped.assert_called_once_with(shim, "rec-1", summary)
    assert shim.close_recording.call_count == 2


def test_recording_stop_remains_available_after_recording_kill_switch() -> None:
    client = _make_client()
    summary = RecordingSummary(recording_id="rec-1")
    workflow = WorkflowRecording(recording_id="rec-1", steps=[])
    recording = SimpleNamespace(
        stop=AsyncMock(return_value=summary),
        fetch=AsyncMock(return_value=workflow),
    )
    shim = SimpleNamespace(
        machine_id="machine-1",
        allowed_root="C:\\Users\\Ada\\Projects",
        capabilities=["recording"],
        recording=recording,
        close_recording=MagicMock(),
    )
    manager = MagicMock()
    manager.get_or_create_shim_for_session = AsyncMock(return_value=shim)
    recording_flag = AsyncMock(return_value=False)

    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=_owned_session()),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_workflow_recording_enabled",
            recording_flag,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
        patch(
            "backend.copilot.sdk.recording_tools.get_recording_state",
            AsyncMock(return_value=None),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.mark_recording_stopped",
            AsyncMock(),
        ),
        patch(
            "backend.copilot.sdk.recording_tools.register_recording_stopped",
            MagicMock(),
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/stop",
            json={"recording_id": "rec-1"},
        )

    assert response.status_code == 200
    recording_flag.assert_not_awaited()
    manager.get_or_create_shim_for_session.assert_awaited_once_with(
        "session-1", timeout=1.0
    )


def test_recording_review_is_authoritative_on_shim() -> None:
    client = _make_client()
    recording_id = "rec_5f327c0d-19c7-4b71-9aa7-90a319912ba0"
    applied = RecordingReviewApplied(recording_id=recording_id, step_count=3)
    recording = SimpleNamespace(apply_review=AsyncMock(return_value=applied))
    shim = SimpleNamespace(capabilities=["recording"], recording=recording)
    register_reviewed = MagicMock()
    mark_reviewed = AsyncMock()
    release = AsyncMock()

    with (
        patch(
            "backend.api.features.local_executor.routes._require_owned_shim",
            AsyncMock(return_value=shim),
        ),
        patch(
            "backend.api.features.local_executor.routes.register_recording_reviewed",
            register_reviewed,
        ),
        patch(
            "backend.api.features.local_executor.routes.mark_recording_reviewed",
            mark_reviewed,
        ),
        patch(
            "backend.api.features.local_executor.routes._release_reviewed_local_shim",
            release,
        ),
    ):
        response = client.post(
            f"/api/copilot/sessions/session-1/executor/recording/{recording_id}/review",
            json={
                "removed_step_seqs": [2],
                "redacted_step_seqs": [3, 4],
            },
        )

    assert response.status_code == 200
    assert response.json() == {"recording_id": recording_id, "step_count": 3}
    recording.apply_review.assert_awaited_once_with(
        recording_id, removed_step_seqs=[2], redacted_step_seqs=[3, 4]
    )
    register_reviewed.assert_called_once_with(shim, recording_id, applied)
    release.assert_awaited_once_with("session-1", "owner-1", shim)
    mark_reviewed.assert_awaited_once_with("session-1", recording_id, step_count=3)


def test_recording_review_validation_error_returns_422() -> None:
    client = _make_client()
    recording = SimpleNamespace(
        apply_review=AsyncMock(
            side_effect=ShimRecordingError(
                "RECORDING_REVIEW_INVALID", "Unknown recording step."
            )
        )
    )
    shim = SimpleNamespace(capabilities=["recording"], recording=recording)

    with patch(
        "backend.api.features.local_executor.routes._require_owned_shim",
        AsyncMock(return_value=shim),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/rec-1/review",
            json={"removed_step_seqs": [99], "redacted_step_seqs": []},
        )

    assert response.status_code == 422
    assert response.json()["detail"]["code"] == "RECORDING_REVIEW_INVALID"


@pytest.mark.parametrize(
    "failure", ["different_binding", "no_data_channel", "mismatched_data_channel"]
)
def test_a_failed_restore_detaches_the_session_it_attached(failure: str) -> None:
    """A review route with no shim restores the chat on its machine, which
    attaches it there. A failure after that left the attachment behind."""
    detach, shim, response = _restore_and_fail(failure, chat_status="idle")

    assert response.status_code == 409
    detach.assert_awaited_once_with(_PRESENCE, "session-1")
    if failure == "mismatched_data_channel":
        shim.kill.assert_awaited_once()


def test_a_failed_restore_leaves_a_running_turns_attachment() -> None:
    """A running turn shares the attachment and detaches it when it ends."""
    detach, _shim, response = _restore_and_fail(
        "no_data_channel", chat_status=CHAT_STATUS_RUNNING
    )

    assert response.status_code == 409
    detach.assert_not_awaited()


_PRESENCE = object()


def _restore_and_fail(failure: str, *, chat_status: str):
    session = _local_owned_session()
    session.chat_status = chat_status
    mismatched = SimpleNamespace(
        machine_id="machine-2",
        allowed_root="C:\\Users\\Ada\\Projects",
        kill=AsyncMock(),
    )

    async def restore(_presence, binding):
        if failure == "different_binding":
            return binding.model_copy(update={"revision": binding.revision + 1})
        return binding

    async def lookup(_session_id: str, *, timeout: float):
        if timeout == 1.0 or failure == "no_data_channel":
            raise TimeoutError("not connected")
        return mismatched

    manager = MagicMock()
    manager.get_or_create_shim_for_session = AsyncMock(side_effect=lookup)
    detach = AsyncMock()
    with (
        patch(
            "backend.api.features.local_executor.routes.get_chat_session_metadata",
            AsyncMock(return_value=session),
        ),
        patch(
            "backend.api.features.local_executor.routes.get_machine_presence",
            AsyncMock(return_value=_PRESENCE),
        ),
        patch(
            "backend.api.features.local_executor.routes.restore_machine_session",
            AsyncMock(side_effect=restore),
        ),
        patch(
            "backend.api.features.local_executor.routes.detach_machine_session",
            detach,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = _make_client().post(
            "/api/copilot/sessions/session-1/executor/recording/start", json={}
        )
    return detach, mismatched, response
