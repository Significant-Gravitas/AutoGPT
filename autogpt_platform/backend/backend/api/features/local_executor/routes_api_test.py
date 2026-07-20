from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from autogpt_libs import auth
from fastapi import FastAPI
from fastapi.testclient import TestClient

from backend.api.features.local_executor.routes import router
from backend.api.features.local_executor.state import RecordingState
from backend.copilot.model import ChatSessionMetadata, LocalExecutionTargetMetadata
from backend.copilot.tools.local_pc_machine import MachineConnectionStaleError
from backend.copilot.tools.local_pc_relay_protocol import RelayPresence
from backend.copilot.tools.local_pc_shim import ShimHello, ShimRecordingError
from backend.copilot.tools.recording_models import (
    RecordingReviewApplied,
    RecordingSummary,
    WorkflowRecording,
)


@pytest.fixture(autouse=True)
def _enabled_local_executor_features():
    with (
        patch(
            "backend.api.features.local_executor.routes.is_local_executor_enabled",
            AsyncMock(return_value=True),
        ),
        patch(
            "backend.api.features.local_executor.routes.is_workflow_recording_enabled",
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
            "backend.api.features.local_executor.routes.get_shim_manager",
            return_value=manager,
        ),
    ):
        response = client.get("/api/copilot/sessions/session-1/executor")

    assert response.status_code == 200
    assert response.json()["kind"] == "none"
    manager.get_hello_async.assert_awaited_once_with("session-1")


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
        "computer_use_features": None,
        "computer_use_features_coarse": None,
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
            "computer_use_features": ["screenshot.capture"],
            "computer_use_features_coarse": ["screenshot"],
            "protocol_version": "1.1",
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
    assert response.json()["computer_use_features_coarse"] == ["screenshot"]


@pytest.mark.parametrize(("approved", "state"), [(True, "approved"), (False, "denied")])
def test_computer_use_consent_is_owner_scoped(approved: bool, state: str) -> None:
    client = _make_client()
    set_consent = AsyncMock(return_value=state)
    hello = ShimHello(
        machine_id="machine-1",
        capabilities=["computer_use"],
        computer_use_features=["screenshot.capture"],
        computer_use_features_coarse=["screenshot"],
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
                    "expected_features_coarse": ["screenshot"],
                    "expected_features": ["screenshot.capture"],
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
        features_coarse=["screenshot"],
        features=["screenshot.capture"],
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
                "expected_features_coarse": ["screenshot"],
                "expected_features": [],
            },
        )

    assert response.status_code == 409


@pytest.mark.parametrize(
    ("expected_machine_id", "expected_features_coarse", "expected_features"),
    [
        ("machine-2", ["screenshot"], ["input.click"]),
        ("machine-1", ["input", "screenshot"], ["input.click"]),
        ("machine-1", ["input.click", "screenshot"], []),
        ("machine-1", ["screenshot"], []),
    ],
)
def test_computer_use_approval_rejects_changed_executor_scope(
    expected_machine_id: str,
    expected_features_coarse: list[str],
    expected_features: list[str],
) -> None:
    client = _make_client()
    set_consent = AsyncMock()
    manager = MagicMock()
    manager.get_hello_async = AsyncMock(
        return_value=ShimHello(
            machine_id="machine-1",
            capabilities=["computer_use"],
            computer_use_features_coarse=["screenshot"],
            computer_use_features=["input.click"],
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
                "expected_features_coarse": expected_features_coarse,
                "expected_features": expected_features,
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
            "backend.api.features.local_executor.routes.register_recording_started",
            register_started,
        ),
        patch(
            "backend.api.features.local_executor.routes.mark_recording_started",
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
            "backend.api.features.local_executor.routes.mark_recording_started",
            AsyncMock(side_effect=ConnectionError("redis unavailable")),
        ),
    ):
        response = client.post(
            "/api/copilot/sessions/session-1/executor/recording/start", json={}
        )

    assert response.status_code == 503
    recording.stop.assert_awaited_once_with("rec-orphan")
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
            "backend.api.features.local_executor.routes.register_recording_stopped",
            register_stopped,
        ),
        patch(
            "backend.api.features.local_executor.routes.mark_recording_stopped",
            mark_stopped,
        ),
        patch(
            "backend.api.features.local_executor.routes.get_recording_state",
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
            "backend.api.features.local_executor.routes.get_recording_state",
            get_state,
        ),
        patch(
            "backend.api.features.local_executor.routes.mark_recording_stopped",
            mark_stopped,
        ),
        patch(
            "backend.api.features.local_executor.routes.register_recording_stopped",
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
            "backend.api.features.local_executor.routes.get_recording_state",
            AsyncMock(return_value=None),
        ),
        patch(
            "backend.api.features.local_executor.routes.mark_recording_stopped",
            AsyncMock(),
        ),
        patch(
            "backend.api.features.local_executor.routes.register_recording_stopped",
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
