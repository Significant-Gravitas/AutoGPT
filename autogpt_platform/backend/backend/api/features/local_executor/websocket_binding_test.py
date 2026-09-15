from datetime import datetime, timezone
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient
from starlette.testclient import WebSocketDenialResponse
from starlette.websockets import WebSocketDisconnect

from backend.api.features.local_executor.routes_test import (
    _AUTH_HEADERS,
    _FAKE_TOKEN_INFO,
    _envelope,
    _make_app,
)
from backend.copilot.model import (
    ChatSessionInfo,
    ChatSessionMetadata,
    LocalExecutionTargetMetadata,
)
from backend.copilot.tools.local_pc_relay import RedisShimRelay
from backend.copilot.tools.local_pc_relay_test import FakeRedis
from backend.copilot.tools.local_pc_shim import ShimConnectionManager, ShimHello


def _session(*, local: bool) -> ChatSessionInfo:
    metadata = ChatSessionMetadata()
    if local:
        metadata.execution_target = LocalExecutionTargetMetadata(
            machine_id="chosen-machine",
            allowed_root="/chosen/workspace",
            directory_ref="chosen-directory",
            root_fingerprint="a" * 64,
            root_grant="chosen-grant",
        )
    return ChatSessionInfo(
        session_id="session-1",
        user_id="owner-1",
        usage=[],
        started_at=datetime.now(timezone.utc),
        updated_at=datetime.now(timezone.utc),
        metadata=metadata,
    )


@pytest.fixture
def authenticated_executor():
    manager = ShimConnectionManager(relay=RedisShimRelay(FakeRedis()))
    with (
        patch(
            "backend.api.features.local_executor.websocket.introspect_token",
            return_value=_FAKE_TOKEN_INFO,
        ),
        patch(
            "backend.api.features.local_executor.websocket.is_local_executor_enabled",
            return_value=True,
        ),
        patch(
            "backend.api.features.local_executor.websocket.get_shim_manager",
            return_value=manager,
        ),
    ):
        yield manager


def test_cloud_session_rejects_local_data_channel(authenticated_executor):
    with (
        patch(
            "backend.api.features.local_executor.websocket.get_chat_session_metadata",
            return_value=_session(local=False),
        ),
        TestClient(_make_app()) as client,
        pytest.raises(WebSocketDenialResponse) as rejected,
    ):
        with client.websocket_connect(
            "/ws/local-executor/session-1", headers=_AUTH_HEADERS
        ):
            pass

    assert rejected.value.status_code == 403
    assert authenticated_executor.get_hello("session-1") is None


@pytest.mark.parametrize(
    ("machine_id", "allowed_root"),
    [
        ("other-machine", "/chosen/workspace"),
        ("chosen-machine", "/other/workspace"),
    ],
)
def test_child_hello_cannot_replace_selected_binding(
    authenticated_executor, machine_id: str, allowed_root: str
):
    existing_hello = ShimHello(
        machine_id="chosen-machine",
        allowed_root="/chosen/workspace",
        platform="linux",
        arch="x86_64",
        capabilities=["shell", "files"],
    )
    authenticated_executor.remember_hello("session-1", existing_hello)
    with (
        patch(
            "backend.api.features.local_executor.websocket.get_chat_session_metadata",
            return_value=_session(local=True),
        ),
        TestClient(_make_app()) as client,
        pytest.raises(WebSocketDisconnect) as rejected,
    ):
        with client.websocket_connect(
            "/ws/local-executor/session-1", headers=_AUTH_HEADERS
        ) as websocket:
            websocket.send_text(
                _envelope(
                    "HELLO",
                    {
                        "machine_id": machine_id,
                        "allowed_root": allowed_root,
                        "platform": "linux",
                        "arch": "x86_64",
                        "capabilities": ["shell", "files"],
                        "protocol_version": "1.1",
                    },
                )
            )
            websocket.receive_text()

    assert rejected.value.code == 4403
    assert authenticated_executor.get_hello("session-1") == existing_hello
