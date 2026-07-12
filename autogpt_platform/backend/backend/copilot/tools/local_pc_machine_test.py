from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .local_pc_machine import (
    MachineConnectionStaleError,
    MachineSessionBinding,
    activate_machine_session,
    attach_machine_session,
    get_machine_presence,
    is_machine_presence,
    restore_machine_session,
)
from .local_pc_relay_protocol import RelayPresence, machine_scope_id


def _presence(**updates) -> RelayPresence:
    values = {
        "session_id": "machine-scope",
        "connection_id": "connection-1",
        "user_id": "user-1",
        "client_id": "autogpt-local-executor",
        "hello": {"connection_kind": "machine", "machine_id": "machine-1"},
        "expires_at": 9_999_999_999,
    }
    values.update(updates)
    return RelayPresence(**values)


def test_machine_scope_is_owner_and_machine_derived() -> None:
    assert machine_scope_id("user-1", "machine-1") == machine_scope_id(
        "user-1", "machine-1"
    )
    assert machine_scope_id("user-1", "machine-1") != machine_scope_id(
        "user-2", "machine-1"
    )


@pytest.mark.parametrize(
    "updates",
    [
        {"user_id": "other-user"},
        {"client_id": "other-client"},
        {"hello": {"connection_kind": "session", "machine_id": "machine-1"}},
        {"hello": {"connection_kind": "machine", "machine_id": "machine-2"}},
        {"expires_at": 0},
    ],
)
def test_machine_presence_rejects_wrong_owner_client_kind_or_machine(updates) -> None:
    assert not is_machine_presence(
        _presence(**updates),
        user_id="user-1",
        client_id="autogpt-local-executor",
        machine_id="machine-1",
    )


@pytest.mark.asyncio
async def test_expected_connection_id_rejects_stale_generation() -> None:
    relay = MagicMock()
    relay.get_presence = AsyncMock(return_value=_presence())
    with patch(
        "backend.copilot.tools.local_pc_machine.get_local_pc_relay",
        return_value=relay,
    ):
        with pytest.raises(MachineConnectionStaleError):
            await get_machine_presence(
                "user-1",
                "autogpt-local-executor",
                "machine-1",
                expected_connection_id="old-connection",
            )


def _binding() -> MachineSessionBinding:
    return MachineSessionBinding(
        session_id="session-1",
        allowed_root="/workspace",
        fingerprint="a" * 64,
        revision=1,
        root_grant="grant-1",
    )


@pytest.mark.asyncio
async def test_attach_sends_machine_generation() -> None:
    rpc = AsyncMock(
        return_value={"type": "SESSION_ATTACHED", "payload": _binding().model_dump()}
    )
    with patch(
        "backend.copilot.tools.local_pc_machine.machine_rpc",
        rpc,
    ):
        result = await attach_machine_session(
            _presence(),
            session_id="session-1",
            browse_id="browse-1",
            directory_ref="directory-1",
        )

    assert result == _binding()
    assert rpc.await_args.args[2]["expected_connection_id"] == "connection-1"


@pytest.mark.asyncio
async def test_activate_uses_minimal_wire_payload_and_preserves_root_grant() -> None:
    response = _binding().model_dump(exclude={"root_grant"})
    rpc = AsyncMock(return_value={"type": "SESSION_ACTIVATED", "payload": response})
    with patch(
        "backend.copilot.tools.local_pc_machine.machine_rpc",
        rpc,
    ):
        result = await activate_machine_session(_presence(), _binding())

    assert result == _binding()
    assert rpc.await_args.args[2] == {"session_id": "session-1", "revision": 1}


@pytest.mark.asyncio
async def test_restore_sends_only_durable_grant() -> None:
    rpc = AsyncMock(
        return_value={"type": "SESSION_RESTORED", "payload": _binding().model_dump()}
    )
    with patch(
        "backend.copilot.tools.local_pc_machine.machine_rpc",
        rpc,
    ):
        result = await restore_machine_session(_presence(), _binding())

    assert result == _binding()
    assert rpc.await_args.args[2] == {
        "session_id": "session-1",
        "root_grant": "grant-1",
    }
