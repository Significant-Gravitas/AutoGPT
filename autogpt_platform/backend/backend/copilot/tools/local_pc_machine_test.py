import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .local_pc_machine import (
    MachineConnectionStaleError,
    MachineControlError,
    MachineSessionBinding,
    activate_machine_session,
    attach_machine_session,
    get_machine_presence,
    is_machine_presence,
    machine_rpc,
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


@pytest.mark.asyncio
@pytest.mark.parametrize("invalid", ["json", "type", "payload", "object"])
async def test_machine_rpc_rejects_invalid_response_and_closes_transport(invalid: str):
    transport = MagicMock()
    transport.send_text = AsyncMock()
    transport.close = AsyncMock()

    async def responses():
        request = json.loads(transport.send_text.await_args.args[0])
        if invalid == "json":
            yield "{"
        elif invalid == "object":
            yield "[]"
        else:
            yield json.dumps(
                {
                    "id": request["id"],
                    "type": "ACK" if invalid == "type" else "DIRECTORY_LIST_RESPONSE",
                    "payload": [] if invalid == "payload" else {},
                }
            )

    transport.iter_text = responses
    relay = MagicMock()
    relay.get_presence = AsyncMock(return_value=_presence())
    relay.open_transport = AsyncMock(return_value=transport)
    with patch(
        "backend.copilot.tools.local_pc_machine.get_local_pc_relay", return_value=relay
    ):
        with pytest.raises(MachineControlError) as error:
            await machine_rpc(_presence(), "DIRECTORY_LIST_REQUEST", {})
    assert error.value.code == "INVALID_MACHINE_RESPONSE"
    transport.close.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["attach", "activate", "restore"])
async def test_binding_response_must_match_requested_session(operation: str):
    payload = _binding().model_dump()
    payload["session_id"] = "another-session"
    rpc = AsyncMock(return_value={"payload": payload})
    with patch("backend.copilot.tools.local_pc_machine.machine_rpc", rpc):
        with pytest.raises(MachineControlError) as error:
            if operation == "attach":
                await attach_machine_session(
                    _presence(),
                    session_id="session-1",
                    browse_id="browse-1",
                    directory_ref="dir-1",
                )
            elif operation == "activate":
                await activate_machine_session(_presence(), _binding())
            else:
                await restore_machine_session(_presence(), _binding())
    assert error.value.code == "INVALID_MACHINE_RESPONSE"


@pytest.mark.asyncio
async def test_only_the_machines_own_error_reply_counts_as_a_refusal():
    """A refusal proves the machine applied nothing; a request it never
    answered may have been applied, and callers must undo it."""

    def transport_answering(reply):
        transport = MagicMock()
        transport.send_text = AsyncMock()
        transport.close = AsyncMock()

        async def responses():
            request = json.loads(transport.send_text.await_args.args[0])
            if reply is None:
                await asyncio.Event().wait()
            yield json.dumps({"id": request["id"], **reply})

        transport.iter_text = responses
        return transport

    async def call(transport):
        relay = MagicMock()
        relay.get_presence = AsyncMock(return_value=_presence())
        relay.open_transport = AsyncMock(return_value=transport)
        with patch(
            "backend.copilot.tools.local_pc_machine.get_local_pc_relay",
            return_value=relay,
        ):
            with pytest.raises(MachineControlError) as error:
                await machine_rpc(_presence(), "ATTACH_SESSION", {}, timeout=0.05)
        return error.value

    refused = await call(
        transport_answering(
            {"type": "ERROR", "payload": {"code": "SESSION_ALREADY_ACTIVE"}}
        )
    )
    unanswered = await call(transport_answering(None))

    assert (refused.code, refused.refused) == ("SESSION_ALREADY_ACTIVE", True)
    assert (unanswered.code, unanswered.refused) == ("MACHINE_REQUEST_TIMEOUT", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("stalled", ["presence", "send"])
async def test_the_rpc_deadline_covers_the_redis_calls_before_the_send(stalled: str):
    """A Redis call that stalls before the request goes out still ends at the
    RPC's deadline, not at whatever the Redis client's own timeout is."""

    async def stall(*_args: object) -> None:
        await asyncio.Event().wait()

    transport = MagicMock()
    transport.close = AsyncMock()
    transport.send_text = AsyncMock(side_effect=stall if stalled == "send" else None)
    relay = MagicMock()
    relay.get_presence = AsyncMock(
        side_effect=stall if stalled == "presence" else None,
        return_value=_presence(),
    )
    relay.open_transport = AsyncMock(return_value=transport)

    with patch(
        "backend.copilot.tools.local_pc_machine.get_local_pc_relay", return_value=relay
    ):
        with pytest.raises(MachineControlError) as error:
            await asyncio.wait_for(
                machine_rpc(_presence(), "DIRECTORY_LIST_REQUEST", {}, timeout=0.05),
                timeout=5,
            )

    assert error.value.code == "MACHINE_REQUEST_TIMEOUT"
    transport.close.assert_awaited_once()
