"""Machine-control discovery and RPCs for persistent Local PC executors."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Any

from pydantic import BaseModel, Field

from backend.data.redis_client import get_redis_async

from .local_pc_relay import get_local_pc_relay
from .local_pc_relay_presence import MAX_DISCOVERY_PRESENCES, owner_presences
from .local_pc_relay_protocol import RelayPresence, machine_scope_id


class MachineControlError(RuntimeError):
    def __init__(
        self,
        code: str,
        message: str,
        *,
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = details or {}


class MachineNotConnectedError(MachineControlError):
    pass


class MachineConnectionStaleError(MachineControlError):
    pass


class MachineSessionBinding(BaseModel):
    session_id: str
    allowed_root: str = Field(min_length=1, max_length=32_767)
    fingerprint: str = Field(pattern=r"^[0-9a-f]{64}$")
    revision: int = Field(ge=1)
    root_grant: str = Field(min_length=1, max_length=131_072)


def is_machine_presence(
    presence: RelayPresence,
    *,
    user_id: str,
    client_id: str,
    machine_id: str | None = None,
) -> bool:
    hello = presence.hello
    return (
        presence.user_id == user_id
        and presence.client_id == client_id
        and presence.expires_at > time.time()
        and hello.get("connection_kind") == "machine"
        and isinstance(hello.get("machine_id"), str)
        and bool(hello["machine_id"])
        and (machine_id is None or hello["machine_id"] == machine_id)
    )


async def list_machine_presences(
    user_id: str,
    client_id: str,
) -> list[RelayPresence]:
    redis = await get_redis_async()
    found: list[RelayPresence] = []
    async for presence in owner_presences(
        redis,
        user_id,
        client_id,
        connection_kind="machine",
        limit=MAX_DISCOVERY_PRESENCES,
    ):
        if is_machine_presence(
            presence,
            user_id=user_id,
            client_id=client_id,
        ):
            found.append(presence)
    found.sort(
        key=lambda presence: (
            str(presence.hello.get("display_name") or "").casefold(),
            str(presence.hello.get("machine_id") or ""),
        )
    )
    return found


async def get_machine_presence(
    user_id: str,
    client_id: str,
    machine_id: str,
    *,
    expected_connection_id: str | None = None,
) -> RelayPresence:
    presence = await get_local_pc_relay().get_presence(
        machine_scope_id(user_id, machine_id)
    )
    if presence is None or not is_machine_presence(
        presence,
        user_id=user_id,
        client_id=client_id,
        machine_id=machine_id,
    ):
        raise MachineNotConnectedError(
            "MACHINE_NOT_CONNECTED",
            "The selected Local PC executor is not connected",
        )
    if (
        expected_connection_id is not None
        and presence.connection_id != expected_connection_id
    ):
        raise MachineConnectionStaleError(
            "MACHINE_CONNECTION_STALE",
            "The Local PC executor reconnected; refresh its folders and try again",
        )
    return presence


async def machine_rpc(
    presence: RelayPresence,
    message_type: str,
    payload: dict[str, Any],
    *,
    timeout: float = 10.0,
) -> dict[str, Any]:
    relay = get_local_pc_relay()
    current = await relay.get_presence(presence.session_id)
    if current is None or current.connection_id != presence.connection_id:
        raise MachineConnectionStaleError(
            "MACHINE_CONNECTION_STALE",
            "The Local PC executor reconnected before the request was sent",
        )

    transport = await relay.open_transport(presence)
    message_id = str(uuid.uuid4())
    envelope = json.dumps(
        {
            "type": message_type,
            "id": message_id,
            "ts": time.time(),
            "payload": payload,
        },
        separators=(",", ":"),
    )
    try:
        await transport.send_text(envelope)
        async with asyncio.timeout(timeout):
            async for raw in transport.iter_text():
                message = json.loads(raw)
                if not isinstance(message, dict) or message.get("id") != message_id:
                    continue
                response_payload = message.get("payload")
                if not isinstance(response_payload, dict):
                    raise MachineControlError(
                        "INVALID_MACHINE_RESPONSE",
                        "The Local PC executor returned an invalid response",
                    )
                if message.get("type") == "ERROR":
                    raise MachineControlError(
                        str(response_payload.get("code") or "MACHINE_REQUEST_FAILED"),
                        str(
                            response_payload.get("message")
                            or "The Local PC executor rejected the request"
                        ),
                        details=(
                            response_payload.get("details")
                            if isinstance(response_payload.get("details"), dict)
                            else None
                        ),
                    )
                current = await relay.get_presence(presence.session_id)
                if current is None or current.connection_id != presence.connection_id:
                    raise MachineConnectionStaleError(
                        "MACHINE_CONNECTION_STALE",
                        "The Local PC executor reconnected while the request was running",
                    )
                return message
    except TimeoutError as exc:
        raise MachineControlError(
            "MACHINE_REQUEST_TIMEOUT",
            "The Local PC executor did not respond in time",
        ) from exc
    except ConnectionError as exc:
        raise MachineConnectionStaleError(
            "MACHINE_CONNECTION_STALE",
            "The Local PC executor disconnected during the request",
        ) from exc
    finally:
        await transport.close()

    raise MachineControlError(
        "MACHINE_REQUEST_FAILED",
        "The Local PC executor closed the request without a response",
    )


async def attach_machine_session(
    presence: RelayPresence,
    *,
    session_id: str,
    browse_id: str,
    directory_ref: str,
) -> MachineSessionBinding:
    message = await machine_rpc(
        presence,
        "ATTACH_SESSION",
        {
            "session_id": session_id,
            "browse_id": browse_id,
            "directory_ref": directory_ref,
            "expected_connection_id": presence.connection_id,
        },
    )
    return MachineSessionBinding.model_validate(message["payload"])


async def activate_machine_session(
    presence: RelayPresence,
    binding: MachineSessionBinding,
) -> MachineSessionBinding:
    message = await machine_rpc(
        presence,
        "ACTIVATE_SESSION",
        {"session_id": binding.session_id, "revision": binding.revision},
    )
    return MachineSessionBinding.model_validate(
        {**message["payload"], "root_grant": binding.root_grant}
    )


async def restore_machine_session(
    presence: RelayPresence,
    binding: MachineSessionBinding,
) -> MachineSessionBinding:
    message = await machine_rpc(
        presence,
        "RESTORE_SESSION",
        {"session_id": binding.session_id, "root_grant": binding.root_grant},
    )
    return MachineSessionBinding.model_validate(message["payload"])


async def detach_machine_session(
    presence: RelayPresence,
    session_id: str,
) -> None:
    await machine_rpc(
        presence,
        "DETACH_SESSION",
        {"session_id": session_id},
        timeout=5.0,
    )
