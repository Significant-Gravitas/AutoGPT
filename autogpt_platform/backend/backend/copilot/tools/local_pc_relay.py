"""Redis-backed transport between executor workers and Local PC WebSockets."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Any

from backend.data.redis_client import get_redis_async

from .local_pc_relay_presence import (
    clear_presence,
    owner_presences,
    read_presence,
    refresh_presence,
    register_presence,
)
from .local_pc_relay_protocol import (
    PLATFORM_REQUEST_TYPES,
    SHIM_RESPONSE_TYPES,
    RelayConnectionReplaced,
    RelayPresence,
    RelayWebSocket,
    decode_stream_entries,
    stream_key,
    validate_envelope,
)
from .local_pc_relay_routing import RelayResponseRouter
from .local_pc_relay_transport import (
    POLL_BLOCK_MS,
    RedisRelayTransport,
    acknowledge_stream_entry,
    append_stream_envelope,
    delete_stream_entry,
)

logger = logging.getLogger(__name__)

_HEARTBEAT_SECONDS = 10.0


class RedisShimRelay:
    def __init__(self, redis_client: Any | None = None) -> None:
        self._redis_client = redis_client

    async def redis(self) -> Any:
        return self._redis_client or await get_redis_async()

    async def get_presence(self, session_id: str) -> RelayPresence | None:
        return await read_presence(await self.redis(), session_id)

    async def wait_for_presence(
        self, session_id: str, *, timeout: float
    ) -> RelayPresence:
        deadline = asyncio.get_running_loop().time() + timeout
        while True:
            presence = await self.get_presence(session_id)
            if presence is not None:
                return presence
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                raise TimeoutError(
                    f"[LocalPC] Shim for session {session_id[:12]} did not connect "
                    f"within {timeout}s"
                )
            await asyncio.sleep(min(0.1, remaining))

    async def open_transport(self, presence: RelayPresence) -> RedisRelayTransport:
        return RedisRelayTransport(self, presence)

    async def is_current(self, presence: RelayPresence) -> bool:
        current = await self.get_presence(presence.session_id)
        return current is not None and current.connection_id == presence.connection_id

    async def _register_presence(
        self,
        session_id: str,
        *,
        hello: dict[str, Any],
        user_id: str,
        client_id: str,
        connection_id: str | None = None,
    ) -> tuple[RelayPresence, RelayPresence | None]:
        return await register_presence(
            await self.redis(),
            session_id,
            hello=hello,
            user_id=user_id,
            client_id=client_id,
            connection_id=connection_id,
        )

    async def _refresh_presence(self, presence: RelayPresence) -> bool:
        return await refresh_presence(await self.redis(), presence)

    async def _clear_presence(self, presence: RelayPresence) -> None:
        await clear_presence(await self.redis(), presence)

    async def _publish_control(self, presence: RelayPresence, *, reason: str) -> None:
        redis = await self.redis()
        key = stream_key(presence.session_id, presence.connection_id, "control")
        envelope = json.dumps(
            {
                "type": "SESSION_REVOKED",
                "id": str(uuid.uuid4()),
                "ts": time.time(),
                "payload": {"reason": reason},
            }
        )
        await append_stream_envelope(redis, key, envelope, maxlen=10)

    async def serve_websocket(
        self,
        session_id: str,
        websocket: RelayWebSocket,
        *,
        hello: dict[str, Any],
        user_id: str,
        client_id: str,
        connection_id: str | None = None,
    ) -> None:
        presence, previous = await self._register_presence(
            session_id,
            hello=hello,
            user_id=user_id,
            client_id=client_id,
            connection_id=connection_id,
        )
        if previous is not None and previous.connection_id != presence.connection_id:
            await self._publish_control(previous, reason="another_shim_connected")

        redis = await self.redis()
        send_lock = asyncio.Lock()
        response_router = RelayResponseRouter(
            redis,
            session_id=session_id,
            connection_id=presence.connection_id,
        )

        async def send_to_shim(raw: str) -> None:
            async with send_lock:
                await websocket.send_text(raw)

        async def pump_requests() -> None:
            key = stream_key(session_id, presence.connection_id, "requests")
            cursor = "0-0"
            while True:
                if not await self.is_current(presence):
                    raise RelayConnectionReplaced
                result = await redis.xread(
                    streams={key: cursor}, block=POLL_BLOCK_MS, count=100
                )
                for message_id, fields in decode_stream_entries(result):
                    cursor = message_id
                    envelope = fields.get("envelope")
                    if envelope is None:
                        await acknowledge_stream_entry(redis, key, message_id, fields)
                        continue
                    message: dict[str, Any] | None = None
                    try:
                        message = validate_envelope(envelope, PLATFORM_REQUEST_TYPES)
                        reply_id = fields.get("reply_id")
                        if reply_id is None:
                            raise ValueError(
                                "Local executor request has no reply stream"
                            )
                        response_router.register_request(message, reply_id)
                        await send_to_shim(envelope)
                    except Exception:
                        if message is not None:
                            response_router.remove_request(message["id"])
                        raise
                    finally:
                        await acknowledge_stream_entry(redis, key, message_id, fields)

        async def pump_responses() -> None:
            async for raw in websocket.iter_text():
                message = validate_envelope(raw, SHIM_RESPONSE_TYPES)
                await response_router.route(raw, message)

        async def pump_control() -> None:
            key = stream_key(session_id, presence.connection_id, "control")
            cursor = "0-0"
            while True:
                result = await redis.xread(
                    streams={key: cursor}, block=POLL_BLOCK_MS, count=10
                )
                for message_id, fields in decode_stream_entries(result):
                    cursor = message_id
                    envelope = fields.get("envelope")
                    if envelope is None:
                        await delete_stream_entry(redis, key, message_id)
                        continue
                    try:
                        message = json.loads(envelope)
                        await send_to_shim(envelope)
                    finally:
                        await delete_stream_entry(redis, key, message_id)
                    reason = str((message.get("payload") or {}).get("reason") or "")
                    close_code = 4427 if reason == "another_shim_connected" else 4428
                    await websocket.close(code=close_code, reason=reason)
                    return

        async def heartbeat() -> None:
            while True:
                await asyncio.sleep(_HEARTBEAT_SECONDS)
                if not await self._refresh_presence(presence):
                    raise RelayConnectionReplaced

        tasks = {
            asyncio.create_task(pump_requests()),
            asyncio.create_task(pump_responses()),
            asyncio.create_task(pump_control()),
            asyncio.create_task(heartbeat()),
        }
        try:
            done, pending = await asyncio.wait(
                tasks, return_when=asyncio.FIRST_COMPLETED
            )
            for task in done:
                task.result()
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
        except RelayConnectionReplaced:
            envelope = json.dumps(
                {
                    "type": "SESSION_REVOKED",
                    "id": str(uuid.uuid4()),
                    "ts": time.time(),
                    "payload": {"reason": "another_shim_connected"},
                }
            )
            try:
                await send_to_shim(envelope)
                await websocket.close(
                    code=4427, reason="Another shim connected for this session"
                )
            except Exception:
                pass
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            await self._clear_presence(presence)

    async def revoke_owner(
        self, user_id: str, client_id: str | None, *, reason: str
    ) -> int:
        redis = await self.redis()
        notified = 0
        async for presence in owner_presences(redis, user_id, client_id):
            await self._publish_control(presence, reason=reason)
            notified += 1
        return notified


_relay: RedisShimRelay | None = None


def get_local_pc_relay() -> RedisShimRelay:
    global _relay
    if _relay is None:
        _relay = RedisShimRelay()
    return _relay
