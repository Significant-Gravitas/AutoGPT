"""
ShimDaemon — the main event loop.

Connects to the AutoGPT platform WebSocket, dispatches incoming messages
to the appropriate handler, and sends back results.

Usage:
    daemon = ShimDaemon(config=ShimConfig(), token_store=KeychainTokenStore())
    asyncio.run(daemon.run())
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
import uuid
from typing import Any

# TODO: pip install websockets
# import websockets

from .config import ShimConfig
from .handlers import CommandHandler, FileHandler, ComputerUseHandler
from .protocol import MessageType, build_hello, parse_message

logger = logging.getLogger(__name__)


class ShimDaemon:
    """
    Main daemon class. Maintains the WebSocket connection to the platform
    and dispatches messages to capability handlers.

    Lifecycle:
        1. __init__: set up handlers
        2. run(): connect loop with exponential backoff
        3. _session(): single connected session
        4. _dispatch(): route a single message to the right handler
        5. On disconnect: back to run() reconnect loop
    """

    def __init__(self, config: ShimConfig, token_store: Any) -> None:
        self.config = config
        self.token_store = token_store
        self._handlers = {
            MessageType.EXECUTE_COMMAND: CommandHandler(config),
            MessageType.FILE_READ: FileHandler(config),
            MessageType.FILE_WRITE: FileHandler(config),
            MessageType.SCREENSHOT_REQUEST: ComputerUseHandler(config),
            MessageType.INPUT_ACTION: ComputerUseHandler(config),
        }
        self._running = False
        self._ws = None  # active websocket connection

    async def run(self) -> None:
        """
        Outer reconnect loop. Runs until stop() is called.

        Uses exponential backoff with jitter so that all shims don't
        reconnect simultaneously after a platform restart.
        """
        self._running = True
        attempt = 0
        while self._running:
            try:
                await self._session()
                attempt = 0  # reset on clean disconnect
            except Exception as exc:
                delay = min(
                    self.config.reconnect_base_delay * (2 ** attempt),
                    self.config.reconnect_max_delay,
                ) + random.uniform(0, 5)
                logger.warning(
                    "Disconnected (%s). Reconnecting in %.1fs (attempt %d)",
                    exc, delay, attempt + 1,
                )
                await asyncio.sleep(delay)
                attempt += 1

    async def stop(self) -> None:
        """Gracefully stop the daemon and close the WebSocket."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _session(self) -> None:
        """
        A single connected session with the platform.

        1. Fetches a fresh access token from the token store
        2. Opens the WebSocket with Authorization header
        3. Sends HELLO, waits for HELLO_ACK
        4. Dispatches messages until disconnect
        """
        token = await self.token_store.get_access_token()
        url = f"{self.config.platform_ws_url}/{_current_session_id()}"

        # TODO: replace stub with real websockets.connect()
        logger.info("Connecting to %s", url)
        raise NotImplementedError(
            "WebSocket connection not yet implemented. "
            "See docs/PROTOCOL.md for the message format."
        )

        # Pseudocode for the real implementation:
        #
        # async with websockets.connect(
        #     url,
        #     extra_headers={"Authorization": f"Bearer {token}"},
        # ) as ws:
        #     self._ws = ws
        #     await self._handshake(ws)
        #     async for raw in ws:
        #         msg = parse_message(raw)
        #         asyncio.create_task(self._dispatch(ws, msg))

    async def _handshake(self, ws: Any) -> dict:
        """
        Send HELLO and wait for HELLO_ACK.
        Raises on auth failure or unsupported protocol version.
        """
        hello = build_hello(self.config)
        await ws.send(json.dumps(hello))
        raw_ack = await asyncio.wait_for(ws.recv(), timeout=10.0)
        ack = parse_message(raw_ack)
        if ack["type"] != MessageType.HELLO_ACK:
            raise ValueError(f"Expected HELLO_ACK, got {ack['type']}")
        logger.info(
            "Connected. Granted capabilities: %s",
            ack["payload"]["granted_capabilities"],
        )
        return ack

    async def _dispatch(self, ws: Any, msg: dict) -> None:
        """
        Route a single incoming message to the appropriate handler.
        Send the result (or ERROR) back over the WebSocket.
        """
        msg_type = msg.get("type")
        msg_id = msg.get("id")

        if msg_type == MessageType.PING:
            await ws.send(json.dumps({
                "type": MessageType.PONG,
                "id": msg_id,
                "ts": time.time(),
                "payload": {},
            }))
            return

        handler = self._handlers.get(msg_type)
        if handler is None:
            logger.warning("Unknown message type: %s", msg_type)
            return

        try:
            result = await handler.handle(msg)
            await ws.send(json.dumps(result))
        except Exception as exc:
            logger.exception("Handler error for %s", msg_type)
            await ws.send(json.dumps({
                "type": MessageType.ERROR,
                "id": msg_id,
                "ts": time.time(),
                "payload": {
                    "code": "INTERNAL_ERROR",
                    "message": str(exc),
                    "fatal": False,
                },
            }))


def _current_session_id() -> str:
    """
    TODO: retrieve the session_id the platform assigned to this shim connection.
    For now returns a placeholder.
    """
    return "SESSION_ID_PLACEHOLDER"
