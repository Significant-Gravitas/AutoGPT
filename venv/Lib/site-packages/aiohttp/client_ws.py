"""WebSocket client for asyncio."""

import asyncio
from typing import Any, Optional, cast

import async_timeout

from .client_exceptions import ClientError
from .client_reqrep import ClientResponse
from .helpers import call_later, set_result
from .http import (
    WS_CLOSED_MESSAGE,
    WS_CLOSING_MESSAGE,
    WebSocketError,
    WSCloseCode,
    WSMessage,
    WSMsgType,
)
from .http_websocket import WebSocketWriter  # WSMessage
from .streams import EofStream, FlowControlDataQueue
from .typedefs import (
    DEFAULT_JSON_DECODER,
    DEFAULT_JSON_ENCODER,
    JSONDecoder,
    JSONEncoder,
)


class ClientWebSocketResponse:
    def __init__(
        self,
        reader: "FlowControlDataQueue[WSMessage]",
        writer: WebSocketWriter,
        protocol: Optional[str],
        response: ClientResponse,
        timeout: float,
        autoclose: bool,
        autoping: bool,
        loop: asyncio.AbstractEventLoop,
        *,
        receive_timeout: Optional[float] = None,
        heartbeat: Optional[float] = None,
        compress: int = 0,
        client_notakeover: bool = False,
    ) -> None:
        self._response = response
        self._conn = response.connection

        self._writer = writer
        self._reader = reader
        self._protocol = protocol
        self._closed = False
        self._closing = False
        self._close_code: Optional[int] = None
        self._timeout = timeout
        self._receive_timeout = receive_timeout
        self._autoclose = autoclose
        self._autoping = autoping
        self._heartbeat = heartbeat
        self._heartbeat_cb: Optional[asyncio.TimerHandle] = None
        if heartbeat is not None:
            self._pong_heartbeat = heartbeat / 2.0
        self._pong_response_cb: Optional[asyncio.TimerHandle] = None
        self._loop = loop
        self._waiting: Optional[asyncio.Future[bool]] = None
        self._exception: Optional[BaseException] = None
        self._compress = compress
        self._client_notakeover = client_notakeover

        self._reset_heartbeat()

    def _cancel_heartbeat(self) -> None:
        if self._pong_response_cb is not None:
            self._pong_response_cb.cancel()
            self._pong_response_cb = None

        if self._heartbeat_cb is not None:
            self._heartbeat_cb.cancel()
            self._heartbeat_cb = None

    def _reset_heartbeat(self) -> None:
        self._cancel_heartbeat()

        if self._heartbeat is not None:
            self._heartbeat_cb = call_later(
                self._send_heartbeat, self._heartbeat, self._loop
            )

    def _send_heartbeat(self) -> None:
        if self._heartbeat is not None and not self._closed:
            # fire-and-forget a task is not perfect but maybe ok for
            # sending ping. Otherwise we need a long-living heartbeat
            # task in the class.
            self._loop.create_task(self._writer.ping())

            if self._pong_response_cb is not None:
                self._pong_response_cb.cancel()
            self._pong_response_cb = call_later(
                self._pong_not_received, self._pong_heartbeat, self._loop
            )

    def _pong_not_received(self) -> None:
        if not self._closed:
            self._closed = True
            self._close_code = WSCloseCode.ABNORMAL_CLOSURE
            self._exception = asyncio.TimeoutError()
            self._response.close()

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def close_code(self) -> Optional[int]:
        return self._close_code

    @property
    def protocol(self) -> Optional[str]:
        return self._protocol

    @property
    def compress(self) -> int:
        return self._compress

    @property
    def client_notakeover(self) -> bool:
        return self._client_notakeover

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """extra info from connection transport"""
        conn = self._response.connection
        if conn is None:
            return default
        transport = conn.transport
        if transport is None:
            return default
        return transport.get_extra_info(name, default)

    def exception(self) -> Optional[BaseException]:
        return self._exception

    async def ping(self, message: bytes = b"") -> None:
        await self._writer.ping(message)

    async def pong(self, message: bytes = b"") -> None:
        await self._writer.pong(message)

    async def send_str(self, data: str, compress: Optional[int] = None) -> None:
        if not isinstance(data, str):
            raise TypeError("data argument must be str (%r)" % type(data))
        await self._writer.send(data, binary=False, compress=compress)

    async def send_bytes(self, data: bytes, compress: Optional[int] = None) -> None:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise TypeError("data argument must be byte-ish (%r)" % type(data))
        await self._writer.send(data, binary=True, compress=compress)

    async def send_json(
        self,
        data: Any,
        compress: Optional[int] = None,
        *,
        dumps: JSONEncoder = DEFAULT_JSON_ENCODER,
    ) -> None:
        await self.send_str(dumps(data), compress=compress)

    async def close(self, *, code: int = WSCloseCode.OK, message: bytes = b"") -> bool:
        # we need to break `receive()` cycle first,
        # `close()` may be called from different task
        if self._waiting is not None and not self._closed:
            self._reader.feed_data(WS_CLOSING_MESSAGE, 0)
            await self._waiting

        if not self._closed:
            self._cancel_heartbeat()
            self._closed = True
            try:
                await self._writer.close(code, message)
            except asyncio.CancelledError:
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                self._response.close()
                raise
            except Exception as exc:
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                self._exception = exc
                self._response.close()
                return True

            if self._closing:
                self._response.close()
                return True

            while True:
                try:
                    async with async_timeout.timeout(self._timeout):
                        msg = await self._reader.read()
                except asyncio.CancelledError:
                    self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                    self._response.close()
                    raise
                except Exception as exc:
                    self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                    self._exception = exc
                    self._response.close()
                    return True

                if msg.type == WSMsgType.CLOSE:
                    self._close_code = msg.data
                    self._response.close()
                    return True
        else:
            return False

    async def receive(self, timeout: Optional[float] = None) -> WSMessage:
        while True:
            if self._waiting is not None:
                raise RuntimeError("Concurrent call to receive() is not allowed")

            if self._closed:
                return WS_CLOSED_MESSAGE
            elif self._closing:
                await self.close()
                return WS_CLOSED_MESSAGE

            try:
                self._waiting = self._loop.create_future()
                try:
                    async with async_timeout.timeout(timeout or self._receive_timeout):
                        msg = await self._reader.read()
                    self._reset_heartbeat()
                finally:
                    waiter = self._waiting
                    self._waiting = None
                    set_result(waiter, True)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                raise
            except EofStream:
                self._close_code = WSCloseCode.OK
                await self.close()
                return WSMessage(WSMsgType.CLOSED, None, None)
            except ClientError:
                self._closed = True
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                return WS_CLOSED_MESSAGE
            except WebSocketError as exc:
                self._close_code = exc.code
                await self.close(code=exc.code)
                return WSMessage(WSMsgType.ERROR, exc, None)
            except Exception as exc:
                self._exception = exc
                self._closing = True
                self._close_code = WSCloseCode.ABNORMAL_CLOSURE
                await self.close()
                return WSMessage(WSMsgType.ERROR, exc, None)

            if msg.type == WSMsgType.CLOSE:
                self._closing = True
                self._close_code = msg.data
                if not self._closed and self._autoclose:
                    await self.close()
            elif msg.type == WSMsgType.CLOSING:
                self._closing = True
            elif msg.type == WSMsgType.PING and self._autoping:
                await self.pong(msg.data)
                continue
            elif msg.type == WSMsgType.PONG and self._autoping:
                continue

            return msg

    async def receive_str(self, *, timeout: Optional[float] = None) -> str:
        msg = await self.receive(timeout)
        if msg.type != WSMsgType.TEXT:
            raise TypeError(f"Received message {msg.type}:{msg.data!r} is not str")
        return cast(str, msg.data)

    async def receive_bytes(self, *, timeout: Optional[float] = None) -> bytes:
        msg = await self.receive(timeout)
        if msg.type != WSMsgType.BINARY:
            raise TypeError(f"Received message {msg.type}:{msg.data!r} is not bytes")
        return cast(bytes, msg.data)

    async def receive_json(
        self,
        *,
        loads: JSONDecoder = DEFAULT_JSON_DECODER,
        timeout: Optional[float] = None,
    ) -> Any:
        data = await self.receive_str(timeout=timeout)
        return loads(data)

    def __aiter__(self) -> "ClientWebSocketResponse":
        return self

    async def __anext__(self) -> WSMessage:
        msg = await self.receive()
        if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSING, WSMsgType.CLOSED):
            raise StopAsyncIteration
        return msg
