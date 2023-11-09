import asyncio
from contextlib import suppress
from typing import Any, Optional, Tuple

from .base_protocol import BaseProtocol
from .client_exceptions import (
    ClientOSError,
    ClientPayloadError,
    ServerDisconnectedError,
    ServerTimeoutError,
)
from .helpers import BaseTimerContext
from .http import HttpResponseParser, RawResponseMessage
from .streams import EMPTY_PAYLOAD, DataQueue, StreamReader


class ResponseHandler(BaseProtocol, DataQueue[Tuple[RawResponseMessage, StreamReader]]):
    """Helper class to adapt between Protocol and StreamReader."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        BaseProtocol.__init__(self, loop=loop)
        DataQueue.__init__(self, loop)

        self._should_close = False

        self._payload: Optional[StreamReader] = None
        self._skip_payload = False
        self._payload_parser = None

        self._timer = None

        self._tail = b""
        self._upgraded = False
        self._parser: Optional[HttpResponseParser] = None

        self._read_timeout: Optional[float] = None
        self._read_timeout_handle: Optional[asyncio.TimerHandle] = None

    @property
    def upgraded(self) -> bool:
        return self._upgraded

    @property
    def should_close(self) -> bool:
        if self._payload is not None and not self._payload.is_eof() or self._upgraded:
            return True

        return (
            self._should_close
            or self._upgraded
            or self.exception() is not None
            or self._payload_parser is not None
            or len(self) > 0
            or bool(self._tail)
        )

    def force_close(self) -> None:
        self._should_close = True

    def close(self) -> None:
        transport = self.transport
        if transport is not None:
            transport.close()
            self.transport = None
            self._payload = None
            self._drop_timeout()

    def is_connected(self) -> bool:
        return self.transport is not None and not self.transport.is_closing()

    def connection_lost(self, exc: Optional[BaseException]) -> None:
        self._drop_timeout()

        if self._payload_parser is not None:
            with suppress(Exception):
                self._payload_parser.feed_eof()

        uncompleted = None
        if self._parser is not None:
            try:
                uncompleted = self._parser.feed_eof()
            except Exception:
                if self._payload is not None:
                    self._payload.set_exception(
                        ClientPayloadError("Response payload is not completed")
                    )

        if not self.is_eof():
            if isinstance(exc, OSError):
                exc = ClientOSError(*exc.args)
            if exc is None:
                exc = ServerDisconnectedError(uncompleted)
            # assigns self._should_close to True as side effect,
            # we do it anyway below
            self.set_exception(exc)

        self._should_close = True
        self._parser = None
        self._payload = None
        self._payload_parser = None
        self._reading_paused = False

        super().connection_lost(exc)

    def eof_received(self) -> None:
        # should call parser.feed_eof() most likely
        self._drop_timeout()

    def pause_reading(self) -> None:
        super().pause_reading()
        self._drop_timeout()

    def resume_reading(self) -> None:
        super().resume_reading()
        self._reschedule_timeout()

    def set_exception(self, exc: BaseException) -> None:
        self._should_close = True
        self._drop_timeout()
        super().set_exception(exc)

    def set_parser(self, parser: Any, payload: Any) -> None:
        # TODO: actual types are:
        #   parser: WebSocketReader
        #   payload: FlowControlDataQueue
        # but they are not generi enough
        # Need an ABC for both types
        self._payload = payload
        self._payload_parser = parser

        self._drop_timeout()

        if self._tail:
            data, self._tail = self._tail, b""
            self.data_received(data)

    def set_response_params(
        self,
        *,
        timer: Optional[BaseTimerContext] = None,
        skip_payload: bool = False,
        read_until_eof: bool = False,
        auto_decompress: bool = True,
        read_timeout: Optional[float] = None,
        read_bufsize: int = 2**16,
    ) -> None:
        self._skip_payload = skip_payload

        self._read_timeout = read_timeout
        self._reschedule_timeout()

        self._parser = HttpResponseParser(
            self,
            self._loop,
            read_bufsize,
            timer=timer,
            payload_exception=ClientPayloadError,
            response_with_body=not skip_payload,
            read_until_eof=read_until_eof,
            auto_decompress=auto_decompress,
        )

        if self._tail:
            data, self._tail = self._tail, b""
            self.data_received(data)

    def _drop_timeout(self) -> None:
        if self._read_timeout_handle is not None:
            self._read_timeout_handle.cancel()
            self._read_timeout_handle = None

    def _reschedule_timeout(self) -> None:
        timeout = self._read_timeout
        if self._read_timeout_handle is not None:
            self._read_timeout_handle.cancel()

        if timeout:
            self._read_timeout_handle = self._loop.call_later(
                timeout, self._on_read_timeout
            )
        else:
            self._read_timeout_handle = None

    def _on_read_timeout(self) -> None:
        exc = ServerTimeoutError("Timeout on reading data from socket")
        self.set_exception(exc)
        if self._payload is not None:
            self._payload.set_exception(exc)

    def data_received(self, data: bytes) -> None:
        self._reschedule_timeout()

        if not data:
            return

        # custom payload parser
        if self._payload_parser is not None:
            eof, tail = self._payload_parser.feed_data(data)
            if eof:
                self._payload = None
                self._payload_parser = None

                if tail:
                    self.data_received(tail)
            return
        else:
            if self._upgraded or self._parser is None:
                # i.e. websocket connection, websocket parser is not set yet
                self._tail += data
            else:
                # parse http messages
                try:
                    messages, upgraded, tail = self._parser.feed_data(data)
                except BaseException as exc:
                    if self.transport is not None:
                        # connection.release() could be called BEFORE
                        # data_received(), the transport is already
                        # closed in this case
                        self.transport.close()
                    # should_close is True after the call
                    self.set_exception(exc)
                    return

                self._upgraded = upgraded

                payload: Optional[StreamReader] = None
                for message, payload in messages:
                    if message.should_close:
                        self._should_close = True

                    self._payload = payload

                    if self._skip_payload or message.code in (204, 304):
                        self.feed_data((message, EMPTY_PAYLOAD), 0)
                    else:
                        self.feed_data((message, payload), 0)
                if payload is not None:
                    # new message(s) was processed
                    # register timeout handler unsubscribing
                    # either on end-of-stream or immediately for
                    # EMPTY_PAYLOAD
                    if payload is not EMPTY_PAYLOAD:
                        payload.on_eof(self._drop_timeout)
                    else:
                        self._drop_timeout()

                if tail:
                    if upgraded:
                        self.data_received(tail)
                    else:
                        self._tail = tail
