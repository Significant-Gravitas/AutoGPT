"""WebSocket protocol versions 13 and 8."""

import asyncio
import collections
import json
import random
import re
import sys
import zlib
from enum import IntEnum
from struct import Struct
from typing import Any, Callable, List, Optional, Pattern, Set, Tuple, Union, cast

from .base_protocol import BaseProtocol
from .helpers import NO_EXTENSIONS
from .streams import DataQueue
from .typedefs import Final

__all__ = (
    "WS_CLOSED_MESSAGE",
    "WS_CLOSING_MESSAGE",
    "WS_KEY",
    "WebSocketReader",
    "WebSocketWriter",
    "WSMessage",
    "WebSocketError",
    "WSMsgType",
    "WSCloseCode",
)


class WSCloseCode(IntEnum):
    OK = 1000
    GOING_AWAY = 1001
    PROTOCOL_ERROR = 1002
    UNSUPPORTED_DATA = 1003
    ABNORMAL_CLOSURE = 1006
    INVALID_TEXT = 1007
    POLICY_VIOLATION = 1008
    MESSAGE_TOO_BIG = 1009
    MANDATORY_EXTENSION = 1010
    INTERNAL_ERROR = 1011
    SERVICE_RESTART = 1012
    TRY_AGAIN_LATER = 1013
    BAD_GATEWAY = 1014


ALLOWED_CLOSE_CODES: Final[Set[int]] = {int(i) for i in WSCloseCode}


class WSMsgType(IntEnum):
    # websocket spec types
    CONTINUATION = 0x0
    TEXT = 0x1
    BINARY = 0x2
    PING = 0x9
    PONG = 0xA
    CLOSE = 0x8

    # aiohttp specific types
    CLOSING = 0x100
    CLOSED = 0x101
    ERROR = 0x102

    text = TEXT
    binary = BINARY
    ping = PING
    pong = PONG
    close = CLOSE
    closing = CLOSING
    closed = CLOSED
    error = ERROR


WS_KEY: Final[bytes] = b"258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


UNPACK_LEN2 = Struct("!H").unpack_from
UNPACK_LEN3 = Struct("!Q").unpack_from
UNPACK_CLOSE_CODE = Struct("!H").unpack
PACK_LEN1 = Struct("!BB").pack
PACK_LEN2 = Struct("!BBH").pack
PACK_LEN3 = Struct("!BBQ").pack
PACK_CLOSE_CODE = Struct("!H").pack
MSG_SIZE: Final[int] = 2**14
DEFAULT_LIMIT: Final[int] = 2**16


_WSMessageBase = collections.namedtuple("_WSMessageBase", ["type", "data", "extra"])


class WSMessage(_WSMessageBase):
    def json(self, *, loads: Callable[[Any], Any] = json.loads) -> Any:
        """Return parsed JSON data.

        .. versionadded:: 0.22
        """
        return loads(self.data)


WS_CLOSED_MESSAGE = WSMessage(WSMsgType.CLOSED, None, None)
WS_CLOSING_MESSAGE = WSMessage(WSMsgType.CLOSING, None, None)


class WebSocketError(Exception):
    """WebSocket protocol parser error."""

    def __init__(self, code: int, message: str) -> None:
        self.code = code
        super().__init__(code, message)

    def __str__(self) -> str:
        return cast(str, self.args[1])


class WSHandshakeError(Exception):
    """WebSocket protocol handshake error."""


native_byteorder: Final[str] = sys.byteorder


# Used by _websocket_mask_python
_XOR_TABLE: Final[List[bytes]] = [bytes(a ^ b for a in range(256)) for b in range(256)]


def _websocket_mask_python(mask: bytes, data: bytearray) -> None:
    """Websocket masking function.

    `mask` is a `bytes` object of length 4; `data` is a `bytearray`
    object of any length. The contents of `data` are masked with `mask`,
    as specified in section 5.3 of RFC 6455.

    Note that this function mutates the `data` argument.

    This pure-python implementation may be replaced by an optimized
    version when available.

    """
    assert isinstance(data, bytearray), data
    assert len(mask) == 4, mask

    if data:
        a, b, c, d = (_XOR_TABLE[n] for n in mask)
        data[::4] = data[::4].translate(a)
        data[1::4] = data[1::4].translate(b)
        data[2::4] = data[2::4].translate(c)
        data[3::4] = data[3::4].translate(d)


if NO_EXTENSIONS:  # pragma: no cover
    _websocket_mask = _websocket_mask_python
else:
    try:
        from ._websocket import _websocket_mask_cython  # type: ignore[import]

        _websocket_mask = _websocket_mask_cython
    except ImportError:  # pragma: no cover
        _websocket_mask = _websocket_mask_python

_WS_DEFLATE_TRAILING: Final[bytes] = bytes([0x00, 0x00, 0xFF, 0xFF])


_WS_EXT_RE: Final[Pattern[str]] = re.compile(
    r"^(?:;\s*(?:"
    r"(server_no_context_takeover)|"
    r"(client_no_context_takeover)|"
    r"(server_max_window_bits(?:=(\d+))?)|"
    r"(client_max_window_bits(?:=(\d+))?)))*$"
)

_WS_EXT_RE_SPLIT: Final[Pattern[str]] = re.compile(r"permessage-deflate([^,]+)?")


def ws_ext_parse(extstr: Optional[str], isserver: bool = False) -> Tuple[int, bool]:
    if not extstr:
        return 0, False

    compress = 0
    notakeover = False
    for ext in _WS_EXT_RE_SPLIT.finditer(extstr):
        defext = ext.group(1)
        # Return compress = 15 when get `permessage-deflate`
        if not defext:
            compress = 15
            break
        match = _WS_EXT_RE.match(defext)
        if match:
            compress = 15
            if isserver:
                # Server never fail to detect compress handshake.
                # Server does not need to send max wbit to client
                if match.group(4):
                    compress = int(match.group(4))
                    # Group3 must match if group4 matches
                    # Compress wbit 8 does not support in zlib
                    # If compress level not support,
                    # CONTINUE to next extension
                    if compress > 15 or compress < 9:
                        compress = 0
                        continue
                if match.group(1):
                    notakeover = True
                # Ignore regex group 5 & 6 for client_max_window_bits
                break
            else:
                if match.group(6):
                    compress = int(match.group(6))
                    # Group5 must match if group6 matches
                    # Compress wbit 8 does not support in zlib
                    # If compress level not support,
                    # FAIL the parse progress
                    if compress > 15 or compress < 9:
                        raise WSHandshakeError("Invalid window size")
                if match.group(2):
                    notakeover = True
                # Ignore regex group 5 & 6 for client_max_window_bits
                break
        # Return Fail if client side and not match
        elif not isserver:
            raise WSHandshakeError("Extension for deflate not supported" + ext.group(1))

    return compress, notakeover


def ws_ext_gen(
    compress: int = 15, isserver: bool = False, server_notakeover: bool = False
) -> str:
    # client_notakeover=False not used for server
    # compress wbit 8 does not support in zlib
    if compress < 9 or compress > 15:
        raise ValueError(
            "Compress wbits must between 9 and 15, " "zlib does not support wbits=8"
        )
    enabledext = ["permessage-deflate"]
    if not isserver:
        enabledext.append("client_max_window_bits")

    if compress < 15:
        enabledext.append("server_max_window_bits=" + str(compress))
    if server_notakeover:
        enabledext.append("server_no_context_takeover")
    # if client_notakeover:
    #     enabledext.append('client_no_context_takeover')
    return "; ".join(enabledext)


class WSParserState(IntEnum):
    READ_HEADER = 1
    READ_PAYLOAD_LENGTH = 2
    READ_PAYLOAD_MASK = 3
    READ_PAYLOAD = 4


class WebSocketReader:
    def __init__(
        self, queue: DataQueue[WSMessage], max_msg_size: int, compress: bool = True
    ) -> None:
        self.queue = queue
        self._max_msg_size = max_msg_size

        self._exc: Optional[BaseException] = None
        self._partial = bytearray()
        self._state = WSParserState.READ_HEADER

        self._opcode: Optional[int] = None
        self._frame_fin = False
        self._frame_opcode: Optional[int] = None
        self._frame_payload = bytearray()

        self._tail = b""
        self._has_mask = False
        self._frame_mask: Optional[bytes] = None
        self._payload_length = 0
        self._payload_length_flag = 0
        self._compressed: Optional[bool] = None
        self._decompressobj: Any = None  # zlib.decompressobj actually
        self._compress = compress

    def feed_eof(self) -> None:
        self.queue.feed_eof()

    def feed_data(self, data: bytes) -> Tuple[bool, bytes]:
        if self._exc:
            return True, data

        try:
            return self._feed_data(data)
        except Exception as exc:
            self._exc = exc
            self.queue.set_exception(exc)
            return True, b""

    def _feed_data(self, data: bytes) -> Tuple[bool, bytes]:
        for fin, opcode, payload, compressed in self.parse_frame(data):
            if compressed and not self._decompressobj:
                self._decompressobj = zlib.decompressobj(wbits=-zlib.MAX_WBITS)
            if opcode == WSMsgType.CLOSE:
                if len(payload) >= 2:
                    close_code = UNPACK_CLOSE_CODE(payload[:2])[0]
                    if close_code < 3000 and close_code not in ALLOWED_CLOSE_CODES:
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            f"Invalid close code: {close_code}",
                        )
                    try:
                        close_message = payload[2:].decode("utf-8")
                    except UnicodeDecodeError as exc:
                        raise WebSocketError(
                            WSCloseCode.INVALID_TEXT, "Invalid UTF-8 text message"
                        ) from exc
                    msg = WSMessage(WSMsgType.CLOSE, close_code, close_message)
                elif payload:
                    raise WebSocketError(
                        WSCloseCode.PROTOCOL_ERROR,
                        f"Invalid close frame: {fin} {opcode} {payload!r}",
                    )
                else:
                    msg = WSMessage(WSMsgType.CLOSE, 0, "")

                self.queue.feed_data(msg, 0)

            elif opcode == WSMsgType.PING:
                self.queue.feed_data(
                    WSMessage(WSMsgType.PING, payload, ""), len(payload)
                )

            elif opcode == WSMsgType.PONG:
                self.queue.feed_data(
                    WSMessage(WSMsgType.PONG, payload, ""), len(payload)
                )

            elif (
                opcode not in (WSMsgType.TEXT, WSMsgType.BINARY)
                and self._opcode is None
            ):
                raise WebSocketError(
                    WSCloseCode.PROTOCOL_ERROR, f"Unexpected opcode={opcode!r}"
                )
            else:
                # load text/binary
                if not fin:
                    # got partial frame payload
                    if opcode != WSMsgType.CONTINUATION:
                        self._opcode = opcode
                    self._partial.extend(payload)
                    if self._max_msg_size and len(self._partial) >= self._max_msg_size:
                        raise WebSocketError(
                            WSCloseCode.MESSAGE_TOO_BIG,
                            "Message size {} exceeds limit {}".format(
                                len(self._partial), self._max_msg_size
                            ),
                        )
                else:
                    # previous frame was non finished
                    # we should get continuation opcode
                    if self._partial:
                        if opcode != WSMsgType.CONTINUATION:
                            raise WebSocketError(
                                WSCloseCode.PROTOCOL_ERROR,
                                "The opcode in non-fin frame is expected "
                                "to be zero, got {!r}".format(opcode),
                            )

                    if opcode == WSMsgType.CONTINUATION:
                        assert self._opcode is not None
                        opcode = self._opcode
                        self._opcode = None

                    self._partial.extend(payload)
                    if self._max_msg_size and len(self._partial) >= self._max_msg_size:
                        raise WebSocketError(
                            WSCloseCode.MESSAGE_TOO_BIG,
                            "Message size {} exceeds limit {}".format(
                                len(self._partial), self._max_msg_size
                            ),
                        )

                    # Decompress process must to be done after all packets
                    # received.
                    if compressed:
                        self._partial.extend(_WS_DEFLATE_TRAILING)
                        payload_merged = self._decompressobj.decompress(
                            self._partial, self._max_msg_size
                        )
                        if self._decompressobj.unconsumed_tail:
                            left = len(self._decompressobj.unconsumed_tail)
                            raise WebSocketError(
                                WSCloseCode.MESSAGE_TOO_BIG,
                                "Decompressed message size {} exceeds limit {}".format(
                                    self._max_msg_size + left, self._max_msg_size
                                ),
                            )
                    else:
                        payload_merged = bytes(self._partial)

                    self._partial.clear()

                    if opcode == WSMsgType.TEXT:
                        try:
                            text = payload_merged.decode("utf-8")
                            self.queue.feed_data(
                                WSMessage(WSMsgType.TEXT, text, ""), len(text)
                            )
                        except UnicodeDecodeError as exc:
                            raise WebSocketError(
                                WSCloseCode.INVALID_TEXT, "Invalid UTF-8 text message"
                            ) from exc
                    else:
                        self.queue.feed_data(
                            WSMessage(WSMsgType.BINARY, payload_merged, ""),
                            len(payload_merged),
                        )

        return False, b""

    def parse_frame(
        self, buf: bytes
    ) -> List[Tuple[bool, Optional[int], bytearray, Optional[bool]]]:
        """Return the next frame from the socket."""
        frames = []
        if self._tail:
            buf, self._tail = self._tail + buf, b""

        start_pos = 0
        buf_length = len(buf)

        while True:
            # read header
            if self._state == WSParserState.READ_HEADER:
                if buf_length - start_pos >= 2:
                    data = buf[start_pos : start_pos + 2]
                    start_pos += 2
                    first_byte, second_byte = data

                    fin = (first_byte >> 7) & 1
                    rsv1 = (first_byte >> 6) & 1
                    rsv2 = (first_byte >> 5) & 1
                    rsv3 = (first_byte >> 4) & 1
                    opcode = first_byte & 0xF

                    # frame-fin = %x0 ; more frames of this message follow
                    #           / %x1 ; final frame of this message
                    # frame-rsv1 = %x0 ;
                    #    1 bit, MUST be 0 unless negotiated otherwise
                    # frame-rsv2 = %x0 ;
                    #    1 bit, MUST be 0 unless negotiated otherwise
                    # frame-rsv3 = %x0 ;
                    #    1 bit, MUST be 0 unless negotiated otherwise
                    #
                    # Remove rsv1 from this test for deflate development
                    if rsv2 or rsv3 or (rsv1 and not self._compress):
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            "Received frame with non-zero reserved bits",
                        )

                    if opcode > 0x7 and fin == 0:
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            "Received fragmented control frame",
                        )

                    has_mask = (second_byte >> 7) & 1
                    length = second_byte & 0x7F

                    # Control frames MUST have a payload
                    # length of 125 bytes or less
                    if opcode > 0x7 and length > 125:
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            "Control frame payload cannot be " "larger than 125 bytes",
                        )

                    # Set compress status if last package is FIN
                    # OR set compress status if this is first fragment
                    # Raise error if not first fragment with rsv1 = 0x1
                    if self._frame_fin or self._compressed is None:
                        self._compressed = True if rsv1 else False
                    elif rsv1:
                        raise WebSocketError(
                            WSCloseCode.PROTOCOL_ERROR,
                            "Received frame with non-zero reserved bits",
                        )

                    self._frame_fin = bool(fin)
                    self._frame_opcode = opcode
                    self._has_mask = bool(has_mask)
                    self._payload_length_flag = length
                    self._state = WSParserState.READ_PAYLOAD_LENGTH
                else:
                    break

            # read payload length
            if self._state == WSParserState.READ_PAYLOAD_LENGTH:
                length = self._payload_length_flag
                if length == 126:
                    if buf_length - start_pos >= 2:
                        data = buf[start_pos : start_pos + 2]
                        start_pos += 2
                        length = UNPACK_LEN2(data)[0]
                        self._payload_length = length
                        self._state = (
                            WSParserState.READ_PAYLOAD_MASK
                            if self._has_mask
                            else WSParserState.READ_PAYLOAD
                        )
                    else:
                        break
                elif length > 126:
                    if buf_length - start_pos >= 8:
                        data = buf[start_pos : start_pos + 8]
                        start_pos += 8
                        length = UNPACK_LEN3(data)[0]
                        self._payload_length = length
                        self._state = (
                            WSParserState.READ_PAYLOAD_MASK
                            if self._has_mask
                            else WSParserState.READ_PAYLOAD
                        )
                    else:
                        break
                else:
                    self._payload_length = length
                    self._state = (
                        WSParserState.READ_PAYLOAD_MASK
                        if self._has_mask
                        else WSParserState.READ_PAYLOAD
                    )

            # read payload mask
            if self._state == WSParserState.READ_PAYLOAD_MASK:
                if buf_length - start_pos >= 4:
                    self._frame_mask = buf[start_pos : start_pos + 4]
                    start_pos += 4
                    self._state = WSParserState.READ_PAYLOAD
                else:
                    break

            if self._state == WSParserState.READ_PAYLOAD:
                length = self._payload_length
                payload = self._frame_payload

                chunk_len = buf_length - start_pos
                if length >= chunk_len:
                    self._payload_length = length - chunk_len
                    payload.extend(buf[start_pos:])
                    start_pos = buf_length
                else:
                    self._payload_length = 0
                    payload.extend(buf[start_pos : start_pos + length])
                    start_pos = start_pos + length

                if self._payload_length == 0:
                    if self._has_mask:
                        assert self._frame_mask is not None
                        _websocket_mask(self._frame_mask, payload)

                    frames.append(
                        (self._frame_fin, self._frame_opcode, payload, self._compressed)
                    )

                    self._frame_payload = bytearray()
                    self._state = WSParserState.READ_HEADER
                else:
                    break

        self._tail = buf[start_pos:]

        return frames


class WebSocketWriter:
    def __init__(
        self,
        protocol: BaseProtocol,
        transport: asyncio.Transport,
        *,
        use_mask: bool = False,
        limit: int = DEFAULT_LIMIT,
        random: Any = random.Random(),
        compress: int = 0,
        notakeover: bool = False,
    ) -> None:
        self.protocol = protocol
        self.transport = transport
        self.use_mask = use_mask
        self.randrange = random.randrange
        self.compress = compress
        self.notakeover = notakeover
        self._closing = False
        self._limit = limit
        self._output_size = 0
        self._compressobj: Any = None  # actually compressobj

    async def _send_frame(
        self, message: bytes, opcode: int, compress: Optional[int] = None
    ) -> None:
        """Send a frame over the websocket with message as its payload."""
        if self._closing and not (opcode & WSMsgType.CLOSE):
            raise ConnectionResetError("Cannot write to closing transport")

        rsv = 0

        # Only compress larger packets (disabled)
        # Does small packet needs to be compressed?
        # if self.compress and opcode < 8 and len(message) > 124:
        if (compress or self.compress) and opcode < 8:
            if compress:
                # Do not set self._compress if compressing is for this frame
                compressobj = zlib.compressobj(level=zlib.Z_BEST_SPEED, wbits=-compress)
            else:  # self.compress
                if not self._compressobj:
                    self._compressobj = zlib.compressobj(
                        level=zlib.Z_BEST_SPEED, wbits=-self.compress
                    )
                compressobj = self._compressobj

            message = compressobj.compress(message)
            message = message + compressobj.flush(
                zlib.Z_FULL_FLUSH if self.notakeover else zlib.Z_SYNC_FLUSH
            )
            if message.endswith(_WS_DEFLATE_TRAILING):
                message = message[:-4]
            rsv = rsv | 0x40

        msg_length = len(message)

        use_mask = self.use_mask
        if use_mask:
            mask_bit = 0x80
        else:
            mask_bit = 0

        if msg_length < 126:
            header = PACK_LEN1(0x80 | rsv | opcode, msg_length | mask_bit)
        elif msg_length < (1 << 16):
            header = PACK_LEN2(0x80 | rsv | opcode, 126 | mask_bit, msg_length)
        else:
            header = PACK_LEN3(0x80 | rsv | opcode, 127 | mask_bit, msg_length)
        if use_mask:
            mask = self.randrange(0, 0xFFFFFFFF)
            mask = mask.to_bytes(4, "big")
            message = bytearray(message)
            _websocket_mask(mask, message)
            self._write(header + mask + message)
            self._output_size += len(header) + len(mask) + len(message)
        else:
            if len(message) > MSG_SIZE:
                self._write(header)
                self._write(message)
            else:
                self._write(header + message)

            self._output_size += len(header) + len(message)

        if self._output_size > self._limit:
            self._output_size = 0
            await self.protocol._drain_helper()

    def _write(self, data: bytes) -> None:
        if self.transport is None or self.transport.is_closing():
            raise ConnectionResetError("Cannot write to closing transport")
        self.transport.write(data)

    async def pong(self, message: bytes = b"") -> None:
        """Send pong message."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        await self._send_frame(message, WSMsgType.PONG)

    async def ping(self, message: bytes = b"") -> None:
        """Send ping message."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        await self._send_frame(message, WSMsgType.PING)

    async def send(
        self,
        message: Union[str, bytes],
        binary: bool = False,
        compress: Optional[int] = None,
    ) -> None:
        """Send a frame over the websocket with message as its payload."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        if binary:
            await self._send_frame(message, WSMsgType.BINARY, compress)
        else:
            await self._send_frame(message, WSMsgType.TEXT, compress)

    async def close(self, code: int = 1000, message: bytes = b"") -> None:
        """Close the websocket, sending the specified code and message."""
        if isinstance(message, str):
            message = message.encode("utf-8")
        try:
            await self._send_frame(
                PACK_CLOSE_CODE(code) + message, opcode=WSMsgType.CLOSE
            )
        finally:
            self._closing = True
