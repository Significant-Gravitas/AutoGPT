"""Http related parsers and protocol."""

import asyncio
import zlib
from typing import Any, Awaitable, Callable, NamedTuple, Optional, Union  # noqa

from multidict import CIMultiDict

from .abc import AbstractStreamWriter
from .base_protocol import BaseProtocol
from .helpers import NO_EXTENSIONS

__all__ = ("StreamWriter", "HttpVersion", "HttpVersion10", "HttpVersion11")


class HttpVersion(NamedTuple):
    major: int
    minor: int


HttpVersion10 = HttpVersion(1, 0)
HttpVersion11 = HttpVersion(1, 1)


_T_OnChunkSent = Optional[Callable[[bytes], Awaitable[None]]]
_T_OnHeadersSent = Optional[Callable[["CIMultiDict[str]"], Awaitable[None]]]


class StreamWriter(AbstractStreamWriter):
    def __init__(
        self,
        protocol: BaseProtocol,
        loop: asyncio.AbstractEventLoop,
        on_chunk_sent: _T_OnChunkSent = None,
        on_headers_sent: _T_OnHeadersSent = None,
    ) -> None:
        self._protocol = protocol

        self.loop = loop
        self.length = None
        self.chunked = False
        self.buffer_size = 0
        self.output_size = 0

        self._eof = False
        self._compress: Any = None
        self._drain_waiter = None

        self._on_chunk_sent: _T_OnChunkSent = on_chunk_sent
        self._on_headers_sent: _T_OnHeadersSent = on_headers_sent

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        return self._protocol.transport

    @property
    def protocol(self) -> BaseProtocol:
        return self._protocol

    def enable_chunking(self) -> None:
        self.chunked = True

    def enable_compression(
        self, encoding: str = "deflate", strategy: int = zlib.Z_DEFAULT_STRATEGY
    ) -> None:
        zlib_mode = 16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
        self._compress = zlib.compressobj(wbits=zlib_mode, strategy=strategy)

    def _write(self, chunk: bytes) -> None:
        size = len(chunk)
        self.buffer_size += size
        self.output_size += size
        transport = self.transport
        if not self._protocol.connected or transport is None or transport.is_closing():
            raise ConnectionResetError("Cannot write to closing transport")
        transport.write(chunk)

    async def write(
        self, chunk: bytes, *, drain: bool = True, LIMIT: int = 0x10000
    ) -> None:
        """Writes chunk of data to a stream.

        write_eof() indicates end of stream.
        writer can't be used after write_eof() method being called.
        write() return drain future.
        """
        if self._on_chunk_sent is not None:
            await self._on_chunk_sent(chunk)

        if isinstance(chunk, memoryview):
            if chunk.nbytes != len(chunk):
                # just reshape it
                chunk = chunk.cast("c")

        if self._compress is not None:
            chunk = self._compress.compress(chunk)
            if not chunk:
                return

        if self.length is not None:
            chunk_len = len(chunk)
            if self.length >= chunk_len:
                self.length = self.length - chunk_len
            else:
                chunk = chunk[: self.length]
                self.length = 0
                if not chunk:
                    return

        if chunk:
            if self.chunked:
                chunk_len_pre = ("%x\r\n" % len(chunk)).encode("ascii")
                chunk = chunk_len_pre + chunk + b"\r\n"

            self._write(chunk)

            if self.buffer_size > LIMIT and drain:
                self.buffer_size = 0
                await self.drain()

    async def write_headers(
        self, status_line: str, headers: "CIMultiDict[str]"
    ) -> None:
        """Write request/response status and headers."""
        if self._on_headers_sent is not None:
            await self._on_headers_sent(headers)

        # status + headers
        buf = _serialize_headers(status_line, headers)
        self._write(buf)

    async def write_eof(self, chunk: bytes = b"") -> None:
        if self._eof:
            return

        if chunk and self._on_chunk_sent is not None:
            await self._on_chunk_sent(chunk)

        if self._compress:
            if chunk:
                chunk = self._compress.compress(chunk)

            chunk = chunk + self._compress.flush()
            if chunk and self.chunked:
                chunk_len = ("%x\r\n" % len(chunk)).encode("ascii")
                chunk = chunk_len + chunk + b"\r\n0\r\n\r\n"
        else:
            if self.chunked:
                if chunk:
                    chunk_len = ("%x\r\n" % len(chunk)).encode("ascii")
                    chunk = chunk_len + chunk + b"\r\n0\r\n\r\n"
                else:
                    chunk = b"0\r\n\r\n"

        if chunk:
            self._write(chunk)

        await self.drain()

        self._eof = True

    async def drain(self) -> None:
        """Flush the write buffer.

        The intended use is to write

          await w.write(data)
          await w.drain()
        """
        if self._protocol.transport is not None:
            await self._protocol._drain_helper()


def _safe_header(string: str) -> str:
    if "\r" in string or "\n" in string:
        raise ValueError(
            "Newline or carriage return detected in headers. "
            "Potential header injection attack."
        )
    return string


def _py_serialize_headers(status_line: str, headers: "CIMultiDict[str]") -> bytes:
    headers_gen = (_safe_header(k) + ": " + _safe_header(v) for k, v in headers.items())
    line = status_line + "\r\n" + "\r\n".join(headers_gen) + "\r\n\r\n"
    return line.encode("utf-8")


_serialize_headers = _py_serialize_headers

try:
    import aiohttp._http_writer as _http_writer  # type: ignore[import]

    _c_serialize_headers = _http_writer._serialize_headers
    if not NO_EXTENSIONS:
        _serialize_headers = _c_serialize_headers
except ImportError:
    pass
