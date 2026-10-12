"""Undo a ``Content-Encoding`` without ever holding more than a set amount.

mitmproxy decodes a compressed body in one call with no bound on the result,
and a few kilobytes of gzip can stand for gigabytes.  The proxy decodes bodies
on the event loop every box shares, so here the bound is part of the decode:
each decoder is asked for at most ``limit + 1`` bytes and the body is refused
the moment it has more to give.  The whole decoded body is never produced
first and measured after.

The encodings are the ones mitmproxy knows (``mitmproxy.net.encoding``), one
per message as there.  Anything else cannot be read, which the caller must
treat as a body it cannot vouch for; so is a stream that is cut short or has
anything but a further gzip member after its end, since a client may still
make something of it.
"""

import zlib
from collections.abc import Callable
from io import BytesIO
from typing import Any

import brotli
import zstandard

_GZIP_MAGIC = b"\x1f\x8b"
_WINDOW = 64 * 1024
# Gzip members or zstd frames in one body.  Each costs a decoder of its own,
# and a box can send 5 MiB of empty ones (a quarter of a million) to hold the
# shared event loop for most of a second.  Real bodies have one, rarely a few.
_MAX_STREAMS = 64


class DecodedTooLarge(Exception):
    """The body decodes to more than the limit."""


class Undecodable(Exception):
    """An encoding this cannot decode within a bound, or corrupt data."""


class _Feed:
    """The body in windows of ``_WINDOW`` bytes.  A decoder copies whatever
    input is left after its stream's end into ``unused_data``; given the whole
    body, many tiny gzip members would cost time quadratic in its size."""

    def __init__(self, raw: bytes):
        self._view = memoryview(raw)
        self._pos = 0
        self.carry = b""  # read past the end of the last stream

    @property
    def done(self) -> bool:
        return not self.carry and self._pos >= len(self._view)

    def take(self) -> bytes:
        if self.carry:
            data, self.carry = self.carry, b""
            return data
        return self._window()

    def head(self, n: int, strip: bytes = b"") -> bytes:
        """At least *n* bytes of what follows (fewer at the end), with any
        leading *strip* bytes dropped for good."""
        while True:
            if strip:
                self.carry = self.carry.lstrip(strip)
            if len(self.carry) >= n or self._pos >= len(self._view):
                return self.carry
            self.carry += self._window()

    def _window(self) -> bytes:
        data = bytes(self._view[self._pos : self._pos + _WINDOW])
        self._pos += len(data)
        return data


def _stream(
    decoder: Any,
    step: Callable[[Any, bytes, int], bytes],
    feed: _Feed,
    out: bytearray,
    limit: int,
) -> None:
    """Decode one complete stream (a gzip member, a zlib or zstd frame) from
    *feed* into *out*.  A stream that stops short is ``Undecodable``: zlib and
    zstd hand back what they have of a cut stream without complaint, and a
    partial body would be scrubbed as if it were the whole one."""
    data = feed.take()
    while True:
        out += step(decoder, data, limit - len(out) + 1)
        if len(out) > limit:
            raise DecodedTooLarge
        if decoder.eof:
            feed.carry = decoder.unused_data
            return
        if feed.done:
            raise Undecodable("truncated stream")
        data = feed.take()


def _zlib_step(decoder: Any, data: bytes, budget: int) -> bytes:
    return decoder.decompress(data, budget)


def _gzip(raw: bytes, limit: int) -> bytes:
    """Every member, as a client decodes it: a gzip body may be several
    members back to back (RFC 1952 2.2), and a value in the second one must
    not pass because only the first was read.  Zero padding after the last
    member is allowed, as Python's ``gzip`` allows it; anything else is not."""
    feed, out = _Feed(raw), bytearray()
    # 32 + 15: gzip or zlib header, detected, as mitmproxy does.
    _stream(zlib.decompressobj(32 + zlib.MAX_WBITS), _zlib_step, feed, out, limit)
    streams = 1
    while head := feed.head(2, strip=b"\0"):
        if not head.startswith(_GZIP_MAGIC):
            raise Undecodable("data after the end of the stream")
        streams += 1
        if streams > _MAX_STREAMS:
            raise Undecodable("too many gzip members")
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
        _stream(decoder, _zlib_step, feed, out, limit)
    return bytes(out)


def _single(raw: bytes, limit: int, wbits: int) -> bytes:
    feed, out = _Feed(raw), bytearray()
    _stream(zlib.decompressobj(wbits), _zlib_step, feed, out, limit)
    if feed.head(1):
        raise Undecodable("data after the end of the stream")
    return bytes(out)


def _deflate(raw: bytes, limit: int) -> bytes:
    try:
        return _single(raw, limit, zlib.MAX_WBITS)
    except zlib.error:
        # Some servers send raw DEFLATE, with no zlib header or checksum.
        return _single(raw, limit, -zlib.MAX_WBITS)


def _brotli(raw: bytes, limit: int) -> bytes:
    decoder = brotli.Decompressor()
    out = decoder.process(raw, output_buffer_limit=limit + 1)
    # Output still pending means the limit stopped it, not the end of the data.
    if len(out) > limit or not decoder.can_accept_more_data():
        raise DecodedTooLarge
    # Data after the end is a ``brotli.error`` from ``process`` itself.
    if not decoder.is_finished():
        raise Undecodable("truncated brotli stream")
    return out


def _zstd(raw: bytes, limit: int) -> bytes:
    reader = zstandard.ZstdDecompressor().stream_reader(
        BytesIO(raw), read_across_frames=True
    )
    out = reader.read(limit + 1)
    if len(out) > limit or reader.read(1):
        raise DecodedTooLarge
    # The reader returns a cut frame's output, or none of it, without an
    # error.  All of it decodes to at most *limit* bytes (the reader ran out
    # first), so a second pass frame by frame, which does not bound its own
    # output, costs no more than the first and can tell a whole frame from a
    # cut one.
    feed, seen = _Feed(raw), bytearray()
    for _ in range(_MAX_STREAMS):
        if not feed.head(1):
            return out
        decoder = zstandard.ZstdDecompressor().decompressobj()
        _stream(decoder, lambda d, data, _: d.decompress(data), feed, seen, limit)
    if feed.head(1):
        raise Undecodable("too many zstd frames")
    return out


def bounded_decode(raw: bytes, content_encoding: str, limit: int) -> bytes:
    """*raw* with its content encoding undone, if that is at most *limit* bytes.

    Raises ``DecodedTooLarge`` or ``Undecodable``.
    """
    encoding = content_encoding.strip().lower()
    if encoding in ("", "none", "identity"):
        if len(raw) > limit:
            raise DecodedTooLarge
        return raw
    if not raw:
        return b""
    try:
        if encoding == "gzip":
            return _gzip(raw, limit)
        if encoding in ("deflate", "deflateraw"):
            return _deflate(raw, limit)
        if encoding == "br":
            return _brotli(raw, limit)
        if encoding == "zstd":
            return _zstd(raw, limit)
    except (zlib.error, brotli.error, zstandard.ZstdError) as e:
        raise Undecodable(type(e).__name__) from None
    raise Undecodable(f"unsupported content encoding: {encoding[:32]!r}")
