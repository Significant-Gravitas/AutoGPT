"""Undo a ``Content-Encoding`` without ever holding more than a set amount.

mitmproxy decodes a compressed body in one call with no bound on the result,
and a few kilobytes of gzip can stand for gigabytes.  The proxy decodes bodies
on the event loop every box shares, so here the bound is part of the decode:
each decoder is asked for at most ``limit + 1`` bytes and the body is refused
the moment it has more to give.  The whole decoded body is never produced
first and measured after.

The encodings are the ones mitmproxy knows (``mitmproxy.net.encoding``), one
per message as there.  Anything else cannot be read, which the caller must
treat as a body it cannot vouch for.
"""

import zlib
from io import BytesIO

import brotli
import zstandard


class DecodedTooLarge(Exception):
    """The body decodes to more than the limit."""


class Undecodable(Exception):
    """An encoding this cannot decode within a bound, or corrupt data."""


def _zlib(raw: bytes, limit: int, wbits: int) -> bytes:
    decoder = zlib.decompressobj(wbits)
    out = decoder.decompress(raw, limit + 1)
    if len(out) > limit or decoder.unconsumed_tail:
        raise DecodedTooLarge
    out += decoder.flush()
    if len(out) > limit:
        raise DecodedTooLarge
    return out


def _deflate(raw: bytes, limit: int) -> bytes:
    try:
        return _zlib(raw, limit, zlib.MAX_WBITS)
    except zlib.error:
        # Some servers send raw DEFLATE, with no zlib header or checksum.
        return _zlib(raw, limit, -zlib.MAX_WBITS)


def _brotli(raw: bytes, limit: int) -> bytes:
    decoder = brotli.Decompressor()
    out = decoder.process(raw, output_buffer_limit=limit + 1)
    # Output still pending means the limit stopped it, not the end of the data.
    if len(out) > limit or not decoder.can_accept_more_data():
        raise DecodedTooLarge
    if not decoder.is_finished():
        raise Undecodable("truncated brotli stream")
    return out


def _zstd(raw: bytes, limit: int) -> bytes:
    reader = zstandard.ZstdDecompressor().stream_reader(
        BytesIO(raw), read_across_frames=True
    )
    out = reader.read(limit + 1)
    if len(out) > limit:
        raise DecodedTooLarge
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
            # 32 + 15: gzip or zlib header, detected, as mitmproxy does.
            return _zlib(raw, limit, 32 + zlib.MAX_WBITS)
        if encoding in ("deflate", "deflateraw"):
            return _deflate(raw, limit)
        if encoding == "br":
            return _brotli(raw, limit)
        if encoding == "zstd":
            return _zstd(raw, limit)
    except (zlib.error, brotli.error, zstandard.ZstdError) as e:
        raise Undecodable(type(e).__name__) from None
    raise Undecodable(f"unsupported content encoding: {encoding[:32]!r}")
