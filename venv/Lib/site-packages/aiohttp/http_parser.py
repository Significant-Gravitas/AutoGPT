import abc
import asyncio
import collections
import re
import string
import zlib
from contextlib import suppress
from enum import IntEnum
from typing import (
    Any,
    Generic,
    List,
    NamedTuple,
    Optional,
    Pattern,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    cast,
)

from multidict import CIMultiDict, CIMultiDictProxy, istr
from yarl import URL

from . import hdrs
from .base_protocol import BaseProtocol
from .helpers import NO_EXTENSIONS, BaseTimerContext
from .http_exceptions import (
    BadHttpMessage,
    BadStatusLine,
    ContentEncodingError,
    ContentLengthError,
    InvalidHeader,
    LineTooLong,
    TransferEncodingError,
)
from .http_writer import HttpVersion, HttpVersion10
from .log import internal_logger
from .streams import EMPTY_PAYLOAD, StreamReader
from .typedefs import Final, RawHeaders

try:
    import brotli

    HAS_BROTLI = True
except ImportError:  # pragma: no cover
    HAS_BROTLI = False


__all__ = (
    "HeadersParser",
    "HttpParser",
    "HttpRequestParser",
    "HttpResponseParser",
    "RawRequestMessage",
    "RawResponseMessage",
)

ASCIISET: Final[Set[str]] = set(string.printable)

# See https://tools.ietf.org/html/rfc7230#section-3.1.1
# and https://tools.ietf.org/html/rfc7230#appendix-B
#
#     method = token
#     tchar = "!" / "#" / "$" / "%" / "&" / "'" / "*" / "+" / "-" / "." /
#             "^" / "_" / "`" / "|" / "~" / DIGIT / ALPHA
#     token = 1*tchar
METHRE: Final[Pattern[str]] = re.compile(r"[!#$%&'*+\-.^_`|~0-9A-Za-z]+")
VERSRE: Final[Pattern[str]] = re.compile(r"HTTP/(\d+).(\d+)")
HDRRE: Final[Pattern[bytes]] = re.compile(rb"[\x00-\x1F\x7F()<>@,;:\[\]={} \t\\\\\"]")


class RawRequestMessage(NamedTuple):
    method: str
    path: str
    version: HttpVersion
    headers: "CIMultiDictProxy[str]"
    raw_headers: RawHeaders
    should_close: bool
    compression: Optional[str]
    upgrade: bool
    chunked: bool
    url: URL


RawResponseMessage = collections.namedtuple(
    "RawResponseMessage",
    [
        "version",
        "code",
        "reason",
        "headers",
        "raw_headers",
        "should_close",
        "compression",
        "upgrade",
        "chunked",
    ],
)


_MsgT = TypeVar("_MsgT", RawRequestMessage, RawResponseMessage)


class ParseState(IntEnum):

    PARSE_NONE = 0
    PARSE_LENGTH = 1
    PARSE_CHUNKED = 2
    PARSE_UNTIL_EOF = 3


class ChunkState(IntEnum):
    PARSE_CHUNKED_SIZE = 0
    PARSE_CHUNKED_CHUNK = 1
    PARSE_CHUNKED_CHUNK_EOF = 2
    PARSE_MAYBE_TRAILERS = 3
    PARSE_TRAILERS = 4


class HeadersParser:
    def __init__(
        self,
        max_line_size: int = 8190,
        max_headers: int = 32768,
        max_field_size: int = 8190,
    ) -> None:
        self.max_line_size = max_line_size
        self.max_headers = max_headers
        self.max_field_size = max_field_size

    def parse_headers(
        self, lines: List[bytes]
    ) -> Tuple["CIMultiDictProxy[str]", RawHeaders]:
        headers: CIMultiDict[str] = CIMultiDict()
        raw_headers = []

        lines_idx = 1
        line = lines[1]
        line_count = len(lines)

        while line:
            # Parse initial header name : value pair.
            try:
                bname, bvalue = line.split(b":", 1)
            except ValueError:
                raise InvalidHeader(line) from None

            bname = bname.strip(b" \t")
            bvalue = bvalue.lstrip()
            if HDRRE.search(bname):
                raise InvalidHeader(bname)
            if len(bname) > self.max_field_size:
                raise LineTooLong(
                    "request header name {}".format(
                        bname.decode("utf8", "xmlcharrefreplace")
                    ),
                    str(self.max_field_size),
                    str(len(bname)),
                )

            header_length = len(bvalue)

            # next line
            lines_idx += 1
            line = lines[lines_idx]

            # consume continuation lines
            continuation = line and line[0] in (32, 9)  # (' ', '\t')

            if continuation:
                bvalue_lst = [bvalue]
                while continuation:
                    header_length += len(line)
                    if header_length > self.max_field_size:
                        raise LineTooLong(
                            "request header field {}".format(
                                bname.decode("utf8", "xmlcharrefreplace")
                            ),
                            str(self.max_field_size),
                            str(header_length),
                        )
                    bvalue_lst.append(line)

                    # next line
                    lines_idx += 1
                    if lines_idx < line_count:
                        line = lines[lines_idx]
                        if line:
                            continuation = line[0] in (32, 9)  # (' ', '\t')
                    else:
                        line = b""
                        break
                bvalue = b"".join(bvalue_lst)
            else:
                if header_length > self.max_field_size:
                    raise LineTooLong(
                        "request header field {}".format(
                            bname.decode("utf8", "xmlcharrefreplace")
                        ),
                        str(self.max_field_size),
                        str(header_length),
                    )

            bvalue = bvalue.strip()
            name = bname.decode("utf-8", "surrogateescape")
            value = bvalue.decode("utf-8", "surrogateescape")

            headers.add(name, value)
            raw_headers.append((bname, bvalue))

        return (CIMultiDictProxy(headers), tuple(raw_headers))


class HttpParser(abc.ABC, Generic[_MsgT]):
    def __init__(
        self,
        protocol: Optional[BaseProtocol] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        limit: int = 2**16,
        max_line_size: int = 8190,
        max_headers: int = 32768,
        max_field_size: int = 8190,
        timer: Optional[BaseTimerContext] = None,
        code: Optional[int] = None,
        method: Optional[str] = None,
        readall: bool = False,
        payload_exception: Optional[Type[BaseException]] = None,
        response_with_body: bool = True,
        read_until_eof: bool = False,
        auto_decompress: bool = True,
    ) -> None:
        self.protocol = protocol
        self.loop = loop
        self.max_line_size = max_line_size
        self.max_headers = max_headers
        self.max_field_size = max_field_size
        self.timer = timer
        self.code = code
        self.method = method
        self.readall = readall
        self.payload_exception = payload_exception
        self.response_with_body = response_with_body
        self.read_until_eof = read_until_eof

        self._lines: List[bytes] = []
        self._tail = b""
        self._upgraded = False
        self._payload = None
        self._payload_parser: Optional[HttpPayloadParser] = None
        self._auto_decompress = auto_decompress
        self._limit = limit
        self._headers_parser = HeadersParser(max_line_size, max_headers, max_field_size)

    @abc.abstractmethod
    def parse_message(self, lines: List[bytes]) -> _MsgT:
        pass

    def feed_eof(self) -> Optional[_MsgT]:
        if self._payload_parser is not None:
            self._payload_parser.feed_eof()
            self._payload_parser = None
        else:
            # try to extract partial message
            if self._tail:
                self._lines.append(self._tail)

            if self._lines:
                if self._lines[-1] != "\r\n":
                    self._lines.append(b"")
                with suppress(Exception):
                    return self.parse_message(self._lines)
        return None

    def feed_data(
        self,
        data: bytes,
        SEP: bytes = b"\r\n",
        EMPTY: bytes = b"",
        CONTENT_LENGTH: istr = hdrs.CONTENT_LENGTH,
        METH_CONNECT: str = hdrs.METH_CONNECT,
        SEC_WEBSOCKET_KEY1: istr = hdrs.SEC_WEBSOCKET_KEY1,
    ) -> Tuple[List[Tuple[_MsgT, StreamReader]], bool, bytes]:

        messages = []

        if self._tail:
            data, self._tail = self._tail + data, b""

        data_len = len(data)
        start_pos = 0
        loop = self.loop

        while start_pos < data_len:

            # read HTTP message (request/response line + headers), \r\n\r\n
            # and split by lines
            if self._payload_parser is None and not self._upgraded:
                pos = data.find(SEP, start_pos)
                # consume \r\n
                if pos == start_pos and not self._lines:
                    start_pos = pos + 2
                    continue

                if pos >= start_pos:
                    # line found
                    self._lines.append(data[start_pos:pos])
                    start_pos = pos + 2

                    # \r\n\r\n found
                    if self._lines[-1] == EMPTY:
                        try:
                            msg: _MsgT = self.parse_message(self._lines)
                        finally:
                            self._lines.clear()

                        def get_content_length() -> Optional[int]:
                            # payload length
                            length_hdr = msg.headers.get(CONTENT_LENGTH)
                            if length_hdr is None:
                                return None

                            try:
                                length = int(length_hdr)
                            except ValueError:
                                raise InvalidHeader(CONTENT_LENGTH)

                            if length < 0:
                                raise InvalidHeader(CONTENT_LENGTH)

                            return length

                        length = get_content_length()
                        # do not support old websocket spec
                        if SEC_WEBSOCKET_KEY1 in msg.headers:
                            raise InvalidHeader(SEC_WEBSOCKET_KEY1)

                        self._upgraded = msg.upgrade

                        method = getattr(msg, "method", self.method)

                        assert self.protocol is not None
                        # calculate payload
                        if (
                            (length is not None and length > 0)
                            or msg.chunked
                            and not msg.upgrade
                        ):
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            payload_parser = HttpPayloadParser(
                                payload,
                                length=length,
                                chunked=msg.chunked,
                                method=method,
                                compression=msg.compression,
                                code=self.code,
                                readall=self.readall,
                                response_with_body=self.response_with_body,
                                auto_decompress=self._auto_decompress,
                            )
                            if not payload_parser.done:
                                self._payload_parser = payload_parser
                        elif method == METH_CONNECT:
                            assert isinstance(msg, RawRequestMessage)
                            payload = StreamReader(
                                self.protocol,
                                timer=self.timer,
                                loop=loop,
                                limit=self._limit,
                            )
                            self._upgraded = True
                            self._payload_parser = HttpPayloadParser(
                                payload,
                                method=msg.method,
                                compression=msg.compression,
                                readall=True,
                                auto_decompress=self._auto_decompress,
                            )
                        else:
                            if (
                                getattr(msg, "code", 100) >= 199
                                and length is None
                                and self.read_until_eof
                            ):
                                payload = StreamReader(
                                    self.protocol,
                                    timer=self.timer,
                                    loop=loop,
                                    limit=self._limit,
                                )
                                payload_parser = HttpPayloadParser(
                                    payload,
                                    length=length,
                                    chunked=msg.chunked,
                                    method=method,
                                    compression=msg.compression,
                                    code=self.code,
                                    readall=True,
                                    response_with_body=self.response_with_body,
                                    auto_decompress=self._auto_decompress,
                                )
                                if not payload_parser.done:
                                    self._payload_parser = payload_parser
                            else:
                                payload = EMPTY_PAYLOAD

                        messages.append((msg, payload))
                else:
                    self._tail = data[start_pos:]
                    data = EMPTY
                    break

            # no parser, just store
            elif self._payload_parser is None and self._upgraded:
                assert not self._lines
                break

            # feed payload
            elif data and start_pos < data_len:
                assert not self._lines
                assert self._payload_parser is not None
                try:
                    eof, data = self._payload_parser.feed_data(data[start_pos:])
                except BaseException as exc:
                    if self.payload_exception is not None:
                        self._payload_parser.payload.set_exception(
                            self.payload_exception(str(exc))
                        )
                    else:
                        self._payload_parser.payload.set_exception(exc)

                    eof = True
                    data = b""

                if eof:
                    start_pos = 0
                    data_len = len(data)
                    self._payload_parser = None
                    continue
            else:
                break

        if data and start_pos < data_len:
            data = data[start_pos:]
        else:
            data = EMPTY

        return messages, self._upgraded, data

    def parse_headers(
        self, lines: List[bytes]
    ) -> Tuple[
        "CIMultiDictProxy[str]", RawHeaders, Optional[bool], Optional[str], bool, bool
    ]:
        """Parses RFC 5322 headers from a stream.

        Line continuations are supported. Returns list of header name
        and value pairs. Header name is in upper case.
        """
        headers, raw_headers = self._headers_parser.parse_headers(lines)
        close_conn = None
        encoding = None
        upgrade = False
        chunked = False

        # keep-alive
        conn = headers.get(hdrs.CONNECTION)
        if conn:
            v = conn.lower()
            if v == "close":
                close_conn = True
            elif v == "keep-alive":
                close_conn = False
            elif v == "upgrade":
                upgrade = True

        # encoding
        enc = headers.get(hdrs.CONTENT_ENCODING)
        if enc:
            enc = enc.lower()
            if enc in ("gzip", "deflate", "br"):
                encoding = enc

        # chunking
        te = headers.get(hdrs.TRANSFER_ENCODING)
        if te is not None:
            if "chunked" == te.lower():
                chunked = True
            else:
                raise BadHttpMessage("Request has invalid `Transfer-Encoding`")

            if hdrs.CONTENT_LENGTH in headers:
                raise BadHttpMessage(
                    "Content-Length can't be present with Transfer-Encoding",
                )

        return (headers, raw_headers, close_conn, encoding, upgrade, chunked)

    def set_upgraded(self, val: bool) -> None:
        """Set connection upgraded (to websocket) mode.

        :param bool val: new state.
        """
        self._upgraded = val


class HttpRequestParser(HttpParser[RawRequestMessage]):
    """Read request status line.

    Exception .http_exceptions.BadStatusLine
    could be raised in case of any errors in status line.
    Returns RawRequestMessage.
    """

    def parse_message(self, lines: List[bytes]) -> RawRequestMessage:
        # request line
        line = lines[0].decode("utf-8", "surrogateescape")
        try:
            method, path, version = line.split(None, 2)
        except ValueError:
            raise BadStatusLine(line) from None

        if len(path) > self.max_line_size:
            raise LineTooLong(
                "Status line is too long", str(self.max_line_size), str(len(path))
            )

        # method
        if not METHRE.match(method):
            raise BadStatusLine(method)

        # version
        try:
            if version.startswith("HTTP/"):
                n1, n2 = version[5:].split(".", 1)
                version_o = HttpVersion(int(n1), int(n2))
            else:
                raise BadStatusLine(version)
        except Exception:
            raise BadStatusLine(version)

        if method == "CONNECT":
            # authority-form,
            # https://datatracker.ietf.org/doc/html/rfc7230#section-5.3.3
            url = URL.build(authority=path, encoded=True)
        elif path.startswith("/"):
            # origin-form,
            # https://datatracker.ietf.org/doc/html/rfc7230#section-5.3.1
            path_part, _hash_separator, url_fragment = path.partition("#")
            path_part, _question_mark_separator, qs_part = path_part.partition("?")

            # NOTE: `yarl.URL.build()` is used to mimic what the Cython-based
            # NOTE: parser does, otherwise it results into the same
            # NOTE: HTTP Request-Line input producing different
            # NOTE: `yarl.URL()` objects
            url = URL.build(
                path=path_part,
                query_string=qs_part,
                fragment=url_fragment,
                encoded=True,
            )
        else:
            # absolute-form for proxy maybe,
            # https://datatracker.ietf.org/doc/html/rfc7230#section-5.3.2
            url = URL(path, encoded=True)

        # read headers
        (
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        ) = self.parse_headers(lines)

        if close is None:  # then the headers weren't set in the request
            if version_o <= HttpVersion10:  # HTTP 1.0 must asks to not close
                close = True
            else:  # HTTP 1.1 must ask to close.
                close = False

        return RawRequestMessage(
            method,
            path,
            version_o,
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
            url,
        )


class HttpResponseParser(HttpParser[RawResponseMessage]):
    """Read response status line and headers.

    BadStatusLine could be raised in case of any errors in status line.
    Returns RawResponseMessage.
    """

    def parse_message(self, lines: List[bytes]) -> RawResponseMessage:
        line = lines[0].decode("utf-8", "surrogateescape")
        try:
            version, status = line.split(None, 1)
        except ValueError:
            raise BadStatusLine(line) from None

        try:
            status, reason = status.split(None, 1)
        except ValueError:
            reason = ""

        if len(reason) > self.max_line_size:
            raise LineTooLong(
                "Status line is too long", str(self.max_line_size), str(len(reason))
            )

        # version
        match = VERSRE.match(version)
        if match is None:
            raise BadStatusLine(line)
        version_o = HttpVersion(int(match.group(1)), int(match.group(2)))

        # The status code is a three-digit number
        try:
            status_i = int(status)
        except ValueError:
            raise BadStatusLine(line) from None

        if status_i > 999:
            raise BadStatusLine(line)

        # read headers
        (
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        ) = self.parse_headers(lines)

        if close is None:
            close = version_o <= HttpVersion10

        return RawResponseMessage(
            version_o,
            status_i,
            reason.strip(),
            headers,
            raw_headers,
            close,
            compression,
            upgrade,
            chunked,
        )


class HttpPayloadParser:
    def __init__(
        self,
        payload: StreamReader,
        length: Optional[int] = None,
        chunked: bool = False,
        compression: Optional[str] = None,
        code: Optional[int] = None,
        method: Optional[str] = None,
        readall: bool = False,
        response_with_body: bool = True,
        auto_decompress: bool = True,
    ) -> None:
        self._length = 0
        self._type = ParseState.PARSE_NONE
        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
        self._chunk_size = 0
        self._chunk_tail = b""
        self._auto_decompress = auto_decompress
        self.done = False

        # payload decompression wrapper
        if response_with_body and compression and self._auto_decompress:
            real_payload: Union[StreamReader, DeflateBuffer] = DeflateBuffer(
                payload, compression
            )
        else:
            real_payload = payload

        # payload parser
        if not response_with_body:
            # don't parse payload if it's not expected to be received
            self._type = ParseState.PARSE_NONE
            real_payload.feed_eof()
            self.done = True

        elif chunked:
            self._type = ParseState.PARSE_CHUNKED
        elif length is not None:
            self._type = ParseState.PARSE_LENGTH
            self._length = length
            if self._length == 0:
                real_payload.feed_eof()
                self.done = True
        else:
            if readall and code != 204:
                self._type = ParseState.PARSE_UNTIL_EOF
            elif method in ("PUT", "POST"):
                internal_logger.warning(  # pragma: no cover
                    "Content-Length or Transfer-Encoding header is required"
                )
                self._type = ParseState.PARSE_NONE
                real_payload.feed_eof()
                self.done = True

        self.payload = real_payload

    def feed_eof(self) -> None:
        if self._type == ParseState.PARSE_UNTIL_EOF:
            self.payload.feed_eof()
        elif self._type == ParseState.PARSE_LENGTH:
            raise ContentLengthError(
                "Not enough data for satisfy content length header."
            )
        elif self._type == ParseState.PARSE_CHUNKED:
            raise TransferEncodingError(
                "Not enough data for satisfy transfer length header."
            )

    def feed_data(
        self, chunk: bytes, SEP: bytes = b"\r\n", CHUNK_EXT: bytes = b";"
    ) -> Tuple[bool, bytes]:
        # Read specified amount of bytes
        if self._type == ParseState.PARSE_LENGTH:
            required = self._length
            chunk_len = len(chunk)

            if required >= chunk_len:
                self._length = required - chunk_len
                self.payload.feed_data(chunk, chunk_len)
                if self._length == 0:
                    self.payload.feed_eof()
                    return True, b""
            else:
                self._length = 0
                self.payload.feed_data(chunk[:required], required)
                self.payload.feed_eof()
                return True, chunk[required:]

        # Chunked transfer encoding parser
        elif self._type == ParseState.PARSE_CHUNKED:
            if self._chunk_tail:
                chunk = self._chunk_tail + chunk
                self._chunk_tail = b""

            while chunk:

                # read next chunk size
                if self._chunk == ChunkState.PARSE_CHUNKED_SIZE:
                    pos = chunk.find(SEP)
                    if pos >= 0:
                        i = chunk.find(CHUNK_EXT, 0, pos)
                        if i >= 0:
                            size_b = chunk[:i]  # strip chunk-extensions
                        else:
                            size_b = chunk[:pos]

                        try:
                            size = int(bytes(size_b), 16)
                        except ValueError:
                            exc = TransferEncodingError(
                                chunk[:pos].decode("ascii", "surrogateescape")
                            )
                            self.payload.set_exception(exc)
                            raise exc from None

                        chunk = chunk[pos + 2 :]
                        if size == 0:  # eof marker
                            self._chunk = ChunkState.PARSE_MAYBE_TRAILERS
                        else:
                            self._chunk = ChunkState.PARSE_CHUNKED_CHUNK
                            self._chunk_size = size
                            self.payload.begin_http_chunk_receiving()
                    else:
                        self._chunk_tail = chunk
                        return False, b""

                # read chunk and feed buffer
                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK:
                    required = self._chunk_size
                    chunk_len = len(chunk)

                    if required > chunk_len:
                        self._chunk_size = required - chunk_len
                        self.payload.feed_data(chunk, chunk_len)
                        return False, b""
                    else:
                        self._chunk_size = 0
                        self.payload.feed_data(chunk[:required], required)
                        chunk = chunk[required:]
                        self._chunk = ChunkState.PARSE_CHUNKED_CHUNK_EOF
                        self.payload.end_http_chunk_receiving()

                # toss the CRLF at the end of the chunk
                if self._chunk == ChunkState.PARSE_CHUNKED_CHUNK_EOF:
                    if chunk[:2] == SEP:
                        chunk = chunk[2:]
                        self._chunk = ChunkState.PARSE_CHUNKED_SIZE
                    else:
                        self._chunk_tail = chunk
                        return False, b""

                # if stream does not contain trailer, after 0\r\n
                # we should get another \r\n otherwise
                # trailers needs to be skiped until \r\n\r\n
                if self._chunk == ChunkState.PARSE_MAYBE_TRAILERS:
                    head = chunk[:2]
                    if head == SEP:
                        # end of stream
                        self.payload.feed_eof()
                        return True, chunk[2:]
                    # Both CR and LF, or only LF may not be received yet. It is
                    # expected that CRLF or LF will be shown at the very first
                    # byte next time, otherwise trailers should come. The last
                    # CRLF which marks the end of response might not be
                    # contained in the same TCP segment which delivered the
                    # size indicator.
                    if not head:
                        return False, b""
                    if head == SEP[:1]:
                        self._chunk_tail = head
                        return False, b""
                    self._chunk = ChunkState.PARSE_TRAILERS

                # read and discard trailer up to the CRLF terminator
                if self._chunk == ChunkState.PARSE_TRAILERS:
                    pos = chunk.find(SEP)
                    if pos >= 0:
                        chunk = chunk[pos + 2 :]
                        self._chunk = ChunkState.PARSE_MAYBE_TRAILERS
                    else:
                        self._chunk_tail = chunk
                        return False, b""

        # Read all bytes until eof
        elif self._type == ParseState.PARSE_UNTIL_EOF:
            self.payload.feed_data(chunk, len(chunk))

        return False, b""


class DeflateBuffer:
    """DeflateStream decompress stream and feed data into specified stream."""

    decompressor: Any

    def __init__(self, out: StreamReader, encoding: Optional[str]) -> None:
        self.out = out
        self.size = 0
        self.encoding = encoding
        self._started_decoding = False

        if encoding == "br":
            if not HAS_BROTLI:  # pragma: no cover
                raise ContentEncodingError(
                    "Can not decode content-encoding: brotli (br). "
                    "Please install `Brotli`"
                )

            class BrotliDecoder:
                # Supports both 'brotlipy' and 'Brotli' packages
                # since they share an import name. The top branches
                # are for 'brotlipy' and bottom branches for 'Brotli'
                def __init__(self) -> None:
                    self._obj = brotli.Decompressor()

                def decompress(self, data: bytes) -> bytes:
                    if hasattr(self._obj, "decompress"):
                        return cast(bytes, self._obj.decompress(data))
                    return cast(bytes, self._obj.process(data))

                def flush(self) -> bytes:
                    if hasattr(self._obj, "flush"):
                        return cast(bytes, self._obj.flush())
                    return b""

            self.decompressor = BrotliDecoder()
        else:
            zlib_mode = 16 + zlib.MAX_WBITS if encoding == "gzip" else zlib.MAX_WBITS
            self.decompressor = zlib.decompressobj(wbits=zlib_mode)

    def set_exception(self, exc: BaseException) -> None:
        self.out.set_exception(exc)

    def feed_data(self, chunk: bytes, size: int) -> None:
        if not size:
            return

        self.size += size

        # RFC1950
        # bits 0..3 = CM = 0b1000 = 8 = "deflate"
        # bits 4..7 = CINFO = 1..7 = windows size.
        if (
            not self._started_decoding
            and self.encoding == "deflate"
            and chunk[0] & 0xF != 8
        ):
            # Change the decoder to decompress incorrectly compressed data
            # Actually we should issue a warning about non-RFC-compliant data.
            self.decompressor = zlib.decompressobj(wbits=-zlib.MAX_WBITS)

        try:
            chunk = self.decompressor.decompress(chunk)
        except Exception:
            raise ContentEncodingError(
                "Can not decode content-encoding: %s" % self.encoding
            )

        self._started_decoding = True

        if chunk:
            self.out.feed_data(chunk, len(chunk))

    def feed_eof(self) -> None:
        chunk = self.decompressor.flush()

        if chunk or self.size > 0:
            self.out.feed_data(chunk, len(chunk))
            if self.encoding == "deflate" and not self.decompressor.eof:
                raise ContentEncodingError("deflate")

        self.out.feed_eof()

    def begin_http_chunk_receiving(self) -> None:
        self.out.begin_http_chunk_receiving()

    def end_http_chunk_receiving(self) -> None:
        self.out.end_http_chunk_receiving()


HttpRequestParserPy = HttpRequestParser
HttpResponseParserPy = HttpResponseParser
RawRequestMessagePy = RawRequestMessage
RawResponseMessagePy = RawResponseMessage

try:
    if not NO_EXTENSIONS:
        from ._http_parser import (  # type: ignore[import,no-redef]
            HttpRequestParser,
            HttpResponseParser,
            RawRequestMessage,
            RawResponseMessage,
        )

        HttpRequestParserC = HttpRequestParser
        HttpResponseParserC = HttpResponseParser
        RawRequestMessageC = RawRequestMessage
        RawResponseMessageC = RawResponseMessage
except ImportError:  # pragma: no cover
    pass
