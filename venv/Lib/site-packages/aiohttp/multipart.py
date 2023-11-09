import base64
import binascii
import json
import re
import uuid
import warnings
import zlib
from collections import deque
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    AsyncIterator,
    Deque,
    Dict,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
)
from urllib.parse import parse_qsl, unquote, urlencode

from multidict import CIMultiDict, CIMultiDictProxy, MultiMapping

from .hdrs import (
    CONTENT_DISPOSITION,
    CONTENT_ENCODING,
    CONTENT_LENGTH,
    CONTENT_TRANSFER_ENCODING,
    CONTENT_TYPE,
)
from .helpers import CHAR, TOKEN, parse_mimetype, reify
from .http import HeadersParser
from .payload import (
    JsonPayload,
    LookupError,
    Order,
    Payload,
    StringPayload,
    get_payload,
    payload_type,
)
from .streams import StreamReader

__all__ = (
    "MultipartReader",
    "MultipartWriter",
    "BodyPartReader",
    "BadContentDispositionHeader",
    "BadContentDispositionParam",
    "parse_content_disposition",
    "content_disposition_filename",
)


if TYPE_CHECKING:  # pragma: no cover
    from .client_reqrep import ClientResponse


class BadContentDispositionHeader(RuntimeWarning):
    pass


class BadContentDispositionParam(RuntimeWarning):
    pass


def parse_content_disposition(
    header: Optional[str],
) -> Tuple[Optional[str], Dict[str, str]]:
    def is_token(string: str) -> bool:
        return bool(string) and TOKEN >= set(string)

    def is_quoted(string: str) -> bool:
        return string[0] == string[-1] == '"'

    def is_rfc5987(string: str) -> bool:
        return is_token(string) and string.count("'") == 2

    def is_extended_param(string: str) -> bool:
        return string.endswith("*")

    def is_continuous_param(string: str) -> bool:
        pos = string.find("*") + 1
        if not pos:
            return False
        substring = string[pos:-1] if string.endswith("*") else string[pos:]
        return substring.isdigit()

    def unescape(text: str, *, chars: str = "".join(map(re.escape, CHAR))) -> str:
        return re.sub(f"\\\\([{chars}])", "\\1", text)

    if not header:
        return None, {}

    disptype, *parts = header.split(";")
    if not is_token(disptype):
        warnings.warn(BadContentDispositionHeader(header))
        return None, {}

    params: Dict[str, str] = {}
    while parts:
        item = parts.pop(0)

        if "=" not in item:
            warnings.warn(BadContentDispositionHeader(header))
            return None, {}

        key, value = item.split("=", 1)
        key = key.lower().strip()
        value = value.lstrip()

        if key in params:
            warnings.warn(BadContentDispositionHeader(header))
            return None, {}

        if not is_token(key):
            warnings.warn(BadContentDispositionParam(item))
            continue

        elif is_continuous_param(key):
            if is_quoted(value):
                value = unescape(value[1:-1])
            elif not is_token(value):
                warnings.warn(BadContentDispositionParam(item))
                continue

        elif is_extended_param(key):
            if is_rfc5987(value):
                encoding, _, value = value.split("'", 2)
                encoding = encoding or "utf-8"
            else:
                warnings.warn(BadContentDispositionParam(item))
                continue

            try:
                value = unquote(value, encoding, "strict")
            except UnicodeDecodeError:  # pragma: nocover
                warnings.warn(BadContentDispositionParam(item))
                continue

        else:
            failed = True
            if is_quoted(value):
                failed = False
                value = unescape(value[1:-1].lstrip("\\/"))
            elif is_token(value):
                failed = False
            elif parts:
                # maybe just ; in filename, in any case this is just
                # one case fix, for proper fix we need to redesign parser
                _value = f"{value};{parts[0]}"
                if is_quoted(_value):
                    parts.pop(0)
                    value = unescape(_value[1:-1].lstrip("\\/"))
                    failed = False

            if failed:
                warnings.warn(BadContentDispositionHeader(header))
                return None, {}

        params[key] = value

    return disptype.lower(), params


def content_disposition_filename(
    params: Mapping[str, str], name: str = "filename"
) -> Optional[str]:
    name_suf = "%s*" % name
    if not params:
        return None
    elif name_suf in params:
        return params[name_suf]
    elif name in params:
        return params[name]
    else:
        parts = []
        fnparams = sorted(
            (key, value) for key, value in params.items() if key.startswith(name_suf)
        )
        for num, (key, value) in enumerate(fnparams):
            _, tail = key.split("*", 1)
            if tail.endswith("*"):
                tail = tail[:-1]
            if tail == str(num):
                parts.append(value)
            else:
                break
        if not parts:
            return None
        value = "".join(parts)
        if "'" in value:
            encoding, _, value = value.split("'", 2)
            encoding = encoding or "utf-8"
            return unquote(value, encoding, "strict")
        return value


class MultipartResponseWrapper:
    """Wrapper around the MultipartReader.

    It takes care about
    underlying connection and close it when it needs in.
    """

    def __init__(
        self,
        resp: "ClientResponse",
        stream: "MultipartReader",
    ) -> None:
        self.resp = resp
        self.stream = stream

    def __aiter__(self) -> "MultipartResponseWrapper":
        return self

    async def __anext__(
        self,
    ) -> Union["MultipartReader", "BodyPartReader"]:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    def at_eof(self) -> bool:
        """Returns True when all response data had been read."""
        return self.resp.content.at_eof()

    async def next(
        self,
    ) -> Optional[Union["MultipartReader", "BodyPartReader"]]:
        """Emits next multipart reader object."""
        item = await self.stream.next()
        if self.stream.at_eof():
            await self.release()
        return item

    async def release(self) -> None:
        """Release the connection gracefully.

        All remaining content is read to the void.
        """
        await self.resp.release()


class BodyPartReader:
    """Multipart reader for single body part."""

    chunk_size = 8192

    def __init__(
        self, boundary: bytes, headers: "CIMultiDictProxy[str]", content: StreamReader
    ) -> None:
        self.headers = headers
        self._boundary = boundary
        self._content = content
        self._at_eof = False
        length = self.headers.get(CONTENT_LENGTH, None)
        self._length = int(length) if length is not None else None
        self._read_bytes = 0
        # TODO: typeing.Deque is not supported by Python 3.5
        self._unread: Deque[bytes] = deque()
        self._prev_chunk: Optional[bytes] = None
        self._content_eof = 0
        self._cache: Dict[str, Any] = {}

    def __aiter__(self) -> AsyncIterator["BodyPartReader"]:
        return self  # type: ignore[return-value]

    async def __anext__(self) -> bytes:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    async def next(self) -> Optional[bytes]:
        item = await self.read()
        if not item:
            return None
        return item

    async def read(self, *, decode: bool = False) -> bytes:
        """Reads body part data.

        decode: Decodes data following by encoding
                method from Content-Encoding header. If it missed
                data remains untouched
        """
        if self._at_eof:
            return b""
        data = bytearray()
        while not self._at_eof:
            data.extend(await self.read_chunk(self.chunk_size))
        if decode:
            return self.decode(data)
        return data

    async def read_chunk(self, size: int = chunk_size) -> bytes:
        """Reads body part content chunk of the specified size.

        size: chunk size
        """
        if self._at_eof:
            return b""
        if self._length:
            chunk = await self._read_chunk_from_length(size)
        else:
            chunk = await self._read_chunk_from_stream(size)

        self._read_bytes += len(chunk)
        if self._read_bytes == self._length:
            self._at_eof = True
        if self._at_eof:
            clrf = await self._content.readline()
            assert (
                b"\r\n" == clrf
            ), "reader did not read all the data or it is malformed"
        return chunk

    async def _read_chunk_from_length(self, size: int) -> bytes:
        # Reads body part content chunk of the specified size.
        # The body part must has Content-Length header with proper value.
        assert self._length is not None, "Content-Length required for chunked read"
        chunk_size = min(size, self._length - self._read_bytes)
        chunk = await self._content.read(chunk_size)
        return chunk

    async def _read_chunk_from_stream(self, size: int) -> bytes:
        # Reads content chunk of body part with unknown length.
        # The Content-Length header for body part is not necessary.
        assert (
            size >= len(self._boundary) + 2
        ), "Chunk size must be greater or equal than boundary length + 2"
        first_chunk = self._prev_chunk is None
        if first_chunk:
            self._prev_chunk = await self._content.read(size)

        chunk = await self._content.read(size)
        self._content_eof += int(self._content.at_eof())
        assert self._content_eof < 3, "Reading after EOF"
        assert self._prev_chunk is not None
        window = self._prev_chunk + chunk
        sub = b"\r\n" + self._boundary
        if first_chunk:
            idx = window.find(sub)
        else:
            idx = window.find(sub, max(0, len(self._prev_chunk) - len(sub)))
        if idx >= 0:
            # pushing boundary back to content
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=DeprecationWarning)
                self._content.unread_data(window[idx:])
            if size > idx:
                self._prev_chunk = self._prev_chunk[:idx]
            chunk = window[len(self._prev_chunk) : idx]
            if not chunk:
                self._at_eof = True
        result = self._prev_chunk
        self._prev_chunk = chunk
        return result

    async def readline(self) -> bytes:
        """Reads body part by line by line."""
        if self._at_eof:
            return b""

        if self._unread:
            line = self._unread.popleft()
        else:
            line = await self._content.readline()

        if line.startswith(self._boundary):
            # the very last boundary may not come with \r\n,
            # so set single rules for everyone
            sline = line.rstrip(b"\r\n")
            boundary = self._boundary
            last_boundary = self._boundary + b"--"
            # ensure that we read exactly the boundary, not something alike
            if sline == boundary or sline == last_boundary:
                self._at_eof = True
                self._unread.append(line)
                return b""
        else:
            next_line = await self._content.readline()
            if next_line.startswith(self._boundary):
                line = line[:-2]  # strip CRLF but only once
            self._unread.append(next_line)

        return line

    async def release(self) -> None:
        """Like read(), but reads all the data to the void."""
        if self._at_eof:
            return
        while not self._at_eof:
            await self.read_chunk(self.chunk_size)

    async def text(self, *, encoding: Optional[str] = None) -> str:
        """Like read(), but assumes that body part contains text data."""
        data = await self.read(decode=True)
        # see https://www.w3.org/TR/html5/forms.html#multipart/form-data-encoding-algorithm # NOQA
        # and https://dvcs.w3.org/hg/xhr/raw-file/tip/Overview.html#dom-xmlhttprequest-send # NOQA
        encoding = encoding or self.get_charset(default="utf-8")
        return data.decode(encoding)

    async def json(self, *, encoding: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Like read(), but assumes that body parts contains JSON data."""
        data = await self.read(decode=True)
        if not data:
            return None
        encoding = encoding or self.get_charset(default="utf-8")
        return cast(Dict[str, Any], json.loads(data.decode(encoding)))

    async def form(self, *, encoding: Optional[str] = None) -> List[Tuple[str, str]]:
        """Like read(), but assumes that body parts contain form urlencoded data."""
        data = await self.read(decode=True)
        if not data:
            return []
        if encoding is not None:
            real_encoding = encoding
        else:
            real_encoding = self.get_charset(default="utf-8")
        return parse_qsl(
            data.rstrip().decode(real_encoding),
            keep_blank_values=True,
            encoding=real_encoding,
        )

    def at_eof(self) -> bool:
        """Returns True if the boundary was reached or False otherwise."""
        return self._at_eof

    def decode(self, data: bytes) -> bytes:
        """Decodes data.

        Decoding is done according the specified Content-Encoding
        or Content-Transfer-Encoding headers value.
        """
        if CONTENT_TRANSFER_ENCODING in self.headers:
            data = self._decode_content_transfer(data)
        if CONTENT_ENCODING in self.headers:
            return self._decode_content(data)
        return data

    def _decode_content(self, data: bytes) -> bytes:
        encoding = self.headers.get(CONTENT_ENCODING, "").lower()

        if encoding == "deflate":
            return zlib.decompress(data, -zlib.MAX_WBITS)
        elif encoding == "gzip":
            return zlib.decompress(data, 16 + zlib.MAX_WBITS)
        elif encoding == "identity":
            return data
        else:
            raise RuntimeError(f"unknown content encoding: {encoding}")

    def _decode_content_transfer(self, data: bytes) -> bytes:
        encoding = self.headers.get(CONTENT_TRANSFER_ENCODING, "").lower()

        if encoding == "base64":
            return base64.b64decode(data)
        elif encoding == "quoted-printable":
            return binascii.a2b_qp(data)
        elif encoding in ("binary", "8bit", "7bit"):
            return data
        else:
            raise RuntimeError(
                "unknown content transfer encoding: {}" "".format(encoding)
            )

    def get_charset(self, default: str) -> str:
        """Returns charset parameter from Content-Type header or default."""
        ctype = self.headers.get(CONTENT_TYPE, "")
        mimetype = parse_mimetype(ctype)
        return mimetype.parameters.get("charset", default)

    @reify
    def name(self) -> Optional[str]:
        """Returns name specified in Content-Disposition header.

        If the header is missing or malformed, returns None.
        """
        _, params = parse_content_disposition(self.headers.get(CONTENT_DISPOSITION))
        return content_disposition_filename(params, "name")

    @reify
    def filename(self) -> Optional[str]:
        """Returns filename specified in Content-Disposition header.

        Returns None if the header is missing or malformed.
        """
        _, params = parse_content_disposition(self.headers.get(CONTENT_DISPOSITION))
        return content_disposition_filename(params, "filename")


@payload_type(BodyPartReader, order=Order.try_first)
class BodyPartReaderPayload(Payload):
    def __init__(self, value: BodyPartReader, *args: Any, **kwargs: Any) -> None:
        super().__init__(value, *args, **kwargs)

        params: Dict[str, str] = {}
        if value.name is not None:
            params["name"] = value.name
        if value.filename is not None:
            params["filename"] = value.filename

        if params:
            self.set_content_disposition("attachment", True, **params)

    async def write(self, writer: Any) -> None:
        field = self._value
        chunk = await field.read_chunk(size=2**16)
        while chunk:
            await writer.write(field.decode(chunk))
            chunk = await field.read_chunk(size=2**16)


class MultipartReader:
    """Multipart body reader."""

    #: Response wrapper, used when multipart readers constructs from response.
    response_wrapper_cls = MultipartResponseWrapper
    #: Multipart reader class, used to handle multipart/* body parts.
    #: None points to type(self)
    multipart_reader_cls = None
    #: Body part reader class for non multipart/* content types.
    part_reader_cls = BodyPartReader

    def __init__(self, headers: Mapping[str, str], content: StreamReader) -> None:
        self.headers = headers
        self._boundary = ("--" + self._get_boundary()).encode()
        self._content = content
        self._last_part: Optional[Union["MultipartReader", BodyPartReader]] = None
        self._at_eof = False
        self._at_bof = True
        self._unread: List[bytes] = []

    def __aiter__(
        self,
    ) -> AsyncIterator["BodyPartReader"]:
        return self  # type: ignore[return-value]

    async def __anext__(
        self,
    ) -> Optional[Union["MultipartReader", BodyPartReader]]:
        part = await self.next()
        if part is None:
            raise StopAsyncIteration
        return part

    @classmethod
    def from_response(
        cls,
        response: "ClientResponse",
    ) -> MultipartResponseWrapper:
        """Constructs reader instance from HTTP response.

        :param response: :class:`~aiohttp.client.ClientResponse` instance
        """
        obj = cls.response_wrapper_cls(
            response, cls(response.headers, response.content)
        )
        return obj

    def at_eof(self) -> bool:
        """Returns True if the final boundary was reached, false otherwise."""
        return self._at_eof

    async def next(
        self,
    ) -> Optional[Union["MultipartReader", BodyPartReader]]:
        """Emits the next multipart body part."""
        # So, if we're at BOF, we need to skip till the boundary.
        if self._at_eof:
            return None
        await self._maybe_release_last_part()
        if self._at_bof:
            await self._read_until_first_boundary()
            self._at_bof = False
        else:
            await self._read_boundary()
        if self._at_eof:  # we just read the last boundary, nothing to do there
            return None
        self._last_part = await self.fetch_next_part()
        return self._last_part

    async def release(self) -> None:
        """Reads all the body parts to the void till the final boundary."""
        while not self._at_eof:
            item = await self.next()
            if item is None:
                break
            await item.release()

    async def fetch_next_part(
        self,
    ) -> Union["MultipartReader", BodyPartReader]:
        """Returns the next body part reader."""
        headers = await self._read_headers()
        return self._get_part_reader(headers)

    def _get_part_reader(
        self,
        headers: "CIMultiDictProxy[str]",
    ) -> Union["MultipartReader", BodyPartReader]:
        """Dispatches the response by the `Content-Type` header.

        Returns a suitable reader instance.

        :param dict headers: Response headers
        """
        ctype = headers.get(CONTENT_TYPE, "")
        mimetype = parse_mimetype(ctype)

        if mimetype.type == "multipart":
            if self.multipart_reader_cls is None:
                return type(self)(headers, self._content)
            return self.multipart_reader_cls(headers, self._content)
        else:
            return self.part_reader_cls(self._boundary, headers, self._content)

    def _get_boundary(self) -> str:
        mimetype = parse_mimetype(self.headers[CONTENT_TYPE])

        assert mimetype.type == "multipart", "multipart/* content type expected"

        if "boundary" not in mimetype.parameters:
            raise ValueError(
                "boundary missed for Content-Type: %s" % self.headers[CONTENT_TYPE]
            )

        boundary = mimetype.parameters["boundary"]
        if len(boundary) > 70:
            raise ValueError("boundary %r is too long (70 chars max)" % boundary)

        return boundary

    async def _readline(self) -> bytes:
        if self._unread:
            return self._unread.pop()
        return await self._content.readline()

    async def _read_until_first_boundary(self) -> None:
        while True:
            chunk = await self._readline()
            if chunk == b"":
                raise ValueError(
                    "Could not find starting boundary %r" % (self._boundary)
                )
            chunk = chunk.rstrip()
            if chunk == self._boundary:
                return
            elif chunk == self._boundary + b"--":
                self._at_eof = True
                return

    async def _read_boundary(self) -> None:
        chunk = (await self._readline()).rstrip()
        if chunk == self._boundary:
            pass
        elif chunk == self._boundary + b"--":
            self._at_eof = True
            epilogue = await self._readline()
            next_line = await self._readline()

            # the epilogue is expected and then either the end of input or the
            # parent multipart boundary, if the parent boundary is found then
            # it should be marked as unread and handed to the parent for
            # processing
            if next_line[:2] == b"--":
                self._unread.append(next_line)
            # otherwise the request is likely missing an epilogue and both
            # lines should be passed to the parent for processing
            # (this handles the old behavior gracefully)
            else:
                self._unread.extend([next_line, epilogue])
        else:
            raise ValueError(f"Invalid boundary {chunk!r}, expected {self._boundary!r}")

    async def _read_headers(self) -> "CIMultiDictProxy[str]":
        lines = [b""]
        while True:
            chunk = await self._content.readline()
            chunk = chunk.strip()
            lines.append(chunk)
            if not chunk:
                break
        parser = HeadersParser()
        headers, raw_headers = parser.parse_headers(lines)
        return headers

    async def _maybe_release_last_part(self) -> None:
        """Ensures that the last read body part is read completely."""
        if self._last_part is not None:
            if not self._last_part.at_eof():
                await self._last_part.release()
            self._unread.extend(self._last_part._unread)
            self._last_part = None


_Part = Tuple[Payload, str, str]


class MultipartWriter(Payload):
    """Multipart body writer."""

    def __init__(self, subtype: str = "mixed", boundary: Optional[str] = None) -> None:
        boundary = boundary if boundary is not None else uuid.uuid4().hex
        # The underlying Payload API demands a str (utf-8), not bytes,
        # so we need to ensure we don't lose anything during conversion.
        # As a result, require the boundary to be ASCII only.
        # In both situations.

        try:
            self._boundary = boundary.encode("ascii")
        except UnicodeEncodeError:
            raise ValueError("boundary should contain ASCII only chars") from None
        ctype = f"multipart/{subtype}; boundary={self._boundary_value}"

        super().__init__(None, content_type=ctype)

        self._parts: List[_Part] = []

    def __enter__(self) -> "MultipartWriter":
        return self

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        pass

    def __iter__(self) -> Iterator[_Part]:
        return iter(self._parts)

    def __len__(self) -> int:
        return len(self._parts)

    def __bool__(self) -> bool:
        return True

    _valid_tchar_regex = re.compile(rb"\A[!#$%&'*+\-.^_`|~\w]+\Z")
    _invalid_qdtext_char_regex = re.compile(rb"[\x00-\x08\x0A-\x1F\x7F]")

    @property
    def _boundary_value(self) -> str:
        """Wrap boundary parameter value in quotes, if necessary.

        Reads self.boundary and returns a unicode sting.
        """
        # Refer to RFCs 7231, 7230, 5234.
        #
        # parameter      = token "=" ( token / quoted-string )
        # token          = 1*tchar
        # quoted-string  = DQUOTE *( qdtext / quoted-pair ) DQUOTE
        # qdtext         = HTAB / SP / %x21 / %x23-5B / %x5D-7E / obs-text
        # obs-text       = %x80-FF
        # quoted-pair    = "\" ( HTAB / SP / VCHAR / obs-text )
        # tchar          = "!" / "#" / "$" / "%" / "&" / "'" / "*"
        #                  / "+" / "-" / "." / "^" / "_" / "`" / "|" / "~"
        #                  / DIGIT / ALPHA
        #                  ; any VCHAR, except delimiters
        # VCHAR           = %x21-7E
        value = self._boundary
        if re.match(self._valid_tchar_regex, value):
            return value.decode("ascii")  # cannot fail

        if re.search(self._invalid_qdtext_char_regex, value):
            raise ValueError("boundary value contains invalid characters")

        # escape %x5C and %x22
        quoted_value_content = value.replace(b"\\", b"\\\\")
        quoted_value_content = quoted_value_content.replace(b'"', b'\\"')

        return '"' + quoted_value_content.decode("ascii") + '"'

    @property
    def boundary(self) -> str:
        return self._boundary.decode("ascii")

    def append(self, obj: Any, headers: Optional[MultiMapping[str]] = None) -> Payload:
        if headers is None:
            headers = CIMultiDict()

        if isinstance(obj, Payload):
            obj.headers.update(headers)
            return self.append_payload(obj)
        else:
            try:
                payload = get_payload(obj, headers=headers)
            except LookupError:
                raise TypeError("Cannot create payload from %r" % obj)
            else:
                return self.append_payload(payload)

    def append_payload(self, payload: Payload) -> Payload:
        """Adds a new body part to multipart writer."""
        # compression
        encoding: Optional[str] = payload.headers.get(
            CONTENT_ENCODING,
            "",
        ).lower()
        if encoding and encoding not in ("deflate", "gzip", "identity"):
            raise RuntimeError(f"unknown content encoding: {encoding}")
        if encoding == "identity":
            encoding = None

        # te encoding
        te_encoding: Optional[str] = payload.headers.get(
            CONTENT_TRANSFER_ENCODING,
            "",
        ).lower()
        if te_encoding not in ("", "base64", "quoted-printable", "binary"):
            raise RuntimeError(
                "unknown content transfer encoding: {}" "".format(te_encoding)
            )
        if te_encoding == "binary":
            te_encoding = None

        # size
        size = payload.size
        if size is not None and not (encoding or te_encoding):
            payload.headers[CONTENT_LENGTH] = str(size)

        self._parts.append((payload, encoding, te_encoding))  # type: ignore[arg-type]
        return payload

    def append_json(
        self, obj: Any, headers: Optional[MultiMapping[str]] = None
    ) -> Payload:
        """Helper to append JSON part."""
        if headers is None:
            headers = CIMultiDict()

        return self.append_payload(JsonPayload(obj, headers=headers))

    def append_form(
        self,
        obj: Union[Sequence[Tuple[str, str]], Mapping[str, str]],
        headers: Optional[MultiMapping[str]] = None,
    ) -> Payload:
        """Helper to append form urlencoded part."""
        assert isinstance(obj, (Sequence, Mapping))

        if headers is None:
            headers = CIMultiDict()

        if isinstance(obj, Mapping):
            obj = list(obj.items())
        data = urlencode(obj, doseq=True)

        return self.append_payload(
            StringPayload(
                data, headers=headers, content_type="application/x-www-form-urlencoded"
            )
        )

    @property
    def size(self) -> Optional[int]:
        """Size of the payload."""
        total = 0
        for part, encoding, te_encoding in self._parts:
            if encoding or te_encoding or part.size is None:
                return None

            total += int(
                2
                + len(self._boundary)
                + 2
                + part.size  # b'--'+self._boundary+b'\r\n'
                + len(part._binary_headers)
                + 2  # b'\r\n'
            )

        total += 2 + len(self._boundary) + 4  # b'--'+self._boundary+b'--\r\n'
        return total

    async def write(self, writer: Any, close_boundary: bool = True) -> None:
        """Write body."""
        for part, encoding, te_encoding in self._parts:
            await writer.write(b"--" + self._boundary + b"\r\n")
            await writer.write(part._binary_headers)

            if encoding or te_encoding:
                w = MultipartPayloadWriter(writer)
                if encoding:
                    w.enable_compression(encoding)
                if te_encoding:
                    w.enable_encoding(te_encoding)
                await part.write(w)  # type: ignore[arg-type]
                await w.write_eof()
            else:
                await part.write(writer)

            await writer.write(b"\r\n")

        if close_boundary:
            await writer.write(b"--" + self._boundary + b"--\r\n")


class MultipartPayloadWriter:
    def __init__(self, writer: Any) -> None:
        self._writer = writer
        self._encoding: Optional[str] = None
        self._compress: Any = None
        self._encoding_buffer: Optional[bytearray] = None

    def enable_encoding(self, encoding: str) -> None:
        if encoding == "base64":
            self._encoding = encoding
            self._encoding_buffer = bytearray()
        elif encoding == "quoted-printable":
            self._encoding = "quoted-printable"

    def enable_compression(
        self, encoding: str = "deflate", strategy: int = zlib.Z_DEFAULT_STRATEGY
    ) -> None:
        zlib_mode = 16 + zlib.MAX_WBITS if encoding == "gzip" else -zlib.MAX_WBITS
        self._compress = zlib.compressobj(wbits=zlib_mode, strategy=strategy)

    async def write_eof(self) -> None:
        if self._compress is not None:
            chunk = self._compress.flush()
            if chunk:
                self._compress = None
                await self.write(chunk)

        if self._encoding == "base64":
            if self._encoding_buffer:
                await self._writer.write(base64.b64encode(self._encoding_buffer))

    async def write(self, chunk: bytes) -> None:
        if self._compress is not None:
            if chunk:
                chunk = self._compress.compress(chunk)
                if not chunk:
                    return

        if self._encoding == "base64":
            buf = self._encoding_buffer
            assert buf is not None
            buf.extend(chunk)

            if buf:
                div, mod = divmod(len(buf), 3)
                enc_chunk, self._encoding_buffer = (buf[: div * 3], buf[div * 3 :])
                if enc_chunk:
                    b64chunk = base64.b64encode(enc_chunk)
                    await self._writer.write(b64chunk)
        elif self._encoding == "quoted-printable":
            await self._writer.write(binascii.b2a_qp(chunk))
        else:
            await self._writer.write(chunk)
