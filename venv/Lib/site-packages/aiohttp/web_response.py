import asyncio
import collections.abc
import datetime
import enum
import json
import math
import time
import warnings
import zlib
from concurrent.futures import Executor
from http.cookies import Morsel, SimpleCookie
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Tuple,
    Union,
    cast,
)

from multidict import CIMultiDict, istr

from . import hdrs, payload
from .abc import AbstractStreamWriter
from .helpers import (
    ETAG_ANY,
    PY_38,
    QUOTED_ETAG_RE,
    ETag,
    HeadersMixin,
    parse_http_date,
    rfc822_formatted_time,
    sentinel,
    validate_etag_value,
)
from .http import RESPONSES, SERVER_SOFTWARE, HttpVersion10, HttpVersion11
from .payload import Payload
from .typedefs import JSONEncoder, LooseHeaders

__all__ = ("ContentCoding", "StreamResponse", "Response", "json_response")


if TYPE_CHECKING:  # pragma: no cover
    from .web_request import BaseRequest

    BaseClass = MutableMapping[str, Any]
else:
    BaseClass = collections.abc.MutableMapping


if not PY_38:
    # allow samesite to be used in python < 3.8
    # already permitted in python 3.8, see https://bugs.python.org/issue29613
    Morsel._reserved["samesite"] = "SameSite"  # type: ignore[attr-defined]


class ContentCoding(enum.Enum):
    # The content codings that we have support for.
    #
    # Additional registered codings are listed at:
    # https://www.iana.org/assignments/http-parameters/http-parameters.xhtml#content-coding
    deflate = "deflate"
    gzip = "gzip"
    identity = "identity"


############################################################
# HTTP Response classes
############################################################


class StreamResponse(BaseClass, HeadersMixin):

    _length_check = True

    def __init__(
        self,
        *,
        status: int = 200,
        reason: Optional[str] = None,
        headers: Optional[LooseHeaders] = None,
    ) -> None:
        self._body = None
        self._keep_alive: Optional[bool] = None
        self._chunked = False
        self._compression = False
        self._compression_force: Optional[ContentCoding] = None
        self._cookies: SimpleCookie[str] = SimpleCookie()

        self._req: Optional[BaseRequest] = None
        self._payload_writer: Optional[AbstractStreamWriter] = None
        self._eof_sent = False
        self._body_length = 0
        self._state: Dict[str, Any] = {}

        if headers is not None:
            self._headers: CIMultiDict[str] = CIMultiDict(headers)
        else:
            self._headers = CIMultiDict()

        self.set_status(status, reason)

    @property
    def prepared(self) -> bool:
        return self._payload_writer is not None

    @property
    def task(self) -> "Optional[asyncio.Task[None]]":
        if self._req:
            return self._req.task
        else:
            return None

    @property
    def status(self) -> int:
        return self._status

    @property
    def chunked(self) -> bool:
        return self._chunked

    @property
    def compression(self) -> bool:
        return self._compression

    @property
    def reason(self) -> str:
        return self._reason

    def set_status(
        self,
        status: int,
        reason: Optional[str] = None,
        _RESPONSES: Mapping[int, Tuple[str, str]] = RESPONSES,
    ) -> None:
        assert not self.prepared, (
            "Cannot change the response status code after " "the headers have been sent"
        )
        self._status = int(status)
        if reason is None:
            try:
                reason = _RESPONSES[self._status][0]
            except Exception:
                reason = ""
        self._reason = reason

    @property
    def keep_alive(self) -> Optional[bool]:
        return self._keep_alive

    def force_close(self) -> None:
        self._keep_alive = False

    @property
    def body_length(self) -> int:
        return self._body_length

    @property
    def output_length(self) -> int:
        warnings.warn("output_length is deprecated", DeprecationWarning)
        assert self._payload_writer
        return self._payload_writer.buffer_size

    def enable_chunked_encoding(self, chunk_size: Optional[int] = None) -> None:
        """Enables automatic chunked transfer encoding."""
        self._chunked = True

        if hdrs.CONTENT_LENGTH in self._headers:
            raise RuntimeError(
                "You can't enable chunked encoding when " "a content length is set"
            )
        if chunk_size is not None:
            warnings.warn("Chunk size is deprecated #1615", DeprecationWarning)

    def enable_compression(
        self, force: Optional[Union[bool, ContentCoding]] = None
    ) -> None:
        """Enables response compression encoding."""
        # Backwards compatibility for when force was a bool <0.17.
        if type(force) == bool:
            force = ContentCoding.deflate if force else ContentCoding.identity
            warnings.warn(
                "Using boolean for force is deprecated #3318", DeprecationWarning
            )
        elif force is not None:
            assert isinstance(force, ContentCoding), (
                "force should one of " "None, bool or " "ContentEncoding"
            )

        self._compression = True
        self._compression_force = force

    @property
    def headers(self) -> "CIMultiDict[str]":
        return self._headers

    @property
    def cookies(self) -> "SimpleCookie[str]":
        return self._cookies

    def set_cookie(
        self,
        name: str,
        value: str,
        *,
        expires: Optional[str] = None,
        domain: Optional[str] = None,
        max_age: Optional[Union[int, str]] = None,
        path: str = "/",
        secure: Optional[bool] = None,
        httponly: Optional[bool] = None,
        version: Optional[str] = None,
        samesite: Optional[str] = None,
    ) -> None:
        """Set or update response cookie.

        Sets new cookie or updates existent with new value.
        Also updates only those params which are not None.
        """
        old = self._cookies.get(name)
        if old is not None and old.coded_value == "":
            # deleted cookie
            self._cookies.pop(name, None)

        self._cookies[name] = value
        c = self._cookies[name]

        if expires is not None:
            c["expires"] = expires
        elif c.get("expires") == "Thu, 01 Jan 1970 00:00:00 GMT":
            del c["expires"]

        if domain is not None:
            c["domain"] = domain

        if max_age is not None:
            c["max-age"] = str(max_age)
        elif "max-age" in c:
            del c["max-age"]

        c["path"] = path

        if secure is not None:
            c["secure"] = secure
        if httponly is not None:
            c["httponly"] = httponly
        if version is not None:
            c["version"] = version
        if samesite is not None:
            c["samesite"] = samesite

    def del_cookie(
        self, name: str, *, domain: Optional[str] = None, path: str = "/"
    ) -> None:
        """Delete cookie.

        Creates new empty expired cookie.
        """
        # TODO: do we need domain/path here?
        self._cookies.pop(name, None)
        self.set_cookie(
            name,
            "",
            max_age=0,
            expires="Thu, 01 Jan 1970 00:00:00 GMT",
            domain=domain,
            path=path,
        )

    @property
    def content_length(self) -> Optional[int]:
        # Just a placeholder for adding setter
        return super().content_length

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        if value is not None:
            value = int(value)
            if self._chunked:
                raise RuntimeError(
                    "You can't set content length when " "chunked encoding is enable"
                )
            self._headers[hdrs.CONTENT_LENGTH] = str(value)
        else:
            self._headers.pop(hdrs.CONTENT_LENGTH, None)

    @property
    def content_type(self) -> str:
        # Just a placeholder for adding setter
        return super().content_type

    @content_type.setter
    def content_type(self, value: str) -> None:
        self.content_type  # read header values if needed
        self._content_type = str(value)
        self._generate_content_type_header()

    @property
    def charset(self) -> Optional[str]:
        # Just a placeholder for adding setter
        return super().charset

    @charset.setter
    def charset(self, value: Optional[str]) -> None:
        ctype = self.content_type  # read header values if needed
        if ctype == "application/octet-stream":
            raise RuntimeError(
                "Setting charset for application/octet-stream "
                "doesn't make sense, setup content_type first"
            )
        assert self._content_dict is not None
        if value is None:
            self._content_dict.pop("charset", None)
        else:
            self._content_dict["charset"] = str(value).lower()
        self._generate_content_type_header()

    @property
    def last_modified(self) -> Optional[datetime.datetime]:
        """The value of Last-Modified HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self._headers.get(hdrs.LAST_MODIFIED))

    @last_modified.setter
    def last_modified(
        self, value: Optional[Union[int, float, datetime.datetime, str]]
    ) -> None:
        if value is None:
            self._headers.pop(hdrs.LAST_MODIFIED, None)
        elif isinstance(value, (int, float)):
            self._headers[hdrs.LAST_MODIFIED] = time.strftime(
                "%a, %d %b %Y %H:%M:%S GMT", time.gmtime(math.ceil(value))
            )
        elif isinstance(value, datetime.datetime):
            self._headers[hdrs.LAST_MODIFIED] = time.strftime(
                "%a, %d %b %Y %H:%M:%S GMT", value.utctimetuple()
            )
        elif isinstance(value, str):
            self._headers[hdrs.LAST_MODIFIED] = value

    @property
    def etag(self) -> Optional[ETag]:
        quoted_value = self._headers.get(hdrs.ETAG)
        if not quoted_value:
            return None
        elif quoted_value == ETAG_ANY:
            return ETag(value=ETAG_ANY)
        match = QUOTED_ETAG_RE.fullmatch(quoted_value)
        if not match:
            return None
        is_weak, value = match.group(1, 2)
        return ETag(
            is_weak=bool(is_weak),
            value=value,
        )

    @etag.setter
    def etag(self, value: Optional[Union[ETag, str]]) -> None:
        if value is None:
            self._headers.pop(hdrs.ETAG, None)
        elif (isinstance(value, str) and value == ETAG_ANY) or (
            isinstance(value, ETag) and value.value == ETAG_ANY
        ):
            self._headers[hdrs.ETAG] = ETAG_ANY
        elif isinstance(value, str):
            validate_etag_value(value)
            self._headers[hdrs.ETAG] = f'"{value}"'
        elif isinstance(value, ETag) and isinstance(value.value, str):
            validate_etag_value(value.value)
            hdr_value = f'W/"{value.value}"' if value.is_weak else f'"{value.value}"'
            self._headers[hdrs.ETAG] = hdr_value
        else:
            raise ValueError(
                f"Unsupported etag type: {type(value)}. "
                f"etag must be str, ETag or None"
            )

    def _generate_content_type_header(
        self, CONTENT_TYPE: istr = hdrs.CONTENT_TYPE
    ) -> None:
        assert self._content_dict is not None
        assert self._content_type is not None
        params = "; ".join(f"{k}={v}" for k, v in self._content_dict.items())
        if params:
            ctype = self._content_type + "; " + params
        else:
            ctype = self._content_type
        self._headers[CONTENT_TYPE] = ctype

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        if coding != ContentCoding.identity:
            assert self._payload_writer is not None
            self._headers[hdrs.CONTENT_ENCODING] = coding.value
            self._payload_writer.enable_compression(coding.value)
            # Compressed payload may have different content length,
            # remove the header
            self._headers.popall(hdrs.CONTENT_LENGTH, None)

    async def _start_compression(self, request: "BaseRequest") -> None:
        if self._compression_force:
            await self._do_start_compression(self._compression_force)
        else:
            accept_encoding = request.headers.get(hdrs.ACCEPT_ENCODING, "").lower()
            for coding in ContentCoding:
                if coding.value in accept_encoding:
                    await self._do_start_compression(coding)
                    return

    async def prepare(self, request: "BaseRequest") -> Optional[AbstractStreamWriter]:
        if self._eof_sent:
            return None
        if self._payload_writer is not None:
            return self._payload_writer

        return await self._start(request)

    async def _start(self, request: "BaseRequest") -> AbstractStreamWriter:
        self._req = request
        writer = self._payload_writer = request._payload_writer

        await self._prepare_headers()
        await request._prepare_hook(self)
        await self._write_headers()

        return writer

    async def _prepare_headers(self) -> None:
        request = self._req
        assert request is not None
        writer = self._payload_writer
        assert writer is not None
        keep_alive = self._keep_alive
        if keep_alive is None:
            keep_alive = request.keep_alive
        self._keep_alive = keep_alive

        version = request.version

        headers = self._headers
        for cookie in self._cookies.values():
            value = cookie.output(header="")[1:]
            headers.add(hdrs.SET_COOKIE, value)

        if self._compression:
            await self._start_compression(request)

        if self._chunked:
            if version != HttpVersion11:
                raise RuntimeError(
                    "Using chunked encoding is forbidden "
                    "for HTTP/{0.major}.{0.minor}".format(request.version)
                )
            writer.enable_chunking()
            headers[hdrs.TRANSFER_ENCODING] = "chunked"
            if hdrs.CONTENT_LENGTH in headers:
                del headers[hdrs.CONTENT_LENGTH]
        elif self._length_check:
            writer.length = self.content_length
            if writer.length is None:
                if version >= HttpVersion11 and self.status != 204:
                    writer.enable_chunking()
                    headers[hdrs.TRANSFER_ENCODING] = "chunked"
                    if hdrs.CONTENT_LENGTH in headers:
                        del headers[hdrs.CONTENT_LENGTH]
                else:
                    keep_alive = False
            # HTTP 1.1: https://tools.ietf.org/html/rfc7230#section-3.3.2
            # HTTP 1.0: https://tools.ietf.org/html/rfc1945#section-10.4
            elif version >= HttpVersion11 and self.status in (100, 101, 102, 103, 204):
                del headers[hdrs.CONTENT_LENGTH]

        if self.status not in (204, 304):
            headers.setdefault(hdrs.CONTENT_TYPE, "application/octet-stream")
        headers.setdefault(hdrs.DATE, rfc822_formatted_time())
        headers.setdefault(hdrs.SERVER, SERVER_SOFTWARE)

        # connection header
        if hdrs.CONNECTION not in headers:
            if keep_alive:
                if version == HttpVersion10:
                    headers[hdrs.CONNECTION] = "keep-alive"
            else:
                if version == HttpVersion11:
                    headers[hdrs.CONNECTION] = "close"

    async def _write_headers(self) -> None:
        request = self._req
        assert request is not None
        writer = self._payload_writer
        assert writer is not None
        # status line
        version = request.version
        status_line = "HTTP/{}.{} {} {}".format(
            version[0], version[1], self._status, self._reason
        )
        await writer.write_headers(status_line, self._headers)

    async def write(self, data: bytes) -> None:
        assert isinstance(
            data, (bytes, bytearray, memoryview)
        ), "data argument must be byte-ish (%r)" % type(data)

        if self._eof_sent:
            raise RuntimeError("Cannot call write() after write_eof()")
        if self._payload_writer is None:
            raise RuntimeError("Cannot call write() before prepare()")

        await self._payload_writer.write(data)

    async def drain(self) -> None:
        assert not self._eof_sent, "EOF has already been sent"
        assert self._payload_writer is not None, "Response has not been started"
        warnings.warn(
            "drain method is deprecated, use await resp.write()",
            DeprecationWarning,
            stacklevel=2,
        )
        await self._payload_writer.drain()

    async def write_eof(self, data: bytes = b"") -> None:
        assert isinstance(
            data, (bytes, bytearray, memoryview)
        ), "data argument must be byte-ish (%r)" % type(data)

        if self._eof_sent:
            return

        assert self._payload_writer is not None, "Response has not been started"

        await self._payload_writer.write_eof(data)
        self._eof_sent = True
        self._req = None
        self._body_length = self._payload_writer.output_size
        self._payload_writer = None

    def __repr__(self) -> str:
        if self._eof_sent:
            info = "eof"
        elif self.prepared:
            assert self._req is not None
            info = f"{self._req.method} {self._req.path} "
        else:
            info = "not prepared"
        return f"<{self.__class__.__name__} {self.reason} {info}>"

    def __getitem__(self, key: str) -> Any:
        return self._state[key]

    def __setitem__(self, key: str, value: Any) -> None:
        self._state[key] = value

    def __delitem__(self, key: str) -> None:
        del self._state[key]

    def __len__(self) -> int:
        return len(self._state)

    def __iter__(self) -> Iterator[str]:
        return iter(self._state)

    def __hash__(self) -> int:
        return hash(id(self))

    def __eq__(self, other: object) -> bool:
        return self is other


class Response(StreamResponse):
    def __init__(
        self,
        *,
        body: Any = None,
        status: int = 200,
        reason: Optional[str] = None,
        text: Optional[str] = None,
        headers: Optional[LooseHeaders] = None,
        content_type: Optional[str] = None,
        charset: Optional[str] = None,
        zlib_executor_size: Optional[int] = None,
        zlib_executor: Optional[Executor] = None,
    ) -> None:
        if body is not None and text is not None:
            raise ValueError("body and text are not allowed together")

        if headers is None:
            real_headers: CIMultiDict[str] = CIMultiDict()
        elif not isinstance(headers, CIMultiDict):
            real_headers = CIMultiDict(headers)
        else:
            real_headers = headers  # = cast('CIMultiDict[str]', headers)

        if content_type is not None and "charset" in content_type:
            raise ValueError("charset must not be in content_type " "argument")

        if text is not None:
            if hdrs.CONTENT_TYPE in real_headers:
                if content_type or charset:
                    raise ValueError(
                        "passing both Content-Type header and "
                        "content_type or charset params "
                        "is forbidden"
                    )
            else:
                # fast path for filling headers
                if not isinstance(text, str):
                    raise TypeError("text argument must be str (%r)" % type(text))
                if content_type is None:
                    content_type = "text/plain"
                if charset is None:
                    charset = "utf-8"
                real_headers[hdrs.CONTENT_TYPE] = content_type + "; charset=" + charset
                body = text.encode(charset)
                text = None
        else:
            if hdrs.CONTENT_TYPE in real_headers:
                if content_type is not None or charset is not None:
                    raise ValueError(
                        "passing both Content-Type header and "
                        "content_type or charset params "
                        "is forbidden"
                    )
            else:
                if content_type is not None:
                    if charset is not None:
                        content_type += "; charset=" + charset
                    real_headers[hdrs.CONTENT_TYPE] = content_type

        super().__init__(status=status, reason=reason, headers=real_headers)

        if text is not None:
            self.text = text
        else:
            self.body = body

        self._compressed_body: Optional[bytes] = None
        self._zlib_executor_size = zlib_executor_size
        self._zlib_executor = zlib_executor

    @property
    def body(self) -> Optional[Union[bytes, Payload]]:
        return self._body

    @body.setter
    def body(
        self,
        body: bytes,
        CONTENT_TYPE: istr = hdrs.CONTENT_TYPE,
        CONTENT_LENGTH: istr = hdrs.CONTENT_LENGTH,
    ) -> None:
        if body is None:
            self._body: Optional[bytes] = None
            self._body_payload: bool = False
        elif isinstance(body, (bytes, bytearray)):
            self._body = body
            self._body_payload = False
        else:
            try:
                self._body = body = payload.PAYLOAD_REGISTRY.get(body)
            except payload.LookupError:
                raise ValueError("Unsupported body type %r" % type(body))

            self._body_payload = True

            headers = self._headers

            # set content-length header if needed
            if not self._chunked and CONTENT_LENGTH not in headers:
                size = body.size
                if size is not None:
                    headers[CONTENT_LENGTH] = str(size)

            # set content-type
            if CONTENT_TYPE not in headers:
                headers[CONTENT_TYPE] = body.content_type

            # copy payload headers
            if body.headers:
                for (key, value) in body.headers.items():
                    if key not in headers:
                        headers[key] = value

        self._compressed_body = None

    @property
    def text(self) -> Optional[str]:
        if self._body is None:
            return None
        return self._body.decode(self.charset or "utf-8")

    @text.setter
    def text(self, text: str) -> None:
        assert text is None or isinstance(
            text, str
        ), "text argument must be str (%r)" % type(text)

        if self.content_type == "application/octet-stream":
            self.content_type = "text/plain"
        if self.charset is None:
            self.charset = "utf-8"

        self._body = text.encode(self.charset)
        self._body_payload = False
        self._compressed_body = None

    @property
    def content_length(self) -> Optional[int]:
        if self._chunked:
            return None

        if hdrs.CONTENT_LENGTH in self._headers:
            return super().content_length

        if self._compressed_body is not None:
            # Return length of the compressed body
            return len(self._compressed_body)
        elif self._body_payload:
            # A payload without content length, or a compressed payload
            return None
        elif self._body is not None:
            return len(self._body)
        else:
            return 0

    @content_length.setter
    def content_length(self, value: Optional[int]) -> None:
        raise RuntimeError("Content length is set automatically")

    async def write_eof(self, data: bytes = b"") -> None:
        if self._eof_sent:
            return
        if self._compressed_body is None:
            body: Optional[Union[bytes, Payload]] = self._body
        else:
            body = self._compressed_body
        assert not data, f"data arg is not supported, got {data!r}"
        assert self._req is not None
        assert self._payload_writer is not None
        if body is not None:
            if self._req._method == hdrs.METH_HEAD or self._status in [204, 304]:
                await super().write_eof()
            elif self._body_payload:
                payload = cast(Payload, body)
                await payload.write(self._payload_writer)
                await super().write_eof()
            else:
                await super().write_eof(cast(bytes, body))
        else:
            await super().write_eof()

    async def _start(self, request: "BaseRequest") -> AbstractStreamWriter:
        if not self._chunked and hdrs.CONTENT_LENGTH not in self._headers:
            if not self._body_payload:
                if self._body is not None:
                    self._headers[hdrs.CONTENT_LENGTH] = str(len(self._body))
                else:
                    self._headers[hdrs.CONTENT_LENGTH] = "0"

        return await super()._start(request)

    def _compress_body(self, zlib_mode: int) -> None:
        assert zlib_mode > 0
        compressobj = zlib.compressobj(wbits=zlib_mode)
        body_in = self._body
        assert body_in is not None
        self._compressed_body = compressobj.compress(body_in) + compressobj.flush()

    async def _do_start_compression(self, coding: ContentCoding) -> None:
        if self._body_payload or self._chunked:
            return await super()._do_start_compression(coding)

        if coding != ContentCoding.identity:
            # Instead of using _payload_writer.enable_compression,
            # compress the whole body
            zlib_mode = (
                16 + zlib.MAX_WBITS if coding == ContentCoding.gzip else zlib.MAX_WBITS
            )
            body_in = self._body
            assert body_in is not None
            if (
                self._zlib_executor_size is not None
                and len(body_in) > self._zlib_executor_size
            ):
                await asyncio.get_event_loop().run_in_executor(
                    self._zlib_executor, self._compress_body, zlib_mode
                )
            else:
                self._compress_body(zlib_mode)

            body_out = self._compressed_body
            assert body_out is not None

            self._headers[hdrs.CONTENT_ENCODING] = coding.value
            self._headers[hdrs.CONTENT_LENGTH] = str(len(body_out))


def json_response(
    data: Any = sentinel,
    *,
    text: Optional[str] = None,
    body: Optional[bytes] = None,
    status: int = 200,
    reason: Optional[str] = None,
    headers: Optional[LooseHeaders] = None,
    content_type: str = "application/json",
    dumps: JSONEncoder = json.dumps,
) -> Response:
    if data is not sentinel:
        if text or body:
            raise ValueError("only one of data, text, or body should be specified")
        else:
            text = dumps(data)
    return Response(
        text=text,
        body=body,
        status=status,
        reason=reason,
        headers=headers,
        content_type=content_type,
    )
