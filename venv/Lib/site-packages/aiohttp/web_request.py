import asyncio
import datetime
import io
import re
import socket
import string
import tempfile
import types
import warnings
from http.cookies import SimpleCookie
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    Mapping,
    MutableMapping,
    Optional,
    Pattern,
    Tuple,
    Union,
    cast,
)
from urllib.parse import parse_qsl

import attr
from multidict import CIMultiDict, CIMultiDictProxy, MultiDict, MultiDictProxy
from yarl import URL

from . import hdrs
from .abc import AbstractStreamWriter
from .helpers import (
    DEBUG,
    ETAG_ANY,
    LIST_QUOTED_ETAG_RE,
    ChainMapProxy,
    ETag,
    HeadersMixin,
    parse_http_date,
    reify,
    sentinel,
)
from .http_parser import RawRequestMessage
from .http_writer import HttpVersion
from .multipart import BodyPartReader, MultipartReader
from .streams import EmptyStreamReader, StreamReader
from .typedefs import (
    DEFAULT_JSON_DECODER,
    Final,
    JSONDecoder,
    LooseHeaders,
    RawHeaders,
    StrOrURL,
)
from .web_exceptions import HTTPRequestEntityTooLarge
from .web_response import StreamResponse

__all__ = ("BaseRequest", "FileField", "Request")


if TYPE_CHECKING:  # pragma: no cover
    from .web_app import Application
    from .web_protocol import RequestHandler
    from .web_urldispatcher import UrlMappingMatchInfo


@attr.s(auto_attribs=True, frozen=True, slots=True)
class FileField:
    name: str
    filename: str
    file: io.BufferedReader
    content_type: str
    headers: "CIMultiDictProxy[str]"


_TCHAR: Final[str] = string.digits + string.ascii_letters + r"!#$%&'*+.^_`|~-"
# '-' at the end to prevent interpretation as range in a char class

_TOKEN: Final[str] = rf"[{_TCHAR}]+"

_QDTEXT: Final[str] = r"[{}]".format(
    r"".join(chr(c) for c in (0x09, 0x20, 0x21) + tuple(range(0x23, 0x7F)))
)
# qdtext includes 0x5C to escape 0x5D ('\]')
# qdtext excludes obs-text (because obsoleted, and encoding not specified)

_QUOTED_PAIR: Final[str] = r"\\[\t !-~]"

_QUOTED_STRING: Final[str] = r'"(?:{quoted_pair}|{qdtext})*"'.format(
    qdtext=_QDTEXT, quoted_pair=_QUOTED_PAIR
)

_FORWARDED_PAIR: Final[
    str
] = r"({token})=({token}|{quoted_string})(:\d{{1,4}})?".format(
    token=_TOKEN, quoted_string=_QUOTED_STRING
)

_QUOTED_PAIR_REPLACE_RE: Final[Pattern[str]] = re.compile(r"\\([\t !-~])")
# same pattern as _QUOTED_PAIR but contains a capture group

_FORWARDED_PAIR_RE: Final[Pattern[str]] = re.compile(_FORWARDED_PAIR)

############################################################
# HTTP Request
############################################################


class BaseRequest(MutableMapping[str, Any], HeadersMixin):

    POST_METHODS = {
        hdrs.METH_PATCH,
        hdrs.METH_POST,
        hdrs.METH_PUT,
        hdrs.METH_TRACE,
        hdrs.METH_DELETE,
    }

    ATTRS = HeadersMixin.ATTRS | frozenset(
        [
            "_message",
            "_protocol",
            "_payload_writer",
            "_payload",
            "_headers",
            "_method",
            "_version",
            "_rel_url",
            "_post",
            "_read_bytes",
            "_state",
            "_cache",
            "_task",
            "_client_max_size",
            "_loop",
            "_transport_sslcontext",
            "_transport_peername",
        ]
    )

    def __init__(
        self,
        message: RawRequestMessage,
        payload: StreamReader,
        protocol: "RequestHandler",
        payload_writer: AbstractStreamWriter,
        task: "asyncio.Task[None]",
        loop: asyncio.AbstractEventLoop,
        *,
        client_max_size: int = 1024**2,
        state: Optional[Dict[str, Any]] = None,
        scheme: Optional[str] = None,
        host: Optional[str] = None,
        remote: Optional[str] = None,
    ) -> None:
        if state is None:
            state = {}
        self._message = message
        self._protocol = protocol
        self._payload_writer = payload_writer

        self._payload = payload
        self._headers = message.headers
        self._method = message.method
        self._version = message.version
        self._cache: Dict[str, Any] = {}
        url = message.url
        if url.is_absolute():
            # absolute URL is given,
            # override auto-calculating url, host, and scheme
            # all other properties should be good
            self._cache["url"] = url
            self._cache["host"] = url.host
            self._cache["scheme"] = url.scheme
            self._rel_url = url.relative()
        else:
            self._rel_url = message.url
        self._post: Optional[MultiDictProxy[Union[str, bytes, FileField]]] = None
        self._read_bytes: Optional[bytes] = None

        self._state = state
        self._task = task
        self._client_max_size = client_max_size
        self._loop = loop

        transport = self._protocol.transport
        assert transport is not None
        self._transport_sslcontext = transport.get_extra_info("sslcontext")
        self._transport_peername = transport.get_extra_info("peername")

        if scheme is not None:
            self._cache["scheme"] = scheme
        if host is not None:
            self._cache["host"] = host
        if remote is not None:
            self._cache["remote"] = remote

    def clone(
        self,
        *,
        method: str = sentinel,
        rel_url: StrOrURL = sentinel,
        headers: LooseHeaders = sentinel,
        scheme: str = sentinel,
        host: str = sentinel,
        remote: str = sentinel,
    ) -> "BaseRequest":
        """Clone itself with replacement some attributes.

        Creates and returns a new instance of Request object. If no parameters
        are given, an exact copy is returned. If a parameter is not passed, it
        will reuse the one from the current request object.
        """
        if self._read_bytes:
            raise RuntimeError("Cannot clone request " "after reading its content")

        dct: Dict[str, Any] = {}
        if method is not sentinel:
            dct["method"] = method
        if rel_url is not sentinel:
            new_url = URL(rel_url)
            dct["url"] = new_url
            dct["path"] = str(new_url)
        if headers is not sentinel:
            # a copy semantic
            dct["headers"] = CIMultiDictProxy(CIMultiDict(headers))
            dct["raw_headers"] = tuple(
                (k.encode("utf-8"), v.encode("utf-8")) for k, v in headers.items()
            )

        message = self._message._replace(**dct)

        kwargs = {}
        if scheme is not sentinel:
            kwargs["scheme"] = scheme
        if host is not sentinel:
            kwargs["host"] = host
        if remote is not sentinel:
            kwargs["remote"] = remote

        return self.__class__(
            message,
            self._payload,
            self._protocol,
            self._payload_writer,
            self._task,
            self._loop,
            client_max_size=self._client_max_size,
            state=self._state.copy(),
            **kwargs,
        )

    @property
    def task(self) -> "asyncio.Task[None]":
        return self._task

    @property
    def protocol(self) -> "RequestHandler":
        return self._protocol

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        if self._protocol is None:
            return None
        return self._protocol.transport

    @property
    def writer(self) -> AbstractStreamWriter:
        return self._payload_writer

    @reify
    def message(self) -> RawRequestMessage:
        warnings.warn("Request.message is deprecated", DeprecationWarning, stacklevel=3)
        return self._message

    @reify
    def rel_url(self) -> URL:
        return self._rel_url

    @reify
    def loop(self) -> asyncio.AbstractEventLoop:
        warnings.warn(
            "request.loop property is deprecated", DeprecationWarning, stacklevel=2
        )
        return self._loop

    # MutableMapping API

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

    ########

    @reify
    def secure(self) -> bool:
        """A bool indicating if the request is handled with SSL."""
        return self.scheme == "https"

    @reify
    def forwarded(self) -> Tuple[Mapping[str, str], ...]:
        """A tuple containing all parsed Forwarded header(s).

        Makes an effort to parse Forwarded headers as specified by RFC 7239:

        - It adds one (immutable) dictionary per Forwarded 'field-value', ie
          per proxy. The element corresponds to the data in the Forwarded
          field-value added by the first proxy encountered by the client. Each
          subsequent item corresponds to those added by later proxies.
        - It checks that every value has valid syntax in general as specified
          in section 4: either a 'token' or a 'quoted-string'.
        - It un-escapes found escape sequences.
        - It does NOT validate 'by' and 'for' contents as specified in section
          6.
        - It does NOT validate 'host' contents (Host ABNF).
        - It does NOT validate 'proto' contents for valid URI scheme names.

        Returns a tuple containing one or more immutable dicts
        """
        elems = []
        for field_value in self._message.headers.getall(hdrs.FORWARDED, ()):
            length = len(field_value)
            pos = 0
            need_separator = False
            elem: Dict[str, str] = {}
            elems.append(types.MappingProxyType(elem))
            while 0 <= pos < length:
                match = _FORWARDED_PAIR_RE.match(field_value, pos)
                if match is not None:  # got a valid forwarded-pair
                    if need_separator:
                        # bad syntax here, skip to next comma
                        pos = field_value.find(",", pos)
                    else:
                        name, value, port = match.groups()
                        if value[0] == '"':
                            # quoted string: remove quotes and unescape
                            value = _QUOTED_PAIR_REPLACE_RE.sub(r"\1", value[1:-1])
                        if port:
                            value += port
                        elem[name.lower()] = value
                        pos += len(match.group(0))
                        need_separator = True
                elif field_value[pos] == ",":  # next forwarded-element
                    need_separator = False
                    elem = {}
                    elems.append(types.MappingProxyType(elem))
                    pos += 1
                elif field_value[pos] == ";":  # next forwarded-pair
                    need_separator = False
                    pos += 1
                elif field_value[pos] in " \t":
                    # Allow whitespace even between forwarded-pairs, though
                    # RFC 7239 doesn't. This simplifies code and is in line
                    # with Postel's law.
                    pos += 1
                else:
                    # bad syntax here, skip to next comma
                    pos = field_value.find(",", pos)
        return tuple(elems)

    @reify
    def scheme(self) -> str:
        """A string representing the scheme of the request.

        Hostname is resolved in this order:

        - overridden value by .clone(scheme=new_scheme) call.
        - type of connection to peer: HTTPS if socket is SSL, HTTP otherwise.

        'http' or 'https'.
        """
        if self._transport_sslcontext:
            return "https"
        else:
            return "http"

    @reify
    def method(self) -> str:
        """Read only property for getting HTTP method.

        The value is upper-cased str like 'GET', 'POST', 'PUT' etc.
        """
        return self._method

    @reify
    def version(self) -> HttpVersion:
        """Read only property for getting HTTP version of request.

        Returns aiohttp.protocol.HttpVersion instance.
        """
        return self._version

    @reify
    def host(self) -> str:
        """Hostname of the request.

        Hostname is resolved in this order:

        - overridden value by .clone(host=new_host) call.
        - HOST HTTP header
        - socket.getfqdn() value
        """
        host = self._message.headers.get(hdrs.HOST)
        if host is not None:
            return host
        return socket.getfqdn()

    @reify
    def remote(self) -> Optional[str]:
        """Remote IP of client initiated HTTP request.

        The IP is resolved in this order:

        - overridden value by .clone(remote=new_remote) call.
        - peername of opened socket
        """
        if self._transport_peername is None:
            return None
        if isinstance(self._transport_peername, (list, tuple)):
            return str(self._transport_peername[0])
        return str(self._transport_peername)

    @reify
    def url(self) -> URL:
        url = URL.build(scheme=self.scheme, host=self.host)
        return url.join(self._rel_url)

    @reify
    def path(self) -> str:
        """The URL including *PATH INFO* without the host or scheme.

        E.g., ``/app/blog``
        """
        return self._rel_url.path

    @reify
    def path_qs(self) -> str:
        """The URL including PATH_INFO and the query string.

        E.g, /app/blog?id=10
        """
        return str(self._rel_url)

    @reify
    def raw_path(self) -> str:
        """The URL including raw *PATH INFO* without the host or scheme.

        Warning, the path is unquoted and may contains non valid URL characters

        E.g., ``/my%2Fpath%7Cwith%21some%25strange%24characters``
        """
        return self._message.path

    @reify
    def query(self) -> "MultiDictProxy[str]":
        """A multidict with all the variables in the query string."""
        return MultiDictProxy(self._rel_url.query)

    @reify
    def query_string(self) -> str:
        """The query string in the URL.

        E.g., id=10
        """
        return self._rel_url.query_string

    @reify
    def headers(self) -> "CIMultiDictProxy[str]":
        """A case-insensitive multidict proxy with all headers."""
        return self._headers

    @reify
    def raw_headers(self) -> RawHeaders:
        """A sequence of pairs for all headers."""
        return self._message.raw_headers

    @reify
    def if_modified_since(self) -> Optional[datetime.datetime]:
        """The value of If-Modified-Since HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self.headers.get(hdrs.IF_MODIFIED_SINCE))

    @reify
    def if_unmodified_since(self) -> Optional[datetime.datetime]:
        """The value of If-Unmodified-Since HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self.headers.get(hdrs.IF_UNMODIFIED_SINCE))

    @staticmethod
    def _etag_values(etag_header: str) -> Iterator[ETag]:
        """Extract `ETag` objects from raw header."""
        if etag_header == ETAG_ANY:
            yield ETag(
                is_weak=False,
                value=ETAG_ANY,
            )
        else:
            for match in LIST_QUOTED_ETAG_RE.finditer(etag_header):
                is_weak, value, garbage = match.group(2, 3, 4)
                # Any symbol captured by 4th group means
                # that the following sequence is invalid.
                if garbage:
                    break

                yield ETag(
                    is_weak=bool(is_weak),
                    value=value,
                )

    @classmethod
    def _if_match_or_none_impl(
        cls, header_value: Optional[str]
    ) -> Optional[Tuple[ETag, ...]]:
        if not header_value:
            return None

        return tuple(cls._etag_values(header_value))

    @reify
    def if_match(self) -> Optional[Tuple[ETag, ...]]:
        """The value of If-Match HTTP header, or None.

        This header is represented as a `tuple` of `ETag` objects.
        """
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_MATCH))

    @reify
    def if_none_match(self) -> Optional[Tuple[ETag, ...]]:
        """The value of If-None-Match HTTP header, or None.

        This header is represented as a `tuple` of `ETag` objects.
        """
        return self._if_match_or_none_impl(self.headers.get(hdrs.IF_NONE_MATCH))

    @reify
    def if_range(self) -> Optional[datetime.datetime]:
        """The value of If-Range HTTP header, or None.

        This header is represented as a `datetime` object.
        """
        return parse_http_date(self.headers.get(hdrs.IF_RANGE))

    @reify
    def keep_alive(self) -> bool:
        """Is keepalive enabled by client?"""
        return not self._message.should_close

    @reify
    def cookies(self) -> Mapping[str, str]:
        """Return request cookies.

        A read-only dictionary-like object.
        """
        raw = self.headers.get(hdrs.COOKIE, "")
        parsed: SimpleCookie[str] = SimpleCookie(raw)
        return MappingProxyType({key: val.value for key, val in parsed.items()})

    @reify
    def http_range(self) -> slice:
        """The content of Range HTTP header.

        Return a slice instance.

        """
        rng = self._headers.get(hdrs.RANGE)
        start, end = None, None
        if rng is not None:
            try:
                pattern = r"^bytes=(\d*)-(\d*)$"
                start, end = re.findall(pattern, rng)[0]
            except IndexError:  # pattern was not found in header
                raise ValueError("range not in acceptable format")

            end = int(end) if end else None
            start = int(start) if start else None

            if start is None and end is not None:
                # end with no start is to return tail of content
                start = -end
                end = None

            if start is not None and end is not None:
                # end is inclusive in range header, exclusive for slice
                end += 1

                if start >= end:
                    raise ValueError("start cannot be after end")

            if start is end is None:  # No valid range supplied
                raise ValueError("No start or end of range specified")

        return slice(start, end, 1)

    @reify
    def content(self) -> StreamReader:
        """Return raw payload stream."""
        return self._payload

    @property
    def has_body(self) -> bool:
        """Return True if request's HTTP BODY can be read, False otherwise."""
        warnings.warn(
            "Deprecated, use .can_read_body #2005", DeprecationWarning, stacklevel=2
        )
        return not self._payload.at_eof()

    @property
    def can_read_body(self) -> bool:
        """Return True if request's HTTP BODY can be read, False otherwise."""
        return not self._payload.at_eof()

    @reify
    def body_exists(self) -> bool:
        """Return True if request has HTTP BODY, False otherwise."""
        return type(self._payload) is not EmptyStreamReader

    async def release(self) -> None:
        """Release request.

        Eat unread part of HTTP BODY if present.
        """
        while not self._payload.at_eof():
            await self._payload.readany()

    async def read(self) -> bytes:
        """Read request body if present.

        Returns bytes object with full request content.
        """
        if self._read_bytes is None:
            body = bytearray()
            while True:
                chunk = await self._payload.readany()
                body.extend(chunk)
                if self._client_max_size:
                    body_size = len(body)
                    if body_size >= self._client_max_size:
                        raise HTTPRequestEntityTooLarge(
                            max_size=self._client_max_size, actual_size=body_size
                        )
                if not chunk:
                    break
            self._read_bytes = bytes(body)
        return self._read_bytes

    async def text(self) -> str:
        """Return BODY as text using encoding from .charset."""
        bytes_body = await self.read()
        encoding = self.charset or "utf-8"
        return bytes_body.decode(encoding)

    async def json(self, *, loads: JSONDecoder = DEFAULT_JSON_DECODER) -> Any:
        """Return BODY as JSON."""
        body = await self.text()
        return loads(body)

    async def multipart(self) -> MultipartReader:
        """Return async iterator to process BODY as multipart."""
        return MultipartReader(self._headers, self._payload)

    async def post(self) -> "MultiDictProxy[Union[str, bytes, FileField]]":
        """Return POST parameters."""
        if self._post is not None:
            return self._post
        if self._method not in self.POST_METHODS:
            self._post = MultiDictProxy(MultiDict())
            return self._post

        content_type = self.content_type
        if content_type not in (
            "",
            "application/x-www-form-urlencoded",
            "multipart/form-data",
        ):
            self._post = MultiDictProxy(MultiDict())
            return self._post

        out: MultiDict[Union[str, bytes, FileField]] = MultiDict()

        if content_type == "multipart/form-data":
            multipart = await self.multipart()
            max_size = self._client_max_size

            field = await multipart.next()
            while field is not None:
                size = 0
                field_ct = field.headers.get(hdrs.CONTENT_TYPE)

                if isinstance(field, BodyPartReader):
                    assert field.name is not None

                    # Note that according to RFC 7578, the Content-Type header
                    # is optional, even for files, so we can't assume it's
                    # present.
                    # https://tools.ietf.org/html/rfc7578#section-4.4
                    if field.filename:
                        # store file in temp file
                        tmp = tempfile.TemporaryFile()
                        chunk = await field.read_chunk(size=2**16)
                        while chunk:
                            chunk = field.decode(chunk)
                            tmp.write(chunk)
                            size += len(chunk)
                            if 0 < max_size < size:
                                tmp.close()
                                raise HTTPRequestEntityTooLarge(
                                    max_size=max_size, actual_size=size
                                )
                            chunk = await field.read_chunk(size=2**16)
                        tmp.seek(0)

                        if field_ct is None:
                            field_ct = "application/octet-stream"

                        ff = FileField(
                            field.name,
                            field.filename,
                            cast(io.BufferedReader, tmp),
                            field_ct,
                            field.headers,
                        )
                        out.add(field.name, ff)
                    else:
                        # deal with ordinary data
                        value = await field.read(decode=True)
                        if field_ct is None or field_ct.startswith("text/"):
                            charset = field.get_charset(default="utf-8")
                            out.add(field.name, value.decode(charset))
                        else:
                            out.add(field.name, value)
                        size += len(value)
                        if 0 < max_size < size:
                            raise HTTPRequestEntityTooLarge(
                                max_size=max_size, actual_size=size
                            )
                else:
                    raise ValueError(
                        "To decode nested multipart you need " "to use custom reader",
                    )

                field = await multipart.next()
        else:
            data = await self.read()
            if data:
                charset = self.charset or "utf-8"
                out.extend(
                    parse_qsl(
                        data.rstrip().decode(charset),
                        keep_blank_values=True,
                        encoding=charset,
                    )
                )

        self._post = MultiDictProxy(out)
        return self._post

    def get_extra_info(self, name: str, default: Any = None) -> Any:
        """Extra info from protocol transport"""
        protocol = self._protocol
        if protocol is None:
            return default

        transport = protocol.transport
        if transport is None:
            return default

        return transport.get_extra_info(name, default)

    def __repr__(self) -> str:
        ascii_encodable_path = self.path.encode("ascii", "backslashreplace").decode(
            "ascii"
        )
        return "<{} {} {} >".format(
            self.__class__.__name__, self._method, ascii_encodable_path
        )

    def __eq__(self, other: object) -> bool:
        return id(self) == id(other)

    def __bool__(self) -> bool:
        return True

    async def _prepare_hook(self, response: StreamResponse) -> None:
        return

    def _cancel(self, exc: BaseException) -> None:
        self._payload.set_exception(exc)


class Request(BaseRequest):

    ATTRS = BaseRequest.ATTRS | frozenset(["_match_info"])

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)

        # matchdict, route_name, handler
        # or information about traversal lookup

        # initialized after route resolving
        self._match_info: Optional[UrlMappingMatchInfo] = None

    if DEBUG:

        def __setattr__(self, name: str, val: Any) -> None:
            if name not in self.ATTRS:
                warnings.warn(
                    "Setting custom {}.{} attribute "
                    "is discouraged".format(self.__class__.__name__, name),
                    DeprecationWarning,
                    stacklevel=2,
                )
            super().__setattr__(name, val)

    def clone(
        self,
        *,
        method: str = sentinel,
        rel_url: StrOrURL = sentinel,
        headers: LooseHeaders = sentinel,
        scheme: str = sentinel,
        host: str = sentinel,
        remote: str = sentinel,
    ) -> "Request":
        ret = super().clone(
            method=method,
            rel_url=rel_url,
            headers=headers,
            scheme=scheme,
            host=host,
            remote=remote,
        )
        new_ret = cast(Request, ret)
        new_ret._match_info = self._match_info
        return new_ret

    @reify
    def match_info(self) -> "UrlMappingMatchInfo":
        """Result of route resolving."""
        match_info = self._match_info
        assert match_info is not None
        return match_info

    @property
    def app(self) -> "Application":
        """Application instance."""
        match_info = self._match_info
        assert match_info is not None
        return match_info.current_app

    @property
    def config_dict(self) -> ChainMapProxy:
        match_info = self._match_info
        assert match_info is not None
        lst = match_info.apps
        app = self.app
        idx = lst.index(app)
        sublist = list(reversed(lst[: idx + 1]))
        return ChainMapProxy(sublist)

    async def _prepare_hook(self, response: StreamResponse) -> None:
        match_info = self._match_info
        if match_info is None:
            return
        for app in match_info._apps:
            await app.on_response_prepare.send(self, response)
