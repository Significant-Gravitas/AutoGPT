"""HTTP related errors."""

import asyncio
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

from .http_parser import RawResponseMessage
from .typedefs import LooseHeaders

try:
    import ssl

    SSLContext = ssl.SSLContext
except ImportError:  # pragma: no cover
    ssl = SSLContext = None  # type: ignore[assignment]


if TYPE_CHECKING:  # pragma: no cover
    from .client_reqrep import ClientResponse, ConnectionKey, Fingerprint, RequestInfo
else:
    RequestInfo = ClientResponse = ConnectionKey = None

__all__ = (
    "ClientError",
    "ClientConnectionError",
    "ClientOSError",
    "ClientConnectorError",
    "ClientProxyConnectionError",
    "ClientSSLError",
    "ClientConnectorSSLError",
    "ClientConnectorCertificateError",
    "ServerConnectionError",
    "ServerTimeoutError",
    "ServerDisconnectedError",
    "ServerFingerprintMismatch",
    "ClientResponseError",
    "ClientHttpProxyError",
    "WSServerHandshakeError",
    "ContentTypeError",
    "ClientPayloadError",
    "InvalidURL",
)


class ClientError(Exception):
    """Base class for client connection errors."""


class ClientResponseError(ClientError):
    """Connection error during reading response.

    request_info: instance of RequestInfo
    """

    def __init__(
        self,
        request_info: RequestInfo,
        history: Tuple[ClientResponse, ...],
        *,
        code: Optional[int] = None,
        status: Optional[int] = None,
        message: str = "",
        headers: Optional[LooseHeaders] = None,
    ) -> None:
        self.request_info = request_info
        if code is not None:
            if status is not None:
                raise ValueError(
                    "Both code and status arguments are provided; "
                    "code is deprecated, use status instead"
                )
            warnings.warn(
                "code argument is deprecated, use status instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if status is not None:
            self.status = status
        elif code is not None:
            self.status = code
        else:
            self.status = 0
        self.message = message
        self.headers = headers
        self.history = history
        self.args = (request_info, history)

    def __str__(self) -> str:
        return "{}, message={!r}, url={!r}".format(
            self.status,
            self.message,
            self.request_info.real_url,
        )

    def __repr__(self) -> str:
        args = f"{self.request_info!r}, {self.history!r}"
        if self.status != 0:
            args += f", status={self.status!r}"
        if self.message != "":
            args += f", message={self.message!r}"
        if self.headers is not None:
            args += f", headers={self.headers!r}"
        return f"{type(self).__name__}({args})"

    @property
    def code(self) -> int:
        warnings.warn(
            "code property is deprecated, use status instead",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.status

    @code.setter
    def code(self, value: int) -> None:
        warnings.warn(
            "code property is deprecated, use status instead",
            DeprecationWarning,
            stacklevel=2,
        )
        self.status = value


class ContentTypeError(ClientResponseError):
    """ContentType found is not valid."""


class WSServerHandshakeError(ClientResponseError):
    """websocket server handshake error."""


class ClientHttpProxyError(ClientResponseError):
    """HTTP proxy error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
    proxy responds with status other than ``200 OK``
    on ``CONNECT`` request.
    """


class TooManyRedirects(ClientResponseError):
    """Client was redirected too many times."""


class ClientConnectionError(ClientError):
    """Base class for client socket errors."""


class ClientOSError(ClientConnectionError, OSError):
    """OSError error."""


class ClientConnectorError(ClientOSError):
    """Client connector error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        a connection can not be established.
    """

    def __init__(self, connection_key: ConnectionKey, os_error: OSError) -> None:
        self._conn_key = connection_key
        self._os_error = os_error
        super().__init__(os_error.errno, os_error.strerror)
        self.args = (connection_key, os_error)

    @property
    def os_error(self) -> OSError:
        return self._os_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> Optional[int]:
        return self._conn_key.port

    @property
    def ssl(self) -> Union[SSLContext, None, bool, "Fingerprint"]:
        return self._conn_key.ssl

    def __str__(self) -> str:
        return "Cannot connect to host {0.host}:{0.port} ssl:{1} [{2}]".format(
            self, self.ssl if self.ssl is not None else "default", self.strerror
        )

    # OSError.__reduce__ does too much black magick
    __reduce__ = BaseException.__reduce__


class ClientProxyConnectionError(ClientConnectorError):
    """Proxy connection error.

    Raised in :class:`aiohttp.connector.TCPConnector` if
        connection to proxy can not be established.
    """


class UnixClientConnectorError(ClientConnectorError):
    """Unix connector error.

    Raised in :py:class:`aiohttp.connector.UnixConnector`
    if connection to unix socket can not be established.
    """

    def __init__(
        self, path: str, connection_key: ConnectionKey, os_error: OSError
    ) -> None:
        self._path = path
        super().__init__(connection_key, os_error)

    @property
    def path(self) -> str:
        return self._path

    def __str__(self) -> str:
        return "Cannot connect to unix socket {0.path} ssl:{1} [{2}]".format(
            self, self.ssl if self.ssl is not None else "default", self.strerror
        )


class ServerConnectionError(ClientConnectionError):
    """Server connection errors."""


class ServerDisconnectedError(ServerConnectionError):
    """Server disconnected."""

    def __init__(self, message: Union[RawResponseMessage, str, None] = None) -> None:
        if message is None:
            message = "Server disconnected"

        self.args = (message,)
        self.message = message


class ServerTimeoutError(ServerConnectionError, asyncio.TimeoutError):
    """Server timeout error."""


class ServerFingerprintMismatch(ServerConnectionError):
    """SSL certificate does not match expected fingerprint."""

    def __init__(self, expected: bytes, got: bytes, host: str, port: int) -> None:
        self.expected = expected
        self.got = got
        self.host = host
        self.port = port
        self.args = (expected, got, host, port)

    def __repr__(self) -> str:
        return "<{} expected={!r} got={!r} host={!r} port={!r}>".format(
            self.__class__.__name__, self.expected, self.got, self.host, self.port
        )


class ClientPayloadError(ClientError):
    """Response payload error."""


class InvalidURL(ClientError, ValueError):
    """Invalid URL.

    URL used for fetching is malformed, e.g. it doesn't contains host
    part.
    """

    # Derive from ValueError for backward compatibility

    def __init__(self, url: Any) -> None:
        # The type of url is not yarl.URL because the exception can be raised
        # on URL(url) call
        super().__init__(url)

    @property
    def url(self) -> Any:
        return self.args[0]

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} {self.url}>"


class ClientSSLError(ClientConnectorError):
    """Base error for ssl.*Errors."""


if ssl is not None:
    cert_errors = (ssl.CertificateError,)
    cert_errors_bases = (
        ClientSSLError,
        ssl.CertificateError,
    )

    ssl_errors = (ssl.SSLError,)
    ssl_error_bases = (ClientSSLError, ssl.SSLError)
else:  # pragma: no cover
    cert_errors = tuple()
    cert_errors_bases = (
        ClientSSLError,
        ValueError,
    )

    ssl_errors = tuple()
    ssl_error_bases = (ClientSSLError,)


class ClientConnectorSSLError(*ssl_error_bases):  # type: ignore[misc]
    """Response ssl error."""


class ClientConnectorCertificateError(*cert_errors_bases):  # type: ignore[misc]
    """Response certificate error."""

    def __init__(
        self, connection_key: ConnectionKey, certificate_error: Exception
    ) -> None:
        self._conn_key = connection_key
        self._certificate_error = certificate_error
        self.args = (connection_key, certificate_error)

    @property
    def certificate_error(self) -> Exception:
        return self._certificate_error

    @property
    def host(self) -> str:
        return self._conn_key.host

    @property
    def port(self) -> Optional[int]:
        return self._conn_key.port

    @property
    def ssl(self) -> bool:
        return self._conn_key.is_ssl

    def __str__(self) -> str:
        return (
            "Cannot connect to host {0.host}:{0.port} ssl:{0.ssl} "
            "[{0.certificate_error.__class__.__name__}: "
            "{0.certificate_error.args}]".format(self)
        )
