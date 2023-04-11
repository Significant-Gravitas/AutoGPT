import asyncio
import logging
from abc import ABC, abstractmethod
from collections.abc import Sized
from http.cookies import BaseCookie, Morsel
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
)

from multidict import CIMultiDict
from yarl import URL

from .helpers import get_running_loop
from .typedefs import LooseCookies

if TYPE_CHECKING:  # pragma: no cover
    from .web_app import Application
    from .web_exceptions import HTTPException
    from .web_request import BaseRequest, Request
    from .web_response import StreamResponse
else:
    BaseRequest = Request = Application = StreamResponse = None
    HTTPException = None


class AbstractRouter(ABC):
    def __init__(self) -> None:
        self._frozen = False

    def post_init(self, app: Application) -> None:
        """Post init stage.

        Not an abstract method for sake of backward compatibility,
        but if the router wants to be aware of the application
        it can override this.
        """

    @property
    def frozen(self) -> bool:
        return self._frozen

    def freeze(self) -> None:
        """Freeze router."""
        self._frozen = True

    @abstractmethod
    async def resolve(self, request: Request) -> "AbstractMatchInfo":
        """Return MATCH_INFO for given request"""


class AbstractMatchInfo(ABC):
    @property  # pragma: no branch
    @abstractmethod
    def handler(self) -> Callable[[Request], Awaitable[StreamResponse]]:
        """Execute matched request handler"""

    @property
    @abstractmethod
    def expect_handler(self) -> Callable[[Request], Awaitable[None]]:
        """Expect handler for 100-continue processing"""

    @property  # pragma: no branch
    @abstractmethod
    def http_exception(self) -> Optional[HTTPException]:
        """HTTPException instance raised on router's resolving, or None"""

    @abstractmethod  # pragma: no branch
    def get_info(self) -> Dict[str, Any]:
        """Return a dict with additional info useful for introspection"""

    @property  # pragma: no branch
    @abstractmethod
    def apps(self) -> Tuple[Application, ...]:
        """Stack of nested applications.

        Top level application is left-most element.

        """

    @abstractmethod
    def add_app(self, app: Application) -> None:
        """Add application to the nested apps stack."""

    @abstractmethod
    def freeze(self) -> None:
        """Freeze the match info.

        The method is called after route resolution.

        After the call .add_app() is forbidden.

        """


class AbstractView(ABC):
    """Abstract class based view."""

    def __init__(self, request: Request) -> None:
        self._request = request

    @property
    def request(self) -> Request:
        """Request instance."""
        return self._request

    @abstractmethod
    def __await__(self) -> Generator[Any, None, StreamResponse]:
        """Execute the view handler."""


class AbstractResolver(ABC):
    """Abstract DNS resolver."""

    @abstractmethod
    async def resolve(self, host: str, port: int, family: int) -> List[Dict[str, Any]]:
        """Return IP address for given hostname"""

    @abstractmethod
    async def close(self) -> None:
        """Release resolver"""


if TYPE_CHECKING:  # pragma: no cover
    IterableBase = Iterable[Morsel[str]]
else:
    IterableBase = Iterable


ClearCookiePredicate = Callable[["Morsel[str]"], bool]


class AbstractCookieJar(Sized, IterableBase):
    """Abstract Cookie Jar."""

    def __init__(self, *, loop: Optional[asyncio.AbstractEventLoop] = None) -> None:
        self._loop = get_running_loop(loop)

    @abstractmethod
    def clear(self, predicate: Optional[ClearCookiePredicate] = None) -> None:
        """Clear all cookies if no predicate is passed."""

    @abstractmethod
    def clear_domain(self, domain: str) -> None:
        """Clear all cookies for domain and all subdomains."""

    @abstractmethod
    def update_cookies(self, cookies: LooseCookies, response_url: URL = URL()) -> None:
        """Update cookies."""

    @abstractmethod
    def filter_cookies(self, request_url: URL) -> "BaseCookie[str]":
        """Return the jar's cookies filtered by their attributes."""


class AbstractStreamWriter(ABC):
    """Abstract stream writer."""

    buffer_size = 0
    output_size = 0
    length: Optional[int] = 0

    @abstractmethod
    async def write(self, chunk: bytes) -> None:
        """Write chunk into stream."""

    @abstractmethod
    async def write_eof(self, chunk: bytes = b"") -> None:
        """Write last chunk."""

    @abstractmethod
    async def drain(self) -> None:
        """Flush the write buffer."""

    @abstractmethod
    def enable_compression(self, encoding: str = "deflate") -> None:
        """Enable HTTP body compression"""

    @abstractmethod
    def enable_chunking(self) -> None:
        """Enable HTTP chunked mode"""

    @abstractmethod
    async def write_headers(
        self, status_line: str, headers: "CIMultiDict[str]"
    ) -> None:
        """Write HTTP headers"""


class AbstractAccessLogger(ABC):
    """Abstract writer to access log."""

    def __init__(self, logger: logging.Logger, log_format: str) -> None:
        self.logger = logger
        self.log_format = log_format

    @abstractmethod
    def log(self, request: BaseRequest, response: StreamResponse, time: float) -> None:
        """Emit log to logger."""
