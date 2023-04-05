"""Utilities shared by tests."""

import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterator,
    List,
    Optional,
    Type,
    Union,
    cast,
)
from unittest import mock

from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL

import aiohttp
from aiohttp.client import _RequestContextManager, _WSRequestContextManager

from . import ClientSession, hdrs
from .abc import AbstractCookieJar
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import PY_38, sentinel
from .http import HttpVersion, RawRequestMessage
from .web import (
    Application,
    AppRunner,
    BaseRunner,
    Request,
    Server,
    ServerRunner,
    SockSite,
    UrlMappingMatchInfo,
)
from .web_protocol import _RequestHandler

if TYPE_CHECKING:  # pragma: no cover
    from ssl import SSLContext
else:
    SSLContext = None

if PY_38:
    from unittest import IsolatedAsyncioTestCase as TestCase
else:
    from asynctest import TestCase  # type: ignore[no-redef]

REUSE_ADDRESS = os.name == "posix" and sys.platform != "cygwin"


def get_unused_port_socket(
    host: str, family: socket.AddressFamily = socket.AF_INET
) -> socket.socket:
    return get_port_socket(host, 0, family)


def get_port_socket(
    host: str, port: int, family: socket.AddressFamily
) -> socket.socket:
    s = socket.socket(family, socket.SOCK_STREAM)
    if REUSE_ADDRESS:
        # Windows has different semantics for SO_REUSEADDR,
        # so don't set it. Ref:
        # https://docs.microsoft.com/en-us/windows/win32/winsock/using-so-reuseaddr-and-so-exclusiveaddruse
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    return s


def unused_port() -> int:
    """Return a port that is unused on the current host."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return cast(int, s.getsockname()[1])


class BaseTestServer(ABC):
    __test__ = False

    def __init__(
        self,
        *,
        scheme: Union[str, object] = sentinel,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        skip_url_asserts: bool = False,
        socket_factory: Callable[
            [str, int, socket.AddressFamily], socket.socket
        ] = get_port_socket,
        **kwargs: Any,
    ) -> None:
        self._loop = loop
        self.runner: Optional[BaseRunner] = None
        self._root: Optional[URL] = None
        self.host = host
        self.port = port
        self._closed = False
        self.scheme = scheme
        self.skip_url_asserts = skip_url_asserts
        self.socket_factory = socket_factory

    async def start_server(
        self, loop: Optional[asyncio.AbstractEventLoop] = None, **kwargs: Any
    ) -> None:
        if self.runner:
            return
        self._loop = loop
        self._ssl = kwargs.pop("ssl", None)
        self.runner = await self._make_runner(**kwargs)
        await self.runner.setup()
        if not self.port:
            self.port = 0
        try:
            version = ipaddress.ip_address(self.host).version
        except ValueError:
            version = 4
        family = socket.AF_INET6 if version == 6 else socket.AF_INET
        _sock = self.socket_factory(self.host, self.port, family)
        self.host, self.port = _sock.getsockname()[:2]
        site = SockSite(self.runner, sock=_sock, ssl_context=self._ssl)
        await site.start()
        server = site._server
        assert server is not None
        sockets = server.sockets
        assert sockets is not None
        self.port = sockets[0].getsockname()[1]
        if self.scheme is sentinel:
            if self._ssl:
                scheme = "https"
            else:
                scheme = "http"
            self.scheme = scheme
        self._root = URL(f"{self.scheme}://{self.host}:{self.port}")

    @abstractmethod  # pragma: no cover
    async def _make_runner(self, **kwargs: Any) -> BaseRunner:
        pass

    def make_url(self, path: str) -> URL:
        assert self._root is not None
        url = URL(path)
        if not self.skip_url_asserts:
            assert not url.is_absolute()
            return self._root.join(url)
        else:
            return URL(str(self._root) + path)

    @property
    def started(self) -> bool:
        return self.runner is not None

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def handler(self) -> Server:
        # for backward compatibility
        # web.Server instance
        runner = self.runner
        assert runner is not None
        assert runner.server is not None
        return runner.server

    async def close(self) -> None:
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run when the object is garbage collected, and on
        exit when used as a context manager.

        """
        if self.started and not self.closed:
            assert self.runner is not None
            await self.runner.cleanup()
            self._root = None
            self.port = None
            self._closed = True

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "BaseTestServer":
        await self.start_server(loop=self._loop)
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ) -> None:
        await self.close()


class TestServer(BaseTestServer):
    def __init__(
        self,
        app: Application,
        *,
        scheme: Union[str, object] = sentinel,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        **kwargs: Any,
    ):
        self.app = app
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, **kwargs: Any) -> BaseRunner:
        return AppRunner(self.app, **kwargs)


class RawTestServer(BaseTestServer):
    def __init__(
        self,
        handler: _RequestHandler,
        *,
        scheme: Union[str, object] = sentinel,
        host: str = "127.0.0.1",
        port: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        self._handler = handler
        super().__init__(scheme=scheme, host=host, port=port, **kwargs)

    async def _make_runner(self, debug: bool = True, **kwargs: Any) -> ServerRunner:
        srv = Server(self._handler, loop=self._loop, debug=debug, **kwargs)
        return ServerRunner(srv, debug=debug, **kwargs)


class TestClient:
    """
    A test client implementation.

    To write functional tests for aiohttp based servers.

    """

    __test__ = False

    def __init__(
        self,
        server: BaseTestServer,
        *,
        cookie_jar: Optional[AbstractCookieJar] = None,
        loop: Optional[asyncio.AbstractEventLoop] = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(server, BaseTestServer):
            raise TypeError(
                "server must be TestServer " "instance, found type: %r" % type(server)
            )
        self._server = server
        self._loop = loop
        if cookie_jar is None:
            cookie_jar = aiohttp.CookieJar(unsafe=True, loop=loop)
        self._session = ClientSession(loop=loop, cookie_jar=cookie_jar, **kwargs)
        self._closed = False
        self._responses: List[ClientResponse] = []
        self._websockets: List[ClientWebSocketResponse] = []

    async def start_server(self) -> None:
        await self._server.start_server(loop=self._loop)

    @property
    def host(self) -> str:
        return self._server.host

    @property
    def port(self) -> Optional[int]:
        return self._server.port

    @property
    def server(self) -> BaseTestServer:
        return self._server

    @property
    def app(self) -> Optional[Application]:
        return cast(Optional[Application], getattr(self._server, "app", None))

    @property
    def session(self) -> ClientSession:
        """An internal aiohttp.ClientSession.

        Unlike the methods on the TestClient, client session requests
        do not automatically include the host in the url queried, and
        will require an absolute path to the resource.

        """
        return self._session

    def make_url(self, path: str) -> URL:
        return self._server.make_url(path)

    async def _request(self, method: str, path: str, **kwargs: Any) -> ClientResponse:
        resp = await self._session.request(method, self.make_url(path), **kwargs)
        # save it to close later
        self._responses.append(resp)
        return resp

    def request(self, method: str, path: str, **kwargs: Any) -> _RequestContextManager:
        """Routes a request to tested http server.

        The interface is identical to aiohttp.ClientSession.request,
        except the loop kwarg is overridden by the instance used by the
        test server.

        """
        return _RequestContextManager(self._request(method, path, **kwargs))

    def get(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP GET request."""
        return _RequestContextManager(self._request(hdrs.METH_GET, path, **kwargs))

    def post(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP POST request."""
        return _RequestContextManager(self._request(hdrs.METH_POST, path, **kwargs))

    def options(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP OPTIONS request."""
        return _RequestContextManager(self._request(hdrs.METH_OPTIONS, path, **kwargs))

    def head(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP HEAD request."""
        return _RequestContextManager(self._request(hdrs.METH_HEAD, path, **kwargs))

    def put(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP PUT request."""
        return _RequestContextManager(self._request(hdrs.METH_PUT, path, **kwargs))

    def patch(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP PATCH request."""
        return _RequestContextManager(self._request(hdrs.METH_PATCH, path, **kwargs))

    def delete(self, path: str, **kwargs: Any) -> _RequestContextManager:
        """Perform an HTTP PATCH request."""
        return _RequestContextManager(self._request(hdrs.METH_DELETE, path, **kwargs))

    def ws_connect(self, path: str, **kwargs: Any) -> _WSRequestContextManager:
        """Initiate websocket connection.

        The api corresponds to aiohttp.ClientSession.ws_connect.

        """
        return _WSRequestContextManager(self._ws_connect(path, **kwargs))

    async def _ws_connect(self, path: str, **kwargs: Any) -> ClientWebSocketResponse:
        ws = await self._session.ws_connect(self.make_url(path), **kwargs)
        self._websockets.append(ws)
        return ws

    async def close(self) -> None:
        """Close all fixtures created by the test client.

        After that point, the TestClient is no longer usable.

        This is an idempotent function: running close multiple times
        will not have any additional effects.

        close is also run on exit when used as a(n) (asynchronous)
        context manager.

        """
        if not self._closed:
            for resp in self._responses:
                resp.close()
            for ws in self._websockets:
                await ws.close()
            await self._session.close()
            await self._server.close()
            self._closed = True

    def __enter__(self) -> None:
        raise TypeError("Use async with instead")

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        # __exit__ should exist in pair with __enter__ but never executed
        pass  # pragma: no cover

    async def __aenter__(self) -> "TestClient":
        await self.start_server()
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc: Optional[BaseException],
        tb: Optional[TracebackType],
    ) -> None:
        await self.close()


class AioHTTPTestCase(TestCase):
    """A base class to allow for unittest web applications using aiohttp.

    Provides the following:

    * self.client (aiohttp.test_utils.TestClient): an aiohttp test client.
    * self.loop (asyncio.BaseEventLoop): the event loop in which the
        application and server are running.
    * self.app (aiohttp.web.Application): the application returned by
        self.get_application()

    Note that the TestClient's methods are asynchronous: you have to
    execute function on the test client using asynchronous methods.
    """

    async def get_application(self) -> Application:
        """Get application.

        This method should be overridden
        to return the aiohttp.web.Application
        object to test.
        """
        return self.get_app()

    def get_app(self) -> Application:
        """Obsolete method used to constructing web application.

        Use .get_application() coroutine instead.
        """
        raise RuntimeError("Did you forget to define get_application()?")

    def setUp(self) -> None:
        if not PY_38:
            asyncio.get_event_loop().run_until_complete(self.asyncSetUp())

    async def asyncSetUp(self) -> None:
        try:
            self.loop = asyncio.get_running_loop()
        except (AttributeError, RuntimeError):  # AttributeError->py36
            self.loop = asyncio.get_event_loop_policy().get_event_loop()

        return await self.setUpAsync()

    async def setUpAsync(self) -> None:
        self.app = await self.get_application()
        self.server = await self.get_server(self.app)
        self.client = await self.get_client(self.server)

        await self.client.start_server()

    def tearDown(self) -> None:
        if not PY_38:
            self.loop.run_until_complete(self.asyncTearDown())

    async def asyncTearDown(self) -> None:
        return await self.tearDownAsync()

    async def tearDownAsync(self) -> None:
        await self.client.close()

    async def get_server(self, app: Application) -> TestServer:
        """Return a TestServer instance."""
        return TestServer(app, loop=self.loop)

    async def get_client(self, server: TestServer) -> TestClient:
        """Return a TestClient instance."""
        return TestClient(server, loop=self.loop)


def unittest_run_loop(func: Any, *args: Any, **kwargs: Any) -> Any:
    """
    A decorator dedicated to use with asynchronous AioHTTPTestCase test methods.

    In 3.8+, this does nothing.
    """
    warnings.warn(
        "Decorator `@unittest_run_loop` is no longer needed in aiohttp 3.8+",
        DeprecationWarning,
        stacklevel=2,
    )
    return func


_LOOP_FACTORY = Callable[[], asyncio.AbstractEventLoop]


@contextlib.contextmanager
def loop_context(
    loop_factory: _LOOP_FACTORY = asyncio.new_event_loop, fast: bool = False
) -> Iterator[asyncio.AbstractEventLoop]:
    """A contextmanager that creates an event_loop, for test purposes.

    Handles the creation and cleanup of a test loop.
    """
    loop = setup_test_loop(loop_factory)
    yield loop
    teardown_test_loop(loop, fast=fast)


def setup_test_loop(
    loop_factory: _LOOP_FACTORY = asyncio.new_event_loop,
) -> asyncio.AbstractEventLoop:
    """Create and return an asyncio.BaseEventLoop instance.

    The caller should also call teardown_test_loop,
    once they are done with the loop.
    """
    loop = loop_factory()
    try:
        module = loop.__class__.__module__
        skip_watcher = "uvloop" in module
    except AttributeError:  # pragma: no cover
        # Just in case
        skip_watcher = True
    asyncio.set_event_loop(loop)
    if sys.platform != "win32" and not skip_watcher:
        policy = asyncio.get_event_loop_policy()
        watcher: asyncio.AbstractChildWatcher
        try:  # Python >= 3.8
            # Refs:
            # * https://github.com/pytest-dev/pytest-xdist/issues/620
            # * https://stackoverflow.com/a/58614689/595220
            # * https://bugs.python.org/issue35621
            # * https://github.com/python/cpython/pull/14344
            watcher = asyncio.ThreadedChildWatcher()
        except AttributeError:  # Python < 3.8
            watcher = asyncio.SafeChildWatcher()
        watcher.attach_loop(loop)
        with contextlib.suppress(NotImplementedError):
            policy.set_child_watcher(watcher)
    return loop


def teardown_test_loop(loop: asyncio.AbstractEventLoop, fast: bool = False) -> None:
    """Teardown and cleanup an event_loop created by setup_test_loop."""
    closed = loop.is_closed()
    if not closed:
        loop.call_soon(loop.stop)
        loop.run_forever()
        loop.close()

    if not fast:
        gc.collect()

    asyncio.set_event_loop(None)


def _create_app_mock() -> mock.MagicMock:
    def get_dict(app: Any, key: str) -> Any:
        return app.__app_dict[key]

    def set_dict(app: Any, key: str, value: Any) -> None:
        app.__app_dict[key] = value

    app = mock.MagicMock(spec=Application)
    app.__app_dict = {}
    app.__getitem__ = get_dict
    app.__setitem__ = set_dict

    app._debug = False
    app.on_response_prepare = Signal(app)
    app.on_response_prepare.freeze()
    return app


def _create_transport(sslcontext: Optional[SSLContext] = None) -> mock.Mock:
    transport = mock.Mock()

    def get_extra_info(key: str) -> Optional[SSLContext]:
        if key == "sslcontext":
            return sslcontext
        else:
            return None

    transport.get_extra_info.side_effect = get_extra_info
    return transport


def make_mocked_request(
    method: str,
    path: str,
    headers: Any = None,
    *,
    match_info: Any = sentinel,
    version: HttpVersion = HttpVersion(1, 1),
    closing: bool = False,
    app: Any = None,
    writer: Any = sentinel,
    protocol: Any = sentinel,
    transport: Any = sentinel,
    payload: Any = sentinel,
    sslcontext: Optional[SSLContext] = None,
    client_max_size: int = 1024**2,
    loop: Any = ...,
) -> Request:
    """Creates mocked web.Request testing purposes.

    Useful in unit tests, when spinning full web server is overkill or
    specific conditions and errors are hard to trigger.
    """
    task = mock.Mock()
    if loop is ...:
        loop = mock.Mock()
        loop.create_future.return_value = ()

    if version < HttpVersion(1, 1):
        closing = True

    if headers:
        headers = CIMultiDictProxy(CIMultiDict(headers))
        raw_hdrs = tuple(
            (k.encode("utf-8"), v.encode("utf-8")) for k, v in headers.items()
        )
    else:
        headers = CIMultiDictProxy(CIMultiDict())
        raw_hdrs = ()

    chunked = "chunked" in headers.get(hdrs.TRANSFER_ENCODING, "").lower()

    message = RawRequestMessage(
        method,
        path,
        version,
        headers,
        raw_hdrs,
        closing,
        None,
        False,
        chunked,
        URL(path),
    )
    if app is None:
        app = _create_app_mock()

    if transport is sentinel:
        transport = _create_transport(sslcontext)

    if protocol is sentinel:
        protocol = mock.Mock()
        protocol.transport = transport

    if writer is sentinel:
        writer = mock.Mock()
        writer.write_headers = make_mocked_coro(None)
        writer.write = make_mocked_coro(None)
        writer.write_eof = make_mocked_coro(None)
        writer.drain = make_mocked_coro(None)
        writer.transport = transport

    protocol.transport = transport
    protocol.writer = writer

    if payload is sentinel:
        payload = mock.Mock()

    req = Request(
        message, payload, protocol, writer, task, loop, client_max_size=client_max_size
    )

    match_info = UrlMappingMatchInfo(
        {} if match_info is sentinel else match_info, mock.Mock()
    )
    match_info.add_app(app)
    req._match_info = match_info

    return req


def make_mocked_coro(
    return_value: Any = sentinel, raise_exception: Any = sentinel
) -> Any:
    """Creates a coroutine mock."""

    async def mock_coro(*args: Any, **kwargs: Any) -> Any:
        if raise_exception is not sentinel:
            raise raise_exception
        if not inspect.isawaitable(return_value):
            return return_value
        await return_value

    return mock.Mock(wraps=mock_coro)
