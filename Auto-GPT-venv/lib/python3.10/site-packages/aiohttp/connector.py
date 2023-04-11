import asyncio
import functools
import random
import sys
import traceback
import warnings
from collections import defaultdict, deque
from contextlib import suppress
from http.cookies import SimpleCookie
from itertools import cycle, islice
from time import monotonic
from types import TracebackType
from typing import (
    TYPE_CHECKING,
    Any,
    Awaitable,
    Callable,
    DefaultDict,
    Dict,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    Union,
    cast,
)

import attr

from . import hdrs, helpers
from .abc import AbstractResolver
from .client_exceptions import (
    ClientConnectionError,
    ClientConnectorCertificateError,
    ClientConnectorError,
    ClientConnectorSSLError,
    ClientHttpProxyError,
    ClientProxyConnectionError,
    ServerFingerprintMismatch,
    UnixClientConnectorError,
    cert_errors,
    ssl_errors,
)
from .client_proto import ResponseHandler
from .client_reqrep import ClientRequest, Fingerprint, _merge_ssl_params
from .helpers import (
    PY_36,
    ceil_timeout,
    get_running_loop,
    is_ip_address,
    noop,
    sentinel,
)
from .http import RESPONSES
from .locks import EventResultOrError
from .resolver import DefaultResolver

try:
    import ssl

    SSLContext = ssl.SSLContext
except ImportError:  # pragma: no cover
    ssl = None  # type: ignore[assignment]
    SSLContext = object  # type: ignore[misc,assignment]


__all__ = ("BaseConnector", "TCPConnector", "UnixConnector", "NamedPipeConnector")


if TYPE_CHECKING:  # pragma: no cover
    from .client import ClientTimeout
    from .client_reqrep import ConnectionKey
    from .tracing import Trace


class _DeprecationWaiter:
    __slots__ = ("_awaitable", "_awaited")

    def __init__(self, awaitable: Awaitable[Any]) -> None:
        self._awaitable = awaitable
        self._awaited = False

    def __await__(self) -> Any:
        self._awaited = True
        return self._awaitable.__await__()

    def __del__(self) -> None:
        if not self._awaited:
            warnings.warn(
                "Connector.close() is a coroutine, "
                "please use await connector.close()",
                DeprecationWarning,
            )


class Connection:

    _source_traceback = None
    _transport = None

    def __init__(
        self,
        connector: "BaseConnector",
        key: "ConnectionKey",
        protocol: ResponseHandler,
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        self._key = key
        self._connector = connector
        self._loop = loop
        self._protocol: Optional[ResponseHandler] = protocol
        self._callbacks: List[Callable[[], None]] = []

        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

    def __repr__(self) -> str:
        return f"Connection<{self._key}>"

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._protocol is not None:
            if PY_36:
                kwargs = {"source": self}
            else:
                kwargs = {}
            _warnings.warn(f"Unclosed connection {self!r}", ResourceWarning, **kwargs)
            if self._loop.is_closed():
                return

            self._connector._release(self._key, self._protocol, should_close=True)

            context = {"client_connection": self, "message": "Unclosed connection"}
            if self._source_traceback is not None:
                context["source_traceback"] = self._source_traceback
            self._loop.call_exception_handler(context)

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        warnings.warn(
            "connector.loop property is deprecated", DeprecationWarning, stacklevel=2
        )
        return self._loop

    @property
    def transport(self) -> Optional[asyncio.Transport]:
        if self._protocol is None:
            return None
        return self._protocol.transport

    @property
    def protocol(self) -> Optional[ResponseHandler]:
        return self._protocol

    def add_callback(self, callback: Callable[[], None]) -> None:
        if callback is not None:
            self._callbacks.append(callback)

    def _notify_release(self) -> None:
        callbacks, self._callbacks = self._callbacks[:], []

        for cb in callbacks:
            with suppress(Exception):
                cb()

    def close(self) -> None:
        self._notify_release()

        if self._protocol is not None:
            self._connector._release(self._key, self._protocol, should_close=True)
            self._protocol = None

    def release(self) -> None:
        self._notify_release()

        if self._protocol is not None:
            self._connector._release(
                self._key, self._protocol, should_close=self._protocol.should_close
            )
            self._protocol = None

    @property
    def closed(self) -> bool:
        return self._protocol is None or not self._protocol.is_connected()


class _TransportPlaceholder:
    """placeholder for BaseConnector.connect function"""

    def close(self) -> None:
        pass


class BaseConnector:
    """Base connector class.

    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    loop - Optional event loop.
    """

    _closed = True  # prevent AttributeError in __del__ if ctor was failed
    _source_traceback = None

    # abort transport after 2 seconds (cleanup broken connections)
    _cleanup_closed_period = 2.0

    def __init__(
        self,
        *,
        keepalive_timeout: Union[object, None, float] = sentinel,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:

        if force_close:
            if keepalive_timeout is not None and keepalive_timeout is not sentinel:
                raise ValueError(
                    "keepalive_timeout cannot " "be set if force_close is True"
                )
        else:
            if keepalive_timeout is sentinel:
                keepalive_timeout = 15.0

        loop = get_running_loop(loop)

        self._closed = False
        if loop.get_debug():
            self._source_traceback = traceback.extract_stack(sys._getframe(1))

        self._conns: Dict[ConnectionKey, List[Tuple[ResponseHandler, float]]] = {}
        self._limit = limit
        self._limit_per_host = limit_per_host
        self._acquired: Set[ResponseHandler] = set()
        self._acquired_per_host: DefaultDict[
            ConnectionKey, Set[ResponseHandler]
        ] = defaultdict(set)
        self._keepalive_timeout = cast(float, keepalive_timeout)
        self._force_close = force_close

        # {host_key: FIFO list of waiters}
        self._waiters = defaultdict(deque)  # type: ignore[var-annotated]

        self._loop = loop
        self._factory = functools.partial(ResponseHandler, loop=loop)

        self.cookies: SimpleCookie[str] = SimpleCookie()

        # start keep-alive connection cleanup task
        self._cleanup_handle: Optional[asyncio.TimerHandle] = None

        # start cleanup closed transports task
        self._cleanup_closed_handle: Optional[asyncio.TimerHandle] = None
        self._cleanup_closed_disabled = not enable_cleanup_closed
        self._cleanup_closed_transports: List[Optional[asyncio.Transport]] = []
        self._cleanup_closed()

    def __del__(self, _warnings: Any = warnings) -> None:
        if self._closed:
            return
        if not self._conns:
            return

        conns = [repr(c) for c in self._conns.values()]

        self._close()

        if PY_36:
            kwargs = {"source": self}
        else:
            kwargs = {}
        _warnings.warn(f"Unclosed connector {self!r}", ResourceWarning, **kwargs)
        context = {
            "connector": self,
            "connections": conns,
            "message": "Unclosed connector",
        }
        if self._source_traceback is not None:
            context["source_traceback"] = self._source_traceback
        self._loop.call_exception_handler(context)

    def __enter__(self) -> "BaseConnector":
        warnings.warn(
            '"with Connector():" is deprecated, '
            'use "async with Connector():" instead',
            DeprecationWarning,
        )
        return self

    def __exit__(self, *exc: Any) -> None:
        self._close()

    async def __aenter__(self) -> "BaseConnector":
        return self

    async def __aexit__(
        self,
        exc_type: Optional[Type[BaseException]] = None,
        exc_value: Optional[BaseException] = None,
        exc_traceback: Optional[TracebackType] = None,
    ) -> None:
        await self.close()

    @property
    def force_close(self) -> bool:
        """Ultimately close connection on releasing if True."""
        return self._force_close

    @property
    def limit(self) -> int:
        """The total number for simultaneous connections.

        If limit is 0 the connector has no limit.
        The default limit size is 100.
        """
        return self._limit

    @property
    def limit_per_host(self) -> int:
        """The limit for simultaneous connections to the same endpoint.

        Endpoints are the same if they are have equal
        (host, port, is_ssl) triple.
        """
        return self._limit_per_host

    def _cleanup(self) -> None:
        """Cleanup unused transports."""
        if self._cleanup_handle:
            self._cleanup_handle.cancel()
            # _cleanup_handle should be unset, otherwise _release() will not
            # recreate it ever!
            self._cleanup_handle = None

        now = self._loop.time()
        timeout = self._keepalive_timeout

        if self._conns:
            connections = {}
            deadline = now - timeout
            for key, conns in self._conns.items():
                alive = []
                for proto, use_time in conns:
                    if proto.is_connected():
                        if use_time - deadline < 0:
                            transport = proto.transport
                            proto.close()
                            if key.is_ssl and not self._cleanup_closed_disabled:
                                self._cleanup_closed_transports.append(transport)
                        else:
                            alive.append((proto, use_time))
                    else:
                        transport = proto.transport
                        proto.close()
                        if key.is_ssl and not self._cleanup_closed_disabled:
                            self._cleanup_closed_transports.append(transport)

                if alive:
                    connections[key] = alive

            self._conns = connections

        if self._conns:
            self._cleanup_handle = helpers.weakref_handle(
                self, "_cleanup", timeout, self._loop
            )

    def _drop_acquired_per_host(
        self, key: "ConnectionKey", val: ResponseHandler
    ) -> None:
        acquired_per_host = self._acquired_per_host
        if key not in acquired_per_host:
            return
        conns = acquired_per_host[key]
        conns.remove(val)
        if not conns:
            del self._acquired_per_host[key]

    def _cleanup_closed(self) -> None:
        """Double confirmation for transport close.

        Some broken ssl servers may leave socket open without proper close.
        """
        if self._cleanup_closed_handle:
            self._cleanup_closed_handle.cancel()

        for transport in self._cleanup_closed_transports:
            if transport is not None:
                transport.abort()

        self._cleanup_closed_transports = []

        if not self._cleanup_closed_disabled:
            self._cleanup_closed_handle = helpers.weakref_handle(
                self, "_cleanup_closed", self._cleanup_closed_period, self._loop
            )

    def close(self) -> Awaitable[None]:
        """Close all opened transports."""
        self._close()
        return _DeprecationWaiter(noop())

    def _close(self) -> None:
        if self._closed:
            return

        self._closed = True

        try:
            if self._loop.is_closed():
                return

            # cancel cleanup task
            if self._cleanup_handle:
                self._cleanup_handle.cancel()

            # cancel cleanup close task
            if self._cleanup_closed_handle:
                self._cleanup_closed_handle.cancel()

            for data in self._conns.values():
                for proto, t0 in data:
                    proto.close()

            for proto in self._acquired:
                proto.close()

            for transport in self._cleanup_closed_transports:
                if transport is not None:
                    transport.abort()

        finally:
            self._conns.clear()
            self._acquired.clear()
            self._waiters.clear()
            self._cleanup_handle = None
            self._cleanup_closed_transports.clear()
            self._cleanup_closed_handle = None

    @property
    def closed(self) -> bool:
        """Is connector closed.

        A readonly property.
        """
        return self._closed

    def _available_connections(self, key: "ConnectionKey") -> int:
        """
        Return number of available connections.

        The limit, limit_per_host and the connection key are taken into account.

        If it returns less than 1 means that there are no connections
        available.
        """
        if self._limit:
            # total calc available connections
            available = self._limit - len(self._acquired)

            # check limit per host
            if (
                self._limit_per_host
                and available > 0
                and key in self._acquired_per_host
            ):
                acquired = self._acquired_per_host.get(key)
                assert acquired is not None
                available = self._limit_per_host - len(acquired)

        elif self._limit_per_host and key in self._acquired_per_host:
            # check limit per host
            acquired = self._acquired_per_host.get(key)
            assert acquired is not None
            available = self._limit_per_host - len(acquired)
        else:
            available = 1

        return available

    async def connect(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> Connection:
        """Get from pool or create new connection."""
        key = req.connection_key
        available = self._available_connections(key)

        # Wait if there are no available connections or if there are/were
        # waiters (i.e. don't steal connection from a waiter about to wake up)
        if available <= 0 or key in self._waiters:
            fut = self._loop.create_future()

            # This connection will now count towards the limit.
            self._waiters[key].append(fut)

            if traces:
                for trace in traces:
                    await trace.send_connection_queued_start()

            try:
                await fut
            except BaseException as e:
                if key in self._waiters:
                    # remove a waiter even if it was cancelled, normally it's
                    #  removed when it's notified
                    try:
                        self._waiters[key].remove(fut)
                    except ValueError:  # fut may no longer be in list
                        pass

                raise e
            finally:
                if key in self._waiters and not self._waiters[key]:
                    del self._waiters[key]

            if traces:
                for trace in traces:
                    await trace.send_connection_queued_end()

        proto = self._get(key)
        if proto is None:
            placeholder = cast(ResponseHandler, _TransportPlaceholder())
            self._acquired.add(placeholder)
            self._acquired_per_host[key].add(placeholder)

            if traces:
                for trace in traces:
                    await trace.send_connection_create_start()

            try:
                proto = await self._create_connection(req, traces, timeout)
                if self._closed:
                    proto.close()
                    raise ClientConnectionError("Connector is closed.")
            except BaseException:
                if not self._closed:
                    self._acquired.remove(placeholder)
                    self._drop_acquired_per_host(key, placeholder)
                    self._release_waiter()
                raise
            else:
                if not self._closed:
                    self._acquired.remove(placeholder)
                    self._drop_acquired_per_host(key, placeholder)

            if traces:
                for trace in traces:
                    await trace.send_connection_create_end()
        else:
            if traces:
                # Acquire the connection to prevent race conditions with limits
                placeholder = cast(ResponseHandler, _TransportPlaceholder())
                self._acquired.add(placeholder)
                self._acquired_per_host[key].add(placeholder)
                for trace in traces:
                    await trace.send_connection_reuseconn()
                self._acquired.remove(placeholder)
                self._drop_acquired_per_host(key, placeholder)

        self._acquired.add(proto)
        self._acquired_per_host[key].add(proto)
        return Connection(self, key, proto, self._loop)

    def _get(self, key: "ConnectionKey") -> Optional[ResponseHandler]:
        try:
            conns = self._conns[key]
        except KeyError:
            return None

        t1 = self._loop.time()
        while conns:
            proto, t0 = conns.pop()
            if proto.is_connected():
                if t1 - t0 > self._keepalive_timeout:
                    transport = proto.transport
                    proto.close()
                    # only for SSL transports
                    if key.is_ssl and not self._cleanup_closed_disabled:
                        self._cleanup_closed_transports.append(transport)
                else:
                    if not conns:
                        # The very last connection was reclaimed: drop the key
                        del self._conns[key]
                    return proto
            else:
                transport = proto.transport
                proto.close()
                if key.is_ssl and not self._cleanup_closed_disabled:
                    self._cleanup_closed_transports.append(transport)

        # No more connections: drop the key
        del self._conns[key]
        return None

    def _release_waiter(self) -> None:
        """
        Iterates over all waiters until one to be released is found.

        The one to be released is not finsihed and
        belongs to a host that has available connections.
        """
        if not self._waiters:
            return

        # Having the dict keys ordered this avoids to iterate
        # at the same order at each call.
        queues = list(self._waiters.keys())
        random.shuffle(queues)

        for key in queues:
            if self._available_connections(key) < 1:
                continue

            waiters = self._waiters[key]
            while waiters:
                waiter = waiters.popleft()
                if not waiter.done():
                    waiter.set_result(None)
                    return

    def _release_acquired(self, key: "ConnectionKey", proto: ResponseHandler) -> None:
        if self._closed:
            # acquired connection is already released on connector closing
            return

        try:
            self._acquired.remove(proto)
            self._drop_acquired_per_host(key, proto)
        except KeyError:  # pragma: no cover
            # this may be result of undetermenistic order of objects
            # finalization due garbage collection.
            pass
        else:
            self._release_waiter()

    def _release(
        self,
        key: "ConnectionKey",
        protocol: ResponseHandler,
        *,
        should_close: bool = False,
    ) -> None:
        if self._closed:
            # acquired connection is already released on connector closing
            return

        self._release_acquired(key, protocol)

        if self._force_close:
            should_close = True

        if should_close or protocol.should_close:
            transport = protocol.transport
            protocol.close()

            if key.is_ssl and not self._cleanup_closed_disabled:
                self._cleanup_closed_transports.append(transport)
        else:
            conns = self._conns.get(key)
            if conns is None:
                conns = self._conns[key] = []
            conns.append((protocol, self._loop.time()))

            if self._cleanup_handle is None:
                self._cleanup_handle = helpers.weakref_handle(
                    self, "_cleanup", self._keepalive_timeout, self._loop
                )

    async def _create_connection(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        raise NotImplementedError()


class _DNSCacheTable:
    def __init__(self, ttl: Optional[float] = None) -> None:
        self._addrs_rr: Dict[Tuple[str, int], Tuple[Iterator[Dict[str, Any]], int]] = {}
        self._timestamps: Dict[Tuple[str, int], float] = {}
        self._ttl = ttl

    def __contains__(self, host: object) -> bool:
        return host in self._addrs_rr

    def add(self, key: Tuple[str, int], addrs: List[Dict[str, Any]]) -> None:
        self._addrs_rr[key] = (cycle(addrs), len(addrs))

        if self._ttl:
            self._timestamps[key] = monotonic()

    def remove(self, key: Tuple[str, int]) -> None:
        self._addrs_rr.pop(key, None)

        if self._ttl:
            self._timestamps.pop(key, None)

    def clear(self) -> None:
        self._addrs_rr.clear()
        self._timestamps.clear()

    def next_addrs(self, key: Tuple[str, int]) -> List[Dict[str, Any]]:
        loop, length = self._addrs_rr[key]
        addrs = list(islice(loop, length))
        # Consume one more element to shift internal state of `cycle`
        next(loop)
        return addrs

    def expired(self, key: Tuple[str, int]) -> bool:
        if self._ttl is None:
            return False

        return self._timestamps[key] + self._ttl < monotonic()


class TCPConnector(BaseConnector):
    """TCP connector.

    verify_ssl - Set to True to check ssl certifications.
    fingerprint - Pass the binary sha256
        digest of the expected certificate in DER format to verify
        that the certificate the server presents matches. See also
        https://en.wikipedia.org/wiki/Transport_Layer_Security#Certificate_pinning
    resolver - Enable DNS lookups and use this
        resolver
    use_dns_cache - Use memory cache for DNS lookups.
    ttl_dns_cache - Max seconds having cached a DNS entry, None forever.
    family - socket address family
    local_addr - local tuple of (host, port) to bind socket to

    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    enable_cleanup_closed - Enables clean-up closed ssl transports.
                            Disabled by default.
    loop - Optional event loop.
    """

    def __init__(
        self,
        *,
        verify_ssl: bool = True,
        fingerprint: Optional[bytes] = None,
        use_dns_cache: bool = True,
        ttl_dns_cache: Optional[int] = 10,
        family: int = 0,
        ssl_context: Optional[SSLContext] = None,
        ssl: Union[None, bool, Fingerprint, SSLContext] = None,
        local_addr: Optional[Tuple[str, int]] = None,
        resolver: Optional[AbstractResolver] = None,
        keepalive_timeout: Union[None, float, object] = sentinel,
        force_close: bool = False,
        limit: int = 100,
        limit_per_host: int = 0,
        enable_cleanup_closed: bool = False,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        super().__init__(
            keepalive_timeout=keepalive_timeout,
            force_close=force_close,
            limit=limit,
            limit_per_host=limit_per_host,
            enable_cleanup_closed=enable_cleanup_closed,
            loop=loop,
        )

        self._ssl = _merge_ssl_params(ssl, verify_ssl, ssl_context, fingerprint)
        if resolver is None:
            resolver = DefaultResolver(loop=self._loop)
        self._resolver = resolver

        self._use_dns_cache = use_dns_cache
        self._cached_hosts = _DNSCacheTable(ttl=ttl_dns_cache)
        self._throttle_dns_events: Dict[Tuple[str, int], EventResultOrError] = {}
        self._family = family
        self._local_addr = local_addr

    def close(self) -> Awaitable[None]:
        """Close all ongoing DNS calls."""
        for ev in self._throttle_dns_events.values():
            ev.cancel()

        return super().close()

    @property
    def family(self) -> int:
        """Socket family like AF_INET."""
        return self._family

    @property
    def use_dns_cache(self) -> bool:
        """True if local DNS caching is enabled."""
        return self._use_dns_cache

    def clear_dns_cache(
        self, host: Optional[str] = None, port: Optional[int] = None
    ) -> None:
        """Remove specified host/port or clear all dns local cache."""
        if host is not None and port is not None:
            self._cached_hosts.remove((host, port))
        elif host is not None or port is not None:
            raise ValueError("either both host and port " "or none of them are allowed")
        else:
            self._cached_hosts.clear()

    async def _resolve_host(
        self, host: str, port: int, traces: Optional[List["Trace"]] = None
    ) -> List[Dict[str, Any]]:
        if is_ip_address(host):
            return [
                {
                    "hostname": host,
                    "host": host,
                    "port": port,
                    "family": self._family,
                    "proto": 0,
                    "flags": 0,
                }
            ]

        if not self._use_dns_cache:

            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_start(host)

            res = await self._resolver.resolve(host, port, family=self._family)

            if traces:
                for trace in traces:
                    await trace.send_dns_resolvehost_end(host)

            return res

        key = (host, port)

        if (key in self._cached_hosts) and (not self._cached_hosts.expired(key)):
            # get result early, before any await (#4014)
            result = self._cached_hosts.next_addrs(key)

            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            return result

        if key in self._throttle_dns_events:
            # get event early, before any await (#4014)
            event = self._throttle_dns_events[key]
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_hit(host)
            await event.wait()
        else:
            # update dict early, before any await (#4014)
            self._throttle_dns_events[key] = EventResultOrError(self._loop)
            if traces:
                for trace in traces:
                    await trace.send_dns_cache_miss(host)
            try:

                if traces:
                    for trace in traces:
                        await trace.send_dns_resolvehost_start(host)

                addrs = await self._resolver.resolve(host, port, family=self._family)
                if traces:
                    for trace in traces:
                        await trace.send_dns_resolvehost_end(host)

                self._cached_hosts.add(key, addrs)
                self._throttle_dns_events[key].set()
            except BaseException as e:
                # any DNS exception, independently of the implementation
                # is set for the waiters to raise the same exception.
                self._throttle_dns_events[key].set(exc=e)
                raise
            finally:
                self._throttle_dns_events.pop(key)

        return self._cached_hosts.next_addrs(key)

    async def _create_connection(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        """Create connection.

        Has same keyword arguments as BaseEventLoop.create_connection.
        """
        if req.proxy:
            _, proto = await self._create_proxy_connection(req, traces, timeout)
        else:
            _, proto = await self._create_direct_connection(req, traces, timeout)

        return proto

    @staticmethod
    @functools.lru_cache(None)
    def _make_ssl_context(verified: bool) -> SSLContext:
        if verified:
            return ssl.create_default_context()
        else:
            sslcontext = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
            sslcontext.options |= ssl.OP_NO_SSLv2
            sslcontext.options |= ssl.OP_NO_SSLv3
            sslcontext.check_hostname = False
            sslcontext.verify_mode = ssl.CERT_NONE
            try:
                sslcontext.options |= ssl.OP_NO_COMPRESSION
            except AttributeError as attr_err:
                warnings.warn(
                    "{!s}: The Python interpreter is compiled "
                    "against OpenSSL < 1.0.0. Ref: "
                    "https://docs.python.org/3/library/ssl.html"
                    "#ssl.OP_NO_COMPRESSION".format(attr_err),
                )
            sslcontext.set_default_verify_paths()
            return sslcontext

    def _get_ssl_context(self, req: "ClientRequest") -> Optional[SSLContext]:
        """Logic to get the correct SSL context

        0. if req.ssl is false, return None

        1. if ssl_context is specified in req, use it
        2. if _ssl_context is specified in self, use it
        3. otherwise:
            1. if verify_ssl is not specified in req, use self.ssl_context
               (will generate a default context according to self.verify_ssl)
            2. if verify_ssl is True in req, generate a default SSL context
            3. if verify_ssl is False in req, generate a SSL context that
               won't verify
        """
        if req.is_ssl():
            if ssl is None:  # pragma: no cover
                raise RuntimeError("SSL is not supported.")
            sslcontext = req.ssl
            if isinstance(sslcontext, ssl.SSLContext):
                return sslcontext
            if sslcontext is not None:
                # not verified or fingerprinted
                return self._make_ssl_context(False)
            sslcontext = self._ssl
            if isinstance(sslcontext, ssl.SSLContext):
                return sslcontext
            if sslcontext is not None:
                # not verified or fingerprinted
                return self._make_ssl_context(False)
            return self._make_ssl_context(True)
        else:
            return None

    def _get_fingerprint(self, req: "ClientRequest") -> Optional["Fingerprint"]:
        ret = req.ssl
        if isinstance(ret, Fingerprint):
            return ret
        ret = self._ssl
        if isinstance(ret, Fingerprint):
            return ret
        return None

    async def _wrap_create_connection(
        self,
        *args: Any,
        req: "ClientRequest",
        timeout: "ClientTimeout",
        client_error: Type[Exception] = ClientConnectorError,
        **kwargs: Any,
    ) -> Tuple[asyncio.Transport, ResponseHandler]:
        try:
            async with ceil_timeout(timeout.sock_connect):
                return await self._loop.create_connection(*args, **kwargs)  # type: ignore[return-value]  # noqa
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc

    def _fail_on_no_start_tls(self, req: "ClientRequest") -> None:
        """Raise a :py:exc:`RuntimeError` on missing ``start_tls()``.

        One case is that :py:meth:`asyncio.loop.start_tls` is not yet
        implemented under Python 3.6. It is necessary for TLS-in-TLS so
        that it is possible to send HTTPS queries through HTTPS proxies.

        This doesn't affect regular HTTP requests, though.
        """
        if not req.is_ssl():
            return

        proxy_url = req.proxy
        assert proxy_url is not None
        if proxy_url.scheme != "https":
            return

        self._check_loop_for_start_tls()

    def _check_loop_for_start_tls(self) -> None:
        try:
            self._loop.start_tls
        except AttributeError as attr_exc:
            raise RuntimeError(
                "An HTTPS request is being sent through an HTTPS proxy. "
                "This needs support for TLS in TLS but it is not implemented "
                "in your runtime for the stdlib asyncio.\n\n"
                "Please upgrade to Python 3.7 or higher. For more details, "
                "please see:\n"
                "* https://bugs.python.org/issue37179\n"
                "* https://github.com/python/cpython/pull/28073\n"
                "* https://docs.aiohttp.org/en/stable/"
                "client_advanced.html#proxy-support\n"
                "* https://github.com/aio-libs/aiohttp/discussions/6044\n",
            ) from attr_exc

    def _loop_supports_start_tls(self) -> bool:
        try:
            self._check_loop_for_start_tls()
        except RuntimeError:
            return False
        else:
            return True

    def _warn_about_tls_in_tls(
        self,
        underlying_transport: asyncio.Transport,
        req: "ClientRequest",
    ) -> None:
        """Issue a warning if the requested URL has HTTPS scheme."""
        if req.request_info.url.scheme != "https":
            return

        asyncio_supports_tls_in_tls = getattr(
            underlying_transport,
            "_start_tls_compatible",
            False,
        )

        if asyncio_supports_tls_in_tls:
            return

        warnings.warn(
            "An HTTPS request is being sent through an HTTPS proxy. "
            "This support for TLS in TLS is known to be disabled "
            "in the stdlib asyncio. This is why you'll probably see "
            "an error in the log below.\n\n"
            "It is possible to enable it via monkeypatching under "
            "Python 3.7 or higher. For more details, see:\n"
            "* https://bugs.python.org/issue37179\n"
            "* https://github.com/python/cpython/pull/28073\n\n"
            "You can temporarily patch this as follows:\n"
            "* https://docs.aiohttp.org/en/stable/client_advanced.html#proxy-support\n"
            "* https://github.com/aio-libs/aiohttp/discussions/6044\n",
            RuntimeWarning,
            source=self,
            # Why `4`? At least 3 of the calls in the stack originate
            # from the methods in this class.
            stacklevel=3,
        )

    async def _start_tls_connection(
        self,
        underlying_transport: asyncio.Transport,
        req: "ClientRequest",
        timeout: "ClientTimeout",
        client_error: Type[Exception] = ClientConnectorError,
    ) -> Tuple[asyncio.BaseTransport, ResponseHandler]:
        """Wrap the raw TCP transport with TLS."""
        tls_proto = self._factory()  # Create a brand new proto for TLS

        # Safety of the `cast()` call here is based on the fact that
        # internally `_get_ssl_context()` only returns `None` when
        # `req.is_ssl()` evaluates to `False` which is never gonna happen
        # in this code path. Of course, it's rather fragile
        # maintainability-wise but this is to be solved separately.
        sslcontext = cast(ssl.SSLContext, self._get_ssl_context(req))

        try:
            async with ceil_timeout(timeout.sock_connect):
                try:
                    tls_transport = await self._loop.start_tls(
                        underlying_transport,
                        tls_proto,
                        sslcontext,
                        server_hostname=req.host,
                        ssl_handshake_timeout=timeout.total,
                    )
                except BaseException:
                    # We need to close the underlying transport since
                    # `start_tls()` probably failed before it had a
                    # chance to do this:
                    underlying_transport.close()
                    raise
        except cert_errors as exc:
            raise ClientConnectorCertificateError(req.connection_key, exc) from exc
        except ssl_errors as exc:
            raise ClientConnectorSSLError(req.connection_key, exc) from exc
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise client_error(req.connection_key, exc) from exc
        except TypeError as type_err:
            # Example cause looks like this:
            # TypeError: transport <asyncio.sslproto._SSLProtocolTransport
            # object at 0x7f760615e460> is not supported by start_tls()

            raise ClientConnectionError(
                "Cannot initialize a TLS-in-TLS connection to host "
                f"{req.host!s}:{req.port:d} through an underlying connection "
                f"to an HTTPS proxy {req.proxy!s} ssl:{req.ssl or 'default'} "
                f"[{type_err!s}]"
            ) from type_err
        else:
            tls_proto.connection_made(
                tls_transport
            )  # Kick the state machine of the new TLS protocol

        return tls_transport, tls_proto

    async def _create_direct_connection(
        self,
        req: "ClientRequest",
        traces: List["Trace"],
        timeout: "ClientTimeout",
        *,
        client_error: Type[Exception] = ClientConnectorError,
    ) -> Tuple[asyncio.Transport, ResponseHandler]:
        sslcontext = self._get_ssl_context(req)
        fingerprint = self._get_fingerprint(req)

        host = req.url.raw_host
        assert host is not None
        port = req.port
        assert port is not None
        host_resolved = asyncio.ensure_future(
            self._resolve_host(host, port, traces=traces), loop=self._loop
        )
        try:
            # Cancelling this lookup should not cancel the underlying lookup
            #  or else the cancel event will get broadcast to all the waiters
            #  across all connections.
            hosts = await asyncio.shield(host_resolved)
        except asyncio.CancelledError:

            def drop_exception(fut: "asyncio.Future[List[Dict[str, Any]]]") -> None:
                with suppress(Exception, asyncio.CancelledError):
                    fut.result()

            host_resolved.add_done_callback(drop_exception)
            raise
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            # in case of proxy it is not ClientProxyConnectionError
            # it is problem of resolving proxy ip itself
            raise ClientConnectorError(req.connection_key, exc) from exc

        last_exc: Optional[Exception] = None

        for hinfo in hosts:
            host = hinfo["host"]
            port = hinfo["port"]

            try:
                transp, proto = await self._wrap_create_connection(
                    self._factory,
                    host,
                    port,
                    timeout=timeout,
                    ssl=sslcontext,
                    family=hinfo["family"],
                    proto=hinfo["proto"],
                    flags=hinfo["flags"],
                    server_hostname=hinfo["hostname"] if sslcontext else None,
                    local_addr=self._local_addr,
                    req=req,
                    client_error=client_error,
                )
            except ClientConnectorError as exc:
                last_exc = exc
                continue

            if req.is_ssl() and fingerprint:
                try:
                    fingerprint.check(transp)
                except ServerFingerprintMismatch as exc:
                    transp.close()
                    if not self._cleanup_closed_disabled:
                        self._cleanup_closed_transports.append(transp)
                    last_exc = exc
                    continue

            return transp, proto
        else:
            assert last_exc is not None
            raise last_exc

    async def _create_proxy_connection(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> Tuple[asyncio.BaseTransport, ResponseHandler]:
        self._fail_on_no_start_tls(req)
        runtime_has_start_tls = self._loop_supports_start_tls()

        headers: Dict[str, str] = {}
        if req.proxy_headers is not None:
            headers = req.proxy_headers  # type: ignore[assignment]
        headers[hdrs.HOST] = req.headers[hdrs.HOST]

        url = req.proxy
        assert url is not None
        proxy_req = ClientRequest(
            hdrs.METH_GET,
            url,
            headers=headers,
            auth=req.proxy_auth,
            loop=self._loop,
            ssl=req.ssl,
        )

        # create connection to proxy server
        transport, proto = await self._create_direct_connection(
            proxy_req, [], timeout, client_error=ClientProxyConnectionError
        )

        # Many HTTP proxies has buggy keepalive support.  Let's not
        # reuse connection but close it after processing every
        # response.
        proto.force_close()

        auth = proxy_req.headers.pop(hdrs.AUTHORIZATION, None)
        if auth is not None:
            if not req.is_ssl():
                req.headers[hdrs.PROXY_AUTHORIZATION] = auth
            else:
                proxy_req.headers[hdrs.PROXY_AUTHORIZATION] = auth

        if req.is_ssl():
            if runtime_has_start_tls:
                self._warn_about_tls_in_tls(transport, req)

            # For HTTPS requests over HTTP proxy
            # we must notify proxy to tunnel connection
            # so we send CONNECT command:
            #   CONNECT www.python.org:443 HTTP/1.1
            #   Host: www.python.org
            #
            # next we must do TLS handshake and so on
            # to do this we must wrap raw socket into secure one
            # asyncio handles this perfectly
            proxy_req.method = hdrs.METH_CONNECT
            proxy_req.url = req.url
            key = attr.evolve(
                req.connection_key, proxy=None, proxy_auth=None, proxy_headers_hash=None
            )
            conn = Connection(self, key, proto, self._loop)
            proxy_resp = await proxy_req.send(conn)
            try:
                protocol = conn._protocol
                assert protocol is not None

                # read_until_eof=True will ensure the connection isn't closed
                # once the response is received and processed allowing
                # START_TLS to work on the connection below.
                protocol.set_response_params(read_until_eof=runtime_has_start_tls)
                resp = await proxy_resp.start(conn)
            except BaseException:
                proxy_resp.close()
                conn.close()
                raise
            else:
                conn._protocol = None
                conn._transport = None
                try:
                    if resp.status != 200:
                        message = resp.reason
                        if message is None:
                            message = RESPONSES[resp.status][0]
                        raise ClientHttpProxyError(
                            proxy_resp.request_info,
                            resp.history,
                            status=resp.status,
                            message=message,
                            headers=resp.headers,
                        )
                    if not runtime_has_start_tls:
                        rawsock = transport.get_extra_info("socket", default=None)
                        if rawsock is None:
                            raise RuntimeError(
                                "Transport does not expose socket instance"
                            )
                        # Duplicate the socket, so now we can close proxy transport
                        rawsock = rawsock.dup()
                except BaseException:
                    # It shouldn't be closed in `finally` because it's fed to
                    # `loop.start_tls()` and the docs say not to touch it after
                    # passing there.
                    transport.close()
                    raise
                finally:
                    if not runtime_has_start_tls:
                        transport.close()

                if not runtime_has_start_tls:
                    # HTTP proxy with support for upgrade to HTTPS
                    sslcontext = self._get_ssl_context(req)
                    return await self._wrap_create_connection(
                        self._factory,
                        timeout=timeout,
                        ssl=sslcontext,
                        sock=rawsock,
                        server_hostname=req.host,
                        req=req,
                    )

                return await self._start_tls_connection(
                    # Access the old transport for the last time before it's
                    # closed and forgotten forever:
                    transport,
                    req=req,
                    timeout=timeout,
                )
            finally:
                proxy_resp.close()

        return transport, proto


class UnixConnector(BaseConnector):
    """Unix socket connector.

    path - Unix socket path.
    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """

    def __init__(
        self,
        path: str,
        force_close: bool = False,
        keepalive_timeout: Union[object, float, None] = sentinel,
        limit: int = 100,
        limit_per_host: int = 0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__(
            force_close=force_close,
            keepalive_timeout=keepalive_timeout,
            limit=limit,
            limit_per_host=limit_per_host,
            loop=loop,
        )
        self._path = path

    @property
    def path(self) -> str:
        """Path to unix socket."""
        return self._path

    async def _create_connection(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        try:
            async with ceil_timeout(timeout.sock_connect):
                _, proto = await self._loop.create_unix_connection(
                    self._factory, self._path
                )
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise UnixClientConnectorError(self.path, req.connection_key, exc) from exc

        return cast(ResponseHandler, proto)


class NamedPipeConnector(BaseConnector):
    """Named pipe connector.

    Only supported by the proactor event loop.
    See also: https://docs.python.org/3.7/library/asyncio-eventloop.html

    path - Windows named pipe path.
    keepalive_timeout - (optional) Keep-alive timeout.
    force_close - Set to True to force close and do reconnect
        after each request (and between redirects).
    limit - The total number of simultaneous connections.
    limit_per_host - Number of simultaneous connections to one host.
    loop - Optional event loop.
    """

    def __init__(
        self,
        path: str,
        force_close: bool = False,
        keepalive_timeout: Union[object, float, None] = sentinel,
        limit: int = 100,
        limit_per_host: int = 0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ) -> None:
        super().__init__(
            force_close=force_close,
            keepalive_timeout=keepalive_timeout,
            limit=limit,
            limit_per_host=limit_per_host,
            loop=loop,
        )
        if not isinstance(
            self._loop, asyncio.ProactorEventLoop  # type: ignore[attr-defined]
        ):
            raise RuntimeError(
                "Named Pipes only available in proactor " "loop under windows"
            )
        self._path = path

    @property
    def path(self) -> str:
        """Path to the named pipe."""
        return self._path

    async def _create_connection(
        self, req: "ClientRequest", traces: List["Trace"], timeout: "ClientTimeout"
    ) -> ResponseHandler:
        try:
            async with ceil_timeout(timeout.sock_connect):
                _, proto = await self._loop.create_pipe_connection(  # type: ignore[attr-defined] # noqa: E501
                    self._factory, self._path
                )
                # the drain is required so that the connection_made is called
                # and transport is set otherwise it is not set before the
                # `assert conn.transport is not None`
                # in client.py's _request method
                await asyncio.sleep(0)
                # other option is to manually set transport like
                # `proto.transport = trans`
        except OSError as exc:
            if exc.errno is None and isinstance(exc, asyncio.TimeoutError):
                raise
            raise ClientConnectorError(req.connection_key, exc) from exc

        return cast(ResponseHandler, proto)
