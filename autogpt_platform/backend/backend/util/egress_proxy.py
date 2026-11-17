"""A loopback SOCKS5 proxy that only connects to public internet addresses.

`backend.util.request.Requests` checks every hop of every request it makes, but
programs that do their own networking (the Chromium behind the Copilot browser
tools) never pass through it: their redirects, page scripts, subresources and
DNS lookups would reach internal services unchecked. Pointing such a program at
this proxy puts the same rule on every connection it opens. The proxy resolves
the target itself, refuses it unless every address is public, and dials the
exact address it checked, so a DNS answer cannot change between the check and
the connection.
"""

import asyncio
import concurrent.futures
import ipaddress
import logging
import os
import socket
import threading

from backend.util.request import resolve_and_check_blocked

logger = logging.getLogger(__name__)

_SETUP_TIMEOUT = 30
_CONNECT_TIMEOUT = 10
_STARTUP_TIMEOUT = 10
_CHUNK_SIZE = 64 * 1024

# SOCKS5 wire values (RFC 1928).
_SOCKS_VERSION = 5
_NO_AUTH = 0
_NO_ACCEPTABLE_METHODS = 0xFF
_CONNECT = 1
_ADDR_IPV4 = 1
_ADDR_DOMAIN = 3
_ADDR_IPV6 = 4
_SUCCEEDED = 0
_NOT_ALLOWED = 2
_HOST_UNREACHABLE = 4
_COMMAND_NOT_SUPPORTED = 7
_ADDRESS_TYPE_NOT_SUPPORTED = 8

_lock = threading.Lock()
_proxy_url: str | None = None
_proxy_pid: int | None = None


def egress_proxy_url() -> str:
    """Return this process's proxy as a ``socks5://`` URL, starting it on first
    use."""
    global _proxy_url, _proxy_pid
    with _lock:
        if _proxy_url is None or _proxy_pid != os.getpid():
            _proxy_url = _start()
            _proxy_pid = os.getpid()
        return _proxy_url


def _start() -> str:
    # Its own thread and loop, so the proxy outlives whichever event loop first
    # asked for it: the Copilot executor runs each turn on a worker loop.
    loop = asyncio.new_event_loop()
    started: concurrent.futures.Future[int] = concurrent.futures.Future()

    def serve() -> None:
        asyncio.set_event_loop(loop)
        try:
            server = loop.run_until_complete(
                asyncio.start_server(_handle, "127.0.0.1", 0)
            )
        except Exception as e:
            started.set_exception(e)
            return
        started.set_result(server.sockets[0].getsockname()[1])
        loop.run_forever()

    threading.Thread(target=serve, name="egress-proxy", daemon=True).start()
    return f"socks5://127.0.0.1:{started.result(timeout=_STARTUP_TIMEOUT)}"


# IPv6 ranges that carry an IPv4 address a gateway may forward to: NAT64
# (RFC 6052, RFC 8215), 6to4 (RFC 3056) and IPv4-compatible (RFC 4291).
# ipaddress.is_global passes some of these depending on the Python version, and
# on a network that routes them they reach the embedded, possibly private, IPv4.
_IPV4_EMBEDDING_NETWORKS = (
    ipaddress.ip_network("64:ff9b::/96"),
    ipaddress.ip_network("64:ff9b:1::/48"),
    ipaddress.ip_network("2002::/16"),
    ipaddress.ip_network("::/96"),
)


def _is_public(address: str) -> bool:
    ip = ipaddress.ip_address(address)
    if isinstance(ip, ipaddress.IPv6Address):
        if ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        elif ip.teredo or any(ip in net for net in _IPV4_EMBEDDING_NETWORKS):
            return False
    return ip.is_global


async def public_addresses(host: str) -> list[str]:
    """Resolve *host* and return its addresses.

    Raises ``ValueError`` unless every address is public: outside the shared
    SSRF blocklist, globally routable, and not an IPv6 address that embeds an
    IPv4 one.
    """
    if not host or not host.isascii():
        raise ValueError(f"Invalid host {host!r}")
    addresses = await resolve_and_check_blocked(host)
    for address in addresses:
        if not _is_public(address):
            raise ValueError(
                f"Access to non-public IP address {address} "
                f"for hostname {host} is not allowed."
            )
    return addresses


async def _handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    upstream: tuple[asyncio.StreamReader, asyncio.StreamWriter] | None = None
    try:
        upstream = await asyncio.wait_for(_open(reader, writer), _SETUP_TIMEOUT)
        if upstream is not None:
            upstream_reader, upstream_writer = upstream
            await asyncio.gather(
                _pipe(reader, upstream_writer), _pipe(upstream_reader, writer)
            )
    except (asyncio.IncompleteReadError, OSError):
        pass
    finally:
        writer.close()
        if upstream is not None:
            upstream[1].close()


async def _open(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
    """Read the client's SOCKS5 request and connect it if the target is public.
    Returns None after replying with an error."""
    version, method_count = await reader.readexactly(2)
    methods = await reader.readexactly(method_count)
    if version != _SOCKS_VERSION or _NO_AUTH not in methods:
        writer.write(bytes([_SOCKS_VERSION, _NO_ACCEPTABLE_METHODS]))
        return None
    writer.write(bytes([_SOCKS_VERSION, _NO_AUTH]))

    _, command, _, address_type = await reader.readexactly(4)
    if address_type == _ADDR_IPV4:
        host = socket.inet_ntop(socket.AF_INET, await reader.readexactly(4))
    elif address_type == _ADDR_IPV6:
        host = socket.inet_ntop(socket.AF_INET6, await reader.readexactly(16))
    elif address_type == _ADDR_DOMAIN:
        length = (await reader.readexactly(1))[0]
        host = (await reader.readexactly(length)).decode("ascii", "replace")
    else:
        await _reply(writer, _ADDRESS_TYPE_NOT_SUPPORTED)
        return None
    port = int.from_bytes(await reader.readexactly(2), "big")
    if command != _CONNECT:
        await _reply(writer, _COMMAND_NOT_SUPPORTED)
        return None

    try:
        addresses = await public_addresses(host)
    except ValueError as e:
        logger.info(f"Egress proxy refused {host!r} port {port}: {e}")
        await _reply(writer, _NOT_ALLOWED)
        return None

    # Dial the checked addresses, never the name: resolving it again could
    # return a different answer. Only IPv4 is dialed: a network may run NAT64
    # under any prefix its operator picks, and such an address forwards to an
    # embedded, possibly private, IPv4 one while looking like a public IPv6
    # address. No address check can tell the two apart.
    dialable = [
        a for a in dict.fromkeys(addresses) if ipaddress.ip_address(a).version == 4
    ]
    if not dialable:
        logger.info(f"Egress proxy refused {host!r} port {port}: no IPv4 address")
        await _reply(writer, _NOT_ALLOWED)
        return None
    for address in dialable:
        try:
            upstream = await asyncio.wait_for(
                asyncio.open_connection(address, port), _CONNECT_TIMEOUT
            )
        except OSError:
            continue
        try:
            await _reply(writer, _SUCCEEDED)
        except BaseException:
            upstream[1].close()
            raise
        return upstream

    await _reply(writer, _HOST_UNREACHABLE)
    return None


async def _reply(writer: asyncio.StreamWriter, code: int) -> None:
    writer.write(bytes([_SOCKS_VERSION, code, 0, _ADDR_IPV4, 0, 0, 0, 0, 0, 0]))
    await writer.drain()


async def _pipe(source: asyncio.StreamReader, sink: asyncio.StreamWriter) -> None:
    """Copy *source* into *sink* until either side closes, then close *sink* so
    the opposite direction ends too."""
    try:
        while data := await source.read(_CHUNK_SIZE):
            sink.write(data)
            await sink.drain()
    except OSError:
        pass
    finally:
        sink.close()
