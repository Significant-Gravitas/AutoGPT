# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2017 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""Talk to a DNS server."""

from typing import Any, Dict, Optional, Tuple, Union

import base64
import contextlib
import socket
import struct
import time

import dns.asyncbackend
import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.transaction

from dns._asyncbackend import NullContext
from dns.query import (
    _compute_times,
    _matches_destination,
    BadResponse,
    ssl,
    UDPMode,
    _have_httpx,
    _have_http2,
    NoDOH,
    NoDOQ,
)

if _have_httpx:
    import httpx

# for brevity
_lltuple = dns.inet.low_level_address_tuple


def _source_tuple(af, address, port):
    # Make a high level source tuple, or return None if address and port
    # are both None
    if address or port:
        if address is None:
            if af == socket.AF_INET:
                address = "0.0.0.0"
            elif af == socket.AF_INET6:
                address = "::"
            else:
                raise NotImplementedError(f"unknown address family {af}")
        return (address, port)
    else:
        return None


def _timeout(expiration, now=None):
    if expiration:
        if not now:
            now = time.time()
        return max(expiration - now, 0)
    else:
        return None


async def send_udp(
    sock: dns.asyncbackend.DatagramSocket,
    what: Union[dns.message.Message, bytes],
    destination: Any,
    expiration: Optional[float] = None,
) -> Tuple[int, float]:
    """Send a DNS message to the specified UDP socket.

    *sock*, a ``dns.asyncbackend.DatagramSocket``.

    *what*, a ``bytes`` or ``dns.message.Message``, the message to send.

    *destination*, a destination tuple appropriate for the address family
    of the socket, specifying where to send the query.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.  The expiration value is meaningless for the asyncio backend, as
    asyncio's transport sendto() never blocks.

    Returns an ``(int, float)`` tuple of bytes sent and the sent time.
    """

    if isinstance(what, dns.message.Message):
        what = what.to_wire()
    sent_time = time.time()
    n = await sock.sendto(what, destination, _timeout(expiration, sent_time))
    return (n, sent_time)


async def receive_udp(
    sock: dns.asyncbackend.DatagramSocket,
    destination: Optional[Any] = None,
    expiration: Optional[float] = None,
    ignore_unexpected: bool = False,
    one_rr_per_rrset: bool = False,
    keyring: Optional[Dict[dns.name.Name, dns.tsig.Key]] = None,
    request_mac: Optional[bytes] = b"",
    ignore_trailing: bool = False,
    raise_on_truncation: bool = False,
) -> Any:
    """Read a DNS message from a UDP socket.

    *sock*, a ``dns.asyncbackend.DatagramSocket``.

    See :py:func:`dns.query.receive_udp()` for the documentation of the other
    parameters, and exceptions.

    Returns a ``(dns.message.Message, float, tuple)`` tuple of the received message, the
    received time, and the address where the message arrived from.
    """

    wire = b""
    while 1:
        (wire, from_address) = await sock.recvfrom(65535, _timeout(expiration))
        if _matches_destination(
            sock.family, from_address, destination, ignore_unexpected
        ):
            break
    received_time = time.time()
    r = dns.message.from_wire(
        wire,
        keyring=keyring,
        request_mac=request_mac,
        one_rr_per_rrset=one_rr_per_rrset,
        ignore_trailing=ignore_trailing,
        raise_on_truncation=raise_on_truncation,
    )
    return (r, received_time, from_address)


async def udp(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 53,
    source: Optional[str] = None,
    source_port: int = 0,
    ignore_unexpected: bool = False,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    raise_on_truncation: bool = False,
    sock: Optional[dns.asyncbackend.DatagramSocket] = None,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> dns.message.Message:
    """Return the response obtained after sending a query via UDP.

    *sock*, a ``dns.asyncbackend.DatagramSocket``, or ``None``,
    the socket to use for the query.  If ``None``, the default, a
    socket is created.  Note that if a socket is provided, the
    *source*, *source_port*, and *backend* are ignored.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.udp()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """
    wire = q.to_wire()
    (begin_time, expiration) = _compute_times(timeout)
    af = dns.inet.af_for_address(where)
    destination = _lltuple((where, port), af)
    if sock:
        cm: contextlib.AbstractAsyncContextManager = NullContext(sock)
    else:
        if not backend:
            backend = dns.asyncbackend.get_default_backend()
        stuple = _source_tuple(af, source, source_port)
        if backend.datagram_connection_required():
            dtuple = (where, port)
        else:
            dtuple = None
        cm = await backend.make_socket(af, socket.SOCK_DGRAM, 0, stuple, dtuple)
    async with cm as s:
        await send_udp(s, wire, destination, expiration)
        (r, received_time, _) = await receive_udp(
            s,
            destination,
            expiration,
            ignore_unexpected,
            one_rr_per_rrset,
            q.keyring,
            q.mac,
            ignore_trailing,
            raise_on_truncation,
        )
        r.time = received_time - begin_time
        if not q.is_response(r):
            raise BadResponse
        return r


async def udp_with_fallback(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 53,
    source: Optional[str] = None,
    source_port: int = 0,
    ignore_unexpected: bool = False,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    udp_sock: Optional[dns.asyncbackend.DatagramSocket] = None,
    tcp_sock: Optional[dns.asyncbackend.StreamSocket] = None,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> Tuple[dns.message.Message, bool]:
    """Return the response to the query, trying UDP first and falling back
    to TCP if UDP results in a truncated response.

    *udp_sock*, a ``dns.asyncbackend.DatagramSocket``, or ``None``,
    the socket to use for the UDP query.  If ``None``, the default, a
    socket is created.  Note that if a socket is provided the *source*,
    *source_port*, and *backend* are ignored for the UDP query.

    *tcp_sock*, a ``dns.asyncbackend.StreamSocket``, or ``None``, the
    socket to use for the TCP query.  If ``None``, the default, a
    socket is created.  Note that if a socket is provided *where*,
    *source*, *source_port*, and *backend*  are ignored for the TCP query.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.udp_with_fallback()` for the documentation
    of the other parameters, exceptions, and return type of this
    method.
    """
    try:
        response = await udp(
            q,
            where,
            timeout,
            port,
            source,
            source_port,
            ignore_unexpected,
            one_rr_per_rrset,
            ignore_trailing,
            True,
            udp_sock,
            backend,
        )
        return (response, False)
    except dns.message.Truncated:
        response = await tcp(
            q,
            where,
            timeout,
            port,
            source,
            source_port,
            one_rr_per_rrset,
            ignore_trailing,
            tcp_sock,
            backend,
        )
        return (response, True)


async def send_tcp(
    sock: dns.asyncbackend.StreamSocket,
    what: Union[dns.message.Message, bytes],
    expiration: Optional[float] = None,
) -> Tuple[int, float]:
    """Send a DNS message to the specified TCP socket.

    *sock*, a ``dns.asyncbackend.StreamSocket``.

    See :py:func:`dns.query.send_tcp()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """

    if isinstance(what, dns.message.Message):
        wire = what.to_wire()
    else:
        wire = what
    l = len(wire)
    # copying the wire into tcpmsg is inefficient, but lets us
    # avoid writev() or doing a short write that would get pushed
    # onto the net
    tcpmsg = struct.pack("!H", l) + wire
    sent_time = time.time()
    await sock.sendall(tcpmsg, _timeout(expiration, sent_time))
    return (len(tcpmsg), sent_time)


async def _read_exactly(sock, count, expiration):
    """Read the specified number of bytes from stream.  Keep trying until we
    either get the desired amount, or we hit EOF.
    """
    s = b""
    while count > 0:
        n = await sock.recv(count, _timeout(expiration))
        if n == b"":
            raise EOFError
        count = count - len(n)
        s = s + n
    return s


async def receive_tcp(
    sock: dns.asyncbackend.StreamSocket,
    expiration: Optional[float] = None,
    one_rr_per_rrset: bool = False,
    keyring: Optional[Dict[dns.name.Name, dns.tsig.Key]] = None,
    request_mac: Optional[bytes] = b"",
    ignore_trailing: bool = False,
) -> Tuple[dns.message.Message, float]:
    """Read a DNS message from a TCP socket.

    *sock*, a ``dns.asyncbackend.StreamSocket``.

    See :py:func:`dns.query.receive_tcp()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """

    ldata = await _read_exactly(sock, 2, expiration)
    (l,) = struct.unpack("!H", ldata)
    wire = await _read_exactly(sock, l, expiration)
    received_time = time.time()
    r = dns.message.from_wire(
        wire,
        keyring=keyring,
        request_mac=request_mac,
        one_rr_per_rrset=one_rr_per_rrset,
        ignore_trailing=ignore_trailing,
    )
    return (r, received_time)


async def tcp(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 53,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    sock: Optional[dns.asyncbackend.StreamSocket] = None,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> dns.message.Message:
    """Return the response obtained after sending a query via TCP.

    *sock*, a ``dns.asyncbacket.StreamSocket``, or ``None``, the
    socket to use for the query.  If ``None``, the default, a socket
    is created.  Note that if a socket is provided
    *where*, *port*, *source*, *source_port*, and *backend* are ignored.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.tcp()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """

    wire = q.to_wire()
    (begin_time, expiration) = _compute_times(timeout)
    if sock:
        # Verify that the socket is connected, as if it's not connected,
        # it's not writable, and the polling in send_tcp() will time out or
        # hang forever.
        await sock.getpeername()
        cm: contextlib.AbstractAsyncContextManager = NullContext(sock)
    else:
        # These are simple (address, port) pairs, not family-dependent tuples
        # you pass to low-level socket code.
        af = dns.inet.af_for_address(where)
        stuple = _source_tuple(af, source, source_port)
        dtuple = (where, port)
        if not backend:
            backend = dns.asyncbackend.get_default_backend()
        cm = await backend.make_socket(
            af, socket.SOCK_STREAM, 0, stuple, dtuple, timeout
        )
    async with cm as s:
        await send_tcp(s, wire, expiration)
        (r, received_time) = await receive_tcp(
            s, expiration, one_rr_per_rrset, q.keyring, q.mac, ignore_trailing
        )
        r.time = received_time - begin_time
        if not q.is_response(r):
            raise BadResponse
        return r


async def tls(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 853,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    sock: Optional[dns.asyncbackend.StreamSocket] = None,
    backend: Optional[dns.asyncbackend.Backend] = None,
    ssl_context: Optional[ssl.SSLContext] = None,
    server_hostname: Optional[str] = None,
) -> dns.message.Message:
    """Return the response obtained after sending a query via TLS.

    *sock*, an ``asyncbackend.StreamSocket``, or ``None``, the socket
    to use for the query.  If ``None``, the default, a socket is
    created.  Note that if a socket is provided, it must be a
    connected SSL stream socket, and *where*, *port*,
    *source*, *source_port*, *backend*, *ssl_context*, and *server_hostname*
    are ignored.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.tls()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """
    (begin_time, expiration) = _compute_times(timeout)
    if sock:
        cm: contextlib.AbstractAsyncContextManager = NullContext(sock)
    else:
        if ssl_context is None:
            # See the comment about ssl.create_default_context() in query.py
            ssl_context = ssl.create_default_context()  # lgtm[py/insecure-protocol]
            ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
            if server_hostname is None:
                ssl_context.check_hostname = False
        else:
            ssl_context = None
            server_hostname = None
        af = dns.inet.af_for_address(where)
        stuple = _source_tuple(af, source, source_port)
        dtuple = (where, port)
        if not backend:
            backend = dns.asyncbackend.get_default_backend()
        cm = await backend.make_socket(
            af,
            socket.SOCK_STREAM,
            0,
            stuple,
            dtuple,
            timeout,
            ssl_context,
            server_hostname,
        )
    async with cm as s:
        timeout = _timeout(expiration)
        response = await tcp(
            q,
            where,
            timeout,
            port,
            source,
            source_port,
            one_rr_per_rrset,
            ignore_trailing,
            s,
            backend,
        )
        end_time = time.time()
        response.time = end_time - begin_time
        return response


async def https(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 443,
    source: Optional[str] = None,
    source_port: int = 0,  # pylint: disable=W0613
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    client: Optional["httpx.AsyncClient"] = None,
    path: str = "/dns-query",
    post: bool = True,
    verify: Union[bool, str] = True,
) -> dns.message.Message:
    """Return the response obtained after sending a query via DNS-over-HTTPS.

    *client*, a ``httpx.AsyncClient``.  If provided, the client to use for
    the query.

    Unlike the other dnspython async functions, a backend cannot be provided
    in this function because httpx always auto-detects the async backend.

    See :py:func:`dns.query.https()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """

    if not _have_httpx:
        raise NoDOH("httpx is not available.")  # pragma: no cover

    wire = q.to_wire()
    try:
        af = dns.inet.af_for_address(where)
    except ValueError:
        af = None
    transport = None
    headers = {"accept": "application/dns-message"}
    if af is not None:
        if af == socket.AF_INET:
            url = "https://{}:{}{}".format(where, port, path)
        elif af == socket.AF_INET6:
            url = "https://[{}]:{}{}".format(where, port, path)
    else:
        url = where
    if source is not None:
        transport = httpx.AsyncHTTPTransport(local_address=source[0])

    if client:
        cm: contextlib.AbstractAsyncContextManager = NullContext(client)
    else:
        cm = httpx.AsyncClient(
            http1=True, http2=_have_http2, verify=verify, transport=transport
        )

    async with cm as the_client:
        # see https://tools.ietf.org/html/rfc8484#section-4.1.1 for DoH
        # GET and POST examples
        if post:
            headers.update(
                {
                    "content-type": "application/dns-message",
                    "content-length": str(len(wire)),
                }
            )
            response = await the_client.post(
                url, headers=headers, content=wire, timeout=timeout
            )
        else:
            wire = base64.urlsafe_b64encode(wire).rstrip(b"=")
            twire = wire.decode()  # httpx does a repr() if we give it bytes
            response = await the_client.get(
                url, headers=headers, timeout=timeout, params={"dns": twire}
            )

    # see https://tools.ietf.org/html/rfc8484#section-4.2.1 for info about DoH
    # status codes
    if response.status_code < 200 or response.status_code > 299:
        raise ValueError(
            "{} responded with status code {}"
            "\nResponse body: {!r}".format(
                where, response.status_code, response.content
            )
        )
    r = dns.message.from_wire(
        response.content,
        keyring=q.keyring,
        request_mac=q.request_mac,
        one_rr_per_rrset=one_rr_per_rrset,
        ignore_trailing=ignore_trailing,
    )
    r.time = response.elapsed.total_seconds()
    if not q.is_response(r):
        raise BadResponse
    return r


async def inbound_xfr(
    where: str,
    txn_manager: dns.transaction.TransactionManager,
    query: Optional[dns.message.Message] = None,
    port: int = 53,
    timeout: Optional[float] = None,
    lifetime: Optional[float] = None,
    source: Optional[str] = None,
    source_port: int = 0,
    udp_mode: UDPMode = UDPMode.NEVER,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> None:
    """Conduct an inbound transfer and apply it via a transaction from the
    txn_manager.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.inbound_xfr()` for the documentation of
    the other parameters, exceptions, and return type of this method.
    """
    if query is None:
        (query, serial) = dns.xfr.make_query(txn_manager)
    else:
        serial = dns.xfr.extract_serial_from_query(query)
    rdtype = query.question[0].rdtype
    is_ixfr = rdtype == dns.rdatatype.IXFR
    origin = txn_manager.from_wire_origin()
    wire = query.to_wire()
    af = dns.inet.af_for_address(where)
    stuple = _source_tuple(af, source, source_port)
    dtuple = (where, port)
    (_, expiration) = _compute_times(lifetime)
    retry = True
    while retry:
        retry = False
        if is_ixfr and udp_mode != UDPMode.NEVER:
            sock_type = socket.SOCK_DGRAM
            is_udp = True
        else:
            sock_type = socket.SOCK_STREAM
            is_udp = False
        if not backend:
            backend = dns.asyncbackend.get_default_backend()
        s = await backend.make_socket(
            af, sock_type, 0, stuple, dtuple, _timeout(expiration)
        )
        async with s:
            if is_udp:
                await s.sendto(wire, dtuple, _timeout(expiration))
            else:
                tcpmsg = struct.pack("!H", len(wire)) + wire
                await s.sendall(tcpmsg, expiration)
            with dns.xfr.Inbound(txn_manager, rdtype, serial, is_udp) as inbound:
                done = False
                tsig_ctx = None
                while not done:
                    (_, mexpiration) = _compute_times(timeout)
                    if mexpiration is None or (
                        expiration is not None and mexpiration > expiration
                    ):
                        mexpiration = expiration
                    if is_udp:
                        destination = _lltuple((where, port), af)
                        while True:
                            timeout = _timeout(mexpiration)
                            (rwire, from_address) = await s.recvfrom(65535, timeout)
                            if _matches_destination(
                                af, from_address, destination, True
                            ):
                                break
                    else:
                        ldata = await _read_exactly(s, 2, mexpiration)
                        (l,) = struct.unpack("!H", ldata)
                        rwire = await _read_exactly(s, l, mexpiration)
                    is_ixfr = rdtype == dns.rdatatype.IXFR
                    r = dns.message.from_wire(
                        rwire,
                        keyring=query.keyring,
                        request_mac=query.mac,
                        xfr=True,
                        origin=origin,
                        tsig_ctx=tsig_ctx,
                        multi=(not is_udp),
                        one_rr_per_rrset=is_ixfr,
                    )
                    try:
                        done = inbound.process_message(r)
                    except dns.xfr.UseTCP:
                        assert is_udp  # should not happen if we used TCP!
                        if udp_mode == UDPMode.ONLY:
                            raise
                        done = True
                        retry = True
                        udp_mode = UDPMode.NEVER
                        continue
                    tsig_ctx = r.tsig_ctx
                if not retry and query.keyring and not r.had_tsig:
                    raise dns.exception.FormError("missing TSIG")


async def quic(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 853,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    connection: Optional[dns.quic.AsyncQuicConnection] = None,
    verify: Union[bool, str] = True,
    backend: Optional[dns.asyncbackend.Backend] = None,
) -> dns.message.Message:
    """Return the response obtained after sending an asynchronous query via
    DNS-over-QUIC.

    *backend*, a ``dns.asyncbackend.Backend``, or ``None``.  If ``None``,
    the default, then dnspython will use the default backend.

    See :py:func:`dns.query.quic()` for the documentation of the other
    parameters, exceptions, and return type of this method.
    """

    if not dns.quic.have_quic:
        raise NoDOQ("DNS-over-QUIC is not available.")  # pragma: no cover

    q.id = 0
    wire = q.to_wire()
    the_connection: dns.quic.AsyncQuicConnection
    if connection:
        cfactory = dns.quic.null_factory
        mfactory = dns.quic.null_factory
        the_connection = connection
    else:
        (cfactory, mfactory) = dns.quic.factories_for_backend(backend)

    async with cfactory() as context:
        async with mfactory(context, verify_mode=verify) as the_manager:
            if not connection:
                the_connection = the_manager.connect(where, port, source, source_port)
            start = time.time()
            stream = await the_connection.make_stream()
            async with stream:
                await stream.send(wire, True)
                wire = await stream.receive(timeout)
            finish = time.time()
        r = dns.message.from_wire(
            wire,
            keyring=q.keyring,
            request_mac=q.request_mac,
            one_rr_per_rrset=one_rr_per_rrset,
            ignore_trailing=ignore_trailing,
        )
    r.time = max(finish - start, 0.0)
    if not q.is_response(r):
        raise BadResponse
    return r
