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
import enum
import errno
import os
import selectors
import socket
import struct
import time
import urllib.parse

import dns.exception
import dns.inet
import dns.name
import dns.message
import dns.quic
import dns.rcode
import dns.rdataclass
import dns.rdatatype
import dns.serial
import dns.transaction
import dns.tsig
import dns.xfr

try:
    import requests
    from requests_toolbelt.adapters.source import SourceAddressAdapter
    from requests_toolbelt.adapters.host_header_ssl import HostHeaderSSLAdapter

    _have_requests = True
except ImportError:  # pragma: no cover
    _have_requests = False

_have_httpx = False
_have_http2 = False
try:
    import httpx

    _have_httpx = True
    try:
        # See if http2 support is available.
        with httpx.Client(http2=True):
            _have_http2 = True
    except Exception:
        pass
except ImportError:  # pragma: no cover
    pass

have_doh = _have_requests or _have_httpx

try:
    import ssl
except ImportError:  # pragma: no cover

    class ssl:  # type: ignore
        class WantReadException(Exception):
            pass

        class WantWriteException(Exception):
            pass

        class SSLContext:
            pass

        class SSLSocket:
            pass

        @classmethod
        def create_default_context(cls, *args, **kwargs):
            raise Exception("no ssl support")


# Function used to create a socket.  Can be overridden if needed in special
# situations.
socket_factory = socket.socket


class UnexpectedSource(dns.exception.DNSException):
    """A DNS query response came from an unexpected address or port."""


class BadResponse(dns.exception.FormError):
    """A DNS query response does not respond to the question asked."""


class NoDOH(dns.exception.DNSException):
    """DNS over HTTPS (DOH) was requested but the requests module is not
    available."""


class NoDOQ(dns.exception.DNSException):
    """DNS over QUIC (DOQ) was requested but the aioquic module is not
    available."""


# for backwards compatibility
TransferError = dns.xfr.TransferError


def _compute_times(timeout):
    now = time.time()
    if timeout is None:
        return (now, None)
    else:
        return (now, now + timeout)


def _wait_for(fd, readable, writable, _, expiration):
    # Use the selected selector class to wait for any of the specified
    # events.  An "expiration" absolute time is converted into a relative
    # timeout.
    #
    # The unused parameter is 'error', which is always set when
    # selecting for read or write, and we have no error-only selects.

    if readable and isinstance(fd, ssl.SSLSocket) and fd.pending() > 0:
        return True
    sel = _selector_class()
    events = 0
    if readable:
        events |= selectors.EVENT_READ
    if writable:
        events |= selectors.EVENT_WRITE
    if events:
        sel.register(fd, events)
    if expiration is None:
        timeout = None
    else:
        timeout = expiration - time.time()
        if timeout <= 0.0:
            raise dns.exception.Timeout
    if not sel.select(timeout):
        raise dns.exception.Timeout


def _set_selector_class(selector_class):
    # Internal API. Do not use.

    global _selector_class

    _selector_class = selector_class


if hasattr(selectors, "PollSelector"):
    # Prefer poll() on platforms that support it because it has no
    # limits on the maximum value of a file descriptor (plus it will
    # be more efficient for high values).
    #
    # We ignore typing here as we can't say _selector_class is Any
    # on python < 3.8 due to a bug.
    _selector_class = selectors.PollSelector  # type: ignore
else:
    _selector_class = selectors.SelectSelector  # type: ignore


def _wait_for_readable(s, expiration):
    _wait_for(s, True, False, True, expiration)


def _wait_for_writable(s, expiration):
    _wait_for(s, False, True, True, expiration)


def _addresses_equal(af, a1, a2):
    # Convert the first value of the tuple, which is a textual format
    # address into binary form, so that we are not confused by different
    # textual representations of the same address
    try:
        n1 = dns.inet.inet_pton(af, a1[0])
        n2 = dns.inet.inet_pton(af, a2[0])
    except dns.exception.SyntaxError:
        return False
    return n1 == n2 and a1[1:] == a2[1:]


def _matches_destination(af, from_address, destination, ignore_unexpected):
    # Check that from_address is appropriate for a response to a query
    # sent to destination.
    if not destination:
        return True
    if _addresses_equal(af, from_address, destination) or (
        dns.inet.is_multicast(destination[0]) and from_address[1:] == destination[1:]
    ):
        return True
    elif ignore_unexpected:
        return False
    raise UnexpectedSource(
        f"got a response from {from_address} instead of " f"{destination}"
    )


def _destination_and_source(
    where, port, source, source_port, where_must_be_address=True
):
    # Apply defaults and compute destination and source tuples
    # suitable for use in connect(), sendto(), or bind().
    af = None
    destination = None
    try:
        af = dns.inet.af_for_address(where)
        destination = where
    except Exception:
        if where_must_be_address:
            raise
        # URLs are ok so eat the exception
    if source:
        saf = dns.inet.af_for_address(source)
        if af:
            # We know the destination af, so source had better agree!
            if saf != af:
                raise ValueError(
                    "different address families for source " + "and destination"
                )
        else:
            # We didn't know the destination af, but we know the source,
            # so that's our af.
            af = saf
    if source_port and not source:
        # Caller has specified a source_port but not an address, so we
        # need to return a source, and we need to use the appropriate
        # wildcard address as the address.
        if af == socket.AF_INET:
            source = "0.0.0.0"
        elif af == socket.AF_INET6:
            source = "::"
        else:
            raise ValueError("source_port specified but address family is unknown")
    # Convert high-level (address, port) tuples into low-level address
    # tuples.
    if destination:
        destination = dns.inet.low_level_address_tuple((destination, port), af)
    if source:
        source = dns.inet.low_level_address_tuple((source, source_port), af)
    return (af, destination, source)


def _make_socket(af, type, source, ssl_context=None, server_hostname=None):
    s = socket_factory(af, type)
    try:
        s.setblocking(False)
        if source is not None:
            s.bind(source)
        if ssl_context:
            # LGTM gets a false positive here, as our default context is OK
            return ssl_context.wrap_socket(
                s,
                do_handshake_on_connect=False,  # lgtm[py/insecure-protocol]
                server_hostname=server_hostname,
            )
        else:
            return s
    except Exception:
        s.close()
        raise


def https(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 443,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    session: Optional[Any] = None,
    path: str = "/dns-query",
    post: bool = True,
    bootstrap_address: Optional[str] = None,
    verify: Union[bool, str] = True,
) -> dns.message.Message:
    """Return the response obtained after sending a query via DNS-over-HTTPS.

    *q*, a ``dns.message.Message``, the query to send.

    *where*, a ``str``, the nameserver IP address or the full URL. If an IP address is
    given, the URL will be constructed using the following schema:
    https://<IP-address>:<port>/<path>.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the query
    times out. If ``None``, the default, wait forever.

    *port*, a ``int``, the port to send the query to. The default is 443.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying the source
    address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message. The default is
    0.

    *one_rr_per_rrset*, a ``bool``. If ``True``, put each RR into its own RRset.

    *ignore_trailing*, a ``bool``. If ``True``, ignore trailing junk at end of the
    received message.

    *session*, an ``httpx.Client`` or ``requests.session.Session``.  If provided, the
    client/session to use to send the queries.

    *path*, a ``str``. If *where* is an IP address, then *path* will be used to
    construct the URL to send the DNS query to.

    *post*, a ``bool``. If ``True``, the default, POST method will be used.

    *bootstrap_address*, a ``str``, the IP address to use to bypass the system's DNS
    resolver.

    *verify*, a ``bool`` or ``str``.  If a ``True``, then TLS certificate verification
    of the server is done using the default CA bundle; if ``False``, then no
    verification is done; if a `str` then it specifies the path to a certificate file or
    directory which will be used for verification.

    Returns a ``dns.message.Message``.
    """

    if not have_doh:
        raise NoDOH("Neither httpx nor requests is available.")  # pragma: no cover

    _httpx_ok = _have_httpx

    wire = q.to_wire()
    (af, _, source) = _destination_and_source(where, port, source, source_port, False)
    transport_adapter = None
    transport = None
    headers = {"accept": "application/dns-message"}
    if af is not None:
        if af == socket.AF_INET:
            url = "https://{}:{}{}".format(where, port, path)
        elif af == socket.AF_INET6:
            url = "https://[{}]:{}{}".format(where, port, path)
    elif bootstrap_address is not None:
        _httpx_ok = False
        split_url = urllib.parse.urlsplit(where)
        if split_url.hostname is None:
            raise ValueError("DoH URL has no hostname")
        headers["Host"] = split_url.hostname
        url = where.replace(split_url.hostname, bootstrap_address)
        if _have_requests:
            transport_adapter = HostHeaderSSLAdapter()
    else:
        url = where
    if source is not None:
        # set source port and source address
        if _have_httpx:
            if source_port == 0:
                transport = httpx.HTTPTransport(local_address=source[0], verify=verify)
            else:
                _httpx_ok = False
        if _have_requests:
            transport_adapter = SourceAddressAdapter(source)

    if session:
        if _have_httpx:
            _is_httpx = isinstance(session, httpx.Client)
        else:
            _is_httpx = False
        if _is_httpx and not _httpx_ok:
            raise NoDOH(
                "Session is httpx, but httpx cannot be used for "
                "the requested operation."
            )
    else:
        _is_httpx = _httpx_ok

    if not _httpx_ok and not _have_requests:
        raise NoDOH(
            "Cannot use httpx for this operation, and requests is not available."
        )

    if session:
        cm: contextlib.AbstractContextManager = contextlib.nullcontext(session)
    elif _is_httpx:
        cm = httpx.Client(
            http1=True, http2=_have_http2, verify=verify, transport=transport
        )
    else:
        cm = requests.sessions.Session()
    with cm as session:
        if transport_adapter and not _is_httpx:
            session.mount(url, transport_adapter)

        # see https://tools.ietf.org/html/rfc8484#section-4.1.1 for DoH
        # GET and POST examples
        if post:
            headers.update(
                {
                    "content-type": "application/dns-message",
                    "content-length": str(len(wire)),
                }
            )
            if _is_httpx:
                response = session.post(
                    url, headers=headers, content=wire, timeout=timeout
                )
            else:
                response = session.post(
                    url, headers=headers, data=wire, timeout=timeout, verify=verify
                )
        else:
            wire = base64.urlsafe_b64encode(wire).rstrip(b"=")
            if _is_httpx:
                twire = wire.decode()  # httpx does a repr() if we give it bytes
                response = session.get(
                    url, headers=headers, timeout=timeout, params={"dns": twire}
                )
            else:
                response = session.get(
                    url,
                    headers=headers,
                    timeout=timeout,
                    verify=verify,
                    params={"dns": wire},
                )

    # see https://tools.ietf.org/html/rfc8484#section-4.2.1 for info about DoH
    # status codes
    if response.status_code < 200 or response.status_code > 299:
        raise ValueError(
            "{} responded with status code {}"
            "\nResponse body: {}".format(where, response.status_code, response.content)
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


def _udp_recv(sock, max_size, expiration):
    """Reads a datagram from the socket.
    A Timeout exception will be raised if the operation is not completed
    by the expiration time.
    """
    while True:
        try:
            return sock.recvfrom(max_size)
        except BlockingIOError:
            _wait_for_readable(sock, expiration)


def _udp_send(sock, data, destination, expiration):
    """Sends the specified datagram to destination over the socket.
    A Timeout exception will be raised if the operation is not completed
    by the expiration time.
    """
    while True:
        try:
            if destination:
                return sock.sendto(data, destination)
            else:
                return sock.send(data)
        except BlockingIOError:  # pragma: no cover
            _wait_for_writable(sock, expiration)


def send_udp(
    sock: Any,
    what: Union[dns.message.Message, bytes],
    destination: Any,
    expiration: Optional[float] = None,
) -> Tuple[int, float]:
    """Send a DNS message to the specified UDP socket.

    *sock*, a ``socket``.

    *what*, a ``bytes`` or ``dns.message.Message``, the message to send.

    *destination*, a destination tuple appropriate for the address family
    of the socket, specifying where to send the query.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.

    Returns an ``(int, float)`` tuple of bytes sent and the sent time.
    """

    if isinstance(what, dns.message.Message):
        what = what.to_wire()
    sent_time = time.time()
    n = _udp_send(sock, what, destination, expiration)
    return (n, sent_time)


def receive_udp(
    sock: Any,
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

    *sock*, a ``socket``.

    *destination*, a destination tuple appropriate for the address family
    of the socket, specifying where the message is expected to arrive from.
    When receiving a response, this would be where the associated query was
    sent.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.

    *ignore_unexpected*, a ``bool``.  If ``True``, ignore responses from
    unexpected sources.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *keyring*, a ``dict``, the keyring to use for TSIG.

    *request_mac*, a ``bytes`` or ``None``, the MAC of the request (for TSIG).

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    *raise_on_truncation*, a ``bool``.  If ``True``, raise an exception if
    the TC bit is set.

    Raises if the message is malformed, if network errors occur, of if
    there is a timeout.

    If *destination* is not ``None``, returns a ``(dns.message.Message, float)``
    tuple of the received message and the received time.

    If *destination* is ``None``, returns a
    ``(dns.message.Message, float, tuple)``
    tuple of the received message, the received time, and the address where
    the message arrived from.
    """

    wire = b""
    while True:
        (wire, from_address) = _udp_recv(sock, 65535, expiration)
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
    if destination:
        return (r, received_time)
    else:
        return (r, received_time, from_address)


def udp(
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
    sock: Optional[Any] = None,
) -> dns.message.Message:
    """Return the response obtained after sending a query via UDP.

    *q*, a ``dns.message.Message``, the query to send

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the
    query times out.  If ``None``, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *ignore_unexpected*, a ``bool``.  If ``True``, ignore responses from
    unexpected sources.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    *raise_on_truncation*, a ``bool``.  If ``True``, raise an exception if
    the TC bit is set.

    *sock*, a ``socket.socket``, or ``None``, the socket to use for the
    query.  If ``None``, the default, a socket is created.  Note that
    if a socket is provided, it must be a nonblocking datagram socket,
    and the *source* and *source_port* are ignored.

    Returns a ``dns.message.Message``.
    """

    wire = q.to_wire()
    (af, destination, source) = _destination_and_source(
        where, port, source, source_port
    )
    (begin_time, expiration) = _compute_times(timeout)
    if sock:
        cm: contextlib.AbstractContextManager = contextlib.nullcontext(sock)
    else:
        cm = _make_socket(af, socket.SOCK_DGRAM, source)
    with cm as s:
        send_udp(s, wire, destination, expiration)
        (r, received_time) = receive_udp(
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
    assert (
        False  # help mypy figure out we can't get here  lgtm[py/unreachable-statement]
    )


def udp_with_fallback(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 53,
    source: Optional[str] = None,
    source_port: int = 0,
    ignore_unexpected: bool = False,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    udp_sock: Optional[Any] = None,
    tcp_sock: Optional[Any] = None,
) -> Tuple[dns.message.Message, bool]:
    """Return the response to the query, trying UDP first and falling back
    to TCP if UDP results in a truncated response.

    *q*, a ``dns.message.Message``, the query to send

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the
    query times out.  If ``None``, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *ignore_unexpected*, a ``bool``.  If ``True``, ignore responses from
    unexpected sources.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    *udp_sock*, a ``socket.socket``, or ``None``, the socket to use for the
    UDP query.  If ``None``, the default, a socket is created.  Note that
    if a socket is provided, it must be a nonblocking datagram socket,
    and the *source* and *source_port* are ignored for the UDP query.

    *tcp_sock*, a ``socket.socket``, or ``None``, the connected socket to use for the
    TCP query.  If ``None``, the default, a socket is created.  Note that
    if a socket is provided, it must be a nonblocking connected stream
    socket, and *where*, *source* and *source_port* are ignored for the TCP
    query.

    Returns a (``dns.message.Message``, tcp) tuple where tcp is ``True``
    if and only if TCP was used.
    """
    try:
        response = udp(
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
        )
        return (response, False)
    except dns.message.Truncated:
        response = tcp(
            q,
            where,
            timeout,
            port,
            source,
            source_port,
            one_rr_per_rrset,
            ignore_trailing,
            tcp_sock,
        )
        return (response, True)


def _net_read(sock, count, expiration):
    """Read the specified number of bytes from sock.  Keep trying until we
    either get the desired amount, or we hit EOF.
    A Timeout exception will be raised if the operation is not completed
    by the expiration time.
    """
    s = b""
    while count > 0:
        try:
            n = sock.recv(count)
            if n == b"":
                raise EOFError
            count -= len(n)
            s += n
        except (BlockingIOError, ssl.SSLWantReadError):
            _wait_for_readable(sock, expiration)
        except ssl.SSLWantWriteError:  # pragma: no cover
            _wait_for_writable(sock, expiration)
    return s


def _net_write(sock, data, expiration):
    """Write the specified data to the socket.
    A Timeout exception will be raised if the operation is not completed
    by the expiration time.
    """
    current = 0
    l = len(data)
    while current < l:
        try:
            current += sock.send(data[current:])
        except (BlockingIOError, ssl.SSLWantWriteError):
            _wait_for_writable(sock, expiration)
        except ssl.SSLWantReadError:  # pragma: no cover
            _wait_for_readable(sock, expiration)


def send_tcp(
    sock: Any,
    what: Union[dns.message.Message, bytes],
    expiration: Optional[float] = None,
) -> Tuple[int, float]:
    """Send a DNS message to the specified TCP socket.

    *sock*, a ``socket``.

    *what*, a ``bytes`` or ``dns.message.Message``, the message to send.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.

    Returns an ``(int, float)`` tuple of bytes sent and the sent time.
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
    _net_write(sock, tcpmsg, expiration)
    return (len(tcpmsg), sent_time)


def receive_tcp(
    sock: Any,
    expiration: Optional[float] = None,
    one_rr_per_rrset: bool = False,
    keyring: Optional[Dict[dns.name.Name, dns.tsig.Key]] = None,
    request_mac: Optional[bytes] = b"",
    ignore_trailing: bool = False,
) -> Tuple[dns.message.Message, float]:
    """Read a DNS message from a TCP socket.

    *sock*, a ``socket``.

    *expiration*, a ``float`` or ``None``, the absolute time at which
    a timeout exception should be raised.  If ``None``, no timeout will
    occur.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *keyring*, a ``dict``, the keyring to use for TSIG.

    *request_mac*, a ``bytes`` or ``None``, the MAC of the request (for TSIG).

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    Raises if the message is malformed, if network errors occur, of if
    there is a timeout.

    Returns a ``(dns.message.Message, float)`` tuple of the received message
    and the received time.
    """

    ldata = _net_read(sock, 2, expiration)
    (l,) = struct.unpack("!H", ldata)
    wire = _net_read(sock, l, expiration)
    received_time = time.time()
    r = dns.message.from_wire(
        wire,
        keyring=keyring,
        request_mac=request_mac,
        one_rr_per_rrset=one_rr_per_rrset,
        ignore_trailing=ignore_trailing,
    )
    return (r, received_time)


def _connect(s, address, expiration):
    err = s.connect_ex(address)
    if err == 0:
        return
    if err in (errno.EINPROGRESS, errno.EWOULDBLOCK, errno.EALREADY):
        _wait_for_writable(s, expiration)
        err = s.getsockopt(socket.SOL_SOCKET, socket.SO_ERROR)
    if err != 0:
        raise OSError(err, os.strerror(err))


def tcp(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 53,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    sock: Optional[Any] = None,
) -> dns.message.Message:
    """Return the response obtained after sending a query via TCP.

    *q*, a ``dns.message.Message``, the query to send

    *where*, a ``str`` containing an IPv4 or IPv6 address, where
    to send the message.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the
    query times out.  If ``None``, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    *sock*, a ``socket.socket``, or ``None``, the connected socket to use for the
    query.  If ``None``, the default, a socket is created.  Note that
    if a socket is provided, it must be a nonblocking connected stream
    socket, and *where*, *port*, *source* and *source_port* are ignored.

    Returns a ``dns.message.Message``.
    """

    wire = q.to_wire()
    (begin_time, expiration) = _compute_times(timeout)
    if sock:
        cm: contextlib.AbstractContextManager = contextlib.nullcontext(sock)
    else:
        (af, destination, source) = _destination_and_source(
            where, port, source, source_port
        )
        cm = _make_socket(af, socket.SOCK_STREAM, source)
    with cm as s:
        if not sock:
            _connect(s, destination, expiration)
        send_tcp(s, wire, expiration)
        (r, received_time) = receive_tcp(
            s, expiration, one_rr_per_rrset, q.keyring, q.mac, ignore_trailing
        )
        r.time = received_time - begin_time
        if not q.is_response(r):
            raise BadResponse
        return r
    assert (
        False  # help mypy figure out we can't get here  lgtm[py/unreachable-statement]
    )


def _tls_handshake(s, expiration):
    while True:
        try:
            s.do_handshake()
            return
        except ssl.SSLWantReadError:
            _wait_for_readable(s, expiration)
        except ssl.SSLWantWriteError:  # pragma: no cover
            _wait_for_writable(s, expiration)


def tls(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 853,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    sock: Optional[ssl.SSLSocket] = None,
    ssl_context: Optional[ssl.SSLContext] = None,
    server_hostname: Optional[str] = None,
) -> dns.message.Message:
    """Return the response obtained after sending a query via TLS.

    *q*, a ``dns.message.Message``, the query to send

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the
    query times out.  If ``None``, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 853.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *one_rr_per_rrset*, a ``bool``.  If ``True``, put each RR into its own
    RRset.

    *ignore_trailing*, a ``bool``.  If ``True``, ignore trailing
    junk at end of the received message.

    *sock*, an ``ssl.SSLSocket``, or ``None``, the socket to use for
    the query.  If ``None``, the default, a socket is created.  Note
    that if a socket is provided, it must be a nonblocking connected
    SSL stream socket, and *where*, *port*, *source*, *source_port*,
    and *ssl_context* are ignored.

    *ssl_context*, an ``ssl.SSLContext``, the context to use when establishing
    a TLS connection. If ``None``, the default, creates one with the default
    configuration.

    *server_hostname*, a ``str`` containing the server's hostname.  The
    default is ``None``, which means that no hostname is known, and if an
    SSL context is created, hostname checking will be disabled.

    Returns a ``dns.message.Message``.

    """

    if sock:
        #
        # If a socket was provided, there's no special TLS handling needed.
        #
        return tcp(
            q,
            where,
            timeout,
            port,
            source,
            source_port,
            one_rr_per_rrset,
            ignore_trailing,
            sock,
        )

    wire = q.to_wire()
    (begin_time, expiration) = _compute_times(timeout)
    (af, destination, source) = _destination_and_source(
        where, port, source, source_port
    )
    if ssl_context is None and not sock:
        ssl_context = ssl.create_default_context()
        ssl_context.minimum_version = ssl.TLSVersion.TLSv1_2
        if server_hostname is None:
            ssl_context.check_hostname = False

    with _make_socket(
        af,
        socket.SOCK_STREAM,
        source,
        ssl_context=ssl_context,
        server_hostname=server_hostname,
    ) as s:
        _connect(s, destination, expiration)
        _tls_handshake(s, expiration)
        send_tcp(s, wire, expiration)
        (r, received_time) = receive_tcp(
            s, expiration, one_rr_per_rrset, q.keyring, q.mac, ignore_trailing
        )
        r.time = received_time - begin_time
        if not q.is_response(r):
            raise BadResponse
        return r
    assert (
        False  # help mypy figure out we can't get here  lgtm[py/unreachable-statement]
    )


def quic(
    q: dns.message.Message,
    where: str,
    timeout: Optional[float] = None,
    port: int = 853,
    source: Optional[str] = None,
    source_port: int = 0,
    one_rr_per_rrset: bool = False,
    ignore_trailing: bool = False,
    connection: Optional[dns.quic.SyncQuicConnection] = None,
    verify: Union[bool, str] = True,
) -> dns.message.Message:
    """Return the response obtained after sending a query via DNS-over-QUIC.

    *q*, a ``dns.message.Message``, the query to send.

    *where*, a ``str``, the nameserver IP address.

    *timeout*, a ``float`` or ``None``, the number of seconds to wait before the query
    times out. If ``None``, the default, wait forever.

    *port*, a ``int``, the port to send the query to. The default is 853.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying the source
    address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message. The default is
    0.

    *one_rr_per_rrset*, a ``bool``. If ``True``, put each RR into its own RRset.

    *ignore_trailing*, a ``bool``. If ``True``, ignore trailing junk at end of the
    received message.

    *connection*, a ``dns.quic.SyncQuicConnection``.  If provided, the
    connection to use to send the query.

    *verify*, a ``bool`` or ``str``.  If a ``True``, then TLS certificate verification
    of the server is done using the default CA bundle; if ``False``, then no
    verification is done; if a `str` then it specifies the path to a certificate file or
    directory which will be used for verification.

    Returns a ``dns.message.Message``.
    """

    if not dns.quic.have_quic:
        raise NoDOQ("DNS-over-QUIC is not available.")  # pragma: no cover

    q.id = 0
    wire = q.to_wire()
    the_connection: dns.quic.SyncQuicConnection
    the_manager: dns.quic.SyncQuicManager
    if connection:
        manager: contextlib.AbstractContextManager = contextlib.nullcontext(None)
        the_connection = connection
    else:
        manager = dns.quic.SyncQuicManager(verify_mode=verify)
        the_manager = manager  # for type checking happiness

    with manager:
        if not connection:
            the_connection = the_manager.connect(where, port, source, source_port)
        start = time.time()
        with the_connection.make_stream() as stream:
            stream.send(wire, True)
            wire = stream.receive(timeout)
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


def xfr(
    where: str,
    zone: Union[dns.name.Name, str],
    rdtype: Union[dns.rdatatype.RdataType, str] = dns.rdatatype.AXFR,
    rdclass: Union[dns.rdataclass.RdataClass, str] = dns.rdataclass.IN,
    timeout: Optional[float] = None,
    port: int = 53,
    keyring: Optional[Dict[dns.name.Name, dns.tsig.Key]] = None,
    keyname: Optional[Union[dns.name.Name, str]] = None,
    relativize: bool = True,
    lifetime: Optional[float] = None,
    source: Optional[str] = None,
    source_port: int = 0,
    serial: int = 0,
    use_udp: bool = False,
    keyalgorithm: Union[dns.name.Name, str] = dns.tsig.default_algorithm,
) -> Any:
    """Return a generator for the responses to a zone transfer.

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *zone*, a ``dns.name.Name`` or ``str``, the name of the zone to transfer.

    *rdtype*, an ``int`` or ``str``, the type of zone transfer.  The
    default is ``dns.rdatatype.AXFR``.  ``dns.rdatatype.IXFR`` can be
    used to do an incremental transfer instead.

    *rdclass*, an ``int`` or ``str``, the class of the zone transfer.
    The default is ``dns.rdataclass.IN``.

    *timeout*, a ``float``, the number of seconds to wait for each
    response message.  If None, the default, wait forever.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *keyring*, a ``dict``, the keyring to use for TSIG.

    *keyname*, a ``dns.name.Name`` or ``str``, the name of the TSIG
    key to use.

    *relativize*, a ``bool``.  If ``True``, all names in the zone will be
    relativized to the zone origin.  It is essential that the
    relativize setting matches the one specified to
    ``dns.zone.from_xfr()`` if using this generator to make a zone.

    *lifetime*, a ``float``, the total number of seconds to spend
    doing the transfer.  If ``None``, the default, then there is no
    limit on the time the transfer may take.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *serial*, an ``int``, the SOA serial number to use as the base for
    an IXFR diff sequence (only meaningful if *rdtype* is
    ``dns.rdatatype.IXFR``).

    *use_udp*, a ``bool``.  If ``True``, use UDP (only meaningful for IXFR).

    *keyalgorithm*, a ``dns.name.Name`` or ``str``, the TSIG algorithm to use.

    Raises on errors, and so does the generator.

    Returns a generator of ``dns.message.Message`` objects.
    """

    if isinstance(zone, str):
        zone = dns.name.from_text(zone)
    rdtype = dns.rdatatype.RdataType.make(rdtype)
    q = dns.message.make_query(zone, rdtype, rdclass)
    if rdtype == dns.rdatatype.IXFR:
        rrset = dns.rrset.from_text(zone, 0, "IN", "SOA", ". . %u 0 0 0 0" % serial)
        q.authority.append(rrset)
    if keyring is not None:
        q.use_tsig(keyring, keyname, algorithm=keyalgorithm)
    wire = q.to_wire()
    (af, destination, source) = _destination_and_source(
        where, port, source, source_port
    )
    if use_udp and rdtype != dns.rdatatype.IXFR:
        raise ValueError("cannot do a UDP AXFR")
    sock_type = socket.SOCK_DGRAM if use_udp else socket.SOCK_STREAM
    with _make_socket(af, sock_type, source) as s:
        (_, expiration) = _compute_times(lifetime)
        _connect(s, destination, expiration)
        l = len(wire)
        if use_udp:
            _udp_send(s, wire, None, expiration)
        else:
            tcpmsg = struct.pack("!H", l) + wire
            _net_write(s, tcpmsg, expiration)
        done = False
        delete_mode = True
        expecting_SOA = False
        soa_rrset = None
        if relativize:
            origin = zone
            oname = dns.name.empty
        else:
            origin = None
            oname = zone
        tsig_ctx = None
        while not done:
            (_, mexpiration) = _compute_times(timeout)
            if mexpiration is None or (
                expiration is not None and mexpiration > expiration
            ):
                mexpiration = expiration
            if use_udp:
                (wire, _) = _udp_recv(s, 65535, mexpiration)
            else:
                ldata = _net_read(s, 2, mexpiration)
                (l,) = struct.unpack("!H", ldata)
                wire = _net_read(s, l, mexpiration)
            is_ixfr = rdtype == dns.rdatatype.IXFR
            r = dns.message.from_wire(
                wire,
                keyring=q.keyring,
                request_mac=q.mac,
                xfr=True,
                origin=origin,
                tsig_ctx=tsig_ctx,
                multi=True,
                one_rr_per_rrset=is_ixfr,
            )
            rcode = r.rcode()
            if rcode != dns.rcode.NOERROR:
                raise TransferError(rcode)
            tsig_ctx = r.tsig_ctx
            answer_index = 0
            if soa_rrset is None:
                if not r.answer or r.answer[0].name != oname:
                    raise dns.exception.FormError("No answer or RRset not for qname")
                rrset = r.answer[0]
                if rrset.rdtype != dns.rdatatype.SOA:
                    raise dns.exception.FormError("first RRset is not an SOA")
                answer_index = 1
                soa_rrset = rrset.copy()
                if rdtype == dns.rdatatype.IXFR:
                    if dns.serial.Serial(soa_rrset[0].serial) <= serial:
                        #
                        # We're already up-to-date.
                        #
                        done = True
                    else:
                        expecting_SOA = True
            #
            # Process SOAs in the answer section (other than the initial
            # SOA in the first message).
            #
            for rrset in r.answer[answer_index:]:
                if done:
                    raise dns.exception.FormError("answers after final SOA")
                if rrset.rdtype == dns.rdatatype.SOA and rrset.name == oname:
                    if expecting_SOA:
                        if rrset[0].serial != serial:
                            raise dns.exception.FormError("IXFR base serial mismatch")
                        expecting_SOA = False
                    elif rdtype == dns.rdatatype.IXFR:
                        delete_mode = not delete_mode
                    #
                    # If this SOA RRset is equal to the first we saw then we're
                    # finished. If this is an IXFR we also check that we're
                    # seeing the record in the expected part of the response.
                    #
                    if rrset == soa_rrset and (
                        rdtype == dns.rdatatype.AXFR
                        or (rdtype == dns.rdatatype.IXFR and delete_mode)
                    ):
                        done = True
                elif expecting_SOA:
                    #
                    # We made an IXFR request and are expecting another
                    # SOA RR, but saw something else, so this must be an
                    # AXFR response.
                    #
                    rdtype = dns.rdatatype.AXFR
                    expecting_SOA = False
            if done and q.keyring and not r.had_tsig:
                raise dns.exception.FormError("missing TSIG")
            yield r


class UDPMode(enum.IntEnum):
    """How should UDP be used in an IXFR from :py:func:`inbound_xfr()`?

    NEVER means "never use UDP; always use TCP"
    TRY_FIRST means "try to use UDP but fall back to TCP if needed"
    ONLY means "raise ``dns.xfr.UseTCP`` if trying UDP does not succeed"
    """

    NEVER = 0
    TRY_FIRST = 1
    ONLY = 2


def inbound_xfr(
    where: str,
    txn_manager: dns.transaction.TransactionManager,
    query: Optional[dns.message.Message] = None,
    port: int = 53,
    timeout: Optional[float] = None,
    lifetime: Optional[float] = None,
    source: Optional[str] = None,
    source_port: int = 0,
    udp_mode: UDPMode = UDPMode.NEVER,
) -> None:
    """Conduct an inbound transfer and apply it via a transaction from the
    txn_manager.

    *where*, a ``str`` containing an IPv4 or IPv6 address,  where
    to send the message.

    *txn_manager*, a ``dns.transaction.TransactionManager``, the txn_manager
    for this transfer (typically a ``dns.zone.Zone``).

    *query*, the query to send.  If not supplied, a default query is
    constructed using information from the *txn_manager*.

    *port*, an ``int``, the port send the message to.  The default is 53.

    *timeout*, a ``float``, the number of seconds to wait for each
    response message.  If None, the default, wait forever.

    *lifetime*, a ``float``, the total number of seconds to spend
    doing the transfer.  If ``None``, the default, then there is no
    limit on the time the transfer may take.

    *source*, a ``str`` containing an IPv4 or IPv6 address, specifying
    the source address.  The default is the wildcard address.

    *source_port*, an ``int``, the port from which to send the message.
    The default is 0.

    *udp_mode*, a ``dns.query.UDPMode``, determines how UDP is used
    for IXFRs.  The default is ``dns.UDPMode.NEVER``, i.e. only use
    TCP.  Other possibilities are ``dns.UDPMode.TRY_FIRST``, which
    means "try UDP but fallback to TCP if needed", and
    ``dns.UDPMode.ONLY``, which means "try UDP and raise
    ``dns.xfr.UseTCP`` if it does not succeed.

    Raises on errors.
    """
    if query is None:
        (query, serial) = dns.xfr.make_query(txn_manager)
    else:
        serial = dns.xfr.extract_serial_from_query(query)
    rdtype = query.question[0].rdtype
    is_ixfr = rdtype == dns.rdatatype.IXFR
    origin = txn_manager.from_wire_origin()
    wire = query.to_wire()
    (af, destination, source) = _destination_and_source(
        where, port, source, source_port
    )
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
        with _make_socket(af, sock_type, source) as s:
            _connect(s, destination, expiration)
            if is_udp:
                _udp_send(s, wire, None, expiration)
            else:
                tcpmsg = struct.pack("!H", len(wire)) + wire
                _net_write(s, tcpmsg, expiration)
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
                        (rwire, _) = _udp_recv(s, 65535, mexpiration)
                    else:
                        ldata = _net_read(s, 2, mexpiration)
                        (l,) = struct.unpack("!H", ldata)
                        rwire = _net_read(s, l, mexpiration)
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
