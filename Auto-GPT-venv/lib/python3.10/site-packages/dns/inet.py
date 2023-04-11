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

"""Generic Internet address helper functions."""

from typing import Any, Optional, Tuple

import socket

import dns.ipv4
import dns.ipv6


# We assume that AF_INET and AF_INET6 are always defined.  We keep
# these here for the benefit of any old code (unlikely though that
# is!).
AF_INET = socket.AF_INET
AF_INET6 = socket.AF_INET6


def inet_pton(family: int, text: str) -> bytes:
    """Convert the textual form of a network address into its binary form.

    *family* is an ``int``, the address family.

    *text* is a ``str``, the textual address.

    Raises ``NotImplementedError`` if the address family specified is not
    implemented.

    Returns a ``bytes``.
    """

    if family == AF_INET:
        return dns.ipv4.inet_aton(text)
    elif family == AF_INET6:
        return dns.ipv6.inet_aton(text, True)
    else:
        raise NotImplementedError


def inet_ntop(family: int, address: bytes) -> str:
    """Convert the binary form of a network address into its textual form.

    *family* is an ``int``, the address family.

    *address* is a ``bytes``, the network address in binary form.

    Raises ``NotImplementedError`` if the address family specified is not
    implemented.

    Returns a ``str``.
    """

    if family == AF_INET:
        return dns.ipv4.inet_ntoa(address)
    elif family == AF_INET6:
        return dns.ipv6.inet_ntoa(address)
    else:
        raise NotImplementedError


def af_for_address(text: str) -> int:
    """Determine the address family of a textual-form network address.

    *text*, a ``str``, the textual address.

    Raises ``ValueError`` if the address family cannot be determined
    from the input.

    Returns an ``int``.
    """

    try:
        dns.ipv4.inet_aton(text)
        return AF_INET
    except Exception:
        try:
            dns.ipv6.inet_aton(text, True)
            return AF_INET6
        except Exception:
            raise ValueError


def is_multicast(text: str) -> bool:
    """Is the textual-form network address a multicast address?

    *text*, a ``str``, the textual address.

    Raises ``ValueError`` if the address family cannot be determined
    from the input.

    Returns a ``bool``.
    """

    try:
        first = dns.ipv4.inet_aton(text)[0]
        return first >= 224 and first <= 239
    except Exception:
        try:
            first = dns.ipv6.inet_aton(text, True)[0]
            return first == 255
        except Exception:
            raise ValueError


def is_address(text: str) -> bool:
    """Is the specified string an IPv4 or IPv6 address?

    *text*, a ``str``, the textual address.

    Returns a ``bool``.
    """

    try:
        dns.ipv4.inet_aton(text)
        return True
    except Exception:
        try:
            dns.ipv6.inet_aton(text, True)
            return True
        except Exception:
            return False


def low_level_address_tuple(
    high_tuple: Tuple[str, int], af: Optional[int] = None
) -> Any:
    """Given a "high-level" address tuple, i.e.
    an (address, port) return the appropriate "low-level" address tuple
    suitable for use in socket calls.

    If an *af* other than ``None`` is provided, it is assumed the
    address in the high-level tuple is valid and has that af.  If af
    is ``None``, then af_for_address will be called.
    """
    address, port = high_tuple
    if af is None:
        af = af_for_address(address)
    if af == AF_INET:
        return (address, port)
    elif af == AF_INET6:
        i = address.find("%")
        if i < 0:
            # no scope, shortcut!
            return (address, port, 0, 0)
        # try to avoid getaddrinfo()
        addrpart = address[:i]
        scope = address[i + 1 :]
        if scope.isdigit():
            return (addrpart, port, 0, int(scope))
        try:
            return (addrpart, port, 0, socket.if_nametoindex(scope))
        except AttributeError:  # pragma: no cover  (we can't really test this)
            ai_flags = socket.AI_NUMERICHOST
            ((*_, tup), *_) = socket.getaddrinfo(address, port, flags=ai_flags)
            return tup
    else:
        raise NotImplementedError(f"unknown address family {af}")
