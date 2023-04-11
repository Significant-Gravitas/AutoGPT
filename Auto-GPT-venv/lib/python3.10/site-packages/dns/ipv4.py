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

"""IPv4 helper functions."""

from typing import Union

import struct

import dns.exception


def inet_ntoa(address: bytes) -> str:
    """Convert an IPv4 address in binary form to text form.

    *address*, a ``bytes``, the IPv4 address in binary form.

    Returns a ``str``.
    """

    if len(address) != 4:
        raise dns.exception.SyntaxError
    return "%u.%u.%u.%u" % (address[0], address[1], address[2], address[3])


def inet_aton(text: Union[str, bytes]) -> bytes:
    """Convert an IPv4 address in text form to binary form.

    *text*, a ``str`` or ``bytes``, the IPv4 address in textual form.

    Returns a ``bytes``.
    """

    if not isinstance(text, bytes):
        btext = text.encode()
    else:
        btext = text
    parts = btext.split(b".")
    if len(parts) != 4:
        raise dns.exception.SyntaxError
    for part in parts:
        if not part.isdigit():
            raise dns.exception.SyntaxError
        if len(part) > 1 and part[0] == ord("0"):
            # No leading zeros
            raise dns.exception.SyntaxError
    try:
        b = [int(part) for part in parts]
        return struct.pack("BBBB", *b)
    except Exception:
        raise dns.exception.SyntaxError
