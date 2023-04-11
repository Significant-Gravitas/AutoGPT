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

"""DNS TTL conversion."""

from typing import Union

import dns.exception

# Technically TTLs are supposed to be between 0 and 2**31 - 1, with values
# greater than that interpreted as 0, but we do not impose this policy here
# as values > 2**31 - 1 occur in real world data.
#
# We leave it to applications to impose tighter bounds if desired.
MAX_TTL = 2**32 - 1


class BadTTL(dns.exception.SyntaxError):
    """DNS TTL value is not well-formed."""


def from_text(text: str) -> int:
    """Convert the text form of a TTL to an integer.

    The BIND 8 units syntax for TTLs (e.g. '1w6d4h3m10s') is supported.

    *text*, a ``str``, the textual TTL.

    Raises ``dns.ttl.BadTTL`` if the TTL is not well-formed.

    Returns an ``int``.
    """

    if text.isdigit():
        total = int(text)
    elif len(text) == 0:
        raise BadTTL
    else:
        total = 0
        current = 0
        need_digit = True
        for c in text:
            if c.isdigit():
                current *= 10
                current += int(c)
                need_digit = False
            else:
                if need_digit:
                    raise BadTTL
                c = c.lower()
                if c == "w":
                    total += current * 604800
                elif c == "d":
                    total += current * 86400
                elif c == "h":
                    total += current * 3600
                elif c == "m":
                    total += current * 60
                elif c == "s":
                    total += current
                else:
                    raise BadTTL("unknown unit '%s'" % c)
                current = 0
                need_digit = True
        if not current == 0:
            raise BadTTL("trailing integer")
    if total < 0 or total > MAX_TTL:
        raise BadTTL("TTL should be between 0 and 2**32 - 1 (inclusive)")
    return total


def make(value: Union[int, str]) -> int:
    if isinstance(value, int):
        return value
    elif isinstance(value, str):
        return dns.ttl.from_text(value)
    else:
        raise ValueError("cannot convert value to TTL")
