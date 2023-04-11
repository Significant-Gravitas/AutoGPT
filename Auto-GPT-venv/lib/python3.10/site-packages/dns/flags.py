# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2001-2017 Nominum, Inc.
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

"""DNS Message Flags."""

from typing import Any

import enum

# Standard DNS flags


class Flag(enum.IntFlag):
    #: Query Response
    QR = 0x8000
    #: Authoritative Answer
    AA = 0x0400
    #: Truncated Response
    TC = 0x0200
    #: Recursion Desired
    RD = 0x0100
    #: Recursion Available
    RA = 0x0080
    #: Authentic Data
    AD = 0x0020
    #: Checking Disabled
    CD = 0x0010


# EDNS flags


class EDNSFlag(enum.IntFlag):
    #: DNSSEC answer OK
    DO = 0x8000


def _from_text(text: str, enum_class: Any) -> int:
    flags = 0
    tokens = text.split()
    for t in tokens:
        flags |= enum_class[t.upper()]
    return flags


def _to_text(flags: int, enum_class: Any) -> str:
    text_flags = []
    for k, v in enum_class.__members__.items():
        if flags & v != 0:
            text_flags.append(k)
    return " ".join(text_flags)


def from_text(text: str) -> int:
    """Convert a space-separated list of flag text values into a flags
    value.

    Returns an ``int``
    """

    return _from_text(text, Flag)


def to_text(flags: int) -> str:
    """Convert a flags value into a space-separated list of flag text
    values.

    Returns a ``str``.
    """

    return _to_text(flags, Flag)


def edns_from_text(text: str) -> int:
    """Convert a space-separated list of EDNS flag text values into a EDNS
    flags value.

    Returns an ``int``
    """

    return _from_text(text, EDNSFlag)


def edns_to_text(flags: int) -> str:
    """Convert an EDNS flags value into a space-separated list of EDNS flag
    text values.

    Returns a ``str``.
    """

    return _to_text(flags, EDNSFlag)


### BEGIN generated Flag constants

QR = Flag.QR
AA = Flag.AA
TC = Flag.TC
RD = Flag.RD
RA = Flag.RA
AD = Flag.AD
CD = Flag.CD

### END generated Flag constants

### BEGIN generated EDNSFlag constants

DO = EDNSFlag.DO

### END generated EDNSFlag constants
