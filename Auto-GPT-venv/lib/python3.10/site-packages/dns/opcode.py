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

"""DNS Opcodes."""

import dns.enum
import dns.exception


class Opcode(dns.enum.IntEnum):
    #: Query
    QUERY = 0
    #: Inverse Query (historical)
    IQUERY = 1
    #: Server Status (unspecified and unimplemented anywhere)
    STATUS = 2
    #: Notify
    NOTIFY = 4
    #: Dynamic Update
    UPDATE = 5

    @classmethod
    def _maximum(cls):
        return 15

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownOpcode


class UnknownOpcode(dns.exception.DNSException):
    """An DNS opcode is unknown."""


def from_text(text: str) -> Opcode:
    """Convert text into an opcode.

    *text*, a ``str``, the textual opcode

    Raises ``dns.opcode.UnknownOpcode`` if the opcode is unknown.

    Returns an ``int``.
    """

    return Opcode.from_text(text)


def from_flags(flags: int) -> Opcode:
    """Extract an opcode from DNS message flags.

    *flags*, an ``int``, the DNS flags.

    Returns an ``int``.
    """

    return Opcode((flags & 0x7800) >> 11)


def to_flags(value: Opcode) -> int:
    """Convert an opcode to a value suitable for ORing into DNS message
    flags.

    *value*, an ``int``, the DNS opcode value.

    Returns an ``int``.
    """

    return (value << 11) & 0x7800


def to_text(value: Opcode) -> str:
    """Convert an opcode to text.

    *value*, an ``int`` the opcode value,

    Raises ``dns.opcode.UnknownOpcode`` if the opcode is unknown.

    Returns a ``str``.
    """

    return Opcode.to_text(value)


def is_update(flags: int) -> bool:
    """Is the opcode in flags UPDATE?

    *flags*, an ``int``, the DNS message flags.

    Returns a ``bool``.
    """

    return from_flags(flags) == Opcode.UPDATE


### BEGIN generated Opcode constants

QUERY = Opcode.QUERY
IQUERY = Opcode.IQUERY
STATUS = Opcode.STATUS
NOTIFY = Opcode.NOTIFY
UPDATE = Opcode.UPDATE

### END generated Opcode constants
