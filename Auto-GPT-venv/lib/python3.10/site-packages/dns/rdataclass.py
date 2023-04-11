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

"""DNS Rdata Classes."""

import dns.enum
import dns.exception


class RdataClass(dns.enum.IntEnum):
    """DNS Rdata Class"""

    RESERVED0 = 0
    IN = 1
    INTERNET = IN
    CH = 3
    CHAOS = CH
    HS = 4
    HESIOD = HS
    NONE = 254
    ANY = 255

    @classmethod
    def _maximum(cls):
        return 65535

    @classmethod
    def _short_name(cls):
        return "class"

    @classmethod
    def _prefix(cls):
        return "CLASS"

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownRdataclass


_metaclasses = {RdataClass.NONE, RdataClass.ANY}


class UnknownRdataclass(dns.exception.DNSException):
    """A DNS class is unknown."""


def from_text(text: str) -> RdataClass:
    """Convert text into a DNS rdata class value.

    The input text can be a defined DNS RR class mnemonic or
    instance of the DNS generic class syntax.

    For example, "IN" and "CLASS1" will both result in a value of 1.

    Raises ``dns.rdatatype.UnknownRdataclass`` if the class is unknown.

    Raises ``ValueError`` if the rdata class value is not >= 0 and <= 65535.

    Returns a ``dns.rdataclass.RdataClass``.
    """

    return RdataClass.from_text(text)


def to_text(value: RdataClass) -> str:
    """Convert a DNS rdata class value to text.

    If the value has a known mnemonic, it will be used, otherwise the
    DNS generic class syntax will be used.

    Raises ``ValueError`` if the rdata class value is not >= 0 and <= 65535.

    Returns a ``str``.
    """

    return RdataClass.to_text(value)


def is_metaclass(rdclass: RdataClass) -> bool:
    """True if the specified class is a metaclass.

    The currently defined metaclasses are ANY and NONE.

    *rdclass* is a ``dns.rdataclass.RdataClass``.
    """

    if rdclass in _metaclasses:
        return True
    return False


### BEGIN generated RdataClass constants

RESERVED0 = RdataClass.RESERVED0
IN = RdataClass.IN
INTERNET = RdataClass.INTERNET
CH = RdataClass.CH
CHAOS = RdataClass.CHAOS
HS = RdataClass.HS
HESIOD = RdataClass.HESIOD
NONE = RdataClass.NONE
ANY = RdataClass.ANY

### END generated RdataClass constants
