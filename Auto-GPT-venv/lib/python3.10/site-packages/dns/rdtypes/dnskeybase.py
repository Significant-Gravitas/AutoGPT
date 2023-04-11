# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2004-2007, 2009-2011 Nominum, Inc.
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

import base64
import enum
import struct

import dns.exception
import dns.immutable
import dns.dnssectypes
import dns.rdata

# wildcard import
__all__ = ["SEP", "REVOKE", "ZONE"]  # noqa: F822


class Flag(enum.IntFlag):
    SEP = 0x0001
    REVOKE = 0x0080
    ZONE = 0x0100


@dns.immutable.immutable
class DNSKEYBase(dns.rdata.Rdata):

    """Base class for rdata that is like a DNSKEY record"""

    __slots__ = ["flags", "protocol", "algorithm", "key"]

    def __init__(self, rdclass, rdtype, flags, protocol, algorithm, key):
        super().__init__(rdclass, rdtype)
        self.flags = self._as_uint16(flags)
        self.protocol = self._as_uint8(protocol)
        self.algorithm = dns.dnssectypes.Algorithm.make(algorithm)
        self.key = self._as_bytes(key)

    def to_text(self, origin=None, relativize=True, **kw):
        return "%d %d %d %s" % (
            self.flags,
            self.protocol,
            self.algorithm,
            dns.rdata._base64ify(self.key, **kw),
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        flags = tok.get_uint16()
        protocol = tok.get_uint8()
        algorithm = tok.get_string()
        b64 = tok.concatenate_remaining_identifiers().encode()
        key = base64.b64decode(b64)
        return cls(rdclass, rdtype, flags, protocol, algorithm, key)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        header = struct.pack("!HBB", self.flags, self.protocol, self.algorithm)
        file.write(header)
        file.write(self.key)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        header = parser.get_struct("!HBB")
        key = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], header[2], key)


### BEGIN generated Flag constants

SEP = Flag.SEP
REVOKE = Flag.REVOKE
ZONE = Flag.ZONE

### END generated Flag constants
