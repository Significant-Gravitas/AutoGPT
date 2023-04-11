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

import struct
import binascii

import dns.exception
import dns.immutable
import dns.rdata


@dns.immutable.immutable
class NSEC3PARAM(dns.rdata.Rdata):

    """NSEC3PARAM record"""

    __slots__ = ["algorithm", "flags", "iterations", "salt"]

    def __init__(self, rdclass, rdtype, algorithm, flags, iterations, salt):
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_uint8(algorithm)
        self.flags = self._as_uint8(flags)
        self.iterations = self._as_uint16(iterations)
        self.salt = self._as_bytes(salt, True, 255)

    def to_text(self, origin=None, relativize=True, **kw):
        if self.salt == b"":
            salt = "-"
        else:
            salt = binascii.hexlify(self.salt).decode()
        return "%u %u %u %s" % (self.algorithm, self.flags, self.iterations, salt)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        algorithm = tok.get_uint8()
        flags = tok.get_uint8()
        iterations = tok.get_uint16()
        salt = tok.get_string()
        if salt == "-":
            salt = ""
        else:
            salt = binascii.unhexlify(salt.encode())
        return cls(rdclass, rdtype, algorithm, flags, iterations, salt)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        l = len(self.salt)
        file.write(struct.pack("!BBHB", self.algorithm, self.flags, self.iterations, l))
        file.write(self.salt)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        (algorithm, flags, iterations) = parser.get_struct("!BBH")
        salt = parser.get_counted_bytes()
        return cls(rdclass, rdtype, algorithm, flags, iterations, salt)
