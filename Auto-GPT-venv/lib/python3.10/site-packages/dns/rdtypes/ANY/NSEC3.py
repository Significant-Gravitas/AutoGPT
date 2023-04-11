# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2004-2017 Nominum, Inc.
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
import binascii
import struct

import dns.exception
import dns.immutable
import dns.rdata
import dns.rdatatype
import dns.rdtypes.util


b32_hex_to_normal = bytes.maketrans(
    b"0123456789ABCDEFGHIJKLMNOPQRSTUV", b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
)
b32_normal_to_hex = bytes.maketrans(
    b"ABCDEFGHIJKLMNOPQRSTUVWXYZ234567", b"0123456789ABCDEFGHIJKLMNOPQRSTUV"
)

# hash algorithm constants
SHA1 = 1

# flag constants
OPTOUT = 1


@dns.immutable.immutable
class Bitmap(dns.rdtypes.util.Bitmap):
    type_name = "NSEC3"


@dns.immutable.immutable
class NSEC3(dns.rdata.Rdata):

    """NSEC3 record"""

    __slots__ = ["algorithm", "flags", "iterations", "salt", "next", "windows"]

    def __init__(
        self, rdclass, rdtype, algorithm, flags, iterations, salt, next, windows
    ):
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_uint8(algorithm)
        self.flags = self._as_uint8(flags)
        self.iterations = self._as_uint16(iterations)
        self.salt = self._as_bytes(salt, True, 255)
        self.next = self._as_bytes(next, True, 255)
        if not isinstance(windows, Bitmap):
            windows = Bitmap(windows)
        self.windows = tuple(windows.windows)

    def to_text(self, origin=None, relativize=True, **kw):
        next = base64.b32encode(self.next).translate(b32_normal_to_hex).lower().decode()
        if self.salt == b"":
            salt = "-"
        else:
            salt = binascii.hexlify(self.salt).decode()
        text = Bitmap(self.windows).to_text()
        return "%u %u %u %s %s%s" % (
            self.algorithm,
            self.flags,
            self.iterations,
            salt,
            next,
            text,
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        algorithm = tok.get_uint8()
        flags = tok.get_uint8()
        iterations = tok.get_uint16()
        salt = tok.get_string()
        if salt == "-":
            salt = b""
        else:
            salt = binascii.unhexlify(salt.encode("ascii"))
        next = tok.get_string().encode("ascii").upper().translate(b32_hex_to_normal)
        next = base64.b32decode(next)
        bitmap = Bitmap.from_text(tok)
        return cls(rdclass, rdtype, algorithm, flags, iterations, salt, next, bitmap)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        l = len(self.salt)
        file.write(struct.pack("!BBHB", self.algorithm, self.flags, self.iterations, l))
        file.write(self.salt)
        l = len(self.next)
        file.write(struct.pack("!B", l))
        file.write(self.next)
        Bitmap(self.windows).to_wire(file)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        (algorithm, flags, iterations) = parser.get_struct("!BBH")
        salt = parser.get_counted_bytes()
        next = parser.get_counted_bytes()
        bitmap = Bitmap.from_wire_parser(parser)
        return cls(rdclass, rdtype, algorithm, flags, iterations, salt, next, bitmap)
