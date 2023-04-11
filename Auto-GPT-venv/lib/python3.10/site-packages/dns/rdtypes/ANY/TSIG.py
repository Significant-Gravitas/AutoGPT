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

import base64
import struct

import dns.exception
import dns.immutable
import dns.rcode
import dns.rdata


@dns.immutable.immutable
class TSIG(dns.rdata.Rdata):

    """TSIG record"""

    __slots__ = [
        "algorithm",
        "time_signed",
        "fudge",
        "mac",
        "original_id",
        "error",
        "other",
    ]

    def __init__(
        self,
        rdclass,
        rdtype,
        algorithm,
        time_signed,
        fudge,
        mac,
        original_id,
        error,
        other,
    ):
        """Initialize a TSIG rdata.

        *rdclass*, an ``int`` is the rdataclass of the Rdata.

        *rdtype*, an ``int`` is the rdatatype of the Rdata.

        *algorithm*, a ``dns.name.Name``.

        *time_signed*, an ``int``.

        *fudge*, an ``int`.

        *mac*, a ``bytes``

        *original_id*, an ``int``

        *error*, an ``int``

        *other*, a ``bytes``
        """

        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_name(algorithm)
        self.time_signed = self._as_uint48(time_signed)
        self.fudge = self._as_uint16(fudge)
        self.mac = self._as_bytes(mac)
        self.original_id = self._as_uint16(original_id)
        self.error = dns.rcode.Rcode.make(error)
        self.other = self._as_bytes(other)

    def to_text(self, origin=None, relativize=True, **kw):
        algorithm = self.algorithm.choose_relativity(origin, relativize)
        error = dns.rcode.to_text(self.error, True)
        text = (
            f"{algorithm} {self.time_signed} {self.fudge} "
            + f"{len(self.mac)} {dns.rdata._base64ify(self.mac, 0)} "
            + f"{self.original_id} {error} {len(self.other)}"
        )
        if self.other:
            text += f" {dns.rdata._base64ify(self.other, 0)}"
        return text

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        algorithm = tok.get_name(relativize=False)
        time_signed = tok.get_uint48()
        fudge = tok.get_uint16()
        mac_len = tok.get_uint16()
        mac = base64.b64decode(tok.get_string())
        if len(mac) != mac_len:
            raise SyntaxError("invalid MAC")
        original_id = tok.get_uint16()
        error = dns.rcode.from_text(tok.get_string())
        other_len = tok.get_uint16()
        if other_len > 0:
            other = base64.b64decode(tok.get_string())
            if len(other) != other_len:
                raise SyntaxError("invalid other data")
        else:
            other = b""
        return cls(
            rdclass,
            rdtype,
            algorithm,
            time_signed,
            fudge,
            mac,
            original_id,
            error,
            other,
        )

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        self.algorithm.to_wire(file, None, origin, False)
        file.write(
            struct.pack(
                "!HIHH",
                (self.time_signed >> 32) & 0xFFFF,
                self.time_signed & 0xFFFFFFFF,
                self.fudge,
                len(self.mac),
            )
        )
        file.write(self.mac)
        file.write(struct.pack("!HHH", self.original_id, self.error, len(self.other)))
        file.write(self.other)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        algorithm = parser.get_name()
        time_signed = parser.get_uint48()
        fudge = parser.get_uint16()
        mac = parser.get_counted_bytes(2)
        (original_id, error) = parser.get_struct("!HH")
        other = parser.get_counted_bytes(2)
        return cls(
            rdclass,
            rdtype,
            algorithm,
            time_signed,
            fudge,
            mac,
            original_id,
            error,
            other,
        )
