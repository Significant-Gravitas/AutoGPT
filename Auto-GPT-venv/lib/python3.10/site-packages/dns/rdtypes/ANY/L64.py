# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import struct

import dns.immutable
import dns.rdtypes.util


@dns.immutable.immutable
class L64(dns.rdata.Rdata):

    """L64 record"""

    # see: rfc6742.txt

    __slots__ = ["preference", "locator64"]

    def __init__(self, rdclass, rdtype, preference, locator64):
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        if isinstance(locator64, bytes):
            if len(locator64) != 8:
                raise ValueError("invalid locator64")
            self.locator64 = dns.rdata._hexify(locator64, 4, b":")
        else:
            dns.rdtypes.util.parse_formatted_hex(locator64, 4, 4, ":")
            self.locator64 = locator64

    def to_text(self, origin=None, relativize=True, **kw):
        return f"{self.preference} {self.locator64}"

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        preference = tok.get_uint16()
        locator64 = tok.get_identifier()
        return cls(rdclass, rdtype, preference, locator64)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(struct.pack("!H", self.preference))
        file.write(dns.rdtypes.util.parse_formatted_hex(self.locator64, 4, 4, ":"))

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        preference = parser.get_uint16()
        locator64 = parser.get_remaining()
        return cls(rdclass, rdtype, preference, locator64)
