# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import struct

import dns.immutable
import dns.rdtypes.util


@dns.immutable.immutable
class NID(dns.rdata.Rdata):

    """NID record"""

    # see: rfc6742.txt

    __slots__ = ["preference", "nodeid"]

    def __init__(self, rdclass, rdtype, preference, nodeid):
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        if isinstance(nodeid, bytes):
            if len(nodeid) != 8:
                raise ValueError("invalid nodeid")
            self.nodeid = dns.rdata._hexify(nodeid, 4, b":")
        else:
            dns.rdtypes.util.parse_formatted_hex(nodeid, 4, 4, ":")
            self.nodeid = nodeid

    def to_text(self, origin=None, relativize=True, **kw):
        return f"{self.preference} {self.nodeid}"

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        preference = tok.get_uint16()
        nodeid = tok.get_identifier()
        return cls(rdclass, rdtype, preference, nodeid)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(struct.pack("!H", self.preference))
        file.write(dns.rdtypes.util.parse_formatted_hex(self.nodeid, 4, 4, ":"))

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        preference = parser.get_uint16()
        nodeid = parser.get_remaining()
        return cls(rdclass, rdtype, preference, nodeid)
