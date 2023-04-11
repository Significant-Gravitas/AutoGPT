# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

import struct

import dns.immutable
import dns.rdata


@dns.immutable.immutable
class LP(dns.rdata.Rdata):

    """LP record"""

    # see: rfc6742.txt

    __slots__ = ["preference", "fqdn"]

    def __init__(self, rdclass, rdtype, preference, fqdn):
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        self.fqdn = self._as_name(fqdn)

    def to_text(self, origin=None, relativize=True, **kw):
        fqdn = self.fqdn.choose_relativity(origin, relativize)
        return "%d %s" % (self.preference, fqdn)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        preference = tok.get_uint16()
        fqdn = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, preference, fqdn)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(struct.pack("!H", self.preference))
        self.fqdn.to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        preference = parser.get_uint16()
        fqdn = parser.get_name(origin)
        return cls(rdclass, rdtype, preference, fqdn)
