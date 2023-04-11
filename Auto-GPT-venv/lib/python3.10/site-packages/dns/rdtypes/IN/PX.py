# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2007, 2009-2011 Nominum, Inc.
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

import dns.exception
import dns.immutable
import dns.rdata
import dns.rdtypes.util
import dns.name


@dns.immutable.immutable
class PX(dns.rdata.Rdata):

    """PX record."""

    # see: RFC 2163

    __slots__ = ["preference", "map822", "mapx400"]

    def __init__(self, rdclass, rdtype, preference, map822, mapx400):
        super().__init__(rdclass, rdtype)
        self.preference = self._as_uint16(preference)
        self.map822 = self._as_name(map822)
        self.mapx400 = self._as_name(mapx400)

    def to_text(self, origin=None, relativize=True, **kw):
        map822 = self.map822.choose_relativity(origin, relativize)
        mapx400 = self.mapx400.choose_relativity(origin, relativize)
        return "%d %s %s" % (self.preference, map822, mapx400)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        preference = tok.get_uint16()
        map822 = tok.get_name(origin, relativize, relativize_to)
        mapx400 = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, preference, map822, mapx400)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        pref = struct.pack("!H", self.preference)
        file.write(pref)
        self.map822.to_wire(file, None, origin, canonicalize)
        self.mapx400.to_wire(file, None, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        preference = parser.get_uint16()
        map822 = parser.get_name(origin)
        mapx400 = parser.get_name(origin)
        return cls(rdclass, rdtype, preference, map822, mapx400)

    def _processing_priority(self):
        return self.preference

    @classmethod
    def _processing_order(cls, iterable):
        return dns.rdtypes.util.priority_processing_order(iterable)
