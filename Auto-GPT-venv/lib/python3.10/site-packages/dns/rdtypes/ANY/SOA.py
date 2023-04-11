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
import dns.name


@dns.immutable.immutable
class SOA(dns.rdata.Rdata):

    """SOA record"""

    # see: RFC 1035

    __slots__ = ["mname", "rname", "serial", "refresh", "retry", "expire", "minimum"]

    def __init__(
        self, rdclass, rdtype, mname, rname, serial, refresh, retry, expire, minimum
    ):
        super().__init__(rdclass, rdtype)
        self.mname = self._as_name(mname)
        self.rname = self._as_name(rname)
        self.serial = self._as_uint32(serial)
        self.refresh = self._as_ttl(refresh)
        self.retry = self._as_ttl(retry)
        self.expire = self._as_ttl(expire)
        self.minimum = self._as_ttl(minimum)

    def to_text(self, origin=None, relativize=True, **kw):
        mname = self.mname.choose_relativity(origin, relativize)
        rname = self.rname.choose_relativity(origin, relativize)
        return "%s %s %d %d %d %d %d" % (
            mname,
            rname,
            self.serial,
            self.refresh,
            self.retry,
            self.expire,
            self.minimum,
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        mname = tok.get_name(origin, relativize, relativize_to)
        rname = tok.get_name(origin, relativize, relativize_to)
        serial = tok.get_uint32()
        refresh = tok.get_ttl()
        retry = tok.get_ttl()
        expire = tok.get_ttl()
        minimum = tok.get_ttl()
        return cls(
            rdclass, rdtype, mname, rname, serial, refresh, retry, expire, minimum
        )

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        self.mname.to_wire(file, compress, origin, canonicalize)
        self.rname.to_wire(file, compress, origin, canonicalize)
        five_ints = struct.pack(
            "!IIIII", self.serial, self.refresh, self.retry, self.expire, self.minimum
        )
        file.write(five_ints)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        mname = parser.get_name(origin)
        rname = parser.get_name(origin)
        return cls(rdclass, rdtype, mname, rname, *parser.get_struct("!IIIII"))
