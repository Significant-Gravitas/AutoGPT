# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2006, 2007, 2009-2011 Nominum, Inc.
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

import dns.exception
import dns.immutable
import dns.rdata


@dns.immutable.immutable
class DHCID(dns.rdata.Rdata):

    """DHCID record"""

    # see: RFC 4701

    __slots__ = ["data"]

    def __init__(self, rdclass, rdtype, data):
        super().__init__(rdclass, rdtype)
        self.data = self._as_bytes(data)

    def to_text(self, origin=None, relativize=True, **kw):
        return dns.rdata._base64ify(self.data, **kw)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        b64 = tok.concatenate_remaining_identifiers().encode()
        data = base64.b64decode(b64)
        return cls(rdclass, rdtype, data)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(self.data)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        data = parser.get_remaining()
        return cls(rdclass, rdtype, data)
