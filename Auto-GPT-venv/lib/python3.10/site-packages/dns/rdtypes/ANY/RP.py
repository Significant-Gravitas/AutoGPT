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

import dns.exception
import dns.immutable
import dns.rdata
import dns.name


@dns.immutable.immutable
class RP(dns.rdata.Rdata):

    """RP record"""

    # see: RFC 1183

    __slots__ = ["mbox", "txt"]

    def __init__(self, rdclass, rdtype, mbox, txt):
        super().__init__(rdclass, rdtype)
        self.mbox = self._as_name(mbox)
        self.txt = self._as_name(txt)

    def to_text(self, origin=None, relativize=True, **kw):
        mbox = self.mbox.choose_relativity(origin, relativize)
        txt = self.txt.choose_relativity(origin, relativize)
        return "{} {}".format(str(mbox), str(txt))

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        mbox = tok.get_name(origin, relativize, relativize_to)
        txt = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, mbox, txt)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        self.mbox.to_wire(file, None, origin, canonicalize)
        self.txt.to_wire(file, None, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        mbox = parser.get_name(origin)
        txt = parser.get_name(origin)
        return cls(rdclass, rdtype, mbox, txt)
