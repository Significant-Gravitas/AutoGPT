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

"""NS-like base classes."""

import dns.exception
import dns.immutable
import dns.rdata
import dns.name


@dns.immutable.immutable
class NSBase(dns.rdata.Rdata):

    """Base class for rdata that is like an NS record."""

    __slots__ = ["target"]

    def __init__(self, rdclass, rdtype, target):
        super().__init__(rdclass, rdtype)
        self.target = self._as_name(target)

    def to_text(self, origin=None, relativize=True, **kw):
        target = self.target.choose_relativity(origin, relativize)
        return str(target)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        target = tok.get_name(origin, relativize, relativize_to)
        return cls(rdclass, rdtype, target)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        self.target.to_wire(file, compress, origin, canonicalize)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        target = parser.get_name(origin)
        return cls(rdclass, rdtype, target)


@dns.immutable.immutable
class UncompressedNS(NSBase):

    """Base class for rdata that is like an NS record, but whose name
    is not compressed when convert to DNS wire format, and whose
    digestable form is not downcased."""

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        self.target.to_wire(file, None, origin, False)
