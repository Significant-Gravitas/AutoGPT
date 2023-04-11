# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2007, 2009-2011 Nominum, Inc.
# Copyright (C) 2015 Red Hat, Inc.
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
class URI(dns.rdata.Rdata):

    """URI record"""

    # see RFC 7553

    __slots__ = ["priority", "weight", "target"]

    def __init__(self, rdclass, rdtype, priority, weight, target):
        super().__init__(rdclass, rdtype)
        self.priority = self._as_uint16(priority)
        self.weight = self._as_uint16(weight)
        self.target = self._as_bytes(target, True)
        if len(self.target) == 0:
            raise dns.exception.SyntaxError("URI target cannot be empty")

    def to_text(self, origin=None, relativize=True, **kw):
        return '%d %d "%s"' % (self.priority, self.weight, self.target.decode())

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        priority = tok.get_uint16()
        weight = tok.get_uint16()
        target = tok.get().unescape()
        if not (target.is_quoted_string() or target.is_identifier()):
            raise dns.exception.SyntaxError("URI target must be a string")
        return cls(rdclass, rdtype, priority, weight, target.value)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        two_ints = struct.pack("!HH", self.priority, self.weight)
        file.write(two_ints)
        file.write(self.target)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        (priority, weight) = parser.get_struct("!HH")
        target = parser.get_remaining()
        if len(target) == 0:
            raise dns.exception.FormError("URI target may not be empty")
        return cls(rdclass, rdtype, priority, weight, target)

    def _processing_priority(self):
        return self.priority

    def _processing_weight(self):
        return self.weight

    @classmethod
    def _processing_order(cls, iterable):
        return dns.rdtypes.util.weighted_processing_order(iterable)
