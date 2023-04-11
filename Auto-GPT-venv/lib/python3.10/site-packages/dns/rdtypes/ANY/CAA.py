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
import dns.tokenizer


@dns.immutable.immutable
class CAA(dns.rdata.Rdata):

    """CAA (Certification Authority Authorization) record"""

    # see: RFC 6844

    __slots__ = ["flags", "tag", "value"]

    def __init__(self, rdclass, rdtype, flags, tag, value):
        super().__init__(rdclass, rdtype)
        self.flags = self._as_uint8(flags)
        self.tag = self._as_bytes(tag, True, 255)
        if not tag.isalnum():
            raise ValueError("tag is not alphanumeric")
        self.value = self._as_bytes(value)

    def to_text(self, origin=None, relativize=True, **kw):
        return '%u %s "%s"' % (
            self.flags,
            dns.rdata._escapify(self.tag),
            dns.rdata._escapify(self.value),
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        flags = tok.get_uint8()
        tag = tok.get_string().encode()
        value = tok.get_string().encode()
        return cls(rdclass, rdtype, flags, tag, value)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(struct.pack("!B", self.flags))
        l = len(self.tag)
        assert l < 256
        file.write(struct.pack("!B", l))
        file.write(self.tag)
        file.write(self.value)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        flags = parser.get_uint8()
        tag = parser.get_counted_bytes()
        value = parser.get_remaining()
        return cls(rdclass, rdtype, flags, tag, value)
