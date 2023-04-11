# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2004-2007, 2009-2011, 2016 Nominum, Inc.
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
import dns.rdatatype
import dns.name
import dns.rdtypes.util


@dns.immutable.immutable
class Bitmap(dns.rdtypes.util.Bitmap):
    type_name = "CSYNC"


@dns.immutable.immutable
class CSYNC(dns.rdata.Rdata):

    """CSYNC record"""

    __slots__ = ["serial", "flags", "windows"]

    def __init__(self, rdclass, rdtype, serial, flags, windows):
        super().__init__(rdclass, rdtype)
        self.serial = self._as_uint32(serial)
        self.flags = self._as_uint16(flags)
        if not isinstance(windows, Bitmap):
            windows = Bitmap(windows)
        self.windows = tuple(windows.windows)

    def to_text(self, origin=None, relativize=True, **kw):
        text = Bitmap(self.windows).to_text()
        return "%d %d%s" % (self.serial, self.flags, text)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        serial = tok.get_uint32()
        flags = tok.get_uint16()
        bitmap = Bitmap.from_text(tok)
        return cls(rdclass, rdtype, serial, flags, bitmap)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(struct.pack("!IH", self.serial, self.flags))
        Bitmap(self.windows).to_wire(file)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        (serial, flags) = parser.get_struct("!IH")
        bitmap = Bitmap.from_wire_parser(parser)
        return cls(rdclass, rdtype, serial, flags, bitmap)
