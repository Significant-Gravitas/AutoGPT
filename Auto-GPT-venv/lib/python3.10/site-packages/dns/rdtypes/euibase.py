# Copyright (C) 2015 Red Hat, Inc.
# Author: Petr Spacek <pspacek@redhat.com>
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED 'AS IS' AND RED HAT DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import binascii

import dns.rdata
import dns.immutable


@dns.immutable.immutable
class EUIBase(dns.rdata.Rdata):

    """EUIxx record"""

    # see: rfc7043.txt

    __slots__ = ["eui"]
    # define these in subclasses
    # byte_len = 6  # 0123456789ab (in hex)
    # text_len = byte_len * 3 - 1  # 01-23-45-67-89-ab

    def __init__(self, rdclass, rdtype, eui):
        super().__init__(rdclass, rdtype)
        self.eui = self._as_bytes(eui)
        if len(self.eui) != self.byte_len:
            raise dns.exception.FormError(
                "EUI%s rdata has to have %s bytes" % (self.byte_len * 8, self.byte_len)
            )

    def to_text(self, origin=None, relativize=True, **kw):
        return dns.rdata._hexify(self.eui, chunksize=2, separator=b"-", **kw)

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        text = tok.get_string()
        if len(text) != cls.text_len:
            raise dns.exception.SyntaxError(
                "Input text must have %s characters" % cls.text_len
            )
        for i in range(2, cls.byte_len * 3 - 1, 3):
            if text[i] != "-":
                raise dns.exception.SyntaxError("Dash expected at position %s" % i)
        text = text.replace("-", "")
        try:
            data = binascii.unhexlify(text.encode())
        except (ValueError, TypeError) as ex:
            raise dns.exception.SyntaxError("Hex decoding error: %s" % str(ex))
        return cls(rdclass, rdtype, data)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        file.write(self.eui)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        eui = parser.get_bytes(cls.byte_len)
        return cls(rdclass, rdtype, eui)
