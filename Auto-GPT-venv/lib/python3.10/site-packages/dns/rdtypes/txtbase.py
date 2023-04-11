# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2006-2017 Nominum, Inc.
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

"""TXT-like base class."""

from typing import Any, Dict, Iterable, Optional, Tuple, Union

import struct

import dns.exception
import dns.immutable
import dns.rdata
import dns.tokenizer


@dns.immutable.immutable
class TXTBase(dns.rdata.Rdata):

    """Base class for rdata that is like a TXT record (see RFC 1035)."""

    __slots__ = ["strings"]

    def __init__(
        self,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        strings: Iterable[Union[bytes, str]],
    ):
        """Initialize a TXT-like rdata.

        *rdclass*, an ``int`` is the rdataclass of the Rdata.

        *rdtype*, an ``int`` is the rdatatype of the Rdata.

        *strings*, a tuple of ``bytes``
        """
        super().__init__(rdclass, rdtype)
        self.strings: Tuple[bytes] = self._as_tuple(
            strings, lambda x: self._as_bytes(x, True, 255)
        )

    def to_text(
        self,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        **kw: Dict[str, Any]
    ) -> str:
        txt = ""
        prefix = ""
        for s in self.strings:
            txt += '{}"{}"'.format(prefix, dns.rdata._escapify(s))
            prefix = " "
        return txt

    @classmethod
    def from_text(
        cls,
        rdclass: dns.rdataclass.RdataClass,
        rdtype: dns.rdatatype.RdataType,
        tok: dns.tokenizer.Tokenizer,
        origin: Optional[dns.name.Name] = None,
        relativize: bool = True,
        relativize_to: Optional[dns.name.Name] = None,
    ) -> dns.rdata.Rdata:
        strings = []
        for token in tok.get_remaining():
            token = token.unescape_to_bytes()
            # The 'if' below is always true in the current code, but we
            # are leaving this check in in case things change some day.
            if not (
                token.is_quoted_string() or token.is_identifier()
            ):  # pragma: no cover
                raise dns.exception.SyntaxError("expected a string")
            if len(token.value) > 255:
                raise dns.exception.SyntaxError("string too long")
            strings.append(token.value)
        if len(strings) == 0:
            raise dns.exception.UnexpectedEnd
        return cls(rdclass, rdtype, strings)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        for s in self.strings:
            l = len(s)
            assert l < 256
            file.write(struct.pack("!B", l))
            file.write(s)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        strings = []
        while parser.remaining() > 0:
            s = parser.get_counted_bytes()
            strings.append(s)
        return cls(rdclass, rdtype, strings)
