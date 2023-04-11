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

import struct
import base64

import dns.exception
import dns.immutable
import dns.rdtypes.util


class Gateway(dns.rdtypes.util.Gateway):
    name = "IPSECKEY gateway"


@dns.immutable.immutable
class IPSECKEY(dns.rdata.Rdata):

    """IPSECKEY record"""

    # see: RFC 4025

    __slots__ = ["precedence", "gateway_type", "algorithm", "gateway", "key"]

    def __init__(
        self, rdclass, rdtype, precedence, gateway_type, algorithm, gateway, key
    ):
        super().__init__(rdclass, rdtype)
        gateway = Gateway(gateway_type, gateway)
        self.precedence = self._as_uint8(precedence)
        self.gateway_type = gateway.type
        self.algorithm = self._as_uint8(algorithm)
        self.gateway = gateway.gateway
        self.key = self._as_bytes(key)

    def to_text(self, origin=None, relativize=True, **kw):
        gateway = Gateway(self.gateway_type, self.gateway).to_text(origin, relativize)
        return "%d %d %d %s %s" % (
            self.precedence,
            self.gateway_type,
            self.algorithm,
            gateway,
            dns.rdata._base64ify(self.key, **kw),
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        precedence = tok.get_uint8()
        gateway_type = tok.get_uint8()
        algorithm = tok.get_uint8()
        gateway = Gateway.from_text(
            gateway_type, tok, origin, relativize, relativize_to
        )
        b64 = tok.concatenate_remaining_identifiers().encode()
        key = base64.b64decode(b64)
        return cls(
            rdclass, rdtype, precedence, gateway_type, algorithm, gateway.gateway, key
        )

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        header = struct.pack("!BBB", self.precedence, self.gateway_type, self.algorithm)
        file.write(header)
        Gateway(self.gateway_type, self.gateway).to_wire(
            file, compress, origin, canonicalize
        )
        file.write(self.key)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        header = parser.get_struct("!BBB")
        gateway_type = header[1]
        gateway = Gateway.from_wire_parser(gateway_type, parser, origin)
        key = parser.get_remaining()
        return cls(
            rdclass, rdtype, header[0], gateway_type, header[2], gateway.gateway, key
        )
