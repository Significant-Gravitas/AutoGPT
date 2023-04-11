# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2004-2007, 2009-2011 Nominum, Inc.
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
import struct

import dns.immutable
import dns.exception
import dns.rdata


@dns.immutable.immutable
class TKEY(dns.rdata.Rdata):

    """TKEY Record"""

    __slots__ = [
        "algorithm",
        "inception",
        "expiration",
        "mode",
        "error",
        "key",
        "other",
    ]

    def __init__(
        self,
        rdclass,
        rdtype,
        algorithm,
        inception,
        expiration,
        mode,
        error,
        key,
        other=b"",
    ):
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_name(algorithm)
        self.inception = self._as_uint32(inception)
        self.expiration = self._as_uint32(expiration)
        self.mode = self._as_uint16(mode)
        self.error = self._as_uint16(error)
        self.key = self._as_bytes(key)
        self.other = self._as_bytes(other)

    def to_text(self, origin=None, relativize=True, **kw):
        _algorithm = self.algorithm.choose_relativity(origin, relativize)
        text = "%s %u %u %u %u %s" % (
            str(_algorithm),
            self.inception,
            self.expiration,
            self.mode,
            self.error,
            dns.rdata._base64ify(self.key, 0),
        )
        if len(self.other) > 0:
            text += " %s" % (dns.rdata._base64ify(self.other, 0))

        return text

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        algorithm = tok.get_name(relativize=False)
        inception = tok.get_uint32()
        expiration = tok.get_uint32()
        mode = tok.get_uint16()
        error = tok.get_uint16()
        key_b64 = tok.get_string().encode()
        key = base64.b64decode(key_b64)
        other_b64 = tok.concatenate_remaining_identifiers(True).encode()
        other = base64.b64decode(other_b64)

        return cls(
            rdclass, rdtype, algorithm, inception, expiration, mode, error, key, other
        )

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        self.algorithm.to_wire(file, compress, origin)
        file.write(
            struct.pack("!IIHH", self.inception, self.expiration, self.mode, self.error)
        )
        file.write(struct.pack("!H", len(self.key)))
        file.write(self.key)
        file.write(struct.pack("!H", len(self.other)))
        if len(self.other) > 0:
            file.write(self.other)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        algorithm = parser.get_name(origin)
        inception, expiration, mode, error = parser.get_struct("!IIHH")
        key = parser.get_counted_bytes(2)
        other = parser.get_counted_bytes(2)

        return cls(
            rdclass, rdtype, algorithm, inception, expiration, mode, error, key, other
        )

    # Constants for the mode field - from RFC 2930:
    # 2.5 The Mode Field
    #
    #    The mode field specifies the general scheme for key agreement or
    #    the purpose of the TKEY DNS message.  Servers and resolvers
    #    supporting this specification MUST implement the Diffie-Hellman key
    #    agreement mode and the key deletion mode for queries.  All other
    #    modes are OPTIONAL.  A server supporting TKEY that receives a TKEY
    #    request with a mode it does not support returns the BADMODE error.
    #    The following values of the Mode octet are defined, available, or
    #    reserved:
    #
    #          Value    Description
    #          -----    -----------
    #           0        - reserved, see section 7
    #           1       server assignment
    #           2       Diffie-Hellman exchange
    #           3       GSS-API negotiation
    #           4       resolver assignment
    #           5       key deletion
    #          6-65534   - available, see section 7
    #          65535     - reserved, see section 7
    SERVER_ASSIGNMENT = 1
    DIFFIE_HELLMAN_EXCHANGE = 2
    GSSAPI_NEGOTIATION = 3
    RESOLVER_ASSIGNMENT = 4
    KEY_DELETION = 5
