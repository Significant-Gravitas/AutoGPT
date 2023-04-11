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
import calendar
import struct
import time

import dns.dnssectypes
import dns.immutable
import dns.exception
import dns.rdata
import dns.rdatatype


class BadSigTime(dns.exception.DNSException):

    """Time in DNS SIG or RRSIG resource record cannot be parsed."""


def sigtime_to_posixtime(what):
    if len(what) <= 10 and what.isdigit():
        return int(what)
    if len(what) != 14:
        raise BadSigTime
    year = int(what[0:4])
    month = int(what[4:6])
    day = int(what[6:8])
    hour = int(what[8:10])
    minute = int(what[10:12])
    second = int(what[12:14])
    return calendar.timegm((year, month, day, hour, minute, second, 0, 0, 0))


def posixtime_to_sigtime(what):
    return time.strftime("%Y%m%d%H%M%S", time.gmtime(what))


@dns.immutable.immutable
class RRSIG(dns.rdata.Rdata):

    """RRSIG record"""

    __slots__ = [
        "type_covered",
        "algorithm",
        "labels",
        "original_ttl",
        "expiration",
        "inception",
        "key_tag",
        "signer",
        "signature",
    ]

    def __init__(
        self,
        rdclass,
        rdtype,
        type_covered,
        algorithm,
        labels,
        original_ttl,
        expiration,
        inception,
        key_tag,
        signer,
        signature,
    ):
        super().__init__(rdclass, rdtype)
        self.type_covered = self._as_rdatatype(type_covered)
        self.algorithm = dns.dnssectypes.Algorithm.make(algorithm)
        self.labels = self._as_uint8(labels)
        self.original_ttl = self._as_ttl(original_ttl)
        self.expiration = self._as_uint32(expiration)
        self.inception = self._as_uint32(inception)
        self.key_tag = self._as_uint16(key_tag)
        self.signer = self._as_name(signer)
        self.signature = self._as_bytes(signature)

    def covers(self):
        return self.type_covered

    def to_text(self, origin=None, relativize=True, **kw):
        return "%s %d %d %d %s %s %d %s %s" % (
            dns.rdatatype.to_text(self.type_covered),
            self.algorithm,
            self.labels,
            self.original_ttl,
            posixtime_to_sigtime(self.expiration),
            posixtime_to_sigtime(self.inception),
            self.key_tag,
            self.signer.choose_relativity(origin, relativize),
            dns.rdata._base64ify(self.signature, **kw),
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        type_covered = dns.rdatatype.from_text(tok.get_string())
        algorithm = dns.dnssectypes.Algorithm.from_text(tok.get_string())
        labels = tok.get_int()
        original_ttl = tok.get_ttl()
        expiration = sigtime_to_posixtime(tok.get_string())
        inception = sigtime_to_posixtime(tok.get_string())
        key_tag = tok.get_int()
        signer = tok.get_name(origin, relativize, relativize_to)
        b64 = tok.concatenate_remaining_identifiers().encode()
        signature = base64.b64decode(b64)
        return cls(
            rdclass,
            rdtype,
            type_covered,
            algorithm,
            labels,
            original_ttl,
            expiration,
            inception,
            key_tag,
            signer,
            signature,
        )

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        header = struct.pack(
            "!HBBIIIH",
            self.type_covered,
            self.algorithm,
            self.labels,
            self.original_ttl,
            self.expiration,
            self.inception,
            self.key_tag,
        )
        file.write(header)
        self.signer.to_wire(file, None, origin, canonicalize)
        file.write(self.signature)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        header = parser.get_struct("!HBBIIIH")
        signer = parser.get_name(origin)
        signature = parser.get_remaining()
        return cls(rdclass, rdtype, *header, signer, signature)
