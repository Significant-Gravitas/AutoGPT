# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2005-2007, 2009-2011 Nominum, Inc.
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
import binascii

import dns.rdata
import dns.immutable
import dns.rdatatype


@dns.immutable.immutable
class SSHFP(dns.rdata.Rdata):

    """SSHFP record"""

    # See RFC 4255

    __slots__ = ["algorithm", "fp_type", "fingerprint"]

    def __init__(self, rdclass, rdtype, algorithm, fp_type, fingerprint):
        super().__init__(rdclass, rdtype)
        self.algorithm = self._as_uint8(algorithm)
        self.fp_type = self._as_uint8(fp_type)
        self.fingerprint = self._as_bytes(fingerprint, True)

    def to_text(self, origin=None, relativize=True, **kw):
        kw = kw.copy()
        chunksize = kw.pop("chunksize", 128)
        return "%d %d %s" % (
            self.algorithm,
            self.fp_type,
            dns.rdata._hexify(self.fingerprint, chunksize=chunksize, **kw),
        )

    @classmethod
    def from_text(
        cls, rdclass, rdtype, tok, origin=None, relativize=True, relativize_to=None
    ):
        algorithm = tok.get_uint8()
        fp_type = tok.get_uint8()
        fingerprint = tok.concatenate_remaining_identifiers().encode()
        fingerprint = binascii.unhexlify(fingerprint)
        return cls(rdclass, rdtype, algorithm, fp_type, fingerprint)

    def _to_wire(self, file, compress=None, origin=None, canonicalize=False):
        header = struct.pack("!BB", self.algorithm, self.fp_type)
        file.write(header)
        file.write(self.fingerprint)

    @classmethod
    def from_wire_parser(cls, rdclass, rdtype, parser, origin=None):
        header = parser.get_struct("BB")
        fingerprint = parser.get_remaining()
        return cls(rdclass, rdtype, header[0], header[1], fingerprint)
