# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2001-2017 Nominum, Inc.
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

"""DNS Rdata Types."""

from typing import Dict

import dns.enum
import dns.exception


class RdataType(dns.enum.IntEnum):
    """DNS Rdata Type"""

    TYPE0 = 0
    NONE = 0
    A = 1
    NS = 2
    MD = 3
    MF = 4
    CNAME = 5
    SOA = 6
    MB = 7
    MG = 8
    MR = 9
    NULL = 10
    WKS = 11
    PTR = 12
    HINFO = 13
    MINFO = 14
    MX = 15
    TXT = 16
    RP = 17
    AFSDB = 18
    X25 = 19
    ISDN = 20
    RT = 21
    NSAP = 22
    NSAP_PTR = 23
    SIG = 24
    KEY = 25
    PX = 26
    GPOS = 27
    AAAA = 28
    LOC = 29
    NXT = 30
    SRV = 33
    NAPTR = 35
    KX = 36
    CERT = 37
    A6 = 38
    DNAME = 39
    OPT = 41
    APL = 42
    DS = 43
    SSHFP = 44
    IPSECKEY = 45
    RRSIG = 46
    NSEC = 47
    DNSKEY = 48
    DHCID = 49
    NSEC3 = 50
    NSEC3PARAM = 51
    TLSA = 52
    SMIMEA = 53
    HIP = 55
    NINFO = 56
    CDS = 59
    CDNSKEY = 60
    OPENPGPKEY = 61
    CSYNC = 62
    ZONEMD = 63
    SVCB = 64
    HTTPS = 65
    SPF = 99
    UNSPEC = 103
    NID = 104
    L32 = 105
    L64 = 106
    LP = 107
    EUI48 = 108
    EUI64 = 109
    TKEY = 249
    TSIG = 250
    IXFR = 251
    AXFR = 252
    MAILB = 253
    MAILA = 254
    ANY = 255
    URI = 256
    CAA = 257
    AVC = 258
    AMTRELAY = 260
    TA = 32768
    DLV = 32769

    @classmethod
    def _maximum(cls):
        return 65535

    @classmethod
    def _short_name(cls):
        return "type"

    @classmethod
    def _prefix(cls):
        return "TYPE"

    @classmethod
    def _extra_from_text(cls, text):
        if text.find("-") >= 0:
            try:
                return cls[text.replace("-", "_")]
            except KeyError:
                pass
        return _registered_by_text.get(text)

    @classmethod
    def _extra_to_text(cls, value, current_text):
        if current_text is None:
            return _registered_by_value.get(value)
        if current_text.find("_") >= 0:
            return current_text.replace("_", "-")
        return current_text

    @classmethod
    def _unknown_exception_class(cls):
        return UnknownRdatatype


_registered_by_text: Dict[str, RdataType] = {}
_registered_by_value: Dict[RdataType, str] = {}

_metatypes = {RdataType.OPT}

_singletons = {
    RdataType.SOA,
    RdataType.NXT,
    RdataType.DNAME,
    RdataType.NSEC,
    RdataType.CNAME,
}


class UnknownRdatatype(dns.exception.DNSException):
    """DNS resource record type is unknown."""


def from_text(text: str) -> RdataType:
    """Convert text into a DNS rdata type value.

    The input text can be a defined DNS RR type mnemonic or
    instance of the DNS generic type syntax.

    For example, "NS" and "TYPE2" will both result in a value of 2.

    Raises ``dns.rdatatype.UnknownRdatatype`` if the type is unknown.

    Raises ``ValueError`` if the rdata type value is not >= 0 and <= 65535.

    Returns a ``dns.rdatatype.RdataType``.
    """

    return RdataType.from_text(text)


def to_text(value: RdataType) -> str:
    """Convert a DNS rdata type value to text.

    If the value has a known mnemonic, it will be used, otherwise the
    DNS generic type syntax will be used.

    Raises ``ValueError`` if the rdata type value is not >= 0 and <= 65535.

    Returns a ``str``.
    """

    return RdataType.to_text(value)


def is_metatype(rdtype: RdataType) -> bool:
    """True if the specified type is a metatype.

    *rdtype* is a ``dns.rdatatype.RdataType``.

    The currently defined metatypes are TKEY, TSIG, IXFR, AXFR, MAILA,
    MAILB, ANY, and OPT.

    Returns a ``bool``.
    """

    return (256 > rdtype >= 128) or rdtype in _metatypes


def is_singleton(rdtype: RdataType) -> bool:
    """Is the specified type a singleton type?

    Singleton types can only have a single rdata in an rdataset, or a single
    RR in an RRset.

    The currently defined singleton types are CNAME, DNAME, NSEC, NXT, and
    SOA.

    *rdtype* is an ``int``.

    Returns a ``bool``.
    """

    if rdtype in _singletons:
        return True
    return False


# pylint: disable=redefined-outer-name
def register_type(
    rdtype: RdataType, rdtype_text: str, is_singleton: bool = False
) -> None:
    """Dynamically register an rdatatype.

    *rdtype*, a ``dns.rdatatype.RdataType``, the rdatatype to register.

    *rdtype_text*, a ``str``, the textual form of the rdatatype.

    *is_singleton*, a ``bool``, indicating if the type is a singleton (i.e.
    RRsets of the type can have only one member.)
    """

    _registered_by_text[rdtype_text] = rdtype
    _registered_by_value[rdtype] = rdtype_text
    if is_singleton:
        _singletons.add(rdtype)


### BEGIN generated RdataType constants

TYPE0 = RdataType.TYPE0
NONE = RdataType.NONE
A = RdataType.A
NS = RdataType.NS
MD = RdataType.MD
MF = RdataType.MF
CNAME = RdataType.CNAME
SOA = RdataType.SOA
MB = RdataType.MB
MG = RdataType.MG
MR = RdataType.MR
NULL = RdataType.NULL
WKS = RdataType.WKS
PTR = RdataType.PTR
HINFO = RdataType.HINFO
MINFO = RdataType.MINFO
MX = RdataType.MX
TXT = RdataType.TXT
RP = RdataType.RP
AFSDB = RdataType.AFSDB
X25 = RdataType.X25
ISDN = RdataType.ISDN
RT = RdataType.RT
NSAP = RdataType.NSAP
NSAP_PTR = RdataType.NSAP_PTR
SIG = RdataType.SIG
KEY = RdataType.KEY
PX = RdataType.PX
GPOS = RdataType.GPOS
AAAA = RdataType.AAAA
LOC = RdataType.LOC
NXT = RdataType.NXT
SRV = RdataType.SRV
NAPTR = RdataType.NAPTR
KX = RdataType.KX
CERT = RdataType.CERT
A6 = RdataType.A6
DNAME = RdataType.DNAME
OPT = RdataType.OPT
APL = RdataType.APL
DS = RdataType.DS
SSHFP = RdataType.SSHFP
IPSECKEY = RdataType.IPSECKEY
RRSIG = RdataType.RRSIG
NSEC = RdataType.NSEC
DNSKEY = RdataType.DNSKEY
DHCID = RdataType.DHCID
NSEC3 = RdataType.NSEC3
NSEC3PARAM = RdataType.NSEC3PARAM
TLSA = RdataType.TLSA
SMIMEA = RdataType.SMIMEA
HIP = RdataType.HIP
NINFO = RdataType.NINFO
CDS = RdataType.CDS
CDNSKEY = RdataType.CDNSKEY
OPENPGPKEY = RdataType.OPENPGPKEY
CSYNC = RdataType.CSYNC
ZONEMD = RdataType.ZONEMD
SVCB = RdataType.SVCB
HTTPS = RdataType.HTTPS
SPF = RdataType.SPF
UNSPEC = RdataType.UNSPEC
NID = RdataType.NID
L32 = RdataType.L32
L64 = RdataType.L64
LP = RdataType.LP
EUI48 = RdataType.EUI48
EUI64 = RdataType.EUI64
TKEY = RdataType.TKEY
TSIG = RdataType.TSIG
IXFR = RdataType.IXFR
AXFR = RdataType.AXFR
MAILB = RdataType.MAILB
MAILA = RdataType.MAILA
ANY = RdataType.ANY
URI = RdataType.URI
CAA = RdataType.CAA
AVC = RdataType.AVC
AMTRELAY = RdataType.AMTRELAY
TA = RdataType.TA
DLV = RdataType.DLV

### END generated RdataType constants
