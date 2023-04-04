# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from the asn1ate tool, with manual
#   changes to AES_CCM_ICVlen.subtypeSpec and added comments
#
# Copyright (c) 2018-2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
#  AES-CCM and AES-GCM Algorithms fo use with the Authenticated-Enveloped-Data
#  protecting content type for the Cryptographic Message Syntax (CMS)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5084.txt

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


class AES_CCM_ICVlen(univ.Integer):
    pass


class AES_GCM_ICVlen(univ.Integer):
    pass


AES_CCM_ICVlen.subtypeSpec = constraint.SingleValueConstraint(4, 6, 8, 10, 12, 14, 16)

AES_GCM_ICVlen.subtypeSpec = constraint.ValueRangeConstraint(12, 16)


class CCMParameters(univ.Sequence):
    pass


CCMParameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('aes-nonce', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(7, 13))),
    # The aes-nonce parameter contains 15-L octets, where L is the size of the length field. L=8 is RECOMMENDED.
    # Within the scope of any content-authenticated-encryption key, the nonce value MUST be unique.
    namedtype.DefaultedNamedType('aes-ICVlen', AES_CCM_ICVlen().subtype(value=12))
)


class GCMParameters(univ.Sequence):
    pass


GCMParameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('aes-nonce', univ.OctetString()),
    # The aes-nonce may have any number of bits between 8 and 2^64, but it MUST be a multiple of 8 bits.
    # Within the scope of any content-authenticated-encryption key, the nonce value MUST be unique.
    # A nonce value of 12 octets can be processed more efficiently, so that length is RECOMMENDED.
    namedtype.DefaultedNamedType('aes-ICVlen', AES_GCM_ICVlen().subtype(value=12))
)

aes = _OID(2, 16, 840, 1, 101, 3, 4, 1)

id_aes128_CCM = _OID(aes, 7)

id_aes128_GCM = _OID(aes, 6)

id_aes192_CCM = _OID(aes, 27)

id_aes192_GCM = _OID(aes, 26)

id_aes256_CCM = _OID(aes, 47)

id_aes256_GCM = _OID(aes, 46)


# Map of Algorithm Identifier OIDs to Parameters is added to the
# ones in rfc5280.py

_algorithmIdentifierMapUpdate = {
    id_aes128_CCM: CCMParameters(),
    id_aes128_GCM: GCMParameters(),
    id_aes192_CCM: CCMParameters(),
    id_aes192_GCM: GCMParameters(),
    id_aes256_CCM: CCMParameters(),
    id_aes256_GCM: GCMParameters(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
