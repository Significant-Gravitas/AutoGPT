#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS #5: Password-Based Cryptography Specification, Version 2.1
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8018.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ

from pyasn1_modules import rfc3565
from pyasn1_modules import rfc5280

MAX = float('inf')

def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


# Import from RFC 3565

AES_IV = rfc3565.AES_IV


# Import from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier


# Basic object identifiers

nistAlgorithms = _OID(2, 16, 840, 1, 101, 3, 4)

aes = _OID(nistAlgorithms, 1)

oiw = _OID(1, 3, 14)

rsadsi = _OID(1, 2, 840, 113549)

pkcs = _OID(rsadsi, 1)

digestAlgorithm = _OID(rsadsi, 2)

encryptionAlgorithm = _OID(rsadsi, 3)

pkcs_5 = _OID(pkcs, 5)



# HMAC object identifiers

id_hmacWithSHA1 = _OID(digestAlgorithm, 7)

id_hmacWithSHA224 = _OID(digestAlgorithm, 8)

id_hmacWithSHA256 = _OID(digestAlgorithm, 9)

id_hmacWithSHA384 = _OID(digestAlgorithm, 10)

id_hmacWithSHA512 = _OID(digestAlgorithm, 11)

id_hmacWithSHA512_224 = _OID(digestAlgorithm, 12)

id_hmacWithSHA512_256 = _OID(digestAlgorithm, 13)


# PBES1 object identifiers

pbeWithMD2AndDES_CBC = _OID(pkcs_5, 1)

pbeWithMD2AndRC2_CBC = _OID(pkcs_5, 4)

pbeWithMD5AndDES_CBC = _OID(pkcs_5, 3)

pbeWithMD5AndRC2_CBC = _OID(pkcs_5, 6)

pbeWithSHA1AndDES_CBC = _OID(pkcs_5, 10)

pbeWithSHA1AndRC2_CBC = _OID(pkcs_5, 11)


# Supporting techniques object identifiers

desCBC = _OID(oiw, 3, 2, 7)

des_EDE3_CBC = _OID(encryptionAlgorithm, 7)

rc2CBC = _OID(encryptionAlgorithm, 2)

rc5_CBC_PAD = _OID(encryptionAlgorithm, 9)

aes128_CBC_PAD = _OID(aes, 2)

aes192_CBC_PAD = _OID(aes, 22)

aes256_CBC_PAD = _OID(aes, 42)


# PBES1

class PBEParameter(univ.Sequence):
    pass

PBEParameter.componentType = namedtype.NamedTypes(
    namedtype.NamedType('salt', univ.OctetString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(8, 8))),
    namedtype.NamedType('iterationCount', univ.Integer())
)


# PBES2

id_PBES2 = _OID(pkcs_5, 13)


class PBES2_params(univ.Sequence):
    pass

PBES2_params.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyDerivationFunc', AlgorithmIdentifier()),
    namedtype.NamedType('encryptionScheme', AlgorithmIdentifier())
)


# PBMAC1

id_PBMAC1 = _OID(pkcs_5, 14)


class PBMAC1_params(univ.Sequence):
    pass

PBMAC1_params.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyDerivationFunc', AlgorithmIdentifier()),
    namedtype.NamedType('messageAuthScheme', AlgorithmIdentifier())
)


# PBKDF2

id_PBKDF2 = _OID(pkcs_5, 12)


algid_hmacWithSHA1 = AlgorithmIdentifier()
algid_hmacWithSHA1['algorithm'] = id_hmacWithSHA1
algid_hmacWithSHA1['parameters'] = univ.Null("")


class PBKDF2_params(univ.Sequence):
    pass

PBKDF2_params.componentType = namedtype.NamedTypes(
    namedtype.NamedType('salt', univ.Choice(componentType=namedtype.NamedTypes(
        namedtype.NamedType('specified', univ.OctetString()),
        namedtype.NamedType('otherSource', AlgorithmIdentifier())
    ))),
    namedtype.NamedType('iterationCount', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, MAX))),
    namedtype.OptionalNamedType('keyLength', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, MAX))),
    namedtype.DefaultedNamedType('prf', algid_hmacWithSHA1)
)


# RC2 CBC algorithm parameter

class RC2_CBC_Parameter(univ.Sequence):
    pass

RC2_CBC_Parameter.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('rc2ParameterVersion', univ.Integer()),
    namedtype.NamedType('iv', univ.OctetString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(8, 8)))
)


# RC5 CBC algorithm parameter

class RC5_CBC_Parameters(univ.Sequence):
    pass

RC5_CBC_Parameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version',
        univ.Integer(namedValues=namedval.NamedValues(('v1_0', 16))).subtype(
            subtypeSpec=constraint.SingleValueConstraint(16))),
    namedtype.NamedType('rounds',
        univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(8, 127))),
    namedtype.NamedType('blockSizeInBits',
        univ.Integer().subtype(subtypeSpec=constraint.SingleValueConstraint(64, 128))),
    namedtype.OptionalNamedType('iv', univ.OctetString())
)


# Initialization Vector for AES: OCTET STRING (SIZE(16))

class AES_IV(univ.OctetString):
    pass

AES_IV.subtypeSpec = constraint.ValueSizeConstraint(16, 16)


# Initialization Vector for DES: OCTET STRING (SIZE(8))

class DES_IV(univ.OctetString):
    pass

DES_IV.subtypeSpec = constraint.ValueSizeConstraint(8, 8)


# Update the Algorithm Identifier map

_algorithmIdentifierMapUpdate = {
    # PBKDF2-PRFs
    id_hmacWithSHA1: univ.Null(),
    id_hmacWithSHA224: univ.Null(),
    id_hmacWithSHA256: univ.Null(),
    id_hmacWithSHA384: univ.Null(),
    id_hmacWithSHA512: univ.Null(),
    id_hmacWithSHA512_224: univ.Null(),
    id_hmacWithSHA512_256: univ.Null(),
    # PBES1Algorithms
    pbeWithMD2AndDES_CBC: PBEParameter(),
    pbeWithMD2AndRC2_CBC: PBEParameter(),
    pbeWithMD5AndDES_CBC: PBEParameter(),
    pbeWithMD5AndRC2_CBC: PBEParameter(),
    pbeWithSHA1AndDES_CBC: PBEParameter(),
    pbeWithSHA1AndRC2_CBC: PBEParameter(),
    # PBES2Algorithms
    id_PBES2: PBES2_params(),
    # PBES2-KDFs
    id_PBKDF2: PBKDF2_params(),
    # PBMAC1Algorithms
    id_PBMAC1: PBMAC1_params(),
    # SupportingAlgorithms
    desCBC: DES_IV(),
    des_EDE3_CBC: DES_IV(),
    rc2CBC: RC2_CBC_Parameter(),
    rc5_CBC_PAD: RC5_CBC_Parameters(),
    aes128_CBC_PAD: AES_IV(),
    aes192_CBC_PAD: AES_IV(),
    aes256_CBC_PAD: AES_IV(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
