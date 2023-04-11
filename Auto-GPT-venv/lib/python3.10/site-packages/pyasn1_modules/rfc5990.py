#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Use of the RSA-KEM Key Transport Algorithm in the CMS
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5990.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

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


# Imports from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier


# Useful types and definitions

class NullParms(univ.Null):
    pass


# Object identifier arcs

is18033_2 = _OID(1, 0, 18033, 2)

nistAlgorithm = _OID(2, 16, 840, 1, 101, 3, 4)

pkcs_1 = _OID(1, 2, 840, 113549, 1, 1)

x9_44 = _OID(1, 3, 133, 16, 840, 9, 44)

x9_44_components = _OID(x9_44, 1)


# Types for algorithm identifiers

class Camellia_KeyWrappingScheme(AlgorithmIdentifier):
    pass

class DataEncapsulationMechanism(AlgorithmIdentifier):
    pass

class KDF2_HashFunction(AlgorithmIdentifier):
    pass

class KDF3_HashFunction(AlgorithmIdentifier):
    pass

class KeyDerivationFunction(AlgorithmIdentifier):
    pass

class KeyEncapsulationMechanism(AlgorithmIdentifier):
    pass

class X9_SymmetricKeyWrappingScheme(AlgorithmIdentifier):
    pass


# RSA-KEM Key Transport Algorithm

id_rsa_kem = _OID(1, 2, 840, 113549, 1, 9, 16, 3, 14)


class GenericHybridParameters(univ.Sequence):
    pass

GenericHybridParameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('kem', KeyEncapsulationMechanism()),
    namedtype.NamedType('dem', DataEncapsulationMechanism())
)


rsa_kem = AlgorithmIdentifier()
rsa_kem['algorithm'] = id_rsa_kem
rsa_kem['parameters'] = GenericHybridParameters()


# KEM-RSA Key Encapsulation Mechanism

id_kem_rsa = _OID(is18033_2, 2, 4)


class KeyLength(univ.Integer):
    pass

KeyLength.subtypeSpec = constraint.ValueRangeConstraint(1, MAX)


class RsaKemParameters(univ.Sequence):
    pass

RsaKemParameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyDerivationFunction', KeyDerivationFunction()),
    namedtype.NamedType('keyLength', KeyLength())
)


kem_rsa = AlgorithmIdentifier()
kem_rsa['algorithm'] = id_kem_rsa
kem_rsa['parameters'] = RsaKemParameters()


# Key Derivation Functions

id_kdf_kdf2 = _OID(x9_44_components, 1)

id_kdf_kdf3 = _OID(x9_44_components, 2)


kdf2 = AlgorithmIdentifier()
kdf2['algorithm'] = id_kdf_kdf2
kdf2['parameters'] = KDF2_HashFunction()

kdf3 = AlgorithmIdentifier()
kdf3['algorithm'] = id_kdf_kdf3
kdf3['parameters'] = KDF3_HashFunction()


# Hash Functions

id_sha1 = _OID(1, 3, 14, 3, 2, 26)

id_sha224 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 4)

id_sha256 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 1)

id_sha384 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 2)

id_sha512 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 3)


sha1 = AlgorithmIdentifier()
sha1['algorithm'] = id_sha1
sha1['parameters'] = univ.Null("")

sha224 = AlgorithmIdentifier()
sha224['algorithm'] = id_sha224
sha224['parameters'] = univ.Null("")

sha256 = AlgorithmIdentifier()
sha256['algorithm'] = id_sha256
sha256['parameters'] = univ.Null("")

sha384 = AlgorithmIdentifier()
sha384['algorithm'] = id_sha384
sha384['parameters'] = univ.Null("")

sha512 = AlgorithmIdentifier()
sha512['algorithm'] = id_sha512
sha512['parameters'] = univ.Null("")


# Symmetric Key-Wrapping Schemes

id_aes128_Wrap = _OID(nistAlgorithm, 1, 5)

id_aes192_Wrap = _OID(nistAlgorithm, 1, 25)

id_aes256_Wrap = _OID(nistAlgorithm, 1, 45)

id_alg_CMS3DESwrap = _OID(1, 2, 840, 113549, 1, 9, 16, 3, 6)

id_camellia128_Wrap = _OID(1, 2, 392, 200011, 61, 1, 1, 3, 2)

id_camellia192_Wrap = _OID(1, 2, 392, 200011, 61, 1, 1, 3, 3)

id_camellia256_Wrap = _OID(1, 2, 392, 200011, 61, 1, 1, 3, 4)


aes128_Wrap = AlgorithmIdentifier()
aes128_Wrap['algorithm'] = id_aes128_Wrap
# aes128_Wrap['parameters'] are absent

aes192_Wrap = AlgorithmIdentifier()
aes192_Wrap['algorithm'] = id_aes128_Wrap
# aes192_Wrap['parameters'] are absent

aes256_Wrap = AlgorithmIdentifier()
aes256_Wrap['algorithm'] = id_sha256
# aes256_Wrap['parameters'] are absent

tdes_Wrap = AlgorithmIdentifier()
tdes_Wrap['algorithm'] = id_alg_CMS3DESwrap
tdes_Wrap['parameters'] = univ.Null("")

camellia128_Wrap = AlgorithmIdentifier()
camellia128_Wrap['algorithm'] = id_camellia128_Wrap
# camellia128_Wrap['parameters'] are absent

camellia192_Wrap = AlgorithmIdentifier()
camellia192_Wrap['algorithm'] = id_camellia192_Wrap
# camellia192_Wrap['parameters'] are absent

camellia256_Wrap = AlgorithmIdentifier()
camellia256_Wrap['algorithm'] = id_camellia256_Wrap
# camellia256_Wrap['parameters'] are absent


# Update the Algorithm Identifier map in rfc5280.py.
# Note that the ones that must not have parameters are not added to the map.

_algorithmIdentifierMapUpdate = {
    id_rsa_kem: GenericHybridParameters(),
    id_kem_rsa: RsaKemParameters(),
    id_kdf_kdf2: KDF2_HashFunction(),
    id_kdf_kdf3: KDF3_HashFunction(),
    id_sha1: univ.Null(),
    id_sha224: univ.Null(),
    id_sha256: univ.Null(),
    id_sha384: univ.Null(),
    id_sha512: univ.Null(),
    id_alg_CMS3DESwrap: univ.Null(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
