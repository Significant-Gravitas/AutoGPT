#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS #1: RSA Cryptography Specifications Version 2.2
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8017.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ

from pyasn1_modules import rfc2437
from pyasn1_modules import rfc3447
from pyasn1_modules import rfc4055
from pyasn1_modules import rfc5280

MAX = float('inf')


# Import Algorithm Identifier from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier

class DigestAlgorithm(AlgorithmIdentifier):
    pass

class HashAlgorithm(AlgorithmIdentifier):
    pass

class MaskGenAlgorithm(AlgorithmIdentifier):
    pass

class PSourceAlgorithm(AlgorithmIdentifier):
    pass


# Object identifiers from NIST SHA2

hashAlgs = univ.ObjectIdentifier('2.16.840.1.101.3.4.2')
id_sha256 = rfc4055.id_sha256
id_sha384 = rfc4055.id_sha384
id_sha512 = rfc4055.id_sha512
id_sha224 = rfc4055.id_sha224
id_sha512_224 = hashAlgs + (5, )
id_sha512_256 = hashAlgs + (6, )


# Basic object identifiers

pkcs_1 = univ.ObjectIdentifier('1.2.840.113549.1.1')
rsaEncryption = rfc2437.rsaEncryption
id_RSAES_OAEP = rfc2437.id_RSAES_OAEP
id_pSpecified = rfc2437.id_pSpecified
id_RSASSA_PSS = rfc4055.id_RSASSA_PSS
md2WithRSAEncryption = rfc2437.md2WithRSAEncryption
md5WithRSAEncryption = rfc2437.md5WithRSAEncryption
sha1WithRSAEncryption = rfc2437.sha1WithRSAEncryption
sha224WithRSAEncryption = rfc4055.sha224WithRSAEncryption
sha256WithRSAEncryption = rfc4055.sha256WithRSAEncryption
sha384WithRSAEncryption = rfc4055.sha384WithRSAEncryption
sha512WithRSAEncryption = rfc4055.sha512WithRSAEncryption
sha512_224WithRSAEncryption = pkcs_1 + (15, )
sha512_256WithRSAEncryption = pkcs_1 + (16, )
id_sha1 = rfc2437.id_sha1
id_md2 = univ.ObjectIdentifier('1.2.840.113549.2.2')
id_md5 = univ.ObjectIdentifier('1.2.840.113549.2.5')
id_mgf1 = rfc2437.id_mgf1


# Default parameter values

sha1 = rfc4055.sha1Identifier
SHA1Parameters = univ.Null("")

mgf1SHA1 = rfc4055.mgf1SHA1Identifier

class EncodingParameters(univ.OctetString):
    subtypeSpec = constraint.ValueSizeConstraint(0, MAX)

pSpecifiedEmpty = rfc4055.pSpecifiedEmptyIdentifier

emptyString = EncodingParameters(value='')


# Main structures

class Version(univ.Integer):
    namedValues = namedval.NamedValues(
        ('two-prime', 0),
        ('multi', 1)
    )

class TrailerField(univ.Integer):
    namedValues = namedval.NamedValues(
       ('trailerFieldBC', 1)
    )

RSAPublicKey = rfc2437.RSAPublicKey

OtherPrimeInfo = rfc3447.OtherPrimeInfo
OtherPrimeInfos = rfc3447.OtherPrimeInfos
RSAPrivateKey = rfc3447.RSAPrivateKey

RSAES_OAEP_params = rfc4055.RSAES_OAEP_params
rSAES_OAEP_Default_Identifier = rfc4055.rSAES_OAEP_Default_Identifier

RSASSA_PSS_params = rfc4055.RSASSA_PSS_params
rSASSA_PSS_Default_Identifier = rfc4055.rSASSA_PSS_Default_Identifier


# Syntax for the EMSA-PKCS1-v1_5 hash identifier

class DigestInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('digestAlgorithm', DigestAlgorithm()),
        namedtype.NamedType('digest', univ.OctetString())
    )


# Update the Algorithm Identifier map

_algorithmIdentifierMapUpdate = {
    id_sha1: univ.Null(),
    id_sha224: univ.Null(),
    id_sha256: univ.Null(),
    id_sha384: univ.Null(),
    id_sha512: univ.Null(),
    id_sha512_224: univ.Null(),
    id_sha512_256: univ.Null(),
    id_mgf1: AlgorithmIdentifier(),
    id_pSpecified: univ.OctetString(),
    id_RSAES_OAEP: RSAES_OAEP_params(),
    id_RSASSA_PSS: RSASSA_PSS_params(),
    md2WithRSAEncryption: univ.Null(),
    md5WithRSAEncryption: univ.Null(),
    sha1WithRSAEncryption: univ.Null(),
    sha224WithRSAEncryption: univ.Null(),
    sha256WithRSAEncryption: univ.Null(),
    sha384WithRSAEncryption: univ.Null(),
    sha512WithRSAEncryption: univ.Null(),
    sha512_224WithRSAEncryption: univ.Null(),
    sha512_256WithRSAEncryption: univ.Null(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
