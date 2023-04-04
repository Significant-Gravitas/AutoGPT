#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with a very small amount of assistance from
# asn1ate v.0.6.0.
# Modified by Russ Housley to add maps for opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Additional Algorithms and Identifiers for RSA Cryptography
# for use in Certificates and CRLs
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc4055.txt
#
from pyasn1.type import namedtype
from pyasn1.type import tag
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


id_sha1 = _OID(1, 3, 14, 3, 2, 26)

id_sha256 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 1)

id_sha384 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 2)

id_sha512 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 3)

id_sha224 = _OID(2, 16, 840, 1, 101, 3, 4, 2, 4)

rsaEncryption = _OID(1, 2, 840, 113549, 1, 1, 1)

id_mgf1 = _OID(1, 2, 840, 113549, 1, 1, 8)

id_RSAES_OAEP = _OID(1, 2, 840, 113549, 1, 1, 7)

id_pSpecified = _OID(1, 2, 840, 113549, 1, 1, 9)

id_RSASSA_PSS = _OID(1, 2, 840, 113549, 1, 1, 10)

sha256WithRSAEncryption = _OID(1, 2, 840, 113549, 1, 1, 11)

sha384WithRSAEncryption = _OID(1, 2, 840, 113549, 1, 1, 12)

sha512WithRSAEncryption = _OID(1, 2, 840, 113549, 1, 1, 13)

sha224WithRSAEncryption = _OID(1, 2, 840, 113549, 1, 1, 14)

sha1Identifier = rfc5280.AlgorithmIdentifier()
sha1Identifier['algorithm'] = id_sha1
sha1Identifier['parameters'] = univ.Null("")

sha224Identifier = rfc5280.AlgorithmIdentifier()
sha224Identifier['algorithm'] = id_sha224
sha224Identifier['parameters'] = univ.Null("")

sha256Identifier = rfc5280.AlgorithmIdentifier()
sha256Identifier['algorithm'] = id_sha256
sha256Identifier['parameters'] = univ.Null("")

sha384Identifier = rfc5280.AlgorithmIdentifier()
sha384Identifier['algorithm'] = id_sha384
sha384Identifier['parameters'] = univ.Null("")

sha512Identifier = rfc5280.AlgorithmIdentifier()
sha512Identifier['algorithm'] = id_sha512
sha512Identifier['parameters'] = univ.Null("")

mgf1SHA1Identifier = rfc5280.AlgorithmIdentifier()
mgf1SHA1Identifier['algorithm'] = id_mgf1
mgf1SHA1Identifier['parameters'] = sha1Identifier

mgf1SHA224Identifier = rfc5280.AlgorithmIdentifier()
mgf1SHA224Identifier['algorithm'] = id_mgf1
mgf1SHA224Identifier['parameters'] = sha224Identifier

mgf1SHA256Identifier = rfc5280.AlgorithmIdentifier()
mgf1SHA256Identifier['algorithm'] = id_mgf1
mgf1SHA256Identifier['parameters'] = sha256Identifier

mgf1SHA384Identifier = rfc5280.AlgorithmIdentifier()
mgf1SHA384Identifier['algorithm'] = id_mgf1
mgf1SHA384Identifier['parameters'] = sha384Identifier

mgf1SHA512Identifier = rfc5280.AlgorithmIdentifier()
mgf1SHA512Identifier['algorithm'] = id_mgf1
mgf1SHA512Identifier['parameters'] = sha512Identifier

pSpecifiedEmptyIdentifier = rfc5280.AlgorithmIdentifier()
pSpecifiedEmptyIdentifier['algorithm'] = id_pSpecified
pSpecifiedEmptyIdentifier['parameters'] = univ.OctetString(value='')


class RSAPublicKey(univ.Sequence):
    pass

RSAPublicKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('modulus', univ.Integer()),
    namedtype.NamedType('publicExponent', univ.Integer())
)


class HashAlgorithm(rfc5280.AlgorithmIdentifier):
    pass


class MaskGenAlgorithm(rfc5280.AlgorithmIdentifier):
    pass


class RSAES_OAEP_params(univ.Sequence):
    pass

RSAES_OAEP_params.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('hashFunc', rfc5280.AlgorithmIdentifier().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('maskGenFunc', rfc5280.AlgorithmIdentifier().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
    namedtype.OptionalNamedType('pSourceFunc', rfc5280.AlgorithmIdentifier().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
)

rSAES_OAEP_Default_Params = RSAES_OAEP_params()

rSAES_OAEP_Default_Identifier = rfc5280.AlgorithmIdentifier()
rSAES_OAEP_Default_Identifier['algorithm'] = id_RSAES_OAEP
rSAES_OAEP_Default_Identifier['parameters'] = rSAES_OAEP_Default_Params

rSAES_OAEP_SHA224_Params = RSAES_OAEP_params()
rSAES_OAEP_SHA224_Params['hashFunc'] = sha224Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSAES_OAEP_SHA224_Params['maskGenFunc'] = mgf1SHA224Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSAES_OAEP_SHA224_Identifier = rfc5280.AlgorithmIdentifier()
rSAES_OAEP_SHA224_Identifier['algorithm'] = id_RSAES_OAEP
rSAES_OAEP_SHA224_Identifier['parameters'] = rSAES_OAEP_SHA224_Params

rSAES_OAEP_SHA256_Params = RSAES_OAEP_params()
rSAES_OAEP_SHA256_Params['hashFunc'] = sha256Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSAES_OAEP_SHA256_Params['maskGenFunc'] = mgf1SHA256Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSAES_OAEP_SHA256_Identifier = rfc5280.AlgorithmIdentifier()
rSAES_OAEP_SHA256_Identifier['algorithm'] = id_RSAES_OAEP
rSAES_OAEP_SHA256_Identifier['parameters'] = rSAES_OAEP_SHA256_Params

rSAES_OAEP_SHA384_Params = RSAES_OAEP_params()
rSAES_OAEP_SHA384_Params['hashFunc'] = sha384Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSAES_OAEP_SHA384_Params['maskGenFunc'] = mgf1SHA384Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSAES_OAEP_SHA384_Identifier = rfc5280.AlgorithmIdentifier()
rSAES_OAEP_SHA384_Identifier['algorithm'] = id_RSAES_OAEP
rSAES_OAEP_SHA384_Identifier['parameters'] = rSAES_OAEP_SHA384_Params

rSAES_OAEP_SHA512_Params = RSAES_OAEP_params()
rSAES_OAEP_SHA512_Params['hashFunc'] = sha512Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSAES_OAEP_SHA512_Params['maskGenFunc'] = mgf1SHA512Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSAES_OAEP_SHA512_Identifier = rfc5280.AlgorithmIdentifier()
rSAES_OAEP_SHA512_Identifier['algorithm'] = id_RSAES_OAEP
rSAES_OAEP_SHA512_Identifier['parameters'] = rSAES_OAEP_SHA512_Params


class RSASSA_PSS_params(univ.Sequence):
    pass

RSASSA_PSS_params.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('hashAlgorithm', rfc5280.AlgorithmIdentifier().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('maskGenAlgorithm', rfc5280.AlgorithmIdentifier().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
    namedtype.DefaultedNamedType('saltLength', univ.Integer(value=20).subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.DefaultedNamedType('trailerField', univ.Integer(value=1).subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)

rSASSA_PSS_Default_Params = RSASSA_PSS_params()

rSASSA_PSS_Default_Identifier = rfc5280.AlgorithmIdentifier()
rSASSA_PSS_Default_Identifier['algorithm'] = id_RSASSA_PSS
rSASSA_PSS_Default_Identifier['parameters'] = rSASSA_PSS_Default_Params

rSASSA_PSS_SHA224_Params = RSASSA_PSS_params()
rSASSA_PSS_SHA224_Params['hashAlgorithm'] = sha224Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSASSA_PSS_SHA224_Params['maskGenAlgorithm'] = mgf1SHA224Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSASSA_PSS_SHA224_Identifier = rfc5280.AlgorithmIdentifier()
rSASSA_PSS_SHA224_Identifier['algorithm'] = id_RSASSA_PSS
rSASSA_PSS_SHA224_Identifier['parameters'] = rSASSA_PSS_SHA224_Params

rSASSA_PSS_SHA256_Params = RSASSA_PSS_params()
rSASSA_PSS_SHA256_Params['hashAlgorithm'] = sha256Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSASSA_PSS_SHA256_Params['maskGenAlgorithm'] = mgf1SHA256Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSASSA_PSS_SHA256_Identifier = rfc5280.AlgorithmIdentifier()
rSASSA_PSS_SHA256_Identifier['algorithm'] = id_RSASSA_PSS
rSASSA_PSS_SHA256_Identifier['parameters'] = rSASSA_PSS_SHA256_Params

rSASSA_PSS_SHA384_Params = RSASSA_PSS_params()
rSASSA_PSS_SHA384_Params['hashAlgorithm'] = sha384Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSASSA_PSS_SHA384_Params['maskGenAlgorithm'] = mgf1SHA384Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSASSA_PSS_SHA384_Identifier = rfc5280.AlgorithmIdentifier()
rSASSA_PSS_SHA384_Identifier['algorithm'] = id_RSASSA_PSS
rSASSA_PSS_SHA384_Identifier['parameters'] = rSASSA_PSS_SHA384_Params

rSASSA_PSS_SHA512_Params = RSASSA_PSS_params()
rSASSA_PSS_SHA512_Params['hashAlgorithm'] = sha512Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0), cloneValueFlag=True)
rSASSA_PSS_SHA512_Params['maskGenAlgorithm'] = mgf1SHA512Identifier.subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1), cloneValueFlag=True)

rSASSA_PSS_SHA512_Identifier = rfc5280.AlgorithmIdentifier()
rSASSA_PSS_SHA512_Identifier['algorithm'] = id_RSASSA_PSS
rSASSA_PSS_SHA512_Identifier['parameters'] = rSASSA_PSS_SHA512_Params


# Update the Algorithm Identifier map

_algorithmIdentifierMapUpdate = {
    id_sha1: univ.Null(),
    id_sha224: univ.Null(),
    id_sha256: univ.Null(),
    id_sha384: univ.Null(),
    id_sha512: univ.Null(),
    id_mgf1: rfc5280.AlgorithmIdentifier(),
    id_pSpecified: univ.OctetString(),
    id_RSAES_OAEP: RSAES_OAEP_params(),
    id_RSASSA_PSS: RSASSA_PSS_params(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
