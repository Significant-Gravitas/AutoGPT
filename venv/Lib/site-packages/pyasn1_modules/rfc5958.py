#
# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley.
# Modified by Russ Housley to add a map for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Asymmetric Key Packages, which is essentially version 2 of
#   the PrivateKeyInfo structure in PKCS#8 in RFC 5208
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5958.txt

from pyasn1.type import univ, constraint, namedtype, namedval, tag

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652


MAX = float('inf')


class KeyEncryptionAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class PrivateKeyAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class EncryptedData(univ.OctetString):
    pass


class EncryptedPrivateKeyInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('encryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()),
        namedtype.NamedType('encryptedData', EncryptedData())
    )


class Version(univ.Integer):
    namedValues = namedval.NamedValues(('v1', 0), ('v2', 1))


class PrivateKey(univ.OctetString):
    pass


class Attributes(univ.SetOf):
    componentType = rfc5652.Attribute()


class PublicKey(univ.BitString):
   pass


# OneAsymmetricKey is essentially version 2 of PrivateKeyInfo.
# If publicKey is present, then the version must be v2;
# otherwise, the version should be v1.

class OneAsymmetricKey(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('privateKeyAlgorithm', PrivateKeyAlgorithmIdentifier()),
        namedtype.NamedType('privateKey', PrivateKey()),
        namedtype.OptionalNamedType('attributes', Attributes().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('publicKey', PublicKey().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class PrivateKeyInfo(OneAsymmetricKey):
    pass


# The CMS AsymmetricKeyPackage Content Type

id_ct_KP_aKeyPackage = univ.ObjectIdentifier('2.16.840.1.101.2.1.2.78.5')

class AsymmetricKeyPackage(univ.SequenceOf):
    pass

AsymmetricKeyPackage.componentType = OneAsymmetricKey()
AsymmetricKeyPackage.sizeSpec=constraint.ValueSizeConstraint(1, MAX)
    

# Map of Content Type OIDs to Content Types is added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_KP_aKeyPackage: AsymmetricKeyPackage(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
