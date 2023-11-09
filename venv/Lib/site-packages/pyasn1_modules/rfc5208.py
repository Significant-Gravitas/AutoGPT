#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS#8 syntax
#
# ASN.1 source from:
# http://tools.ietf.org/html/rfc5208
#
# Sample captures could be obtained with "openssl pkcs8 -topk8" command
#
from pyasn1_modules import rfc2251
from pyasn1_modules.rfc2459 import *


class KeyEncryptionAlgorithms(AlgorithmIdentifier):
    pass


class PrivateKeyAlgorithms(AlgorithmIdentifier):
    pass


class EncryptedData(univ.OctetString):
    pass


class EncryptedPrivateKeyInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('encryptionAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('encryptedData', EncryptedData())
    )


class PrivateKey(univ.OctetString):
    pass


class Attributes(univ.SetOf):
    componentType = rfc2251.Attribute()


class Version(univ.Integer):
    namedValues = namedval.NamedValues(('v1', 0), ('v2', 1))


class PrivateKeyInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('privateKeyAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('privateKey', PrivateKey()),
        namedtype.OptionalNamedType('attributes', Attributes().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
    )
