#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS#10 syntax
#
# ASN.1 source from:
# http://tools.ietf.org/html/rfc2314
#
# Sample captures could be obtained with "openssl req" command
#
from pyasn1_modules.rfc2459 import *


class Attributes(univ.SetOf):
    componentType = Attribute()


class Version(univ.Integer):
    pass


class CertificationRequestInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('subject', Name()),
        namedtype.NamedType('subjectPublicKeyInfo', SubjectPublicKeyInfo()),
        namedtype.NamedType('attributes',
                            Attributes().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
    )


class Signature(univ.BitString):
    pass


class SignatureAlgorithmIdentifier(AlgorithmIdentifier):
    pass


class CertificationRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certificationRequestInfo', CertificationRequestInfo()),
        namedtype.NamedType('signatureAlgorithm', SignatureAlgorithmIdentifier()),
        namedtype.NamedType('signature', Signature())
    )
