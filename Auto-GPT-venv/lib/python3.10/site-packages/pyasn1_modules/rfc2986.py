# coding: utf-8
#
# This file is part of pyasn1-modules software.
#
# Created by Joel Johnson with asn1ate tool.
# Modified by Russ Housley to add support for opentypes by importing
#   definitions from rfc5280 so that the same maps are used.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS #10: Certification Request Syntax Specification
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc2986.txt
#
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


AttributeType = rfc5280.AttributeType

AttributeValue = rfc5280.AttributeValue

AttributeTypeAndValue = rfc5280.AttributeTypeAndValue

Attribute = rfc5280.Attribute

RelativeDistinguishedName = rfc5280.RelativeDistinguishedName

RDNSequence = rfc5280.RDNSequence

Name = rfc5280.Name

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier

SubjectPublicKeyInfo = rfc5280.SubjectPublicKeyInfo


class Attributes(univ.SetOf):
    pass


Attributes.componentType = Attribute()


class CertificationRequestInfo(univ.Sequence):
    pass


CertificationRequestInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', univ.Integer()),
    namedtype.NamedType('subject', Name()),
    namedtype.NamedType('subjectPKInfo', SubjectPublicKeyInfo()),
    namedtype.NamedType('attributes',
                        Attributes().subtype(implicitTag=tag.Tag(
                            tag.tagClassContext, tag.tagFormatSimple, 0))
    )
)


class CertificationRequest(univ.Sequence):
    pass


CertificationRequest.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certificationRequestInfo', CertificationRequestInfo()),
    namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
    namedtype.NamedType('signature', univ.BitString())
)
