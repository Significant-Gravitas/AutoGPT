# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Trust Anchor Format
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5914.txt

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280


MAX = float('inf')

Certificate = rfc5280.Certificate

Name = rfc5280.Name

Extensions = rfc5280.Extensions

SubjectPublicKeyInfo = rfc5280.SubjectPublicKeyInfo

TBSCertificate = rfc5280.TBSCertificate

CertificatePolicies = rfc5280.CertificatePolicies

KeyIdentifier = rfc5280.KeyIdentifier

NameConstraints = rfc5280.NameConstraints


class CertPolicyFlags(univ.BitString):
    pass

CertPolicyFlags.namedValues = namedval.NamedValues(
    ('inhibitPolicyMapping', 0),
    ('requireExplicitPolicy', 1),
    ('inhibitAnyPolicy', 2)
)


class CertPathControls(univ.Sequence):
    pass

CertPathControls.componentType = namedtype.NamedTypes(
    namedtype.NamedType('taName', Name()),
    namedtype.OptionalNamedType('certificate', Certificate().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('policySet', CertificatePolicies().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('policyFlags', CertPolicyFlags().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('nameConstr', NameConstraints().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
    namedtype.OptionalNamedType('pathLenConstraint', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4)))
)


class TrustAnchorTitle(char.UTF8String):
    pass

TrustAnchorTitle.subtypeSpec = constraint.ValueSizeConstraint(1, 64)


class TrustAnchorInfoVersion(univ.Integer):
    pass

TrustAnchorInfoVersion.namedValues = namedval.NamedValues(
    ('v1', 1)
)


class TrustAnchorInfo(univ.Sequence):
    pass

TrustAnchorInfo.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', TrustAnchorInfoVersion().subtype(value='v1')),
    namedtype.NamedType('pubKey', SubjectPublicKeyInfo()),
    namedtype.NamedType('keyId', KeyIdentifier()),
    namedtype.OptionalNamedType('taTitle', TrustAnchorTitle()),
    namedtype.OptionalNamedType('certPath', CertPathControls()),
    namedtype.OptionalNamedType('exts', Extensions().subtype(explicitTag=tag.Tag(
        tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('taTitleLangTag', char.UTF8String().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)


class TrustAnchorChoice(univ.Choice):
    pass

TrustAnchorChoice.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certificate', Certificate()),
    namedtype.NamedType('tbsCert', TBSCertificate().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('taInfo', TrustAnchorInfo().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
)


id_ct_trustAnchorList = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.34')

class TrustAnchorList(univ.SequenceOf):
    pass

TrustAnchorList.componentType = TrustAnchorChoice()
TrustAnchorList.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)
