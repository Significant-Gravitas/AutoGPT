#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# An Internet Attribute Certificate Profile for Authorization
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5755.txt
# https://www.rfc-editor.org/rfc/rfc5912.txt (see Section 13)
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652

MAX = float('inf')

# Map for Security Category type to value

securityCategoryMap = { }


# Imports from RFC 5652

ContentInfo = rfc5652.ContentInfo


# Imports from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier

Attribute = rfc5280.Attribute

AuthorityInfoAccessSyntax = rfc5280.AuthorityInfoAccessSyntax

AuthorityKeyIdentifier = rfc5280.AuthorityKeyIdentifier

CertificateSerialNumber = rfc5280.CertificateSerialNumber

CRLDistributionPoints = rfc5280.CRLDistributionPoints

Extensions = rfc5280.Extensions

Extension = rfc5280.Extension

GeneralNames = rfc5280.GeneralNames

GeneralName = rfc5280.GeneralName

UniqueIdentifier = rfc5280.UniqueIdentifier


# Object Identifier arcs

id_pkix = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, ))

id_pe = id_pkix + (1, )

id_kp = id_pkix + (3, )

id_aca = id_pkix + (10, )

id_ad = id_pkix + (48, )

id_at = univ.ObjectIdentifier((2, 5, 4, ))

id_ce = univ.ObjectIdentifier((2, 5, 29, ))


# Attribute Certificate

class AttCertVersion(univ.Integer):
    namedValues = namedval.NamedValues(
        ('v2', 1)
    )


class IssuerSerial(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('issuer', GeneralNames()),
        namedtype.NamedType('serial', CertificateSerialNumber()),
        namedtype.OptionalNamedType('issuerUID', UniqueIdentifier())
    )


class ObjectDigestInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('digestedObjectType',
            univ.Enumerated(namedValues=namedval.NamedValues(
                ('publicKey', 0),
                ('publicKeyCert', 1),
                ('otherObjectTypes', 2)))),
        namedtype.OptionalNamedType('otherObjectTypeID',
            univ.ObjectIdentifier()),
        namedtype.NamedType('digestAlgorithm',
            AlgorithmIdentifier()),
        namedtype.NamedType('objectDigest',
            univ.BitString())
    )


class Holder(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('baseCertificateID',
            IssuerSerial().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('entityName',
            GeneralNames().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('objectDigestInfo',
            ObjectDigestInfo().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatConstructed, 2)))
)


class V2Form(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('issuerName',
            GeneralNames()),
        namedtype.OptionalNamedType('baseCertificateID',
            IssuerSerial().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('objectDigestInfo',
            ObjectDigestInfo().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class AttCertIssuer(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('v1Form', GeneralNames()),
        namedtype.NamedType('v2Form', V2Form().subtype(implicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatConstructed, 0)))
    )


class AttCertValidityPeriod(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('notBeforeTime', useful.GeneralizedTime()),
        namedtype.NamedType('notAfterTime', useful.GeneralizedTime())
    )


class AttributeCertificateInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version',
            AttCertVersion()),
        namedtype.NamedType('holder',
            Holder()),
        namedtype.NamedType('issuer',
            AttCertIssuer()),
        namedtype.NamedType('signature',
            AlgorithmIdentifier()),
        namedtype.NamedType('serialNumber',
            CertificateSerialNumber()),
        namedtype.NamedType('attrCertValidityPeriod',
            AttCertValidityPeriod()),
        namedtype.NamedType('attributes',
            univ.SequenceOf(componentType=Attribute())),
        namedtype.OptionalNamedType('issuerUniqueID',
            UniqueIdentifier()),
        namedtype.OptionalNamedType('extensions',
            Extensions())
    )


class AttributeCertificate(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('acinfo', AttributeCertificateInfo()),
        namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('signatureValue', univ.BitString())
    )


# Attribute Certificate Extensions

id_pe_ac_auditIdentity = id_pe + (4, )

id_ce_noRevAvail = id_ce + (56, )

id_ce_targetInformation = id_ce + (55, )


class TargetCert(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('targetCertificate', IssuerSerial()),
        namedtype.OptionalNamedType('targetName', GeneralName()),
        namedtype.OptionalNamedType('certDigestInfo', ObjectDigestInfo())
    )


class Target(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('targetName',
            GeneralName().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('targetGroup',
            GeneralName().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('targetCert',
            TargetCert().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatConstructed, 2)))
    )


class Targets(univ.SequenceOf):
    componentType = Target()


id_pe_ac_proxying = id_pe + (10, )


class ProxyInfo(univ.SequenceOf):
    componentType = Targets()


id_pe_aaControls = id_pe + (6, )


class AttrSpec(univ.SequenceOf):
    componentType = univ.ObjectIdentifier()


class AAControls(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('pathLenConstraint',
            univ.Integer().subtype(
                subtypeSpec=constraint.ValueRangeConstraint(0, MAX))),
        namedtype.OptionalNamedType('permittedAttrs',
            AttrSpec().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('excludedAttrs',
            AttrSpec().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.DefaultedNamedType('permitUnSpecified',
            univ.Boolean().subtype(value=1))
    )


# Attribute Certificate Attributes

id_aca_authenticationInfo = id_aca + (1, )


id_aca_accessIdentity = id_aca + (2, )


class SvceAuthInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('service', GeneralName()),
        namedtype.NamedType('ident', GeneralName()),
        namedtype.OptionalNamedType('authInfo', univ.OctetString())
    )


id_aca_chargingIdentity = id_aca + (3, )


id_aca_group = id_aca + (4, )


class IetfAttrSyntax(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('policyAuthority',
            GeneralNames().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('values', univ.SequenceOf(
            componentType=univ.Choice(componentType=namedtype.NamedTypes(
                namedtype.NamedType('octets', univ.OctetString()),
                namedtype.NamedType('oid', univ.ObjectIdentifier()),
                namedtype.NamedType('string', char.UTF8String())
            ))
        ))
    )


id_at_role = id_at + (72,)


class RoleSyntax(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('roleAuthority',
            GeneralNames().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('roleName',
            GeneralName().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class ClassList(univ.BitString):
    namedValues = namedval.NamedValues(
        ('unmarked', 0),
        ('unclassified', 1),
        ('restricted', 2),
        ('confidential', 3),
        ('secret', 4),
        ('topSecret', 5)
    )


class SecurityCategory(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type',
            univ.ObjectIdentifier().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('value',
            univ.Any().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1)),
            openType=opentype.OpenType('type', securityCategoryMap))
    )


id_at_clearance = univ.ObjectIdentifier((2, 5, 4, 55, ))


class Clearance(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('policyId',
            univ.ObjectIdentifier()),
        namedtype.DefaultedNamedType('classList',
            ClassList().subtype(value='unclassified')),
        namedtype.OptionalNamedType('securityCategories',
            univ.SetOf(componentType=SecurityCategory()))
    )


id_at_clearance_rfc3281 = univ.ObjectIdentifier((2, 5, 1, 5, 55, ))


class Clearance_rfc3281(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('policyId',
            univ.ObjectIdentifier().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.DefaultedNamedType('classList',
            ClassList().subtype(implicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1)).subtype(
                    value='unclassified')),
        namedtype.OptionalNamedType('securityCategories',
            univ.SetOf(componentType=SecurityCategory()).subtype(
                implicitTag=tag.Tag(
                    tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


id_aca_encAttrs = id_aca + (6, )


class ACClearAttrs(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('acIssuer', GeneralName()),
        namedtype.NamedType('acSerial', univ.Integer()),
        namedtype.NamedType('attrs', univ.SequenceOf(componentType=Attribute()))
    )


# Map of Certificate Extension OIDs to Extensions added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_pe_ac_auditIdentity: univ.OctetString(),
    id_ce_noRevAvail: univ.Null(),
    id_ce_targetInformation: Targets(),
    id_pe_ac_proxying: ProxyInfo(),
    id_pe_aaControls: AAControls(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)


# Map of AttributeType OIDs to AttributeValue added to the
# ones that are in rfc5280.py

_certificateAttributesMapUpdate = {
    id_aca_authenticationInfo: SvceAuthInfo(),
    id_aca_accessIdentity: SvceAuthInfo(),
    id_aca_chargingIdentity: IetfAttrSyntax(),
    id_aca_group: IetfAttrSyntax(),
    id_at_role: RoleSyntax(),
    id_at_clearance: Clearance(),
    id_at_clearance_rfc3281: Clearance_rfc3281(),
    id_aca_encAttrs: ContentInfo(),
}

rfc5280.certificateAttributesMap.update(_certificateAttributesMapUpdate)
