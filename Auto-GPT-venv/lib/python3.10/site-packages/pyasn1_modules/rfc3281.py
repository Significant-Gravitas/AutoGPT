# coding: utf-8
#
# This file is part of pyasn1-modules software.
#
# Created by Stanisław Pitucha with asn1ate tool.
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# An Internet Attribute Certificate Profile for Authorization
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc3281.txt
#
from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc3280

MAX = float('inf')


def _buildOid(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


class ObjectDigestInfo(univ.Sequence):
    pass


ObjectDigestInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('digestedObjectType', univ.Enumerated(
        namedValues=namedval.NamedValues(('publicKey', 0), ('publicKeyCert', 1), ('otherObjectTypes', 2)))),
    namedtype.OptionalNamedType('otherObjectTypeID', univ.ObjectIdentifier()),
    namedtype.NamedType('digestAlgorithm', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('objectDigest', univ.BitString())
)


class IssuerSerial(univ.Sequence):
    pass


IssuerSerial.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuer', rfc3280.GeneralNames()),
    namedtype.NamedType('serial', rfc3280.CertificateSerialNumber()),
    namedtype.OptionalNamedType('issuerUID', rfc3280.UniqueIdentifier())
)


class TargetCert(univ.Sequence):
    pass


TargetCert.componentType = namedtype.NamedTypes(
    namedtype.NamedType('targetCertificate', IssuerSerial()),
    namedtype.OptionalNamedType('targetName', rfc3280.GeneralName()),
    namedtype.OptionalNamedType('certDigestInfo', ObjectDigestInfo())
)


class Target(univ.Choice):
    pass


Target.componentType = namedtype.NamedTypes(
    namedtype.NamedType('targetName', rfc3280.GeneralName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('targetGroup', rfc3280.GeneralName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('targetCert',
                        TargetCert().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
)


class Targets(univ.SequenceOf):
    pass


Targets.componentType = Target()


class ProxyInfo(univ.SequenceOf):
    pass


ProxyInfo.componentType = Targets()

id_at_role = _buildOid(rfc3280.id_at, 72)

id_pe_aaControls = _buildOid(rfc3280.id_pe, 6)

id_ce_targetInformation = _buildOid(rfc3280.id_ce, 55)

id_pe_ac_auditIdentity = _buildOid(rfc3280.id_pe, 4)


class ClassList(univ.BitString):
    pass


ClassList.namedValues = namedval.NamedValues(
    ('unmarked', 0),
    ('unclassified', 1),
    ('restricted', 2),
    ('confidential', 3),
    ('secret', 4),
    ('topSecret', 5)
)


class SecurityCategory(univ.Sequence):
    pass


SecurityCategory.componentType = namedtype.NamedTypes(
    namedtype.NamedType('type', univ.ObjectIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('value', univ.Any().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class Clearance(univ.Sequence):
    pass


Clearance.componentType = namedtype.NamedTypes(
    namedtype.NamedType('policyId', univ.ObjectIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.DefaultedNamedType('classList',
                                 ClassList().subtype(implicitTag=tag.Tag(tag.tagClassContext,
                                                                         tag.tagFormatSimple, 1)).subtype(
                                     value="unclassified")),
    namedtype.OptionalNamedType('securityCategories', univ.SetOf(componentType=SecurityCategory()).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)


class AttCertVersion(univ.Integer):
    pass


AttCertVersion.namedValues = namedval.NamedValues(
    ('v2', 1)
)

id_aca = _buildOid(rfc3280.id_pkix, 10)

id_at_clearance = _buildOid(2, 5, 1, 5, 55)


class AttrSpec(univ.SequenceOf):
    pass


AttrSpec.componentType = univ.ObjectIdentifier()


class AAControls(univ.Sequence):
    pass


AAControls.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('pathLenConstraint',
                                univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, MAX))),
    namedtype.OptionalNamedType('permittedAttrs',
                                AttrSpec().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('excludedAttrs',
                                AttrSpec().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.DefaultedNamedType('permitUnSpecified', univ.Boolean().subtype(value=1))
)


class AttCertValidityPeriod(univ.Sequence):
    pass


AttCertValidityPeriod.componentType = namedtype.NamedTypes(
    namedtype.NamedType('notBeforeTime', useful.GeneralizedTime()),
    namedtype.NamedType('notAfterTime', useful.GeneralizedTime())
)


id_aca_authenticationInfo = _buildOid(id_aca, 1)


class V2Form(univ.Sequence):
    pass


V2Form.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('issuerName', rfc3280.GeneralNames()),
    namedtype.OptionalNamedType('baseCertificateID', IssuerSerial().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('objectDigestInfo', ObjectDigestInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
)


class AttCertIssuer(univ.Choice):
    pass


AttCertIssuer.componentType = namedtype.NamedTypes(
    namedtype.NamedType('v1Form', rfc3280.GeneralNames()),
    namedtype.NamedType('v2Form',
                        V2Form().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
)


class Holder(univ.Sequence):
    pass


Holder.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('baseCertificateID', IssuerSerial().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('entityName', rfc3280.GeneralNames().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('objectDigestInfo', ObjectDigestInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
)


class AttributeCertificateInfo(univ.Sequence):
    pass


AttributeCertificateInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', AttCertVersion()),
    namedtype.NamedType('holder', Holder()),
    namedtype.NamedType('issuer', AttCertIssuer()),
    namedtype.NamedType('signature', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('serialNumber', rfc3280.CertificateSerialNumber()),
    namedtype.NamedType('attrCertValidityPeriod', AttCertValidityPeriod()),
    namedtype.NamedType('attributes', univ.SequenceOf(componentType=rfc3280.Attribute())),
    namedtype.OptionalNamedType('issuerUniqueID', rfc3280.UniqueIdentifier()),
    namedtype.OptionalNamedType('extensions', rfc3280.Extensions())
)


class AttributeCertificate(univ.Sequence):
    pass


AttributeCertificate.componentType = namedtype.NamedTypes(
    namedtype.NamedType('acinfo', AttributeCertificateInfo()),
    namedtype.NamedType('signatureAlgorithm', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('signatureValue', univ.BitString())
)

id_mod = _buildOid(rfc3280.id_pkix, 0)

id_mod_attribute_cert = _buildOid(id_mod, 12)

id_aca_accessIdentity = _buildOid(id_aca, 2)


class RoleSyntax(univ.Sequence):
    pass


RoleSyntax.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('roleAuthority', rfc3280.GeneralNames().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('roleName',
                        rfc3280.GeneralName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)

id_aca_chargingIdentity = _buildOid(id_aca, 3)


class ACClearAttrs(univ.Sequence):
    pass


ACClearAttrs.componentType = namedtype.NamedTypes(
    namedtype.NamedType('acIssuer', rfc3280.GeneralName()),
    namedtype.NamedType('acSerial', univ.Integer()),
    namedtype.NamedType('attrs', univ.SequenceOf(componentType=rfc3280.Attribute()))
)

id_aca_group = _buildOid(id_aca, 4)

id_pe_ac_proxying = _buildOid(rfc3280.id_pe, 10)


class SvceAuthInfo(univ.Sequence):
    pass


SvceAuthInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('service', rfc3280.GeneralName()),
    namedtype.NamedType('ident', rfc3280.GeneralName()),
    namedtype.OptionalNamedType('authInfo', univ.OctetString())
)


class IetfAttrSyntax(univ.Sequence):
    pass


IetfAttrSyntax.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType(
        'policyAuthority', rfc3280.GeneralNames().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))
    ),
    namedtype.NamedType(
        'values', univ.SequenceOf(
            componentType=univ.Choice(
                componentType=namedtype.NamedTypes(
                    namedtype.NamedType('octets', univ.OctetString()),
                    namedtype.NamedType('oid', univ.ObjectIdentifier()),
                    namedtype.NamedType('string', char.UTF8String())
                )
            )
        )
    )
)

id_aca_encAttrs = _buildOid(id_aca, 6)
