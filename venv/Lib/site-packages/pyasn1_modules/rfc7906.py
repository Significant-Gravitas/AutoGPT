#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# NSA's CMS Key Management Attributes
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7906.txt
# https://www.rfc-editor.org/errata/eid5850
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc2634
from pyasn1_modules import rfc4108
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc6010
from pyasn1_modules import rfc6019
from pyasn1_modules import rfc7191

MAX = float('inf')


# Imports From RFC 2634

id_aa_contentHint = rfc2634.id_aa_contentHint

ContentHints = rfc2634.ContentHints

id_aa_securityLabel = rfc2634.id_aa_securityLabel

SecurityPolicyIdentifier = rfc2634.SecurityPolicyIdentifier

SecurityClassification = rfc2634.SecurityClassification

ESSPrivacyMark = rfc2634.ESSPrivacyMark

SecurityCategories= rfc2634.SecurityCategories

ESSSecurityLabel = rfc2634.ESSSecurityLabel


# Imports From RFC 4108

id_aa_communityIdentifiers = rfc4108.id_aa_communityIdentifiers

CommunityIdentifier = rfc4108.CommunityIdentifier

CommunityIdentifiers = rfc4108.CommunityIdentifiers


# Imports From RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier

Name = rfc5280.Name

Certificate = rfc5280.Certificate

GeneralNames = rfc5280.GeneralNames

GeneralName = rfc5280.GeneralName


SubjectInfoAccessSyntax = rfc5280.SubjectInfoAccessSyntax

id_pkix = rfc5280.id_pkix

id_pe = rfc5280.id_pe

id_pe_subjectInfoAccess = rfc5280.id_pe_subjectInfoAccess


# Imports From RFC 6010

CMSContentConstraints = rfc6010.CMSContentConstraints


# Imports From RFC 6019

BinaryTime = rfc6019.BinaryTime

id_aa_binarySigningTime = rfc6019.id_aa_binarySigningTime

BinarySigningTime = rfc6019.BinarySigningTime


# Imports From RFC 5652

Attribute = rfc5652.Attribute

CertificateSet = rfc5652.CertificateSet

CertificateChoices = rfc5652.CertificateChoices

id_contentType = rfc5652.id_contentType

ContentType = rfc5652.ContentType

id_messageDigest = rfc5652.id_messageDigest

MessageDigest = rfc5652.MessageDigest


# Imports From RFC 7191

SIREntityName = rfc7191.SIREntityName

id_aa_KP_keyPkgIdAndReceiptReq = rfc7191.id_aa_KP_keyPkgIdAndReceiptReq

KeyPkgIdentifierAndReceiptReq = rfc7191.KeyPkgIdentifierAndReceiptReq


# Key Province Attribute

id_aa_KP_keyProvinceV2 = univ.ObjectIdentifier('2.16.840.1.101.2.1.5.71')


class KeyProvinceV2(univ.ObjectIdentifier):
    pass


aa_keyProvince_v2 = Attribute()
aa_keyProvince_v2['attrType'] = id_aa_KP_keyProvinceV2
aa_keyProvince_v2['attrValues'][0] = KeyProvinceV2()
 

# Manifest Attribute

id_aa_KP_manifest = univ.ObjectIdentifier('2.16.840.1.101.2.1.5.72')


class ShortTitle(char.PrintableString):
    pass


class Manifest(univ.SequenceOf):
    pass

Manifest.componentType = ShortTitle()
Manifest.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


aa_manifest = Attribute()
aa_manifest['attrType'] = id_aa_KP_manifest
aa_manifest['attrValues'][0] = Manifest()


# Key Algorithm Attribute

id_kma_keyAlgorithm = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.1')


class KeyAlgorithm(univ.Sequence):
    pass

KeyAlgorithm.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyAlg', univ.ObjectIdentifier()),
    namedtype.OptionalNamedType('checkWordAlg', univ.ObjectIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('crcAlg', univ.ObjectIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)


aa_keyAlgorithm = Attribute()
aa_keyAlgorithm['attrType'] = id_kma_keyAlgorithm
aa_keyAlgorithm['attrValues'][0] = KeyAlgorithm()


# User Certificate Attribute

id_at_userCertificate = univ.ObjectIdentifier('2.5.4.36')


aa_userCertificate = Attribute()
aa_userCertificate['attrType'] = id_at_userCertificate
aa_userCertificate['attrValues'][0] =  Certificate()


# Key Package Receivers Attribute

id_kma_keyPkgReceiversV2 = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.16')


class KeyPkgReceiver(univ.Choice):
    pass

KeyPkgReceiver.componentType = namedtype.NamedTypes(
    namedtype.NamedType('sirEntity', SIREntityName().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('community', CommunityIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class KeyPkgReceiversV2(univ.SequenceOf):
    pass

KeyPkgReceiversV2.componentType = KeyPkgReceiver()
KeyPkgReceiversV2.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


aa_keyPackageReceivers_v2 = Attribute()
aa_keyPackageReceivers_v2['attrType'] = id_kma_keyPkgReceiversV2
aa_keyPackageReceivers_v2['attrValues'][0] = KeyPkgReceiversV2()


# TSEC Nomenclature Attribute

id_kma_TSECNomenclature = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.3')


class CharEdition(char.PrintableString):
    pass


class CharEditionRange(univ.Sequence):
    pass

CharEditionRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('firstCharEdition', CharEdition()),
    namedtype.NamedType('lastCharEdition', CharEdition())
)


class NumEdition(univ.Integer):
    pass

NumEdition.subtypeSpec = constraint.ValueRangeConstraint(0, 308915776)


class NumEditionRange(univ.Sequence):
    pass

NumEditionRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('firstNumEdition', NumEdition()),
    namedtype.NamedType('lastNumEdition', NumEdition())
)


class EditionID(univ.Choice):
    pass

EditionID.componentType = namedtype.NamedTypes(
    namedtype.NamedType('char', univ.Choice(componentType=namedtype.NamedTypes(
        namedtype.NamedType('charEdition', CharEdition().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('charEditionRange', CharEditionRange().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
    ))
    ),
    namedtype.NamedType('num', univ.Choice(componentType=namedtype.NamedTypes(
        namedtype.NamedType('numEdition', NumEdition().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
        namedtype.NamedType('numEditionRange', NumEditionRange().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4)))
    ))
    )
)


class Register(univ.Integer):
    pass

Register.subtypeSpec = constraint.ValueRangeConstraint(0, 2147483647)


class RegisterRange(univ.Sequence):
    pass

RegisterRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('firstRegister', Register()),
    namedtype.NamedType('lastRegister', Register())
)


class RegisterID(univ.Choice):
    pass

RegisterID.componentType = namedtype.NamedTypes(
    namedtype.NamedType('register', Register().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5))),
    namedtype.NamedType('registerRange', RegisterRange().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 6)))
)


class SegmentNumber(univ.Integer):
    pass

SegmentNumber.subtypeSpec = constraint.ValueRangeConstraint(1, 127)


class SegmentRange(univ.Sequence):
    pass

SegmentRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('firstSegment', SegmentNumber()),
    namedtype.NamedType('lastSegment', SegmentNumber())
)


class SegmentID(univ.Choice):
    pass

SegmentID.componentType = namedtype.NamedTypes(
    namedtype.NamedType('segmentNumber', SegmentNumber().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 7))),
    namedtype.NamedType('segmentRange', SegmentRange().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 8)))
)


class TSECNomenclature(univ.Sequence):
    pass

TSECNomenclature.componentType = namedtype.NamedTypes(
    namedtype.NamedType('shortTitle', ShortTitle()),
    namedtype.OptionalNamedType('editionID', EditionID()),
    namedtype.OptionalNamedType('registerID', RegisterID()),
    namedtype.OptionalNamedType('segmentID', SegmentID())
)


aa_tsecNomenclature = Attribute()
aa_tsecNomenclature['attrType'] = id_kma_TSECNomenclature
aa_tsecNomenclature['attrValues'][0] = TSECNomenclature()


# Key Purpose Attribute

id_kma_keyPurpose = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.13')


class KeyPurpose(univ.Enumerated):
    pass

KeyPurpose.namedValues = namedval.NamedValues(
    ('n-a', 0),
    ('a', 65),
    ('b', 66),
    ('l', 76),
    ('m', 77),
    ('r', 82),
    ('s', 83),
    ('t', 84),
    ('v', 86),
    ('x', 88),
    ('z', 90)
)


aa_keyPurpose = Attribute()
aa_keyPurpose['attrType'] = id_kma_keyPurpose
aa_keyPurpose['attrValues'][0] = KeyPurpose()


# Key Use Attribute

id_kma_keyUse = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.14')


class KeyUse(univ.Enumerated):
    pass

KeyUse.namedValues = namedval.NamedValues(
    ('n-a', 0),
    ('ffk', 1),
    ('kek', 2),
    ('kpk', 3),
    ('msk', 4),
    ('qkek', 5),
    ('tek', 6),
    ('tsk', 7),
    ('trkek', 8),
    ('nfk', 9),
    ('effk', 10),
    ('ebfk', 11),
    ('aek', 12),
    ('wod', 13),
    ('kesk', 246),
    ('eik', 247),
    ('ask', 248),
    ('kmk', 249),
    ('rsk', 250),
    ('csk', 251),
    ('sak', 252),
    ('rgk', 253),
    ('cek', 254),
    ('exk', 255)
)


aa_keyUse = Attribute()
aa_keyPurpose['attrType'] = id_kma_keyUse
aa_keyPurpose['attrValues'][0] = KeyUse()


# Transport Key Attribute

id_kma_transportKey = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.15')


class TransOp(univ.Enumerated):
    pass

TransOp.namedValues = namedval.NamedValues(
    ('transport', 1),
    ('operational', 2)
)


aa_transportKey = Attribute()
aa_transportKey['attrType'] = id_kma_transportKey
aa_transportKey['attrValues'][0] = TransOp()


# Key Distribution Period Attribute

id_kma_keyDistPeriod = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.5')


class KeyDistPeriod(univ.Sequence):
    pass

KeyDistPeriod.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('doNotDistBefore', BinaryTime().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('doNotDistAfter', BinaryTime())
)


aa_keyDistributionPeriod = Attribute()
aa_keyDistributionPeriod['attrType'] = id_kma_keyDistPeriod
aa_keyDistributionPeriod['attrValues'][0] = KeyDistPeriod()


# Key Validity Period Attribute

id_kma_keyValidityPeriod = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.6')


class KeyValidityPeriod(univ.Sequence):
    pass

KeyValidityPeriod.componentType = namedtype.NamedTypes(
    namedtype.NamedType('doNotUseBefore', BinaryTime()),
    namedtype.OptionalNamedType('doNotUseAfter', BinaryTime())
)


aa_keyValidityPeriod = Attribute()
aa_keyValidityPeriod['attrType'] = id_kma_keyValidityPeriod
aa_keyValidityPeriod['attrValues'][0] = KeyValidityPeriod()


# Key Duration Attribute

id_kma_keyDuration = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.7')


ub_KeyDuration_months = univ.Integer(72)

ub_KeyDuration_hours = univ.Integer(96)

ub_KeyDuration_days = univ.Integer(732)

ub_KeyDuration_weeks = univ.Integer(104)

ub_KeyDuration_years = univ.Integer(100)


class KeyDuration(univ.Choice):
    pass

KeyDuration.componentType = namedtype.NamedTypes(
    namedtype.NamedType('hours', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, ub_KeyDuration_hours)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('days', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, ub_KeyDuration_days))),
    namedtype.NamedType('weeks', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, ub_KeyDuration_weeks)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('months', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, ub_KeyDuration_months)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.NamedType('years', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(1, ub_KeyDuration_years)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)


aa_keyDurationPeriod = Attribute()
aa_keyDurationPeriod['attrType'] = id_kma_keyDuration
aa_keyDurationPeriod['attrValues'][0] = KeyDuration()


# Classification Attribute

id_aa_KP_classification = univ.ObjectIdentifier(id_aa_securityLabel)


id_enumeratedPermissiveAttributes = univ.ObjectIdentifier('2.16.840.1.101.2.1.8.3.1')

id_enumeratedRestrictiveAttributes = univ.ObjectIdentifier('2.16.840.1.101.2.1.8.3.4')

id_informativeAttributes = univ.ObjectIdentifier('2.16.840.1.101.2.1.8.3.3')


class SecurityAttribute(univ.Integer):
    pass

SecurityAttribute.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


class EnumeratedTag(univ.Sequence):
    pass

EnumeratedTag.componentType = namedtype.NamedTypes(
    namedtype.NamedType('tagName', univ.ObjectIdentifier()),
    namedtype.NamedType('attributeList', univ.SetOf(componentType=SecurityAttribute()))
)


class FreeFormField(univ.Choice):
    pass

FreeFormField.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bitSetAttributes', univ.BitString()), # Not permitted in RFC 7906
    namedtype.NamedType('securityAttributes', univ.SetOf(componentType=SecurityAttribute()))
)


class InformativeTag(univ.Sequence):
    pass

InformativeTag.componentType = namedtype.NamedTypes(
    namedtype.NamedType('tagName', univ.ObjectIdentifier()),
    namedtype.NamedType('attributes', FreeFormField())
)


class Classification(ESSSecurityLabel):
    pass


aa_classification = Attribute()
aa_classification['attrType'] = id_aa_KP_classification
aa_classification['attrValues'][0] = Classification()


# Split Identifier Attribute

id_kma_splitID = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.11')


class SplitID(univ.Sequence):
    pass

SplitID.componentType = namedtype.NamedTypes(
    namedtype.NamedType('half', univ.Enumerated(
        namedValues=namedval.NamedValues(('a', 0), ('b', 1)))),
    namedtype.OptionalNamedType('combineAlg', AlgorithmIdentifier())
)


aa_splitIdentifier = Attribute()
aa_splitIdentifier['attrType'] = id_kma_splitID
aa_splitIdentifier['attrValues'][0] = SplitID()


# Key Package Type Attribute

id_kma_keyPkgType = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.12')


class KeyPkgType(univ.ObjectIdentifier):
    pass


aa_keyPackageType = Attribute()
aa_keyPackageType['attrType'] = id_kma_keyPkgType
aa_keyPackageType['attrValues'][0] = KeyPkgType()


# Signature Usage Attribute

id_kma_sigUsageV3 = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.22')


class SignatureUsage(CMSContentConstraints):
    pass


aa_signatureUsage_v3 = Attribute()
aa_signatureUsage_v3['attrType'] = id_kma_sigUsageV3
aa_signatureUsage_v3['attrValues'][0] = SignatureUsage()


# Other Certificate Format Attribute

id_kma_otherCertFormats = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.19')


aa_otherCertificateFormats = Attribute()
aa_signatureUsage_v3['attrType'] = id_kma_otherCertFormats
aa_signatureUsage_v3['attrValues'][0] = CertificateChoices()


# PKI Path Attribute

id_at_pkiPath = univ.ObjectIdentifier('2.5.4.70')


class PkiPath(univ.SequenceOf):
    pass

PkiPath.componentType = Certificate()
PkiPath.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


aa_pkiPath = Attribute()
aa_pkiPath['attrType'] = id_at_pkiPath
aa_pkiPath['attrValues'][0] = PkiPath()


# Useful Certificates Attribute

id_kma_usefulCerts = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.20')


aa_usefulCertificates = Attribute()
aa_usefulCertificates['attrType'] = id_kma_usefulCerts
aa_usefulCertificates['attrValues'][0] = CertificateSet()


# Key Wrap Attribute

id_kma_keyWrapAlgorithm = univ.ObjectIdentifier('2.16.840.1.101.2.1.13.21')


aa_keyWrapAlgorithm  = Attribute()
aa_keyWrapAlgorithm['attrType'] = id_kma_keyWrapAlgorithm
aa_keyWrapAlgorithm['attrValues'][0] = AlgorithmIdentifier()


# Content Decryption Key Identifier Attribute

id_aa_KP_contentDecryptKeyID = univ.ObjectIdentifier('2.16.840.1.101.2.1.5.66')


class ContentDecryptKeyID(univ.OctetString):
    pass


aa_contentDecryptKeyIdentifier = Attribute()
aa_contentDecryptKeyIdentifier['attrType'] = id_aa_KP_contentDecryptKeyID
aa_contentDecryptKeyIdentifier['attrValues'][0] = ContentDecryptKeyID()


# Certificate Pointers Attribute

aa_certificatePointers = Attribute()
aa_certificatePointers['attrType'] = id_pe_subjectInfoAccess
aa_certificatePointers['attrValues'][0] = SubjectInfoAccessSyntax()


# CRL Pointers Attribute

id_aa_KP_crlPointers = univ.ObjectIdentifier('2.16.840.1.101.2.1.5.70')


aa_cRLDistributionPoints = Attribute()
aa_cRLDistributionPoints['attrType'] = id_aa_KP_crlPointers
aa_cRLDistributionPoints['attrValues'][0] = GeneralNames()


# Extended Error Codes

id_errorCodes = univ.ObjectIdentifier('2.16.840.1.101.2.1.22')

id_missingKeyType = univ.ObjectIdentifier('2.16.840.1.101.2.1.22.1')

id_privacyMarkTooLong = univ.ObjectIdentifier('2.16.840.1.101.2.1.22.2')

id_unrecognizedSecurityPolicy = univ.ObjectIdentifier('2.16.840.1.101.2.1.22.3')


# Map of Attribute Type OIDs to Attributes added to the
# ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_aa_contentHint: ContentHints(),
    id_aa_communityIdentifiers: CommunityIdentifiers(),
    id_aa_binarySigningTime: BinarySigningTime(),
    id_contentType: ContentType(),
    id_messageDigest: MessageDigest(),
    id_aa_KP_keyPkgIdAndReceiptReq: KeyPkgIdentifierAndReceiptReq(),
    id_aa_KP_keyProvinceV2: KeyProvinceV2(),
    id_aa_KP_manifest: Manifest(),
    id_kma_keyAlgorithm: KeyAlgorithm(),
    id_at_userCertificate: Certificate(),
    id_kma_keyPkgReceiversV2: KeyPkgReceiversV2(),
    id_kma_TSECNomenclature: TSECNomenclature(),
    id_kma_keyPurpose: KeyPurpose(),
    id_kma_keyUse: KeyUse(),
    id_kma_transportKey: TransOp(),
    id_kma_keyDistPeriod: KeyDistPeriod(),
    id_kma_keyValidityPeriod: KeyValidityPeriod(),
    id_kma_keyDuration: KeyDuration(),
    id_aa_KP_classification: Classification(),
    id_kma_splitID: SplitID(),
    id_kma_keyPkgType: KeyPkgType(),
    id_kma_sigUsageV3: SignatureUsage(),
    id_kma_otherCertFormats: CertificateChoices(),
    id_at_pkiPath: PkiPath(),
    id_kma_usefulCerts: CertificateSet(),
    id_kma_keyWrapAlgorithm: AlgorithmIdentifier(),
    id_aa_KP_contentDecryptKeyID: ContentDecryptKeyID(),
    id_pe_subjectInfoAccess: SubjectInfoAccessSyntax(),
    id_aa_KP_crlPointers: GeneralNames(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)
