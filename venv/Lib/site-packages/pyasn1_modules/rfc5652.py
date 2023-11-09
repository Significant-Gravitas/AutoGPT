# coding: utf-8
#
# This file is part of pyasn1-modules software.
#
# Created by Stanisław Pitucha with asn1ate tool.
# Modified by Russ Housley to add support for opentypes.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# Cryptographic Message Syntax (CMS)
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc5652.txt
#
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc3281
from pyasn1_modules import rfc5280

MAX = float('inf')


def _buildOid(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


cmsContentTypesMap = { }

cmsAttributesMap = { }

otherKeyAttributesMap = { }

otherCertFormatMap = { }

otherRevInfoFormatMap = { }

otherRecipientInfoMap = { }


class AttCertVersionV1(univ.Integer):
    pass


AttCertVersionV1.namedValues = namedval.NamedValues(
    ('v1', 0)
)


class AttributeCertificateInfoV1(univ.Sequence):
    pass


AttributeCertificateInfoV1.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', AttCertVersionV1().subtype(value="v1")),
    namedtype.NamedType(
        'subject', univ.Choice(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType('baseCertificateID', rfc3281.IssuerSerial().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
                namedtype.NamedType('subjectName', rfc5280.GeneralNames().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
            )
        )
    ),
    namedtype.NamedType('issuer', rfc5280.GeneralNames()),
    namedtype.NamedType('signature', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('serialNumber', rfc5280.CertificateSerialNumber()),
    namedtype.NamedType('attCertValidityPeriod', rfc3281.AttCertValidityPeriod()),
    namedtype.NamedType('attributes', univ.SequenceOf(componentType=rfc5280.Attribute())),
    namedtype.OptionalNamedType('issuerUniqueID', rfc5280.UniqueIdentifier()),
    namedtype.OptionalNamedType('extensions', rfc5280.Extensions())
)


class AttributeCertificateV1(univ.Sequence):
    pass


AttributeCertificateV1.componentType = namedtype.NamedTypes(
    namedtype.NamedType('acInfo', AttributeCertificateInfoV1()),
    namedtype.NamedType('signatureAlgorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('signature', univ.BitString())
)


class AttributeValue(univ.Any):
    pass


class Attribute(univ.Sequence):
    pass


Attribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', univ.ObjectIdentifier()),
    namedtype.NamedType('attrValues', univ.SetOf(componentType=AttributeValue()),
        openType=opentype.OpenType('attrType', cmsAttributesMap)
    )
)


class SignedAttributes(univ.SetOf):
    pass


SignedAttributes.componentType = Attribute()
SignedAttributes.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class AttributeCertificateV2(rfc3281.AttributeCertificate):
    pass


class OtherKeyAttribute(univ.Sequence):
    pass


OtherKeyAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyAttrId', univ.ObjectIdentifier()),
    namedtype.OptionalNamedType('keyAttr', univ.Any(),
        openType=opentype.OpenType('keyAttrId', otherKeyAttributesMap)
    )
)


class UnauthAttributes(univ.SetOf):
    pass


UnauthAttributes.componentType = Attribute()
UnauthAttributes.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

id_encryptedData = _buildOid(1, 2, 840, 113549, 1, 7, 6)


class SignatureValue(univ.OctetString):
    pass


class IssuerAndSerialNumber(univ.Sequence):
    pass


IssuerAndSerialNumber.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuer', rfc5280.Name()),
    namedtype.NamedType('serialNumber', rfc5280.CertificateSerialNumber())
)


class SubjectKeyIdentifier(univ.OctetString):
    pass


class RecipientKeyIdentifier(univ.Sequence):
    pass


RecipientKeyIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('subjectKeyIdentifier', SubjectKeyIdentifier()),
    namedtype.OptionalNamedType('date', useful.GeneralizedTime()),
    namedtype.OptionalNamedType('other', OtherKeyAttribute())
)


class KeyAgreeRecipientIdentifier(univ.Choice):
    pass


KeyAgreeRecipientIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()),
    namedtype.NamedType('rKeyId', RecipientKeyIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
)


class EncryptedKey(univ.OctetString):
    pass


class RecipientEncryptedKey(univ.Sequence):
    pass


RecipientEncryptedKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('rid', KeyAgreeRecipientIdentifier()),
    namedtype.NamedType('encryptedKey', EncryptedKey())
)


class RecipientEncryptedKeys(univ.SequenceOf):
    pass


RecipientEncryptedKeys.componentType = RecipientEncryptedKey()


class MessageAuthenticationCode(univ.OctetString):
    pass


class CMSVersion(univ.Integer):
    pass


CMSVersion.namedValues = namedval.NamedValues(
    ('v0', 0),
    ('v1', 1),
    ('v2', 2),
    ('v3', 3),
    ('v4', 4),
    ('v5', 5)
)


class OtherCertificateFormat(univ.Sequence):
    pass


OtherCertificateFormat.componentType = namedtype.NamedTypes(
    namedtype.NamedType('otherCertFormat', univ.ObjectIdentifier()),
    namedtype.NamedType('otherCert', univ.Any(),
        openType=opentype.OpenType('otherCertFormat', otherCertFormatMap)
    )
)


class ExtendedCertificateInfo(univ.Sequence):
    pass


ExtendedCertificateInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('certificate', rfc5280.Certificate()),
    namedtype.NamedType('attributes', UnauthAttributes())
)


class Signature(univ.BitString):
    pass


class SignatureAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class ExtendedCertificate(univ.Sequence):
    pass


ExtendedCertificate.componentType = namedtype.NamedTypes(
    namedtype.NamedType('extendedCertificateInfo', ExtendedCertificateInfo()),
    namedtype.NamedType('signatureAlgorithm', SignatureAlgorithmIdentifier()),
    namedtype.NamedType('signature', Signature())
)


class CertificateChoices(univ.Choice):
    pass


CertificateChoices.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certificate', rfc5280.Certificate()),
    namedtype.NamedType('extendedCertificate', ExtendedCertificate().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('v1AttrCert', AttributeCertificateV1().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('v2AttrCert', AttributeCertificateV2().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.NamedType('other', OtherCertificateFormat().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3)))
)


class CertificateSet(univ.SetOf):
    pass


CertificateSet.componentType = CertificateChoices()


class OtherRevocationInfoFormat(univ.Sequence):
    pass


OtherRevocationInfoFormat.componentType = namedtype.NamedTypes(
    namedtype.NamedType('otherRevInfoFormat', univ.ObjectIdentifier()),
    namedtype.NamedType('otherRevInfo', univ.Any(),
        openType=opentype.OpenType('otherRevInfoFormat', otherRevInfoFormatMap)
    )
)


class RevocationInfoChoice(univ.Choice):
    pass


RevocationInfoChoice.componentType = namedtype.NamedTypes(
    namedtype.NamedType('crl', rfc5280.CertificateList()),
    namedtype.NamedType('other', OtherRevocationInfoFormat().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
)


class RevocationInfoChoices(univ.SetOf):
    pass


RevocationInfoChoices.componentType = RevocationInfoChoice()


class OriginatorInfo(univ.Sequence):
    pass


OriginatorInfo.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('certs', CertificateSet().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('crls', RevocationInfoChoices().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class ContentType(univ.ObjectIdentifier):
    pass


class EncryptedContent(univ.OctetString):
    pass


class ContentEncryptionAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class EncryptedContentInfo(univ.Sequence):
    pass


EncryptedContentInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('contentType', ContentType()),
    namedtype.NamedType('contentEncryptionAlgorithm', ContentEncryptionAlgorithmIdentifier()),
    namedtype.OptionalNamedType('encryptedContent', EncryptedContent().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class UnprotectedAttributes(univ.SetOf):
    pass


UnprotectedAttributes.componentType = Attribute()
UnprotectedAttributes.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class KeyEncryptionAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class KEKIdentifier(univ.Sequence):
    pass


KEKIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyIdentifier', univ.OctetString()),
    namedtype.OptionalNamedType('date', useful.GeneralizedTime()),
    namedtype.OptionalNamedType('other', OtherKeyAttribute())
)


class KEKRecipientInfo(univ.Sequence):
    pass


KEKRecipientInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('kekid', KEKIdentifier()),
    namedtype.NamedType('keyEncryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()),
    namedtype.NamedType('encryptedKey', EncryptedKey())
)


class KeyDerivationAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class PasswordRecipientInfo(univ.Sequence):
    pass


PasswordRecipientInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.OptionalNamedType('keyDerivationAlgorithm', KeyDerivationAlgorithmIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('keyEncryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()),
    namedtype.NamedType('encryptedKey', EncryptedKey())
)


class RecipientIdentifier(univ.Choice):
    pass


RecipientIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()),
    namedtype.NamedType('subjectKeyIdentifier', SubjectKeyIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class KeyTransRecipientInfo(univ.Sequence):
    pass


KeyTransRecipientInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('rid', RecipientIdentifier()),
    namedtype.NamedType('keyEncryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()),
    namedtype.NamedType('encryptedKey', EncryptedKey())
)


class UserKeyingMaterial(univ.OctetString):
    pass


class OriginatorPublicKey(univ.Sequence):
    pass


OriginatorPublicKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('algorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('publicKey', univ.BitString())
)


class OriginatorIdentifierOrKey(univ.Choice):
    pass


OriginatorIdentifierOrKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()),
    namedtype.NamedType('subjectKeyIdentifier', SubjectKeyIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('originatorKey', OriginatorPublicKey().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
)


class KeyAgreeRecipientInfo(univ.Sequence):
    pass


KeyAgreeRecipientInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('originator', OriginatorIdentifierOrKey().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('ukm', UserKeyingMaterial().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('keyEncryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()),
    namedtype.NamedType('recipientEncryptedKeys', RecipientEncryptedKeys())
)


class OtherRecipientInfo(univ.Sequence):
    pass


OtherRecipientInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('oriType', univ.ObjectIdentifier()),
    namedtype.NamedType('oriValue', univ.Any(),
        openType=opentype.OpenType('oriType', otherRecipientInfoMap)
    )
)


class RecipientInfo(univ.Choice):
    pass


RecipientInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('ktri', KeyTransRecipientInfo()),
    namedtype.NamedType('kari', KeyAgreeRecipientInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
    namedtype.NamedType('kekri', KEKRecipientInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))),
    namedtype.NamedType('pwri', PasswordRecipientInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))),
    namedtype.NamedType('ori', OtherRecipientInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4)))
)


class RecipientInfos(univ.SetOf):
    pass


RecipientInfos.componentType = RecipientInfo()
RecipientInfos.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class EnvelopedData(univ.Sequence):
    pass


EnvelopedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.OptionalNamedType('originatorInfo', OriginatorInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('recipientInfos', RecipientInfos()),
    namedtype.NamedType('encryptedContentInfo', EncryptedContentInfo()),
    namedtype.OptionalNamedType('unprotectedAttrs', UnprotectedAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class DigestAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


id_ct_contentInfo = _buildOid(1, 2, 840, 113549, 1, 9, 16, 1, 6)

id_digestedData = _buildOid(1, 2, 840, 113549, 1, 7, 5)


class EncryptedData(univ.Sequence):
    pass


EncryptedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('encryptedContentInfo', EncryptedContentInfo()),
    namedtype.OptionalNamedType('unprotectedAttrs', UnprotectedAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)

id_messageDigest = _buildOid(1, 2, 840, 113549, 1, 9, 4)

id_signedData = _buildOid(1, 2, 840, 113549, 1, 7, 2)


class MessageAuthenticationCodeAlgorithm(rfc5280.AlgorithmIdentifier):
    pass


class UnsignedAttributes(univ.SetOf):
    pass


UnsignedAttributes.componentType = Attribute()
UnsignedAttributes.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class SignerIdentifier(univ.Choice):
    pass


SignerIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()),
    namedtype.NamedType('subjectKeyIdentifier', SubjectKeyIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class SignerInfo(univ.Sequence):
    pass


SignerInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('sid', SignerIdentifier()),
    namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()),
    namedtype.OptionalNamedType('signedAttrs', SignedAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('signatureAlgorithm', SignatureAlgorithmIdentifier()),
    namedtype.NamedType('signature', SignatureValue()),
    namedtype.OptionalNamedType('unsignedAttrs', UnsignedAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class SignerInfos(univ.SetOf):
    pass


SignerInfos.componentType = SignerInfo()


class Countersignature(SignerInfo):
    pass


class ContentInfo(univ.Sequence):
    pass


ContentInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('contentType', ContentType()),
    namedtype.NamedType('content', univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)),
        openType=opentype.OpenType('contentType', cmsContentTypesMap)
    )
)


class EncapsulatedContentInfo(univ.Sequence):
    pass


EncapsulatedContentInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('eContentType', ContentType()),
    namedtype.OptionalNamedType('eContent', univ.OctetString().subtype(
        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)

id_countersignature = _buildOid(1, 2, 840, 113549, 1, 9, 6)

id_data = _buildOid(1, 2, 840, 113549, 1, 7, 1)


class MessageDigest(univ.OctetString):
    pass


class AuthAttributes(univ.SetOf):
    pass


AuthAttributes.componentType = Attribute()
AuthAttributes.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class Time(univ.Choice):
    pass


Time.componentType = namedtype.NamedTypes(
    namedtype.NamedType('utcTime', useful.UTCTime()),
    namedtype.NamedType('generalTime', useful.GeneralizedTime())
)


class AuthenticatedData(univ.Sequence):
    pass


AuthenticatedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.OptionalNamedType('originatorInfo', OriginatorInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('recipientInfos', RecipientInfos()),
    namedtype.NamedType('macAlgorithm', MessageAuthenticationCodeAlgorithm()),
    namedtype.OptionalNamedType('digestAlgorithm', DigestAlgorithmIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('encapContentInfo', EncapsulatedContentInfo()),
    namedtype.OptionalNamedType('authAttrs', AuthAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.NamedType('mac', MessageAuthenticationCode()),
    namedtype.OptionalNamedType('unauthAttrs', UnauthAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3)))
)

id_contentType = _buildOid(1, 2, 840, 113549, 1, 9, 3)


class ExtendedCertificateOrCertificate(univ.Choice):
    pass


ExtendedCertificateOrCertificate.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certificate', rfc5280.Certificate()),
    namedtype.NamedType('extendedCertificate', ExtendedCertificate().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
)


class Digest(univ.OctetString):
    pass


class DigestedData(univ.Sequence):
    pass


DigestedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()),
    namedtype.NamedType('encapContentInfo', EncapsulatedContentInfo()),
    namedtype.NamedType('digest', Digest())
)

id_envelopedData = _buildOid(1, 2, 840, 113549, 1, 7, 3)


class DigestAlgorithmIdentifiers(univ.SetOf):
    pass


DigestAlgorithmIdentifiers.componentType = DigestAlgorithmIdentifier()


class SignedData(univ.Sequence):
    pass


SignedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', CMSVersion()),
    namedtype.NamedType('digestAlgorithms', DigestAlgorithmIdentifiers()),
    namedtype.NamedType('encapContentInfo', EncapsulatedContentInfo()),
    namedtype.OptionalNamedType('certificates', CertificateSet().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('crls', RevocationInfoChoices().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('signerInfos', SignerInfos())
)

id_signingTime = _buildOid(1, 2, 840, 113549, 1, 9, 5)


class SigningTime(Time):
    pass


id_ct_authData = _buildOid(1, 2, 840, 113549, 1, 9, 16, 1, 2)


# CMS Content Type Map

_cmsContentTypesMapUpdate = {
    id_ct_contentInfo: ContentInfo(),
    id_data: univ.OctetString(),
    id_signedData: SignedData(),
    id_envelopedData: EnvelopedData(),
    id_digestedData: DigestedData(),
    id_encryptedData: EncryptedData(),
    id_ct_authData: AuthenticatedData(),
}

cmsContentTypesMap.update(_cmsContentTypesMapUpdate)


# CMS Attribute Map

_cmsAttributesMapUpdate = {
    id_contentType: ContentType(),
    id_messageDigest: MessageDigest(),
    id_signingTime: SigningTime(),
    id_countersignature: Countersignature(),
}

cmsAttributesMap.update(_cmsAttributesMapUpdate)
