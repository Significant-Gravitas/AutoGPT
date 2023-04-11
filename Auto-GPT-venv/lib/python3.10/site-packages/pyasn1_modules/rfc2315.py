#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS#7 message syntax
#
# ASN.1 source from:
# https://opensource.apple.com/source/Security/Security-55179.1/libsecurity_asn1/asn1/pkcs7.asn.auto.html
#
# Sample captures from:
# openssl crl2pkcs7 -nocrl -certfile cert1.cer -out outfile.p7b
#
from pyasn1_modules.rfc2459 import *


class Attribute(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('type', AttributeType()),
        namedtype.NamedType('values', univ.SetOf(componentType=AttributeValue()))
    )


class AttributeValueAssertion(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('attributeType', AttributeType()),
        namedtype.NamedType('attributeValue', AttributeValue(),
                            openType=opentype.OpenType('type', certificateAttributesMap))
    )


pkcs_7 = univ.ObjectIdentifier('1.2.840.113549.1.7')
data = univ.ObjectIdentifier('1.2.840.113549.1.7.1')
signedData = univ.ObjectIdentifier('1.2.840.113549.1.7.2')
envelopedData = univ.ObjectIdentifier('1.2.840.113549.1.7.3')
signedAndEnvelopedData = univ.ObjectIdentifier('1.2.840.113549.1.7.4')
digestedData = univ.ObjectIdentifier('1.2.840.113549.1.7.5')
encryptedData = univ.ObjectIdentifier('1.2.840.113549.1.7.6')


class ContentType(univ.ObjectIdentifier):
    pass


class ContentEncryptionAlgorithmIdentifier(AlgorithmIdentifier):
    pass


class EncryptedContent(univ.OctetString):
    pass


contentTypeMap = {}


class EncryptedContentInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('contentType', ContentType()),
        namedtype.NamedType('contentEncryptionAlgorithm', ContentEncryptionAlgorithmIdentifier()),
        namedtype.OptionalNamedType(
            'encryptedContent', EncryptedContent().subtype(
                implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)
            ),
            openType=opentype.OpenType('contentType', contentTypeMap)
        )
    )


class Version(univ.Integer):  # overrides x509.Version
    pass


class EncryptedData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('encryptedContentInfo', EncryptedContentInfo())
    )


class DigestAlgorithmIdentifier(AlgorithmIdentifier):
    pass


class DigestAlgorithmIdentifiers(univ.SetOf):
    componentType = DigestAlgorithmIdentifier()


class Digest(univ.OctetString):
    pass


class ContentInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('contentType', ContentType()),
        namedtype.OptionalNamedType(
            'content',
            univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)),
            openType=opentype.OpenType('contentType', contentTypeMap)
        )
    )


class DigestedData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()),
        namedtype.NamedType('contentInfo', ContentInfo()),
        namedtype.NamedType('digest', Digest())
    )


class IssuerAndSerialNumber(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('issuer', Name()),
        namedtype.NamedType('serialNumber', CertificateSerialNumber())
    )


class KeyEncryptionAlgorithmIdentifier(AlgorithmIdentifier):
    pass


class EncryptedKey(univ.OctetString):
    pass


class RecipientInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()),
        namedtype.NamedType('keyEncryptionAlgorithm', KeyEncryptionAlgorithmIdentifier()),
        namedtype.NamedType('encryptedKey', EncryptedKey())
    )


class RecipientInfos(univ.SetOf):
    componentType = RecipientInfo()


class Attributes(univ.SetOf):
    componentType = Attribute()


class ExtendedCertificateInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('certificate', Certificate()),
        namedtype.NamedType('attributes', Attributes())
    )


class SignatureAlgorithmIdentifier(AlgorithmIdentifier):
    pass


class Signature(univ.BitString):
    pass


class ExtendedCertificate(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('extendedCertificateInfo', ExtendedCertificateInfo()),
        namedtype.NamedType('signatureAlgorithm', SignatureAlgorithmIdentifier()),
        namedtype.NamedType('signature', Signature())
    )


class ExtendedCertificateOrCertificate(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certificate', Certificate()),
        namedtype.NamedType('extendedCertificate', ExtendedCertificate().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)))
    )


class ExtendedCertificatesAndCertificates(univ.SetOf):
    componentType = ExtendedCertificateOrCertificate()


class SerialNumber(univ.Integer):
    pass


class CRLEntry(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('userCertificate', SerialNumber()),
        namedtype.NamedType('revocationDate', useful.UTCTime())
    )


class TBSCertificateRevocationList(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('signature', AlgorithmIdentifier()),
        namedtype.NamedType('issuer', Name()),
        namedtype.NamedType('lastUpdate', useful.UTCTime()),
        namedtype.NamedType('nextUpdate', useful.UTCTime()),
        namedtype.OptionalNamedType('revokedCertificates', univ.SequenceOf(componentType=CRLEntry()))
    )


class CertificateRevocationList(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsCertificateRevocationList', TBSCertificateRevocationList()),
        namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('signature', univ.BitString())
    )


class CertificateRevocationLists(univ.SetOf):
    componentType = CertificateRevocationList()


class DigestEncryptionAlgorithmIdentifier(AlgorithmIdentifier):
    pass


class EncryptedDigest(univ.OctetString):
    pass


class SignerInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('issuerAndSerialNumber', IssuerAndSerialNumber()),
        namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()),
        namedtype.OptionalNamedType('authenticatedAttributes', Attributes().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.NamedType('digestEncryptionAlgorithm', DigestEncryptionAlgorithmIdentifier()),
        namedtype.NamedType('encryptedDigest', EncryptedDigest()),
        namedtype.OptionalNamedType('unauthenticatedAttributes', Attributes().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class SignerInfos(univ.SetOf):
    componentType = SignerInfo()


class SignedAndEnvelopedData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('recipientInfos', RecipientInfos()),
        namedtype.NamedType('digestAlgorithms', DigestAlgorithmIdentifiers()),
        namedtype.NamedType('encryptedContentInfo', EncryptedContentInfo()),
        namedtype.OptionalNamedType('certificates', ExtendedCertificatesAndCertificates().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('crls', CertificateRevocationLists().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
        namedtype.NamedType('signerInfos', SignerInfos())
    )


class EnvelopedData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('recipientInfos', RecipientInfos()),
        namedtype.NamedType('encryptedContentInfo', EncryptedContentInfo())
    )


class DigestInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('digestAlgorithm', DigestAlgorithmIdentifier()),
        namedtype.NamedType('digest', Digest())
    )


class SignedData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.OptionalNamedType('digestAlgorithms', DigestAlgorithmIdentifiers()),
        namedtype.NamedType('contentInfo', ContentInfo()),
        namedtype.OptionalNamedType('certificates', ExtendedCertificatesAndCertificates().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('crls', CertificateRevocationLists().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
        namedtype.OptionalNamedType('signerInfos', SignerInfos())
    )


class Data(univ.OctetString):
    pass

_contentTypeMapUpdate = {
    data: Data(),
    signedData: SignedData(),
    envelopedData: EnvelopedData(),
    signedAndEnvelopedData: SignedAndEnvelopedData(),
    digestedData: DigestedData(),
    encryptedData: EncryptedData()
}

contentTypeMap.update(_contentTypeMapUpdate)
