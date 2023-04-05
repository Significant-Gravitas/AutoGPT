# This file is being contributed to of pyasn1-modules software.
#
# Created by Russ Housley without assistance from the asn1ate tool.
# Modified by Russ Housley to add support for opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# CMS Key Package Receipt and Error Content Types
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7191.txt

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652

MAX = float('inf')

DistinguishedName = rfc5280.DistinguishedName


# SingleAttribute is the same as Attribute in RFC 5652, except that the
# attrValues SET must have one and only one member

class AttributeValue(univ.Any):
    pass


class AttributeValues(univ.SetOf):
    pass

AttributeValues.componentType = AttributeValue()
AttributeValues.sizeSpec = univ.Set.sizeSpec + constraint.ValueSizeConstraint(1, 1)


class SingleAttribute(univ.Sequence):
    pass

SingleAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', univ.ObjectIdentifier()),
    namedtype.NamedType('attrValues', AttributeValues(),
        openType=opentype.OpenType('attrType', rfc5652.cmsAttributesMap)
    )
)


# SIR Entity Name

class SIREntityNameType(univ.ObjectIdentifier):
    pass


class SIREntityNameValue(univ.Any):
    pass


class SIREntityName(univ.Sequence):
    pass

SIREntityName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('sirenType', SIREntityNameType()),
    namedtype.NamedType('sirenValue', univ.OctetString())
    # CONTAINING the DER-encoded SIREntityNameValue
)


class SIREntityNames(univ.SequenceOf):
    pass

SIREntityNames.componentType = SIREntityName()
SIREntityNames.sizeSpec=constraint.ValueSizeConstraint(1, MAX)


id_dn = univ.ObjectIdentifier('2.16.840.1.101.2.1.16.0')


class siren_dn(SIREntityName):
    def __init__(self):
        SIREntityName.__init__(self)
        self['sirenType'] = id_dn


# Key Package Error CMS Content Type

class EnumeratedErrorCode(univ.Enumerated):
    pass

# Error codes with values <= 33 are aligned with RFC 5934
EnumeratedErrorCode.namedValues = namedval.NamedValues(
    ('decodeFailure', 1),
    ('badContentInfo', 2),
    ('badSignedData', 3),
    ('badEncapContent', 4),
    ('badCertificate', 5),
    ('badSignerInfo', 6),
    ('badSignedAttrs', 7),
    ('badUnsignedAttrs', 8),
    ('missingContent', 9),
    ('noTrustAnchor', 10),
    ('notAuthorized', 11),
    ('badDigestAlgorithm', 12),
    ('badSignatureAlgorithm', 13),
    ('unsupportedKeySize', 14),
    ('unsupportedParameters', 15),
    ('signatureFailure', 16),
    ('insufficientMemory', 17),
    ('incorrectTarget', 23),
    ('missingSignature', 29),
    ('resourcesBusy', 30),
    ('versionNumberMismatch', 31),
    ('revokedCertificate', 33),
    ('ambiguousDecrypt', 60),
    ('noDecryptKey', 61),
    ('badEncryptedData', 62),
    ('badEnvelopedData', 63),
    ('badAuthenticatedData', 64),
    ('badAuthEnvelopedData', 65),
    ('badKeyAgreeRecipientInfo', 66),
    ('badKEKRecipientInfo', 67),
    ('badEncryptContent', 68),
    ('badEncryptAlgorithm', 69),
    ('missingCiphertext', 70),
    ('decryptFailure', 71),
    ('badMACAlgorithm', 72),
    ('badAuthAttrs', 73),
    ('badUnauthAttrs', 74),
    ('invalidMAC', 75),
    ('mismatchedDigestAlg', 76),
    ('missingCertificate', 77),
    ('tooManySigners', 78),
    ('missingSignedAttributes', 79),
    ('derEncodingNotUsed', 80),
    ('missingContentHints', 81),
    ('invalidAttributeLocation', 82),
    ('badMessageDigest', 83),
    ('badKeyPackage', 84),
    ('badAttributes', 85),
    ('attributeComparisonFailure', 86),
    ('unsupportedSymmetricKeyPackage', 87),
    ('unsupportedAsymmetricKeyPackage', 88),
    ('constraintViolation', 89),
    ('ambiguousDefaultValue', 90),
    ('noMatchingRecipientInfo', 91),
    ('unsupportedKeyWrapAlgorithm', 92),
    ('badKeyTransRecipientInfo', 93),
    ('other', 127)
)


class ErrorCodeChoice(univ.Choice):
    pass

ErrorCodeChoice.componentType = namedtype.NamedTypes(
    namedtype.NamedType('enum', EnumeratedErrorCode()),
    namedtype.NamedType('oid', univ.ObjectIdentifier())
)


class KeyPkgID(univ.OctetString):
    pass


class KeyPkgIdentifier(univ.Choice):
    pass

KeyPkgIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pkgID', KeyPkgID()),
    namedtype.NamedType('attribute', SingleAttribute())
)


class KeyPkgVersion(univ.Integer):
    pass


KeyPkgVersion.namedValues = namedval.NamedValues(
    ('v1', 1),
    ('v2', 2)
)

KeyPkgVersion.subtypeSpec = constraint.ValueRangeConstraint(1, 65535)


id_ct_KP_keyPackageError = univ.ObjectIdentifier('2.16.840.1.101.2.1.2.78.6')

class KeyPackageError(univ.Sequence):
    pass

KeyPackageError.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', KeyPkgVersion().subtype(value='v2')),
    namedtype.OptionalNamedType('errorOf', KeyPkgIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('errorBy', SIREntityName()),
    namedtype.NamedType('errorCode', ErrorCodeChoice())
)


# Key Package Receipt CMS Content Type

id_ct_KP_keyPackageReceipt = univ.ObjectIdentifier('2.16.840.1.101.2.1.2.78.3')

class KeyPackageReceipt(univ.Sequence):
    pass

KeyPackageReceipt.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', KeyPkgVersion().subtype(value='v2')),
    namedtype.NamedType('receiptOf', KeyPkgIdentifier()),
    namedtype.NamedType('receivedBy', SIREntityName())
)


# Key Package Receipt Request Attribute

class KeyPkgReceiptReq(univ.Sequence):
    pass

KeyPkgReceiptReq.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('encryptReceipt', univ.Boolean().subtype(value=0)),
    namedtype.OptionalNamedType('receiptsFrom', SIREntityNames().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('receiptsTo', SIREntityNames())
)


id_aa_KP_keyPkgIdAndReceiptReq = univ.ObjectIdentifier('2.16.840.1.101.2.1.5.65')

class KeyPkgIdentifierAndReceiptReq(univ.Sequence):
    pass

KeyPkgIdentifierAndReceiptReq.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pkgID', KeyPkgID()),
    namedtype.OptionalNamedType('receiptReq', KeyPkgReceiptReq())
)


# Map of Attribute Type OIDs to Attributes are added to
# the ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_aa_KP_keyPkgIdAndReceiptReq: KeyPkgIdentifierAndReceiptReq(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)


# Map of CMC Content Type OIDs to CMC Content Types are added to
# the ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_KP_keyPackageError: KeyPackageError(),
    id_ct_KP_keyPackageReceipt: KeyPackageReceipt(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
