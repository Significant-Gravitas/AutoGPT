#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS#9: Selected Attribute Types (Version 2.0)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc2985.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc7292
from pyasn1_modules import rfc5958
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5280


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


MAX = float('inf')


# Imports from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier

Attribute = rfc5280.Attribute

EmailAddress = rfc5280.EmailAddress

Extensions = rfc5280.Extensions

Time = rfc5280.Time

X520countryName = rfc5280.X520countryName

X520SerialNumber = rfc5280.X520SerialNumber


# Imports from RFC 5652

ContentInfo = rfc5652.ContentInfo

ContentType = rfc5652.ContentType

Countersignature = rfc5652.Countersignature

MessageDigest = rfc5652.MessageDigest

SignerInfo = rfc5652.SignerInfo

SigningTime = rfc5652.SigningTime


# Imports from RFC 5958

EncryptedPrivateKeyInfo = rfc5958.EncryptedPrivateKeyInfo


# Imports from RFC 7292

PFX = rfc7292.PFX


# TODO:
# Need a place to import PKCS15Token; it does not yet appear in an RFC


# SingleAttribute is the same as Attribute in RFC 5280, except that the
# attrValues SET must have one and only one member

class AttributeType(univ.ObjectIdentifier):
    pass


class AttributeValue(univ.Any):
    pass


class AttributeValues(univ.SetOf):
    pass

AttributeValues.componentType = AttributeValue()


class SingleAttributeValues(univ.SetOf):
    pass

SingleAttributeValues.componentType = AttributeValue()


class SingleAttribute(univ.Sequence):
    pass

SingleAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('type', AttributeType()),
    namedtype.NamedType('values',
        AttributeValues().subtype(sizeSpec=constraint.ValueSizeConstraint(1, 1)),
        openType=opentype.OpenType('type', rfc5280.certificateAttributesMap)
    )
)


# CMSAttribute is the same as Attribute in RFC 5652, and CMSSingleAttribute
# is the companion where the attrValues SET must have one and only one member

CMSAttribute = rfc5652.Attribute


class CMSSingleAttribute(univ.Sequence):
    pass

CMSSingleAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', AttributeType()),
    namedtype.NamedType('attrValues',
        AttributeValues().subtype(sizeSpec=constraint.ValueSizeConstraint(1, 1)),
        openType=opentype.OpenType('attrType', rfc5652.cmsAttributesMap)
    )
)


# DirectoryString is the same as RFC 5280, except the length is limited to 255

class DirectoryString(univ.Choice):
    pass

DirectoryString.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('printableString', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('universalString', char.UniversalString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('utf8String', char.UTF8String().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255)))
)


# PKCS9String is DirectoryString with an additional choice of IA5String,
# and the SIZE is limited to 255

class PKCS9String(univ.Choice):
    pass

PKCS9String.componentType = namedtype.NamedTypes(
    namedtype.NamedType('ia5String', char.IA5String().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('directoryString', DirectoryString())
)


# Upper Bounds

pkcs_9_ub_pkcs9String = univ.Integer(255)

pkcs_9_ub_challengePassword = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_ub_emailAddress = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_ub_friendlyName = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_ub_match = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_ub_signingDescription = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_ub_unstructuredAddress = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_ub_unstructuredName = univ.Integer(pkcs_9_ub_pkcs9String)


ub_name = univ.Integer(32768)

pkcs_9_ub_placeOfBirth = univ.Integer(ub_name)

pkcs_9_ub_pseudonym = univ.Integer(ub_name)


# Object Identifier Arcs

ietf_at = _OID(1, 3, 6, 1, 5, 5, 7, 9)

id_at = _OID(2, 5, 4)

pkcs_9 = _OID(1, 2, 840, 113549, 1, 9)

pkcs_9_mo = _OID(pkcs_9, 0)

smime = _OID(pkcs_9, 16)

certTypes = _OID(pkcs_9, 22)

crlTypes = _OID(pkcs_9, 23)

pkcs_9_oc = _OID(pkcs_9, 24)

pkcs_9_at = _OID(pkcs_9, 25)

pkcs_9_sx = _OID(pkcs_9, 26)

pkcs_9_mr = _OID(pkcs_9, 27)


# Object Identifiers for Syntaxes for use with LDAP-accessible directories

pkcs_9_sx_pkcs9String = _OID(pkcs_9_sx, 1)

pkcs_9_sx_signingTime = _OID(pkcs_9_sx, 2)


# Object Identifiers for object classes

pkcs_9_oc_pkcsEntity = _OID(pkcs_9_oc, 1)

pkcs_9_oc_naturalPerson = _OID(pkcs_9_oc, 2)


# Object Identifiers for matching rules

pkcs_9_mr_caseIgnoreMatch = _OID(pkcs_9_mr, 1)

pkcs_9_mr_signingTimeMatch = _OID(pkcs_9_mr, 2)


# PKCS #7 PDU

pkcs_9_at_pkcs7PDU = _OID(pkcs_9_at, 5)

pKCS7PDU = Attribute()
pKCS7PDU['type'] = pkcs_9_at_pkcs7PDU
pKCS7PDU['values'][0] = ContentInfo()


# PKCS #12 token

pkcs_9_at_userPKCS12 = _OID(2, 16, 840, 1, 113730, 3, 1, 216)

userPKCS12 = Attribute()
userPKCS12['type'] = pkcs_9_at_userPKCS12
userPKCS12['values'][0] = PFX()


# PKCS #15 token

pkcs_9_at_pkcs15Token = _OID(pkcs_9_at, 1)

# TODO: Once PKCS15Token can be imported, this can be included
# 
# pKCS15Token = Attribute()
# userPKCS12['type'] = pkcs_9_at_pkcs15Token
# userPKCS12['values'][0] = PKCS15Token()


# PKCS #8 encrypted private key information

pkcs_9_at_encryptedPrivateKeyInfo = _OID(pkcs_9_at, 2)

encryptedPrivateKeyInfo = Attribute()
encryptedPrivateKeyInfo['type'] = pkcs_9_at_encryptedPrivateKeyInfo
encryptedPrivateKeyInfo['values'][0] = EncryptedPrivateKeyInfo()


# Electronic-mail address

pkcs_9_at_emailAddress = rfc5280.id_emailAddress

emailAddress = Attribute()
emailAddress['type'] = pkcs_9_at_emailAddress
emailAddress['values'][0] = EmailAddress()


# Unstructured name

pkcs_9_at_unstructuredName = _OID(pkcs_9, 2)

unstructuredName = Attribute()
unstructuredName['type'] = pkcs_9_at_unstructuredName
unstructuredName['values'][0] = PKCS9String()


# Unstructured address

pkcs_9_at_unstructuredAddress = _OID(pkcs_9, 8)

unstructuredAddress = Attribute()
unstructuredAddress['type'] = pkcs_9_at_unstructuredAddress
unstructuredAddress['values'][0] = DirectoryString()


# Date of birth

pkcs_9_at_dateOfBirth = _OID(ietf_at, 1)

dateOfBirth = SingleAttribute()
dateOfBirth['type'] = pkcs_9_at_dateOfBirth
dateOfBirth['values'][0] = useful.GeneralizedTime()


# Place of birth

pkcs_9_at_placeOfBirth = _OID(ietf_at, 2)

placeOfBirth = SingleAttribute()
placeOfBirth['type'] = pkcs_9_at_placeOfBirth
placeOfBirth['values'][0] = DirectoryString()


# Gender

class GenderString(char.PrintableString):
    pass

GenderString.subtypeSpec = constraint.ValueSizeConstraint(1, 1)
GenderString.subtypeSpec = constraint.SingleValueConstraint("M", "F", "m", "f")


pkcs_9_at_gender = _OID(ietf_at, 3)

gender = SingleAttribute()
gender['type'] = pkcs_9_at_gender
gender['values'][0] = GenderString()


# Country of citizenship

pkcs_9_at_countryOfCitizenship = _OID(ietf_at, 4)

countryOfCitizenship = Attribute()
countryOfCitizenship['type'] = pkcs_9_at_countryOfCitizenship
countryOfCitizenship['values'][0] = X520countryName()


#  Country of residence

pkcs_9_at_countryOfResidence = _OID(ietf_at, 5)

countryOfResidence = Attribute()
countryOfResidence['type'] = pkcs_9_at_countryOfResidence
countryOfResidence['values'][0] = X520countryName()


# Pseudonym

id_at_pseudonym = _OID(2, 5, 4, 65)

pseudonym = Attribute()
pseudonym['type'] = id_at_pseudonym
pseudonym['values'][0] = DirectoryString()


# Serial number

id_at_serialNumber = rfc5280.id_at_serialNumber

serialNumber = Attribute()
serialNumber['type'] = id_at_serialNumber
serialNumber['values'][0] = X520SerialNumber()


# Content type

pkcs_9_at_contentType = rfc5652.id_contentType

contentType = CMSSingleAttribute()
contentType['attrType'] = pkcs_9_at_contentType
contentType['attrValues'][0] = ContentType()


# Message digest

pkcs_9_at_messageDigest = rfc5652.id_messageDigest

messageDigest = CMSSingleAttribute()
messageDigest['attrType'] = pkcs_9_at_messageDigest
messageDigest['attrValues'][0] = MessageDigest()


# Signing time

pkcs_9_at_signingTime = rfc5652.id_signingTime

signingTime = CMSSingleAttribute()
signingTime['attrType'] = pkcs_9_at_signingTime
signingTime['attrValues'][0] = SigningTime()


# Random nonce

class RandomNonce(univ.OctetString):
    pass

RandomNonce.subtypeSpec = constraint.ValueSizeConstraint(4, MAX)


pkcs_9_at_randomNonce = _OID(pkcs_9_at, 3)

randomNonce = CMSSingleAttribute()
randomNonce['attrType'] = pkcs_9_at_randomNonce
randomNonce['attrValues'][0] = RandomNonce()


# Sequence number

class SequenceNumber(univ.Integer):
    pass

SequenceNumber.subtypeSpec = constraint.ValueRangeConstraint(1, MAX)


pkcs_9_at_sequenceNumber = _OID(pkcs_9_at, 4)

sequenceNumber = CMSSingleAttribute()
sequenceNumber['attrType'] = pkcs_9_at_sequenceNumber
sequenceNumber['attrValues'][0] = SequenceNumber()


# Countersignature

pkcs_9_at_counterSignature = rfc5652.id_countersignature

counterSignature = CMSAttribute()
counterSignature['attrType'] = pkcs_9_at_counterSignature
counterSignature['attrValues'][0] = Countersignature()


# Challenge password

pkcs_9_at_challengePassword = _OID(pkcs_9, 7)

challengePassword = SingleAttribute()
challengePassword['type'] = pkcs_9_at_challengePassword
challengePassword['values'][0] = DirectoryString()


# Extension request

class ExtensionRequest(Extensions):
    pass


pkcs_9_at_extensionRequest = _OID(pkcs_9, 14)

extensionRequest = SingleAttribute()
extensionRequest['type'] = pkcs_9_at_extensionRequest
extensionRequest['values'][0] = ExtensionRequest()


# Extended-certificate attributes (deprecated)

class AttributeSet(univ.SetOf):
    pass

AttributeSet.componentType = Attribute()


pkcs_9_at_extendedCertificateAttributes = _OID(pkcs_9, 9)

extendedCertificateAttributes = SingleAttribute()
extendedCertificateAttributes['type'] = pkcs_9_at_extendedCertificateAttributes
extendedCertificateAttributes['values'][0] = AttributeSet()


# Friendly name

class FriendlyName(char.BMPString):
    pass

FriendlyName.subtypeSpec = constraint.ValueSizeConstraint(1, pkcs_9_ub_friendlyName)


pkcs_9_at_friendlyName = _OID(pkcs_9, 20)

friendlyName = SingleAttribute()
friendlyName['type'] = pkcs_9_at_friendlyName
friendlyName['values'][0] = FriendlyName()


# Local key identifier

pkcs_9_at_localKeyId = _OID(pkcs_9, 21)

localKeyId = SingleAttribute()
localKeyId['type'] = pkcs_9_at_localKeyId
localKeyId['values'][0] = univ.OctetString()


# Signing description

pkcs_9_at_signingDescription = _OID(pkcs_9, 13)

signingDescription = CMSSingleAttribute()
signingDescription['attrType'] = pkcs_9_at_signingDescription
signingDescription['attrValues'][0] = DirectoryString()


# S/MIME capabilities

class SMIMECapability(AlgorithmIdentifier):
    pass


class SMIMECapabilities(univ.SequenceOf):
    pass

SMIMECapabilities.componentType = SMIMECapability()


pkcs_9_at_smimeCapabilities = _OID(pkcs_9, 15)

smimeCapabilities = CMSSingleAttribute()
smimeCapabilities['attrType'] = pkcs_9_at_smimeCapabilities
smimeCapabilities['attrValues'][0] = SMIMECapabilities()


# Certificate Attribute Map

_certificateAttributesMapUpdate = {
    # Attribute types for use with the "pkcsEntity" object class
    pkcs_9_at_pkcs7PDU: ContentInfo(),
    pkcs_9_at_userPKCS12: PFX(),
    # TODO: Once PKCS15Token can be imported, this can be included
    # pkcs_9_at_pkcs15Token: PKCS15Token(),
    pkcs_9_at_encryptedPrivateKeyInfo: EncryptedPrivateKeyInfo(),
    # Attribute types for use with the "naturalPerson" object class
    pkcs_9_at_emailAddress: EmailAddress(),
    pkcs_9_at_unstructuredName: PKCS9String(),
    pkcs_9_at_unstructuredAddress: DirectoryString(),
    pkcs_9_at_dateOfBirth: useful.GeneralizedTime(),
    pkcs_9_at_placeOfBirth: DirectoryString(),
    pkcs_9_at_gender: GenderString(),
    pkcs_9_at_countryOfCitizenship: X520countryName(),
    pkcs_9_at_countryOfResidence: X520countryName(),
    id_at_pseudonym: DirectoryString(),
    id_at_serialNumber: X520SerialNumber(),
    # Attribute types for use with PKCS #10 certificate requests
    pkcs_9_at_challengePassword: DirectoryString(),
    pkcs_9_at_extensionRequest: ExtensionRequest(),
    pkcs_9_at_extendedCertificateAttributes: AttributeSet(),
}

rfc5280.certificateAttributesMap.update(_certificateAttributesMapUpdate)


# CMS Attribute Map

# Note: pkcs_9_at_smimeCapabilities is not included in the map because
#       the definition in RFC 5751 is preferred, which produces the same
#       encoding, but it allows different parameters for SMIMECapability
#       and AlgorithmIdentifier.

_cmsAttributesMapUpdate = {
    # Attribute types for use in PKCS #7 data (a.k.a. CMS)
    pkcs_9_at_contentType: ContentType(),
    pkcs_9_at_messageDigest: MessageDigest(),
    pkcs_9_at_signingTime: SigningTime(),
    pkcs_9_at_randomNonce: RandomNonce(),
    pkcs_9_at_sequenceNumber: SequenceNumber(),
    pkcs_9_at_counterSignature: Countersignature(),
    # Attributes for use in PKCS #12 "PFX" PDUs or PKCS #15 tokens
    pkcs_9_at_friendlyName: FriendlyName(),
    pkcs_9_at_localKeyId: univ.OctetString(),
    pkcs_9_at_signingDescription: DirectoryString(),
    # pkcs_9_at_smimeCapabilities: SMIMECapabilities(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)
