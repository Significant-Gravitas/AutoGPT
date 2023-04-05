#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# CMS Symmetric Key Package Content Type
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6031.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc5652
from pyasn1_modules import rfc6019


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))
    return univ.ObjectIdentifier(output)


MAX = float('inf')

id_pskc = univ.ObjectIdentifier('1.2.840.113549.1.9.16.12')


# Symmetric Key Package Attributes

id_pskc_manufacturer = _OID(id_pskc, 1)

class at_pskc_manufacturer(char.UTF8String):
    pass


id_pskc_serialNo = _OID(id_pskc, 2)

class at_pskc_serialNo(char.UTF8String):
    pass


id_pskc_model = _OID(id_pskc, 3)

class at_pskc_model(char.UTF8String):
    pass


id_pskc_issueNo = _OID(id_pskc, 4)

class at_pskc_issueNo(char.UTF8String):
    pass


id_pskc_deviceBinding = _OID(id_pskc, 5)

class at_pskc_deviceBinding(char.UTF8String):
    pass


id_pskc_deviceStartDate = _OID(id_pskc, 6)

class at_pskc_deviceStartDate(useful.GeneralizedTime):
    pass


id_pskc_deviceExpiryDate = _OID(id_pskc, 7)

class at_pskc_deviceExpiryDate(useful.GeneralizedTime):
    pass


id_pskc_moduleId = _OID(id_pskc, 8)

class at_pskc_moduleId(char.UTF8String):
    pass


id_pskc_deviceUserId = _OID(id_pskc, 26)

class at_pskc_deviceUserId(char.UTF8String):
    pass


# Symmetric Key Attributes

id_pskc_keyId = _OID(id_pskc, 9)

class at_pskc_keyUserId(char.UTF8String):
    pass


id_pskc_algorithm = _OID(id_pskc, 10)

class at_pskc_algorithm(char.UTF8String):
    pass


id_pskc_issuer = _OID(id_pskc, 11)

class at_pskc_issuer(char.UTF8String):
    pass


id_pskc_keyProfileId = _OID(id_pskc, 12)

class at_pskc_keyProfileId(char.UTF8String):
    pass


id_pskc_keyReference = _OID(id_pskc, 13)

class at_pskc_keyReference(char.UTF8String):
    pass


id_pskc_friendlyName = _OID(id_pskc, 14)

class FriendlyName(univ.Sequence):
    pass

FriendlyName.componentType = namedtype.NamedTypes(
    namedtype.NamedType('friendlyName', char.UTF8String()),
    namedtype.OptionalNamedType('friendlyNameLangTag', char.UTF8String())
)

class at_pskc_friendlyName(FriendlyName):
    pass


id_pskc_algorithmParameters = _OID(id_pskc, 15)

class Encoding(char.UTF8String):
    pass

Encoding.namedValues = namedval.NamedValues(
    ('dec',   "DECIMAL"),
    ('hex',   "HEXADECIMAL"),
    ('alpha', "ALPHANUMERIC"),
    ('b64',   "BASE64"),
    ('bin',   "BINARY")
)

Encoding.subtypeSpec = constraint.SingleValueConstraint(
    "DECIMAL", "HEXADECIMAL", "ALPHANUMERIC", "BASE64", "BINARY" )

class ChallengeFormat(univ.Sequence):
    pass

ChallengeFormat.componentType = namedtype.NamedTypes(
    namedtype.NamedType('encoding', Encoding()),
    namedtype.DefaultedNamedType('checkDigit',
        univ.Boolean().subtype(value=0)),
    namedtype.NamedType('min', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX))),
    namedtype.NamedType('max', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX)))
)

class ResponseFormat(univ.Sequence):
    pass

ResponseFormat.componentType = namedtype.NamedTypes(
    namedtype.NamedType('encoding', Encoding()),
    namedtype.NamedType('length', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX))),
    namedtype.DefaultedNamedType('checkDigit',
        univ.Boolean().subtype(value=0))
)

class PSKCAlgorithmParameters(univ.Choice):
    pass

PSKCAlgorithmParameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('suite', char.UTF8String()),
    namedtype.NamedType('challengeFormat', ChallengeFormat().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('responseFormat', ResponseFormat().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
)

class at_pskc_algorithmParameters(PSKCAlgorithmParameters):
    pass


id_pskc_counter = _OID(id_pskc, 16)

class at_pskc_counter(univ.Integer):
    pass

at_pskc_counter.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


id_pskc_time = _OID(id_pskc, 17)

class at_pskc_time(rfc6019.BinaryTime):
    pass


id_pskc_timeInterval = _OID(id_pskc, 18)

class at_pskc_timeInterval(univ.Integer):
    pass

at_pskc_timeInterval.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


id_pskc_timeDrift = _OID(id_pskc, 19)

class at_pskc_timeDrift(univ.Integer):
    pass

at_pskc_timeDrift.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


id_pskc_valueMAC = _OID(id_pskc, 20)

class ValueMac(univ.Sequence):
    pass

ValueMac.componentType = namedtype.NamedTypes(
    namedtype.NamedType('macAlgorithm', char.UTF8String()),
    namedtype.NamedType('mac', char.UTF8String())
)

class at_pskc_valueMAC(ValueMac):
    pass


id_pskc_keyUserId = _OID(id_pskc, 27)

class at_pskc_keyId(char.UTF8String):
    pass


id_pskc_keyStartDate = _OID(id_pskc, 21)

class at_pskc_keyStartDate(useful.GeneralizedTime):
    pass


id_pskc_keyExpiryDate = _OID(id_pskc, 22)

class at_pskc_keyExpiryDate(useful.GeneralizedTime):
    pass


id_pskc_numberOfTransactions = _OID(id_pskc, 23)

class at_pskc_numberOfTransactions(univ.Integer):
    pass
    
at_pskc_numberOfTransactions.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


id_pskc_keyUsages = _OID(id_pskc, 24)

class PSKCKeyUsage(char.UTF8String):
    pass

PSKCKeyUsage.namedValues = namedval.NamedValues(
    ('otp',       "OTP"),
    ('cr',        "CR"),
    ('encrypt',   "Encrypt"),
    ('integrity', "Integrity"),
    ('verify',    "Verify"),
    ('unlock',    "Unlock"),
    ('decrypt',   "Decrypt"),
    ('keywrap',   "KeyWrap"),
    ('unwrap',    "Unwrap"),
    ('derive',    "Derive"),
    ('generate',  "Generate")
)

PSKCKeyUsage.subtypeSpec = constraint.SingleValueConstraint(
    "OTP", "CR", "Encrypt", "Integrity", "Verify", "Unlock",
    "Decrypt", "KeyWrap", "Unwrap", "Derive", "Generate" )

class PSKCKeyUsages(univ.SequenceOf):
    pass

PSKCKeyUsages.componentType = PSKCKeyUsage()

class at_pskc_keyUsage(PSKCKeyUsages):
    pass


id_pskc_pinPolicy = _OID(id_pskc, 25)

class PINUsageMode(char.UTF8String):
    pass

PINUsageMode.namedValues = namedval.NamedValues(
    ("local",       "Local"),
    ("prepend",     "Prepend"),
    ("append",      "Append"),
    ("algorithmic", "Algorithmic")
)

PINUsageMode.subtypeSpec = constraint.SingleValueConstraint(
    "Local", "Prepend", "Append", "Algorithmic" )

class PINPolicy(univ.Sequence):
    pass

PINPolicy.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('pinKeyId', char.UTF8String().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('pinUsageMode', PINUsageMode().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('maxFailedAttempts', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('minLength', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
    namedtype.OptionalNamedType('maxLength', univ.Integer().subtype(
        subtypeSpec=constraint.ValueRangeConstraint(0, MAX)).subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))),
    namedtype.OptionalNamedType('pinEncoding', Encoding().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5)))
)

class at_pskc_pinPolicy(PINPolicy):
    pass


# Map of Symmetric Key Package Attribute OIDs to Attributes

sKeyPkgAttributesMap = {
     id_pskc_manufacturer: at_pskc_manufacturer(),
     id_pskc_serialNo: at_pskc_serialNo(),
     id_pskc_model: at_pskc_model(),
     id_pskc_issueNo: at_pskc_issueNo(),
     id_pskc_deviceBinding: at_pskc_deviceBinding(),
     id_pskc_deviceStartDate: at_pskc_deviceStartDate(),
     id_pskc_deviceExpiryDate: at_pskc_deviceExpiryDate(),
     id_pskc_moduleId: at_pskc_moduleId(),
     id_pskc_deviceUserId: at_pskc_deviceUserId(),
}


# Map of Symmetric Key Attribute OIDs to Attributes

sKeyAttributesMap = {
     id_pskc_keyId: at_pskc_keyId(),
     id_pskc_algorithm: at_pskc_algorithm(),
     id_pskc_issuer: at_pskc_issuer(),
     id_pskc_keyProfileId: at_pskc_keyProfileId(),
     id_pskc_keyReference: at_pskc_keyReference(),
     id_pskc_friendlyName: at_pskc_friendlyName(),
     id_pskc_algorithmParameters: at_pskc_algorithmParameters(),
     id_pskc_counter: at_pskc_counter(),
     id_pskc_time: at_pskc_time(),
     id_pskc_timeInterval: at_pskc_timeInterval(),
     id_pskc_timeDrift: at_pskc_timeDrift(),
     id_pskc_valueMAC: at_pskc_valueMAC(),
     id_pskc_keyUserId: at_pskc_keyUserId(),
     id_pskc_keyStartDate: at_pskc_keyStartDate(),
     id_pskc_keyExpiryDate: at_pskc_keyExpiryDate(),
     id_pskc_numberOfTransactions: at_pskc_numberOfTransactions(),
     id_pskc_keyUsages: at_pskc_keyUsage(),
     id_pskc_pinPolicy: at_pskc_pinPolicy(),
}


# This definition replaces Attribute() from rfc5652.py; it is the same except
# that opentype is added with sKeyPkgAttributesMap and sKeyAttributesMap

class AttributeType(univ.ObjectIdentifier):
    pass


class AttributeValue(univ.Any):
    pass


class SKeyAttribute(univ.Sequence):
    pass

SKeyAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', AttributeType()),
    namedtype.NamedType('attrValues',
        univ.SetOf(componentType=AttributeValue()),
        openType=opentype.OpenType('attrType', sKeyAttributesMap)
    )
)


class SKeyPkgAttribute(univ.Sequence):
    pass

SKeyPkgAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', AttributeType()),
    namedtype.NamedType('attrValues',
        univ.SetOf(componentType=AttributeValue()),
        openType=opentype.OpenType('attrType', sKeyPkgAttributesMap)
    )
)


# Symmetric Key Package Content Type

id_ct_KP_sKeyPackage = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.25')


class KeyPkgVersion(univ.Integer):
    pass

KeyPkgVersion.namedValues = namedval.NamedValues(
    ('v1', 1)
)


class OneSymmetricKey(univ.Sequence):
    pass

OneSymmetricKey.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('sKeyAttrs',
        univ.SequenceOf(componentType=SKeyAttribute()).subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, MAX))),
    namedtype.OptionalNamedType('sKey', univ.OctetString())
)

OneSymmetricKey.sizeSpec = univ.Sequence.sizeSpec + constraint.ValueSizeConstraint(1, 2)


class SymmetricKeys(univ.SequenceOf):
    pass

SymmetricKeys.componentType = OneSymmetricKey()
SymmetricKeys.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


class SymmetricKeyPackage(univ.Sequence):
    pass

SymmetricKeyPackage.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('version', KeyPkgVersion().subtype(value='v1')),
    namedtype.OptionalNamedType('sKeyPkgAttrs',
        univ.SequenceOf(componentType=SKeyPkgAttribute()).subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, MAX),
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('sKeys', SymmetricKeys())
)


# Map of Content Type OIDs to Content Types are
# added to the ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_KP_sKeyPackage: SymmetricKeyPackage(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
