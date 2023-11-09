# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from the asn1ate tool.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS #12: Personal Information Exchange Syntax v1.1
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7292.txt
# https://www.rfc-editor.org/errata_search.php?rfc=7292

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc2315
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5958


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


# Initialize the maps used in PKCS#12

pkcs12BagTypeMap = { }

pkcs12CertBagMap = { }

pkcs12CRLBagMap = { }

pkcs12SecretBagMap = { }


# Imports from RFC 2315, RFC 5652, and RFC 5958

DigestInfo = rfc2315.DigestInfo


ContentInfo = rfc5652.ContentInfo

PKCS12Attribute = rfc5652.Attribute


EncryptedPrivateKeyInfo = rfc5958.EncryptedPrivateKeyInfo

PrivateKeyInfo = rfc5958.PrivateKeyInfo


# CMSSingleAttribute is the same as Attribute in RFC 5652 except the attrValues
# SET must have one and only one member

class AttributeType(univ.ObjectIdentifier):
    pass


class AttributeValue(univ.Any):
    pass


class AttributeValues(univ.SetOf):
    pass

AttributeValues.componentType = AttributeValue()


class CMSSingleAttribute(univ.Sequence):
    pass

CMSSingleAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', AttributeType()),
    namedtype.NamedType('attrValues',
        AttributeValues().subtype(sizeSpec=constraint.ValueSizeConstraint(1, 1)),
        openType=opentype.OpenType('attrType', rfc5652.cmsAttributesMap)
    )
)


# Object identifier arcs

rsadsi = _OID(1, 2, 840, 113549)

pkcs = _OID(rsadsi, 1)

pkcs_9 = _OID(pkcs, 9)

certTypes = _OID(pkcs_9, 22)

crlTypes = _OID(pkcs_9, 23)

pkcs_12 = _OID(pkcs, 12)


# PBE Algorithm Identifiers and Parameters Structure

pkcs_12PbeIds = _OID(pkcs_12, 1)

pbeWithSHAAnd128BitRC4 = _OID(pkcs_12PbeIds, 1)

pbeWithSHAAnd40BitRC4 = _OID(pkcs_12PbeIds, 2)

pbeWithSHAAnd3_KeyTripleDES_CBC = _OID(pkcs_12PbeIds, 3)

pbeWithSHAAnd2_KeyTripleDES_CBC = _OID(pkcs_12PbeIds, 4)

pbeWithSHAAnd128BitRC2_CBC = _OID(pkcs_12PbeIds, 5)

pbeWithSHAAnd40BitRC2_CBC = _OID(pkcs_12PbeIds, 6)


class Pkcs_12PbeParams(univ.Sequence):
    pass

Pkcs_12PbeParams.componentType = namedtype.NamedTypes(
    namedtype.NamedType('salt', univ.OctetString()),
    namedtype.NamedType('iterations', univ.Integer())
)


# Bag types

bagtypes = _OID(pkcs_12, 10, 1)

class BAG_TYPE(univ.Sequence):
    pass

BAG_TYPE.componentType = namedtype.NamedTypes(
    namedtype.NamedType('id', univ.ObjectIdentifier()),
    namedtype.NamedType('unnamed1', univ.Any(),
        openType=opentype.OpenType('attrType', pkcs12BagTypeMap)
    )
)


id_keyBag = _OID(bagtypes, 1)

class KeyBag(PrivateKeyInfo):
    pass


id_pkcs8ShroudedKeyBag = _OID(bagtypes, 2)

class PKCS8ShroudedKeyBag(EncryptedPrivateKeyInfo):
    pass


id_certBag = _OID(bagtypes, 3)

class CertBag(univ.Sequence):
    pass

CertBag.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certId', univ.ObjectIdentifier()),
    namedtype.NamedType('certValue',
        univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)),
        openType=opentype.OpenType('certId', pkcs12CertBagMap)
    )
)


x509Certificate = CertBag()
x509Certificate['certId'] = _OID(certTypes, 1)
x509Certificate['certValue'] = univ.OctetString()
# DER-encoded X.509 certificate stored in OCTET STRING


sdsiCertificate = CertBag()
sdsiCertificate['certId'] = _OID(certTypes, 2)
sdsiCertificate['certValue'] = char.IA5String()
# Base64-encoded SDSI certificate stored in IA5String


id_CRLBag = _OID(bagtypes, 4)

class CRLBag(univ.Sequence):
    pass

CRLBag.componentType = namedtype.NamedTypes(
    namedtype.NamedType('crlId', univ.ObjectIdentifier()),
    namedtype.NamedType('crlValue',
        univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)),
                openType=opentype.OpenType('crlId', pkcs12CRLBagMap)
    )
)


x509CRL = CRLBag()
x509CRL['crlId'] = _OID(crlTypes, 1)
x509CRL['crlValue'] = univ.OctetString()
# DER-encoded X.509 CRL stored in OCTET STRING


id_secretBag = _OID(bagtypes, 5)

class SecretBag(univ.Sequence):
    pass

SecretBag.componentType = namedtype.NamedTypes(
    namedtype.NamedType('secretTypeId', univ.ObjectIdentifier()),
    namedtype.NamedType('secretValue',
        univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)),
        openType=opentype.OpenType('secretTypeId', pkcs12SecretBagMap)
    )
)


id_safeContentsBag = _OID(bagtypes, 6)

class SafeBag(univ.Sequence):
    pass

SafeBag.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bagId', univ.ObjectIdentifier()),
    namedtype.NamedType('bagValue',
        univ.Any().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)),
        openType=opentype.OpenType('bagId', pkcs12BagTypeMap)
    ),
    namedtype.OptionalNamedType('bagAttributes',
        univ.SetOf(componentType=PKCS12Attribute())
    )
)


class SafeContents(univ.SequenceOf):
    pass

SafeContents.componentType = SafeBag()


# The PFX PDU

class AuthenticatedSafe(univ.SequenceOf):
    pass

AuthenticatedSafe.componentType = ContentInfo()
# Data if unencrypted
# EncryptedData if password-encrypted
# EnvelopedData if public key-encrypted


class MacData(univ.Sequence):
    pass

MacData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('mac', DigestInfo()),
    namedtype.NamedType('macSalt', univ.OctetString()),
    namedtype.DefaultedNamedType('iterations', univ.Integer().subtype(value=1))
    # Note: The default is for historical reasons and its use is deprecated
)


class PFX(univ.Sequence):
    pass

PFX.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version',
        univ.Integer(namedValues=namedval.NamedValues(('v3', 3)))
    ),
    namedtype.NamedType('authSafe', ContentInfo()),
    namedtype.OptionalNamedType('macData', MacData())
)


# Local key identifier (also defined as certificateAttribute in rfc2985.py)

pkcs_9_at_localKeyId = _OID(pkcs_9, 21)

localKeyId = CMSSingleAttribute()
localKeyId['attrType'] = pkcs_9_at_localKeyId
localKeyId['attrValues'][0] = univ.OctetString()


# Friendly name (also defined as certificateAttribute in rfc2985.py)

pkcs_9_ub_pkcs9String = univ.Integer(255)

pkcs_9_ub_friendlyName = univ.Integer(pkcs_9_ub_pkcs9String)

pkcs_9_at_friendlyName = _OID(pkcs_9, 20)

class FriendlyName(char.BMPString):
    pass

FriendlyName.subtypeSpec = constraint.ValueSizeConstraint(1, pkcs_9_ub_friendlyName)


friendlyName = CMSSingleAttribute()
friendlyName['attrType'] = pkcs_9_at_friendlyName
friendlyName['attrValues'][0] = FriendlyName()


# Update the PKCS#12 maps

_pkcs12BagTypeMap = {
    id_keyBag: KeyBag(),
    id_pkcs8ShroudedKeyBag: PKCS8ShroudedKeyBag(),
    id_certBag: CertBag(),
    id_CRLBag: CRLBag(),
    id_secretBag: SecretBag(),
    id_safeContentsBag: SafeBag(),
}

pkcs12BagTypeMap.update(_pkcs12BagTypeMap)


_pkcs12CertBagMap = {
    _OID(certTypes, 1): univ.OctetString(),
    _OID(certTypes, 2): char.IA5String(),
}

pkcs12CertBagMap.update(_pkcs12CertBagMap)


_pkcs12CRLBagMap = {
    _OID(crlTypes, 1): univ.OctetString(),
}

pkcs12CRLBagMap.update(_pkcs12CRLBagMap)


# Update the Algorithm Identifier map

_algorithmIdentifierMapUpdate = {
    pbeWithSHAAnd128BitRC4: Pkcs_12PbeParams(),
    pbeWithSHAAnd40BitRC4: Pkcs_12PbeParams(),
    pbeWithSHAAnd3_KeyTripleDES_CBC: Pkcs_12PbeParams(),
    pbeWithSHAAnd2_KeyTripleDES_CBC: Pkcs_12PbeParams(),
    pbeWithSHAAnd128BitRC2_CBC: Pkcs_12PbeParams(),
    pbeWithSHAAnd40BitRC2_CBC: Pkcs_12PbeParams(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)


# Update the CMS Attribute map

_cmsAttributesMapUpdate = {
    pkcs_9_at_friendlyName: FriendlyName(),
    pkcs_9_at_localKeyId: univ.OctetString(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)
