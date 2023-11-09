# This file is being contributed to of pyasn1-modules software.
#
# Created by Russ Housley without assistance from the asn1ate tool.
# Modified by Russ Housley to add a map for use with opentypes and
#   simplify the code for the object identifier assignment.
#
# Copyright (c) 2018, 2019 Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
#  Authenticated-Enveloped-Data for the Cryptographic Message Syntax (CMS)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5083.txt

from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5652

MAX = float('inf')


# CMS Authenticated-Enveloped-Data Content Type

id_ct_authEnvelopedData = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.23')

class AuthEnvelopedData(univ.Sequence):
    pass

AuthEnvelopedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', rfc5652.CMSVersion()),
    namedtype.OptionalNamedType('originatorInfo', rfc5652.OriginatorInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('recipientInfos', rfc5652.RecipientInfos()),
    namedtype.NamedType('authEncryptedContentInfo', rfc5652.EncryptedContentInfo()),
    namedtype.OptionalNamedType('authAttrs', rfc5652.AuthAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('mac', rfc5652.MessageAuthenticationCode()),
    namedtype.OptionalNamedType('unauthAttrs', rfc5652.UnauthAttributes().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)


# Map of Content Type OIDs to Content Types is added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_authEnvelopedData: AuthEnvelopedData(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
