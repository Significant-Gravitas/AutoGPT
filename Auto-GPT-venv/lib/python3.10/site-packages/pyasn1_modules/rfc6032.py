#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# CMS Encrypted Key Package Content Type
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6032.txt
#

from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5083


# Content Decryption Key Identifier attribute

id_aa_KP_contentDecryptKeyID = univ.ObjectIdentifier('2.16.840.1.101.2.1.5.66')

class ContentDecryptKeyID(univ.OctetString):
    pass

aa_content_decrypt_key_identifier = rfc5652.Attribute()
aa_content_decrypt_key_identifier['attrType'] = id_aa_KP_contentDecryptKeyID
aa_content_decrypt_key_identifier['attrValues'][0] = ContentDecryptKeyID()


# Encrypted Key Package Content Type

id_ct_KP_encryptedKeyPkg = univ.ObjectIdentifier('2.16.840.1.101.2.1.2.78.2')

class EncryptedKeyPackage(univ.Choice):
    pass

EncryptedKeyPackage.componentType = namedtype.NamedTypes(
    namedtype.NamedType('encrypted', rfc5652.EncryptedData()),
    namedtype.NamedType('enveloped', rfc5652.EnvelopedData().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('authEnveloped', rfc5083.AuthEnvelopedData().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


# Map of Attribute Type OIDs to Attributes are
# added to the ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_aa_KP_contentDecryptKeyID: ContentDecryptKeyID(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)


# Map of Content Type OIDs to Content Types are
# added to the ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_KP_encryptedKeyPkg: EncryptedKeyPackage(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
