#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with some assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add a map for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Protecting Multiple Contents with the CMS
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc4073.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5652

MAX = float('inf')


# Content Collection Content Type and Object Identifier

id_ct_contentCollection = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.19')

class ContentCollection(univ.SequenceOf):
    pass

ContentCollection.componentType = rfc5652.ContentInfo()
ContentCollection.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


# Content With Attributes Content Type and Object Identifier

id_ct_contentWithAttrs = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.20')

class ContentWithAttributes(univ.Sequence):
    pass

ContentWithAttributes.componentType = namedtype.NamedTypes(
    namedtype.NamedType('content', rfc5652.ContentInfo()),
    namedtype.NamedType('attrs', univ.SequenceOf(
        componentType=rfc5652.Attribute()).subtype(
            sizeSpec=constraint.ValueSizeConstraint(1, MAX)))
)


# Map of Content Type OIDs to Content Types is added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_contentCollection: ContentCollection(),
    id_ct_contentWithAttrs: ContentWithAttributes(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
