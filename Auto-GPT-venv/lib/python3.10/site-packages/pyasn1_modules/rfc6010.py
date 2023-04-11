#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add maps for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Certificate Extension for CMS Content Constraints (CCC)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6010.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


AttributeType = rfc5280.AttributeType

AttributeValue = rfc5280.AttributeValue


id_ct_anyContentType = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.0')


class AttrConstraint(univ.Sequence):
    pass

AttrConstraint.componentType = namedtype.NamedTypes(
    namedtype.NamedType('attrType', AttributeType()),
    namedtype.NamedType('attrValues', univ.SetOf(
        componentType=AttributeValue()).subtype(subtypeSpec=constraint.ValueSizeConstraint(1, MAX)))
)


class AttrConstraintList(univ.SequenceOf):
    pass

AttrConstraintList.componentType = AttrConstraint()
AttrConstraintList.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


class ContentTypeGeneration(univ.Enumerated):
    pass

ContentTypeGeneration.namedValues = namedval.NamedValues(
    ('canSource', 0),
    ('cannotSource', 1)
)


class ContentTypeConstraint(univ.Sequence):
    pass

ContentTypeConstraint.componentType = namedtype.NamedTypes(
    namedtype.NamedType('contentType', univ.ObjectIdentifier()),
    namedtype.DefaultedNamedType('canSource', ContentTypeGeneration().subtype(value='canSource')),
    namedtype.OptionalNamedType('attrConstraints', AttrConstraintList())
)


# CMS Content Constraints (CCC) Extension and Object Identifier

id_pe_cmsContentConstraints = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.18')

class CMSContentConstraints(univ.SequenceOf):
    pass

CMSContentConstraints.componentType = ContentTypeConstraint()
CMSContentConstraints.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


# Map of Certificate Extension OIDs to Extensions
# To be added to the ones that are in rfc5280.py

_certificateExtensionsMap = {
    id_pe_cmsContentConstraints: CMSContentConstraints(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMap)
