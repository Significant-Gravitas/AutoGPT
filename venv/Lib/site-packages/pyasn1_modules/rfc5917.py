#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Clearance Sponsor Attribute
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5917.txt
# https://www.rfc-editor.org/errata/eid4558
# https://www.rfc-editor.org/errata/eid5883
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280


# DirectoryString is the same as RFC 5280, except for two things:
#   1. the length is limited to 64;
#   2. only the 'utf8String' choice remains because the ASN.1
#      specification says: ( WITH COMPONENTS { utf8String PRESENT } )

class DirectoryString(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('utf8String', char.UTF8String().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(1, 64))),
    )


# Clearance Sponsor Attribute

id_clearanceSponsor = univ.ObjectIdentifier((2, 16, 840, 1, 101, 2, 1, 5, 68))

ub_clearance_sponsor = univ.Integer(64)


at_clearanceSponsor = rfc5280.Attribute()
at_clearanceSponsor['type'] = id_clearanceSponsor
at_clearanceSponsor['values'][0] = DirectoryString()


# Add to the map of Attribute Type OIDs to Attributes in rfc5280.py.

_certificateAttributesMapUpdate = {
    id_clearanceSponsor: DirectoryString(),
}

rfc5280.certificateAttributesMap.update(_certificateAttributesMapUpdate)
