#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Diffie-Hellman Key Agreement
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc2631.txt
# https://www.rfc-editor.org/errata/eid5897
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ


class KeySpecificInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('algorithm', univ.ObjectIdentifier()),
        namedtype.NamedType('counter', univ.OctetString().subtype(
            subtypeSpec=constraint.ValueSizeConstraint(4, 4)))
    )


class OtherInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('keyInfo', KeySpecificInfo()),
        namedtype.OptionalNamedType('partyAInfo', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('suppPubInfo', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )
