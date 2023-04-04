#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add maps for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# X.509 Extensions for IP Addresses and AS Identifiers
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc3779.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280


# IP Address Delegation Extension

id_pe_ipAddrBlocks = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.7')


class IPAddress(univ.BitString):
    pass


class IPAddressRange(univ.Sequence):
    pass

IPAddressRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('min', IPAddress()),
    namedtype.NamedType('max', IPAddress())
)


class IPAddressOrRange(univ.Choice):
    pass

IPAddressOrRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('addressPrefix', IPAddress()),
    namedtype.NamedType('addressRange', IPAddressRange())
)


class IPAddressChoice(univ.Choice):
    pass

IPAddressChoice.componentType = namedtype.NamedTypes(
    namedtype.NamedType('inherit', univ.Null()),
    namedtype.NamedType('addressesOrRanges', univ.SequenceOf(
        componentType=IPAddressOrRange())
    )
)


class IPAddressFamily(univ.Sequence):
    pass

IPAddressFamily.componentType = namedtype.NamedTypes(
    namedtype.NamedType('addressFamily', univ.OctetString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(2, 3))),
    namedtype.NamedType('ipAddressChoice', IPAddressChoice())
)


class IPAddrBlocks(univ.SequenceOf):
    pass

IPAddrBlocks.componentType = IPAddressFamily()


# Autonomous System Identifier Delegation Extension

id_pe_autonomousSysIds = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.8')


class ASId(univ.Integer):
    pass


class ASRange(univ.Sequence):
    pass

ASRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('min', ASId()),
    namedtype.NamedType('max', ASId())
)


class ASIdOrRange(univ.Choice):
    pass

ASIdOrRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('id', ASId()),
    namedtype.NamedType('range', ASRange())
)


class ASIdentifierChoice(univ.Choice):
    pass

ASIdentifierChoice.componentType = namedtype.NamedTypes(
    namedtype.NamedType('inherit', univ.Null()),
    namedtype.NamedType('asIdsOrRanges', univ.SequenceOf(
        componentType=ASIdOrRange())
    )
)


class ASIdentifiers(univ.Sequence):
    pass

ASIdentifiers.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('asnum', ASIdentifierChoice().subtype(
        explicitTag=tag.Tag(tag.tagClassContext,
        tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('rdi', ASIdentifierChoice().subtype(
        explicitTag=tag.Tag(tag.tagClassContext,
        tag.tagFormatConstructed, 1)))
)


# Map of Certificate Extension OIDs to Extensions is added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_pe_ipAddrBlocks: IPAddrBlocks(),
    id_pe_autonomousSysIds: ASIdentifiers(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
