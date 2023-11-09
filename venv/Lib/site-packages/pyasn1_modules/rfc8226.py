# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from the asn1ate tool, with manual
#   changes to implement appropriate constraints and added comments.
# Modified by Russ Housley to add maps for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# JWT Claim Constraints and TN Authorization List for certificate extensions.
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8226.txt (with errata corrected)

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


class JWTClaimName(char.IA5String):
    pass


class JWTClaimNames(univ.SequenceOf):
    pass

JWTClaimNames.componentType = JWTClaimName()
JWTClaimNames.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class JWTClaimPermittedValues(univ.Sequence):
    pass

JWTClaimPermittedValues.componentType = namedtype.NamedTypes(
    namedtype.NamedType('claim', JWTClaimName()),
    namedtype.NamedType('permitted', univ.SequenceOf(
        componentType=char.UTF8String()).subtype(
            sizeSpec=constraint.ValueSizeConstraint(1, MAX)))
)


class JWTClaimPermittedValuesList(univ.SequenceOf):
    pass

JWTClaimPermittedValuesList.componentType = JWTClaimPermittedValues()
JWTClaimPermittedValuesList.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class JWTClaimConstraints(univ.Sequence):
    pass

JWTClaimConstraints.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('mustInclude',
        JWTClaimNames().subtype(explicitTag=tag.Tag(tag.tagClassContext,
            tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('permittedValues',
        JWTClaimPermittedValuesList().subtype(explicitTag=tag.Tag(tag.tagClassContext,
            tag.tagFormatSimple, 1)))
)

JWTClaimConstraints.subtypeSpec = constraint.ConstraintsUnion(
    constraint.WithComponentsConstraint(
        ('mustInclude', constraint.ComponentPresentConstraint())),
    constraint.WithComponentsConstraint(
        ('permittedValues', constraint.ComponentPresentConstraint()))
)


id_pe_JWTClaimConstraints = _OID(1, 3, 6, 1, 5, 5, 7, 1, 27)


class ServiceProviderCode(char.IA5String):
    pass


class TelephoneNumber(char.IA5String):
    pass

TelephoneNumber.subtypeSpec = constraint.ConstraintsIntersection(
    constraint.ValueSizeConstraint(1, 15),
    constraint.PermittedAlphabetConstraint(
        '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '#', '*')
)


class TelephoneNumberRange(univ.Sequence):
    pass

TelephoneNumberRange.componentType = namedtype.NamedTypes(
    namedtype.NamedType('start', TelephoneNumber()),
    namedtype.NamedType('count',
        univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(2, MAX)))
)


class TNEntry(univ.Choice):
    pass

TNEntry.componentType = namedtype.NamedTypes(
    namedtype.NamedType('spc',
        ServiceProviderCode().subtype(explicitTag=tag.Tag(tag.tagClassContext,
            tag.tagFormatSimple, 0))),
    namedtype.NamedType('range',
        TelephoneNumberRange().subtype(explicitTag=tag.Tag(tag.tagClassContext,
            tag.tagFormatConstructed, 1))),
    namedtype.NamedType('one',
        TelephoneNumber().subtype(explicitTag=tag.Tag(tag.tagClassContext,
            tag.tagFormatSimple, 2)))
)


class TNAuthorizationList(univ.SequenceOf):
    pass

TNAuthorizationList.componentType = TNEntry()
TNAuthorizationList.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

id_pe_TNAuthList = _OID(1, 3, 6, 1, 5, 5, 7, 1, 26)


id_ad_stirTNList = _OID(1, 3, 6, 1, 5, 5, 7, 48, 14)


# Map of Certificate Extension OIDs to Extensions added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_pe_TNAuthList: TNAuthorizationList(),
    id_pe_JWTClaimConstraints: JWTClaimConstraints(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
