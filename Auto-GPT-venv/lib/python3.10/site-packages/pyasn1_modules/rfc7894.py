#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Alternative Challenge Password Attributes for EST
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7894.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5652
from pyasn1_modules import rfc6402
from pyasn1_modules import rfc7191


# SingleAttribute is the same as Attribute in RFC 5652, except that the
# attrValues SET must have one and only one member

Attribute = rfc7191.SingleAttribute


# DirectoryString is the same as RFC 5280, except the length is limited to 255

class DirectoryString(univ.Choice):
    pass

DirectoryString.componentType = namedtype.NamedTypes(
    namedtype.NamedType('teletexString', char.TeletexString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('printableString', char.PrintableString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('universalString', char.UniversalString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('utf8String', char.UTF8String().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255))),
    namedtype.NamedType('bmpString', char.BMPString().subtype(
        subtypeSpec=constraint.ValueSizeConstraint(1, 255)))
)


# OTP Challenge Attribute

id_aa_otpChallenge = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.56')

ub_aa_otpChallenge = univ.Integer(255)

otpChallenge = Attribute()
otpChallenge['attrType'] = id_aa_otpChallenge
otpChallenge['attrValues'][0] = DirectoryString()


# Revocation Challenge Attribute

id_aa_revocationChallenge = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.57')

ub_aa_revocationChallenge = univ.Integer(255)

revocationChallenge = Attribute()
revocationChallenge['attrType'] = id_aa_revocationChallenge
revocationChallenge['attrValues'][0] = DirectoryString()


#  EST Identity Linking Attribute

id_aa_estIdentityLinking = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.58')

ub_aa_est_identity_linking = univ.Integer(255)

estIdentityLinking = Attribute()
estIdentityLinking['attrType'] = id_aa_estIdentityLinking
estIdentityLinking['attrValues'][0] = DirectoryString()


# Map of Attribute Type OIDs to Attributes added to the
# ones that are in rfc6402.py

_cmcControlAttributesMapUpdate = {
    id_aa_otpChallenge: DirectoryString(),
    id_aa_revocationChallenge: DirectoryString(),
    id_aa_estIdentityLinking: DirectoryString(),
}

rfc6402.cmcControlAttributesMap.update(_cmcControlAttributesMapUpdate)
