#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with some assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Network Access Identifier (NAI) Realm Name for Certificates
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7585.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import univ

from pyasn1_modules import rfc5280


# NAI Realm Name for Certificates

id_pkix = univ.ObjectIdentifier('1.3.6.1.5.5.7')

id_on = id_pkix + (8, )

id_on_naiRealm = id_on + (8, )


ub_naiRealm_length = univ.Integer(255)


class NAIRealm(char.UTF8String):
    subtypeSpec = constraint.ValueSizeConstraint(1, ub_naiRealm_length)


naiRealm = rfc5280.AnotherName()
naiRealm['type-id'] = id_on_naiRealm
naiRealm['value'] = NAIRealm()


# Map of Other Name OIDs to Other Name is added to the
# ones that are in rfc5280.py

_anotherNameMapUpdate = {
    id_on_naiRealm: NAIRealm(),
}

rfc5280.anotherNameMap.update(_anotherNameMapUpdate)
