#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with some assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Authentication Context Certificate Extension
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7773.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


# Authentication Context Extension

e_legnamnden = univ.ObjectIdentifier('1.2.752.201')

id_eleg_ce = e_legnamnden + (5, )

id_ce_authContext = id_eleg_ce + (1, )


class AuthenticationContext(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('contextType', char.UTF8String()),
        namedtype.OptionalNamedType('contextInfo', char.UTF8String())
    )

class AuthenticationContexts(univ.SequenceOf):
    componentType = AuthenticationContext()
    subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


# Map of Certificate Extension OIDs to Extensions added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_ce_authContext: AuthenticationContexts(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
