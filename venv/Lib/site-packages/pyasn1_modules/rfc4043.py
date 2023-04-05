#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Internet X.509 Public Key Infrastructure Permanent Identifier
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc4043.txt
#

from pyasn1.type import char
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280


id_pkix = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, ))

id_on = id_pkix + (8, )

id_on_permanentIdentifier = id_on + (3, )


class PermanentIdentifier(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('identifierValue', char.UTF8String()),
        namedtype.OptionalNamedType('assigner', univ.ObjectIdentifier())
    )


# Map of Other Name OIDs to Other Name is added to the
# ones that are in rfc5280.py

_anotherNameMapUpdate = {
    id_on_permanentIdentifier: PermanentIdentifier(),
}

rfc5280.anotherNameMap.update(_anotherNameMapUpdate)
