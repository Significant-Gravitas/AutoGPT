#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
#The scrypt Password-Based Key Derivation Function
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8520.txt
# https://www.rfc-editor.org/errata/eid5871
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280

MAX = float('inf')


id_scrypt = univ.ObjectIdentifier('1.3.6.1.4.1.11591.4.11')


class Scrypt_params(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('salt',
            univ.OctetString()),
        namedtype.NamedType('costParameter',
            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))),
        namedtype.NamedType('blockSize',
            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))),
        namedtype.NamedType('parallelizationParameter',
            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX))),
        namedtype.OptionalNamedType('keyLength',
            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, MAX)))
    )


# Update the Algorithm Identifier map in rfc5280.py

_algorithmIdentifierMapUpdate = {
    id_scrypt: Scrypt_params(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
