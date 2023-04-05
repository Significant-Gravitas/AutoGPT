#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# SNMPv3 message syntax
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc3412.txt
#
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc1905


class ScopedPDU(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('contextEngineId', univ.OctetString()),
        namedtype.NamedType('contextName', univ.OctetString()),
        namedtype.NamedType('data', rfc1905.PDUs())
    )


class ScopedPduData(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('plaintext', ScopedPDU()),
        namedtype.NamedType('encryptedPDU', univ.OctetString()),
    )


class HeaderData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('msgID',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, 2147483647))),
        namedtype.NamedType('msgMaxSize',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(484, 2147483647))),
        namedtype.NamedType('msgFlags', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(1, 1))),
        namedtype.NamedType('msgSecurityModel',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(1, 2147483647)))
    )


class SNMPv3Message(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('msgVersion',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, 2147483647))),
        namedtype.NamedType('msgGlobalData', HeaderData()),
        namedtype.NamedType('msgSecurityParameters', univ.OctetString()),
        namedtype.NamedType('msgData', ScopedPduData())
    )
