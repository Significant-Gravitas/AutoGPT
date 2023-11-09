#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# SNMPv3 message syntax
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc3414.txt
#
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ


class UsmSecurityParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('msgAuthoritativeEngineID', univ.OctetString()),
        namedtype.NamedType('msgAuthoritativeEngineBoots',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, 2147483647))),
        namedtype.NamedType('msgAuthoritativeEngineTime',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, 2147483647))),
        namedtype.NamedType('msgUserName',
                            univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(0, 32))),
        namedtype.NamedType('msgAuthenticationParameters', univ.OctetString()),
        namedtype.NamedType('msgPrivacyParameters', univ.OctetString())
    )
