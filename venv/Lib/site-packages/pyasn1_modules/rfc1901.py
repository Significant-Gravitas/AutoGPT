#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# SNMPv2c message syntax
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc1901.txt
#
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ


class Message(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', univ.Integer(namedValues=namedval.NamedValues(('version-2c', 1)))),
        namedtype.NamedType('community', univ.OctetString()),
        namedtype.NamedType('data', univ.Any())
    )
