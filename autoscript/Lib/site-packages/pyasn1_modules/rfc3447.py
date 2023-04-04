#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS#1 syntax
#
# ASN.1 source from:
# ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2-1.asn
#
# Sample captures could be obtained with "openssl genrsa" command
#
from pyasn1.type import constraint
from pyasn1.type import namedval

from pyasn1_modules.rfc2437 import *


class OtherPrimeInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('prime', univ.Integer()),
        namedtype.NamedType('exponent', univ.Integer()),
        namedtype.NamedType('coefficient', univ.Integer())
    )


class OtherPrimeInfos(univ.SequenceOf):
    componentType = OtherPrimeInfo()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class RSAPrivateKey(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', univ.Integer(namedValues=namedval.NamedValues(('two-prime', 0), ('multi', 1)))),
        namedtype.NamedType('modulus', univ.Integer()),
        namedtype.NamedType('publicExponent', univ.Integer()),
        namedtype.NamedType('privateExponent', univ.Integer()),
        namedtype.NamedType('prime1', univ.Integer()),
        namedtype.NamedType('prime2', univ.Integer()),
        namedtype.NamedType('exponent1', univ.Integer()),
        namedtype.NamedType('exponent2', univ.Integer()),
        namedtype.NamedType('coefficient', univ.Integer()),
        namedtype.OptionalNamedType('otherPrimeInfos', OtherPrimeInfos())
    )
