#
# This file is part of pyasn1-modules.
#
# Copyright (c) 2017, Danielle Madeley <danielle@madeley.id.au>
# License: http://snmplabs.com/pyasn1/license.html
#
# Modified by Russ Housley to add maps for use with opentypes.
#
# Algorithms and Identifiers for Internet X.509 Certificates and CRLs
#
# Derived from RFC 3279:
# https://www.rfc-editor.org/rfc/rfc3279.txt
#
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import univ

from pyasn1_modules import rfc5280


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


md2 = _OID(1, 2, 840, 113549, 2, 2)
md5 = _OID(1, 2, 840, 113549, 2, 5)
id_sha1 = _OID(1, 3, 14, 3, 2, 26)
id_dsa = _OID(1, 2, 840, 10040, 4, 1)


class DSAPublicKey(univ.Integer):
    pass


class Dss_Parms(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('p', univ.Integer()),
        namedtype.NamedType('q', univ.Integer()),
        namedtype.NamedType('g', univ.Integer())
    )


id_dsa_with_sha1 = _OID(1, 2, 840, 10040, 4, 3)


class Dss_Sig_Value(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('r', univ.Integer()),
        namedtype.NamedType('s', univ.Integer())
    )


pkcs_1 = _OID(1, 2, 840, 113549, 1, 1)
rsaEncryption = _OID(pkcs_1, 1)
md2WithRSAEncryption = _OID(pkcs_1, 2)
md5WithRSAEncryption = _OID(pkcs_1, 4)
sha1WithRSAEncryption = _OID(pkcs_1, 5)


class RSAPublicKey(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('modulus', univ.Integer()),
        namedtype.NamedType('publicExponent', univ.Integer())
    )


dhpublicnumber = _OID(1, 2, 840, 10046, 2, 1)


class DHPublicKey(univ.Integer):
    pass


class ValidationParms(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('seed', univ.BitString()),
        namedtype.NamedType('pgenCounter', univ.Integer())
    )


class DomainParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('p', univ.Integer()),
        namedtype.NamedType('g', univ.Integer()),
        namedtype.NamedType('q', univ.Integer()),
        namedtype.OptionalNamedType('j', univ.Integer()),
        namedtype.OptionalNamedType('validationParms', ValidationParms())
    )


id_keyExchangeAlgorithm = _OID(2, 16, 840, 1, 101, 2, 1, 1, 22)


class KEA_Parms_Id(univ.OctetString):
    pass


ansi_X9_62 = _OID(1, 2, 840, 10045)


class FieldID(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('fieldType', univ.ObjectIdentifier()),
        namedtype.NamedType('parameters', univ.Any())
    )


id_ecSigType = _OID(ansi_X9_62, 4)
ecdsa_with_SHA1 = _OID(id_ecSigType, 1)


class ECDSA_Sig_Value(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('r', univ.Integer()),
        namedtype.NamedType('s', univ.Integer())
    )


id_fieldType = _OID(ansi_X9_62, 1)
prime_field = _OID(id_fieldType, 1)


class Prime_p(univ.Integer):
    pass


characteristic_two_field = _OID(id_fieldType, 2)


class Characteristic_two(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('m', univ.Integer()),
        namedtype.NamedType('basis', univ.ObjectIdentifier()),
        namedtype.NamedType('parameters', univ.Any())
    )


id_characteristic_two_basis = _OID(characteristic_two_field, 3)
gnBasis = _OID(id_characteristic_two_basis, 1)
tpBasis = _OID(id_characteristic_two_basis, 2)


class Trinomial(univ.Integer):
    pass


ppBasis = _OID(id_characteristic_two_basis, 3)


class Pentanomial(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('k1', univ.Integer()),
        namedtype.NamedType('k2', univ.Integer()),
        namedtype.NamedType('k3', univ.Integer())
    )


class FieldElement(univ.OctetString):
    pass


class ECPoint(univ.OctetString):
    pass


class Curve(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('a', FieldElement()),
        namedtype.NamedType('b', FieldElement()),
        namedtype.OptionalNamedType('seed', univ.BitString())
    )


class ECPVer(univ.Integer):
    namedValues = namedval.NamedValues(
        ('ecpVer1', 1)
    )


class ECParameters(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', ECPVer()),
        namedtype.NamedType('fieldID', FieldID()),
        namedtype.NamedType('curve', Curve()),
        namedtype.NamedType('base', ECPoint()),
        namedtype.NamedType('order', univ.Integer()),
        namedtype.OptionalNamedType('cofactor', univ.Integer())
    )


class EcpkParameters(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('ecParameters', ECParameters()),
        namedtype.NamedType('namedCurve', univ.ObjectIdentifier()),
        namedtype.NamedType('implicitlyCA', univ.Null())
    )


id_publicKeyType = _OID(ansi_X9_62, 2)
id_ecPublicKey = _OID(id_publicKeyType, 1)

ellipticCurve = _OID(ansi_X9_62, 3)

c_TwoCurve = _OID(ellipticCurve, 0)
c2pnb163v1 = _OID(c_TwoCurve, 1)
c2pnb163v2 = _OID(c_TwoCurve, 2)
c2pnb163v3 = _OID(c_TwoCurve, 3)
c2pnb176w1 = _OID(c_TwoCurve, 4)
c2tnb191v1 = _OID(c_TwoCurve, 5)
c2tnb191v2 = _OID(c_TwoCurve, 6)
c2tnb191v3 = _OID(c_TwoCurve, 7)
c2onb191v4 = _OID(c_TwoCurve, 8)
c2onb191v5 = _OID(c_TwoCurve, 9)
c2pnb208w1 = _OID(c_TwoCurve, 10)
c2tnb239v1 = _OID(c_TwoCurve, 11)
c2tnb239v2 = _OID(c_TwoCurve, 12)
c2tnb239v3 = _OID(c_TwoCurve, 13)
c2onb239v4 = _OID(c_TwoCurve, 14)
c2onb239v5 = _OID(c_TwoCurve, 15)
c2pnb272w1 = _OID(c_TwoCurve, 16)
c2pnb304w1 = _OID(c_TwoCurve, 17)
c2tnb359v1 = _OID(c_TwoCurve, 18)
c2pnb368w1 = _OID(c_TwoCurve, 19)
c2tnb431r1 = _OID(c_TwoCurve, 20)

primeCurve = _OID(ellipticCurve, 1)
prime192v1 = _OID(primeCurve, 1)
prime192v2 = _OID(primeCurve, 2)
prime192v3 = _OID(primeCurve, 3)
prime239v1 = _OID(primeCurve, 4)
prime239v2 = _OID(primeCurve, 5)
prime239v3 = _OID(primeCurve, 6)
prime256v1 = _OID(primeCurve, 7)


# Map of Algorithm Identifier OIDs to Parameters added to the
# ones in rfc5280.py.  Do not add OIDs with absent paramaters.

_algorithmIdentifierMapUpdate = {
    md2: univ.Null(""),
    md5: univ.Null(""),
    id_sha1: univ.Null(""),
    id_dsa: Dss_Parms(),
    rsaEncryption: univ.Null(""),
    md2WithRSAEncryption: univ.Null(""),
    md5WithRSAEncryption: univ.Null(""),
    sha1WithRSAEncryption: univ.Null(""),
    dhpublicnumber: DomainParameters(),
    id_keyExchangeAlgorithm: KEA_Parms_Id(),
    id_ecPublicKey: EcpkParameters(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
