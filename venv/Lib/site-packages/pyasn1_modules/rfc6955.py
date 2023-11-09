#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Diffie-Hellman Proof-of-Possession Algorithms
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6955.txt
#

from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc3279
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652


# Imports from RFC 5652

MessageDigest = rfc5652.MessageDigest

IssuerAndSerialNumber = rfc5652.IssuerAndSerialNumber


# Imports from RFC 5280

id_pkix = rfc5280.id_pkix


# Imports from RFC 3279

Dss_Sig_Value = rfc3279.Dss_Sig_Value

DomainParameters = rfc3279.DomainParameters


# Static DH Proof-of-Possession

class DhSigStatic(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('issuerAndSerial', IssuerAndSerialNumber()),
        namedtype.NamedType('hashValue', MessageDigest())
    )


# Object Identifiers

id_dh_sig_hmac_sha1 = id_pkix + (6, 3, )

id_dhPop_static_sha1_hmac_sha1 = univ.ObjectIdentifier(id_dh_sig_hmac_sha1)


id_alg_dh_pop = id_pkix + (6, 4, )

id_alg_dhPop_sha1 = univ.ObjectIdentifier(id_alg_dh_pop)

id_alg_dhPop_sha224 = id_pkix + (6, 5, )

id_alg_dhPop_sha256 = id_pkix + (6, 6, )

id_alg_dhPop_sha384 = id_pkix + (6, 7, )

id_alg_dhPop_sha512 = id_pkix + (6, 8, )


id_alg_dhPop_static_sha224_hmac_sha224 = id_pkix + (6, 15, )

id_alg_dhPop_static_sha256_hmac_sha256 = id_pkix + (6, 16, )

id_alg_dhPop_static_sha384_hmac_sha384 = id_pkix + (6, 17, )

id_alg_dhPop_static_sha512_hmac_sha512 = id_pkix + (6, 18, )


id_alg_ecdhPop_static_sha224_hmac_sha224 = id_pkix + (6, 25, )

id_alg_ecdhPop_static_sha256_hmac_sha256 = id_pkix + (6, 26, )

id_alg_ecdhPop_static_sha384_hmac_sha384 = id_pkix + (6, 27, )

id_alg_ecdhPop_static_sha512_hmac_sha512 = id_pkix + (6, 28, )


# Update the Algorithm Identifier map in rfc5280.py

_algorithmIdentifierMapUpdate = {
    id_alg_dh_pop: DomainParameters(),
    id_alg_dhPop_sha224: DomainParameters(),
    id_alg_dhPop_sha256: DomainParameters(),
    id_alg_dhPop_sha384: DomainParameters(),
    id_alg_dhPop_sha512: DomainParameters(),
    id_dh_sig_hmac_sha1: univ.Null(""),
    id_alg_dhPop_static_sha224_hmac_sha224: univ.Null(""),
    id_alg_dhPop_static_sha256_hmac_sha256: univ.Null(""),
    id_alg_dhPop_static_sha384_hmac_sha384: univ.Null(""),
    id_alg_dhPop_static_sha512_hmac_sha512: univ.Null(""),
    id_alg_ecdhPop_static_sha224_hmac_sha224: univ.Null(""),
    id_alg_ecdhPop_static_sha256_hmac_sha256: univ.Null(""),
    id_alg_ecdhPop_static_sha384_hmac_sha384: univ.Null(""),
    id_alg_ecdhPop_static_sha512_hmac_sha512: univ.Null(""),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
