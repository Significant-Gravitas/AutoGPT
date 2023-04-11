# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add maps for opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Elliptic Curve Cryptography Subject Public Key Information
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5480.txt


# What can be imported from rfc4055.py ?

from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc3279
from pyasn1_modules import rfc5280


# These structures are the same as RFC 3279.

DHPublicKey = rfc3279.DHPublicKey

DSAPublicKey = rfc3279.DSAPublicKey

ValidationParms = rfc3279.ValidationParms

DomainParameters = rfc3279.DomainParameters

ECDSA_Sig_Value = rfc3279.ECDSA_Sig_Value

ECPoint = rfc3279.ECPoint

KEA_Parms_Id = rfc3279.KEA_Parms_Id

RSAPublicKey = rfc3279.RSAPublicKey


# RFC 5480 changed the names of these structures from RFC 3279.

DSS_Parms = rfc3279.Dss_Parms

DSA_Sig_Value = rfc3279.Dss_Sig_Value


# RFC 3279 defines a more complex alternative for ECParameters.
# RFC 5480 narrows the definition to a single CHOICE: namedCurve.

class ECParameters(univ.Choice):
    pass

ECParameters.componentType = namedtype.NamedTypes(
    namedtype.NamedType('namedCurve', univ.ObjectIdentifier())
)


# OIDs for Message Digest Algorithms

id_md2 = univ.ObjectIdentifier('1.2.840.113549.2.2')

id_md5 = univ.ObjectIdentifier('1.2.840.113549.2.5')

id_sha1 = univ.ObjectIdentifier('1.3.14.3.2.26')

id_sha224 = univ.ObjectIdentifier('2.16.840.1.101.3.4.2.4')

id_sha256 = univ.ObjectIdentifier('2.16.840.1.101.3.4.2.1')

id_sha384 = univ.ObjectIdentifier('2.16.840.1.101.3.4.2.2')

id_sha512 = univ.ObjectIdentifier('2.16.840.1.101.3.4.2.3')


# OID for RSA PK Algorithm and Key

rsaEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.1')


# OID for DSA PK Algorithm, Key, and Parameters

id_dsa = univ.ObjectIdentifier('1.2.840.10040.4.1')


# OID for Diffie-Hellman PK Algorithm, Key, and Parameters

dhpublicnumber = univ.ObjectIdentifier('1.2.840.10046.2.1')

# OID for KEA PK Algorithm and Parameters

id_keyExchangeAlgorithm = univ.ObjectIdentifier('2.16.840.1.101.2.1.1.22')


# OIDs for Elliptic Curve Algorithm ID, Key, and Parameters
# Note that ECDSA keys always use this OID

id_ecPublicKey = univ.ObjectIdentifier('1.2.840.10045.2.1')

id_ecDH = univ.ObjectIdentifier('1.3.132.1.12')

id_ecMQV = univ.ObjectIdentifier('1.3.132.1.13')


# OIDs for RSA Signature Algorithms

md2WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.2')

md5WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.4')

sha1WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.5')


# OIDs for DSA Signature Algorithms

id_dsa_with_sha1 = univ.ObjectIdentifier('1.2.840.10040.4.3')

id_dsa_with_sha224 = univ.ObjectIdentifier('2.16.840.1.101.3.4.3.1')

id_dsa_with_sha256 = univ.ObjectIdentifier('2.16.840.1.101.3.4.3.2')


# OIDs for ECDSA Signature Algorithms

ecdsa_with_SHA1 = univ.ObjectIdentifier('1.2.840.10045.4.1')

ecdsa_with_SHA224 = univ.ObjectIdentifier('1.2.840.10045.4.3.1')

ecdsa_with_SHA256 = univ.ObjectIdentifier('1.2.840.10045.4.3.2')

ecdsa_with_SHA384 = univ.ObjectIdentifier('1.2.840.10045.4.3.3')

ecdsa_with_SHA512 = univ.ObjectIdentifier('1.2.840.10045.4.3.4')


# OIDs for Named Elliptic Curves

secp192r1 = univ.ObjectIdentifier('1.2.840.10045.3.1.1')

sect163k1 = univ.ObjectIdentifier('1.3.132.0.1')

sect163r2 = univ.ObjectIdentifier('1.3.132.0.15')

secp224r1 = univ.ObjectIdentifier('1.3.132.0.33')

sect233k1 = univ.ObjectIdentifier('1.3.132.0.26')

sect233r1 = univ.ObjectIdentifier('1.3.132.0.27')

secp256r1 = univ.ObjectIdentifier('1.2.840.10045.3.1.7')

sect283k1 = univ.ObjectIdentifier('1.3.132.0.16')

sect283r1 = univ.ObjectIdentifier('1.3.132.0.17')

secp384r1 = univ.ObjectIdentifier('1.3.132.0.34')

sect409k1 = univ.ObjectIdentifier('1.3.132.0.36')

sect409r1 = univ.ObjectIdentifier('1.3.132.0.37')

secp521r1 = univ.ObjectIdentifier('1.3.132.0.35')

sect571k1 = univ.ObjectIdentifier('1.3.132.0.38')

sect571r1 = univ.ObjectIdentifier('1.3.132.0.39')


# Map of Algorithm Identifier OIDs to Parameters
# The algorithm is not included if the parameters MUST be absent

_algorithmIdentifierMapUpdate = {
    rsaEncryption: univ.Null(),
    md2WithRSAEncryption: univ.Null(),
    md5WithRSAEncryption: univ.Null(),
    sha1WithRSAEncryption: univ.Null(),
    id_dsa: DSS_Parms(),
    dhpublicnumber: DomainParameters(),
    id_keyExchangeAlgorithm: KEA_Parms_Id(),
    id_ecPublicKey: ECParameters(),
    id_ecDH: ECParameters(),
    id_ecMQV: ECParameters(),
}


# Add these Algorithm Identifier map entries to the ones in rfc5280.py

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
