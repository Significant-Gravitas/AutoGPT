#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# PKCS#1 syntax
#
# ASN.1 source from:
# ftp://ftp.rsasecurity.com/pub/pkcs/pkcs-1/pkcs-1v2.asn
#
# Sample captures could be obtained with "openssl genrsa" command
#
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules.rfc2459 import AlgorithmIdentifier

pkcs_1 = univ.ObjectIdentifier('1.2.840.113549.1.1')
rsaEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.1')
md2WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.2')
md4WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.3')
md5WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.4')
sha1WithRSAEncryption = univ.ObjectIdentifier('1.2.840.113549.1.1.5')
rsaOAEPEncryptionSET = univ.ObjectIdentifier('1.2.840.113549.1.1.6')
id_RSAES_OAEP = univ.ObjectIdentifier('1.2.840.113549.1.1.7')
id_mgf1 = univ.ObjectIdentifier('1.2.840.113549.1.1.8')
id_pSpecified = univ.ObjectIdentifier('1.2.840.113549.1.1.9')
id_sha1 = univ.ObjectIdentifier('1.3.14.3.2.26')

MAX = float('inf')


class Version(univ.Integer):
    pass


class RSAPrivateKey(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('modulus', univ.Integer()),
        namedtype.NamedType('publicExponent', univ.Integer()),
        namedtype.NamedType('privateExponent', univ.Integer()),
        namedtype.NamedType('prime1', univ.Integer()),
        namedtype.NamedType('prime2', univ.Integer()),
        namedtype.NamedType('exponent1', univ.Integer()),
        namedtype.NamedType('exponent2', univ.Integer()),
        namedtype.NamedType('coefficient', univ.Integer())
    )


class RSAPublicKey(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('modulus', univ.Integer()),
        namedtype.NamedType('publicExponent', univ.Integer())
    )


# XXX defaults not set
class RSAES_OAEP_params(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('hashFunc', AlgorithmIdentifier().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.NamedType('maskGenFunc', AlgorithmIdentifier().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
        namedtype.NamedType('pSourceFunc', AlgorithmIdentifier().subtype(
            implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
    )
