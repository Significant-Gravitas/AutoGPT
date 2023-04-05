# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Algorithm Identifiers for Ed25519, Ed448, X25519, and X448
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8410.txt

from pyasn1.type import univ
from pyasn1_modules import rfc3565
from pyasn1_modules import rfc4055
from pyasn1_modules import rfc5280


class SignatureAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class KeyEncryptionAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class CurvePrivateKey(univ.OctetString):
    pass


id_X25519 = univ.ObjectIdentifier('1.3.101.110')

id_X448 = univ.ObjectIdentifier('1.3.101.111')

id_Ed25519 = univ.ObjectIdentifier('1.3.101.112')

id_Ed448 = univ.ObjectIdentifier('1.3.101.113')

id_sha512 = rfc4055.id_sha512

id_aes128_wrap = rfc3565.id_aes128_wrap

id_aes256_wrap = rfc3565.id_aes256_wrap
