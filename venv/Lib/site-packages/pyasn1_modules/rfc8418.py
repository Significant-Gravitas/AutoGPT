# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Elliptic Curve Diffie-Hellman (ECDH) Key Agreement Algorithm
#   with X25519 and X448
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8418.txt

from pyasn1.type import univ
from pyasn1_modules import rfc5280


class KeyEncryptionAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


class KeyWrapAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


dhSinglePass_stdDH_sha256kdf_scheme = univ.ObjectIdentifier('1.3.133.16.840.63.0.11.1')

dhSinglePass_stdDH_sha384kdf_scheme = univ.ObjectIdentifier('1.3.133.16.840.63.0.11.2')

dhSinglePass_stdDH_sha512kdf_scheme = univ.ObjectIdentifier('1.3.133.16.840.63.0.11.3')

dhSinglePass_stdDH_hkdf_sha256_scheme = univ.ObjectIdentifier('1.2.840.113549.1.9.16.3.19')

dhSinglePass_stdDH_hkdf_sha384_scheme = univ.ObjectIdentifier('1.2.840.113549.1.9.16.3.20')

dhSinglePass_stdDH_hkdf_sha512_scheme = univ.ObjectIdentifier('1.2.840.113549.1.9.16.3.21')
