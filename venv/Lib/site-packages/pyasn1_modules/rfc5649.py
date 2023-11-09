# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# AES Key Wrap with Padding
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5649.txt

from pyasn1.type import univ

from pyasn1_modules import rfc5280


class AlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


id_aes128_wrap = univ.ObjectIdentifier('2.16.840.1.101.3.4.1.5')

id_aes192_wrap = univ.ObjectIdentifier('2.16.840.1.101.3.4.1.25')

id_aes256_wrap = univ.ObjectIdentifier('2.16.840.1.101.3.4.1.45')


id_aes128_wrap_pad = univ.ObjectIdentifier('2.16.840.1.101.3.4.1.8')

id_aes192_wrap_pad = univ.ObjectIdentifier('2.16.840.1.101.3.4.1.28')

id_aes256_wrap_pad = univ.ObjectIdentifier('2.16.840.1.101.3.4.1.48')
