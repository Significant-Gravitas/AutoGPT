#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with some assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Transport Layer Security (TLS) Feature Certificate Extension
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7633.txt
#

from pyasn1.type import univ

from pyasn1_modules import rfc5280


# TLS Features Extension

id_pe = univ.ObjectIdentifier('1.3.6.1.5.5.7.1')

id_pe_tlsfeature = id_pe + (24, )


class Features(univ.SequenceOf):
    componentType = univ.Integer()


# Map of Certificate Extension OIDs to Extensions added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_pe_tlsfeature: Features(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
