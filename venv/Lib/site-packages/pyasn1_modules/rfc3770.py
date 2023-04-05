#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Certificate Extensions and Attributes Supporting Authentication
#   in PPP and Wireless LAN Networks
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc3770.txt
# https://www.rfc-editor.org/errata/eid234
#

from pyasn1.type import constraint
from pyasn1.type import univ

from pyasn1_modules import rfc5280


MAX = float('inf')


# Extended Key Usage Values

id_kp_eapOverLAN = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.14')

id_kp_eapOverPPP = univ.ObjectIdentifier('1.3.6.1.5.5.7.3.13')


# Wireless LAN SSID Extension

id_pe_wlanSSID = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.13')


class SSID(univ.OctetString):
    pass

SSID.subtypeSpec = constraint.ValueSizeConstraint(1, 32)


class SSIDList(univ.SequenceOf):
    pass

SSIDList.componentType = SSID()
SSIDList.subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


# Wireless LAN SSID Attribute Certificate Attribute
# Uses same syntax as the certificate extension: SSIDList
# Correction for https://www.rfc-editor.org/errata/eid234

id_aca_wlanSSID = univ.ObjectIdentifier('1.3.6.1.5.5.7.10.7')


# Map of Certificate Extension OIDs to Extensions
# To be added to the ones that are in rfc5280.py

_certificateExtensionsMap = {
    id_pe_wlanSSID: SSIDList(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMap)


# Map of AttributeType OIDs to AttributeValue added to the
# ones that are in rfc5280.py

_certificateAttributesMapUpdate = {
    id_aca_wlanSSID: SSIDList(),
}

rfc5280.certificateAttributesMap.update(_certificateAttributesMapUpdate)
