#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Resource Public Key Infrastructure (RPKI) Validation Reconsidered
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8360.txt
# https://www.rfc-editor.org/errata/eid5870
#

from pyasn1.type import univ

from pyasn1_modules import rfc3779
from pyasn1_modules import rfc5280


# IP Address Delegation Extension V2

id_pe_ipAddrBlocks_v2 = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.28')

IPAddrBlocks = rfc3779.IPAddrBlocks


# Autonomous System Identifier Delegation Extension V2

id_pe_autonomousSysIds_v2 = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.29')

ASIdentifiers = rfc3779.ASIdentifiers


# Map of Certificate Extension OIDs to Extensions is added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_pe_ipAddrBlocks_v2: IPAddrBlocks(),
    id_pe_autonomousSysIds_v2: ASIdentifiers(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
