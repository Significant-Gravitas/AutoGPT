#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Authority Clearance Constraints Certificate Extension
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5913.txt
# https://www.rfc-editor.org/errata/eid5890
#

from pyasn1.type import constraint
from pyasn1.type import univ

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5755

MAX = float('inf')


# Authority Clearance Constraints Certificate Extension

id_pe_clearanceConstraints = univ.ObjectIdentifier('1.3.6.1.5.5.7.1.21')

id_pe_authorityClearanceConstraints = id_pe_clearanceConstraints


class AuthorityClearanceConstraints(univ.SequenceOf):
    componentType = rfc5755.Clearance()
    subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


# Map of Certificate Extension OIDs to Extensions added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_pe_clearanceConstraints: AuthorityClearanceConstraints(),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
