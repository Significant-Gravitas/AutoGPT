# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley.
# Modified by Russ Housley to add a map for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# BinaryTime: An Alternate Format for Representing Date and Time
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6019.txt

from pyasn1.type import constraint
from pyasn1.type import univ

from pyasn1_modules import rfc5652

MAX = float('inf')


# BinaryTime: Represent date and time as an integer 

class BinaryTime(univ.Integer):
    pass

BinaryTime.subtypeSpec = constraint.ValueRangeConstraint(0, MAX)


# CMS Attribute for representing signing time in BinaryTime

id_aa_binarySigningTime = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.46')

class BinarySigningTime(BinaryTime):
    pass


# Map of Attribute Type OIDs to Attributes ia added to the
# ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_aa_binarySigningTime: BinarySigningTime(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)
