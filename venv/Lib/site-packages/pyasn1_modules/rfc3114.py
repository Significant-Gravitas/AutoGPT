#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# TEST Company Classification Policies
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc3114.txt
#

from pyasn1.type import char
from pyasn1.type import namedval
from pyasn1.type import univ

from pyasn1_modules import rfc5755


id_smime = univ.ObjectIdentifier((1, 2, 840, 113549, 1, 9, 16, ))

id_tsp = id_smime + (7, )

id_tsp_TEST_Amoco = id_tsp + (1, )

class Amoco_SecurityClassification(univ.Integer):
    namedValues = namedval.NamedValues(
        ('amoco-general', 6),
        ('amoco-confidential', 7),
        ('amoco-highly-confidential', 8)
    )


id_tsp_TEST_Caterpillar = id_tsp + (2, )

class Caterpillar_SecurityClassification(univ.Integer):
    namedValues = namedval.NamedValues(
        ('caterpillar-public', 6),
        ('caterpillar-green', 7),
        ('caterpillar-yellow', 8),
        ('caterpillar-red', 9)
    )


id_tsp_TEST_Whirlpool = id_tsp + (3, )

class Whirlpool_SecurityClassification(univ.Integer):
    namedValues = namedval.NamedValues(
        ('whirlpool-public', 6),
        ('whirlpool-internal', 7),
        ('whirlpool-confidential', 8)
    )


id_tsp_TEST_Whirlpool_Categories = id_tsp + (4, )

class SecurityCategoryValues(univ.SequenceOf):
    componentType = char.UTF8String()

# Example SecurityCategoryValues: "LAW DEPARTMENT USE ONLY"
# Example SecurityCategoryValues: "HUMAN RESOURCES USE ONLY"


# Also, the privacy mark in the security label can contain a string,
# such as: "ATTORNEY-CLIENT PRIVILEGED INFORMATION"


# Map of security category type OIDs to security category added
# to the ones that are in rfc5755.py

_securityCategoryMapUpdate = {
    id_tsp_TEST_Whirlpool_Categories: SecurityCategoryValues(),
}

rfc5755.securityCategoryMap.update(_securityCategoryMapUpdate)
