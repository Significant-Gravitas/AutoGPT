#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Object Identifiers for Test Certificate Policies
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7229.txt
#

from pyasn1.type import univ


id_pkix = univ.ObjectIdentifier('1.3.6.1.5.5.7')

id_TEST = id_pkix + (13, )

id_TEST_certPolicyOne   = id_TEST + (1, )
id_TEST_certPolicyTwo   = id_TEST + (2, )
id_TEST_certPolicyThree = id_TEST + (3, )
id_TEST_certPolicyFour  = id_TEST + (4, )
id_TEST_certPolicyFive  = id_TEST + (5, )
id_TEST_certPolicySix   = id_TEST + (6, )
id_TEST_certPolicySeven = id_TEST + (7, )
id_TEST_certPolicyEight = id_TEST + (8, )
