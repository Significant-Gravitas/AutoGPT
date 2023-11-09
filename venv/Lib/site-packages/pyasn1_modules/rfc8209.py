#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# BGPsec Router PKI Profile
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8209.txt
#

from pyasn1.type import univ


id_kp = univ.ObjectIdentifier('1.3.6.1.5.5.7.3')

id_kp_bgpsec_router = id_kp + (30, )
