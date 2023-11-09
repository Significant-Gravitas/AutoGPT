#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add map for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Additional CMS Revocation Information Choices
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5940.txt
#

from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc2560
from pyasn1_modules import rfc5652


# RevocationInfoChoice for OCSP response:
# The OID is included in otherRevInfoFormat, and
# signed OCSPResponse is included in otherRevInfo

id_ri_ocsp_response = univ.ObjectIdentifier('1.3.6.1.5.5.7.16.2')

OCSPResponse = rfc2560.OCSPResponse


# RevocationInfoChoice for SCVP request/response:
# The OID is included in otherRevInfoFormat, and
# SCVPReqRes is included in otherRevInfo

id_ri_scvp = univ.ObjectIdentifier('1.3.6.1.5.5.7.16.4')

ContentInfo = rfc5652.ContentInfo

class SCVPReqRes(univ.Sequence):
    pass

SCVPReqRes.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('request',
        ContentInfo().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('response', ContentInfo())
)


# Map of Revocation Info Format OIDs to Revocation Info Format
# is added to the ones that are in rfc5652.py

_otherRevInfoFormatMapUpdate = {
     id_ri_ocsp_response: OCSPResponse(),
     id_ri_scvp: SCVPReqRes(),
}

rfc5652.otherRevInfoFormatMap.update(_otherRevInfoFormatMapUpdate)
