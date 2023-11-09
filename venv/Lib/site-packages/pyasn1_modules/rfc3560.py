#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# RSAES-OAEP Key Transport Algorithm in CMS
#
# Notice that all of the things needed in RFC 3560 are also defined
# in RFC 4055.  So, they are all pulled from the RFC 4055 module into
# this one so that people looking a RFC 3560 can easily find them.
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc3560.txt
#

from pyasn1_modules import rfc4055

id_sha1 = rfc4055.id_sha1

id_sha256 = rfc4055.id_sha256

id_sha384 = rfc4055.id_sha384

id_sha512 = rfc4055.id_sha512

id_mgf1 = rfc4055.id_mgf1

rsaEncryption = rfc4055.rsaEncryption

id_RSAES_OAEP = rfc4055.id_RSAES_OAEP

id_pSpecified = rfc4055.id_pSpecified

sha1Identifier = rfc4055.sha1Identifier

sha256Identifier = rfc4055.sha256Identifier

sha384Identifier = rfc4055.sha384Identifier

sha512Identifier = rfc4055.sha512Identifier

mgf1SHA1Identifier = rfc4055.mgf1SHA1Identifier

mgf1SHA256Identifier = rfc4055.mgf1SHA256Identifier

mgf1SHA384Identifier = rfc4055.mgf1SHA384Identifier

mgf1SHA512Identifier = rfc4055.mgf1SHA512Identifier

pSpecifiedEmptyIdentifier = rfc4055.pSpecifiedEmptyIdentifier


class RSAES_OAEP_params(rfc4055.RSAES_OAEP_params):
    pass


rSAES_OAEP_Default_Params = RSAES_OAEP_params()

rSAES_OAEP_Default_Identifier = rfc4055.rSAES_OAEP_Default_Identifier

rSAES_OAEP_SHA256_Params = rfc4055.rSAES_OAEP_SHA256_Params

rSAES_OAEP_SHA256_Identifier = rfc4055.rSAES_OAEP_SHA256_Identifier

rSAES_OAEP_SHA384_Params = rfc4055.rSAES_OAEP_SHA384_Params

rSAES_OAEP_SHA384_Identifier = rfc4055.rSAES_OAEP_SHA384_Identifier

rSAES_OAEP_SHA512_Params = rfc4055.rSAES_OAEP_SHA512_Params

rSAES_OAEP_SHA512_Identifier = rfc4055.rSAES_OAEP_SHA512_Identifier
