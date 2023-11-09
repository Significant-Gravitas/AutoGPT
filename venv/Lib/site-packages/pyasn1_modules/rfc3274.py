#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add a map for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# CMS Compressed Data Content Type
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc3274.txt
#

from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652


class CompressionAlgorithmIdentifier(rfc5280.AlgorithmIdentifier):
    pass


# The CMS Compressed Data Content Type

id_ct_compressedData = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1.9')

class CompressedData(univ.Sequence):
    pass

CompressedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', rfc5652.CMSVersion()), # Always set to 0
    namedtype.NamedType('compressionAlgorithm', CompressionAlgorithmIdentifier()),
    namedtype.NamedType('encapContentInfo', rfc5652.EncapsulatedContentInfo())
)


# Algorithm identifier for the zLib Compression Algorithm
# This includes cpa_zlibCompress as defined in RFC 6268,
# from https://www.rfc-editor.org/rfc/rfc6268.txt

id_alg_zlibCompress = univ.ObjectIdentifier('1.2.840.113549.1.9.16.3.8')

cpa_zlibCompress = rfc5280.AlgorithmIdentifier()
cpa_zlibCompress['algorithm'] = id_alg_zlibCompress
# cpa_zlibCompress['parameters'] are absent


# Map of Content Type OIDs to Content Types is added to thr
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_compressedData: CompressedData(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
