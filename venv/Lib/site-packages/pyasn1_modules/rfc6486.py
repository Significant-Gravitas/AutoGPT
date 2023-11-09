#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# RPKI Manifests
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6486.txt
#

from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import useful
from pyasn1.type import univ

from pyasn1_modules import rfc5652

MAX = float('inf')


id_smime = univ.ObjectIdentifier('1.2.840.113549.1.9.16')

id_ct = id_smime + (1, )

id_ct_rpkiManifest = id_ct + (26, )


class FileAndHash(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('file', char.IA5String()),
        namedtype.NamedType('hash', univ.BitString())
    )


class Manifest(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('version',
            univ.Integer().subtype(explicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0)).subtype(value=0)),
        namedtype.NamedType('manifestNumber',
            univ.Integer().subtype(
                subtypeSpec=constraint.ValueRangeConstraint(0, MAX))),
        namedtype.NamedType('thisUpdate',
            useful.GeneralizedTime()),
        namedtype.NamedType('nextUpdate',
            useful.GeneralizedTime()),
        namedtype.NamedType('fileHashAlg',
            univ.ObjectIdentifier()),
        namedtype.NamedType('fileList',
            univ.SequenceOf(componentType=FileAndHash()).subtype(
                subtypeSpec=constraint.ValueSizeConstraint(0, MAX)))
    )


# Map of Content Type OIDs to Content Types added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_rpkiManifest: Manifest(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
