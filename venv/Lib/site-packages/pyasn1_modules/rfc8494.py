# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Multicast Email (MULE) over Allied Communications Publication 142
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8494.txt

from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ


id_mmhs_CDT = univ.ObjectIdentifier('1.3.26.0.4406.0.4.2')


class AlgorithmID_ShortForm(univ.Integer):
    pass

AlgorithmID_ShortForm.namedValues = namedval.NamedValues(
    ('zlibCompress', 0)
)


class ContentType_ShortForm(univ.Integer):
    pass

ContentType_ShortForm.namedValues = namedval.NamedValues(
    ('unidentified', 0),
    ('external', 1),
    ('p1', 2),
    ('p3', 3),
    ('p7', 4),
    ('mule', 25)
)


class CompressedContentInfo(univ.Sequence):
    pass

CompressedContentInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('unnamed', univ.Choice(componentType=namedtype.NamedTypes(
        namedtype.NamedType('contentType-ShortForm',
            ContentType_ShortForm().subtype(explicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('contentType-OID',
            univ.ObjectIdentifier().subtype(explicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 1)))
    ))),
    namedtype.NamedType('compressedContent',
        univ.OctetString().subtype(explicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class CompressionAlgorithmIdentifier(univ.Choice):
    pass

CompressionAlgorithmIdentifier.componentType = namedtype.NamedTypes(
    namedtype.NamedType('algorithmID-ShortForm',
        AlgorithmID_ShortForm().subtype(explicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('algorithmID-OID',
        univ.ObjectIdentifier().subtype(explicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class CompressedData(univ.Sequence):
    pass

CompressedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('compressionAlgorithm', CompressionAlgorithmIdentifier()),
    namedtype.NamedType('compressedContentInfo', CompressedContentInfo())
)
