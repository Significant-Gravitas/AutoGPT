#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Digital Signatures on Internet-Draft Documents
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8358.txt
#

from pyasn1.type import univ

from pyasn1_modules import rfc5652


id_ct = univ.ObjectIdentifier('1.2.840.113549.1.9.16.1')

id_ct_asciiTextWithCRLF = id_ct + (27, )

id_ct_epub = id_ct + (39, )

id_ct_htmlWithCRLF = id_ct + (38, )

id_ct_pdf = id_ct + (29, )

id_ct_postscript = id_ct + (30, )

id_ct_utf8TextWithCRLF = id_ct + (37, )

id_ct_xml = id_ct + (28, )


# Map of Content Type OIDs to Content Types is added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_asciiTextWithCRLF: univ.OctetString(),
    id_ct_epub: univ.OctetString(),
    id_ct_htmlWithCRLF: univ.OctetString(),
    id_ct_pdf: univ.OctetString(),
    id_ct_postscript: univ.OctetString(),
    id_ct_utf8TextWithCRLF: univ.OctetString(),
    id_ct_xml: univ.OctetString(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
