# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# IKEv2 Certificate Bundle
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7296.txt

from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280


class CertificateOrCRL(univ.Choice):
    pass

CertificateOrCRL.componentType = namedtype.NamedTypes(
    namedtype.NamedType('cert', rfc5280.Certificate().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('crl', rfc5280.CertificateList().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
)


class CertificateBundle(univ.SequenceOf):
    pass

CertificateBundle.componentType = CertificateOrCRL()
