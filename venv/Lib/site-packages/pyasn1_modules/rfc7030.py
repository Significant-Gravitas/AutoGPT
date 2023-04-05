#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Enrollment over Secure Transport (EST)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc7030.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5652

MAX = float('inf')


# Imports from RFC 5652

Attribute = rfc5652.Attribute


# Asymmetric Decrypt Key Identifier Attribute

id_aa_asymmDecryptKeyID = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.54')

class AsymmetricDecryptKeyIdentifier(univ.OctetString):
    pass


aa_asymmDecryptKeyID = Attribute()
aa_asymmDecryptKeyID['attrType'] = id_aa_asymmDecryptKeyID
aa_asymmDecryptKeyID['attrValues'][0] = AsymmetricDecryptKeyIdentifier()


# CSR Attributes

class AttrOrOID(univ.Choice):
    pass

AttrOrOID.componentType = namedtype.NamedTypes(
    namedtype.NamedType('oid', univ.ObjectIdentifier()),
    namedtype.NamedType('attribute', Attribute())
)


class CsrAttrs(univ.SequenceOf):
    pass

CsrAttrs.componentType = AttrOrOID()
CsrAttrs.subtypeSpec=constraint.ValueSizeConstraint(0, MAX)

   
# Update CMS Attribute Map

_cmsAttributesMapUpdate = {
    id_aa_asymmDecryptKeyID: AsymmetricDecryptKeyIdentifier(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)
