#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Storing Validation Parameters in PKCS#8
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8479.txt
#

from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5652


id_attr_validation_parameters = univ.ObjectIdentifier('1.3.6.1.4.1.2312.18.8.1')


class ValidationParams(univ.Sequence):
    pass

ValidationParams.componentType = namedtype.NamedTypes(
    namedtype.NamedType('hashAlg', univ.ObjectIdentifier()),
    namedtype.NamedType('seed', univ.OctetString())
)


at_validation_parameters = rfc5652.Attribute()
at_validation_parameters['attrType'] = id_attr_validation_parameters
at_validation_parameters['attrValues'][0] = ValidationParams()


# Map of Attribute Type OIDs to Attributes added to the
# ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_attr_validation_parameters: ValidationParams(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)
