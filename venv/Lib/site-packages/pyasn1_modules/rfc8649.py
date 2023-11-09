#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# X.509 Certificate Extension for Hash Of Root Key
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc8649.txt
#

from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc5280


id_ce_hashOfRootKey = univ.ObjectIdentifier('1.3.6.1.4.1.51483.2.1')


class HashedRootKey(univ.Sequence):
    pass

HashedRootKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('hashAlg', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('hashValue', univ.OctetString())
)


# Map of Certificate Extension OIDs to Extensions added to the
# ones that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    id_ce_hashOfRootKey: HashedRootKey(),	
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
