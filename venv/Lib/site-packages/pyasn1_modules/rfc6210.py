#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Experiment for Hash Functions with Parameters in the CMS
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6210.txt
#

from pyasn1.type import constraint
from pyasn1.type import univ

from pyasn1_modules import rfc5280


id_alg_MD5_XOR_EXPERIMENT = univ.ObjectIdentifier('1.2.840.113549.1.9.16.3.13')


class MD5_XOR_EXPERIMENT(univ.OctetString):
    pass

MD5_XOR_EXPERIMENT.subtypeSpec = constraint.ValueSizeConstraint(64, 64)


mda_xor_md5_EXPERIMENT = rfc5280.AlgorithmIdentifier()
mda_xor_md5_EXPERIMENT['algorithm'] = id_alg_MD5_XOR_EXPERIMENT
mda_xor_md5_EXPERIMENT['parameters'] = MD5_XOR_EXPERIMENT()


# Map of Algorithm Identifier OIDs to Parameters added to the
# ones that are in rfc5280.py.

_algorithmIdentifierMapUpdate = {
    id_alg_MD5_XOR_EXPERIMENT: MD5_XOR_EXPERIMENT(),
}

rfc5280.algorithmIdentifierMap.update(_algorithmIdentifierMapUpdate)
