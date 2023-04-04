#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with some assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# S/MIME Capabilities for Public Key Definitions
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6664.txt
#

from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5751
from pyasn1_modules import rfc5480
from pyasn1_modules import rfc4055
from pyasn1_modules import rfc3279

MAX = float('inf')


# Imports from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier


# Imports from RFC 3279

dhpublicnumber = rfc3279.dhpublicnumber

Dss_Parms = rfc3279.Dss_Parms

id_dsa = rfc3279.id_dsa

id_ecPublicKey = rfc3279.id_ecPublicKey

rsaEncryption = rfc3279.rsaEncryption


# Imports from RFC 4055

id_mgf1 = rfc4055.id_mgf1

id_RSAES_OAEP = rfc4055.id_RSAES_OAEP

id_RSASSA_PSS = rfc4055.id_RSASSA_PSS


# Imports from RFC 5480

ECParameters = rfc5480.ECParameters

id_ecDH = rfc5480.id_ecDH

id_ecMQV = rfc5480.id_ecMQV


# RSA

class RSAKeySize(univ.Integer):
    # suggested values are 1024, 2048, 3072, 4096, 7680, 8192, and 15360;
    # however, the integer value is not limited to these suggestions
    pass


class RSAKeyCapabilities(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('minKeySize', RSAKeySize()),
        namedtype.OptionalNamedType('maxKeySize', RSAKeySize())
    )


class RsaSsa_Pss_sig_caps(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('hashAlg', AlgorithmIdentifier()),
        namedtype.OptionalNamedType('maskAlg', AlgorithmIdentifier()),
        namedtype.DefaultedNamedType('trailerField', univ.Integer().subtype(value=1))
    )


# Diffie-Hellman and DSA

class DSAKeySize(univ.Integer):
    subtypeSpec = constraint.SingleValueConstraint(1024, 2048, 3072, 7680, 15360)


class DSAKeyCapabilities(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('keySizes', univ.Sequence(componentType=namedtype.NamedTypes(
            namedtype.NamedType('minKeySize',
                DSAKeySize()),
            namedtype.OptionalNamedType('maxKeySize',
                DSAKeySize()),
            namedtype.OptionalNamedType('maxSizeP',
                univ.Integer().subtype(explicitTag=tag.Tag(
                    tag.tagClassContext, tag.tagFormatSimple, 1))),
            namedtype.OptionalNamedType('maxSizeQ',
                univ.Integer().subtype(explicitTag=tag.Tag(
                    tag.tagClassContext, tag.tagFormatSimple, 2))),
            namedtype.OptionalNamedType('maxSizeG',
                univ.Integer().subtype(explicitTag=tag.Tag(
                    tag.tagClassContext, tag.tagFormatSimple, 3)))
        )).subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.NamedType('keyParams',
            Dss_Parms().subtype(explicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


# Elliptic Curve

class EC_SMimeCaps(univ.SequenceOf):
    componentType = ECParameters()
    subtypeSpec=constraint.ValueSizeConstraint(1, MAX)


# Update the SMIMECapabilities Attribute Map in rfc5751.py
#
# The map can either include an entry for scap-sa-rsaSSA-PSS or 
# scap-pk-rsaSSA-PSS, but not both.  One is associated with the
# public key and the other is associated with the signature
# algorithm; however, they use the same OID.  If you need the
# other one in your application, copy the map into a local dict,
# adjust as needed, and pass the local dict to the decoder with
# openTypes=your_local_map.

_smimeCapabilityMapUpdate = {
    rsaEncryption: RSAKeyCapabilities(),
    id_RSASSA_PSS: RSAKeyCapabilities(),
    # id_RSASSA_PSS: RsaSsa_Pss_sig_caps(),
    id_RSAES_OAEP: RSAKeyCapabilities(),
    id_dsa: DSAKeyCapabilities(),
    dhpublicnumber: DSAKeyCapabilities(),
    id_ecPublicKey: EC_SMimeCaps(),
    id_ecDH: EC_SMimeCaps(),
    id_ecMQV: EC_SMimeCaps(),
    id_mgf1: AlgorithmIdentifier(),
}

rfc5751.smimeCapabilityMap.update(_smimeCapabilityMapUpdate)
