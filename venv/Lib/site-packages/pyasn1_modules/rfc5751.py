# This file is being contributed to pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# S/MIME Version 3.2 Message Specification
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5751.txt

from pyasn1.type import namedtype
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc5652
from pyasn1_modules import rfc8018


def _OID(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))
    return univ.ObjectIdentifier(output)


# Imports from RFC 5652 and RFC 8018

IssuerAndSerialNumber = rfc5652.IssuerAndSerialNumber

RecipientKeyIdentifier = rfc5652.RecipientKeyIdentifier

SubjectKeyIdentifier = rfc5652.SubjectKeyIdentifier

rc2CBC = rfc8018.rc2CBC


# S/MIME Capabilities Attribute

smimeCapabilities = univ.ObjectIdentifier('1.2.840.113549.1.9.15')


smimeCapabilityMap = { }


class SMIMECapability(univ.Sequence):
    pass

SMIMECapability.componentType = namedtype.NamedTypes(
    namedtype.NamedType('capabilityID', univ.ObjectIdentifier()),
    namedtype.OptionalNamedType('parameters', univ.Any(),
        openType=opentype.OpenType('capabilityID', smimeCapabilityMap))
)


class SMIMECapabilities(univ.SequenceOf):
    pass

SMIMECapabilities.componentType = SMIMECapability()


class SMIMECapabilitiesParametersForRC2CBC(univ.Integer):
    # which carries the RC2 Key Length (number of bits)
    pass


# S/MIME Encryption Key Preference Attribute

id_smime = univ.ObjectIdentifier('1.2.840.113549.1.9.16')

id_aa = _OID(id_smime, 2)

id_aa_encrypKeyPref = _OID(id_aa, 11)


class SMIMEEncryptionKeyPreference(univ.Choice):
    pass

SMIMEEncryptionKeyPreference.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerAndSerialNumber',
        IssuerAndSerialNumber().subtype(implicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('receipentKeyId',
        # Yes, 'receipentKeyId' is spelled incorrectly, but kept
        # this way for alignment with the ASN.1 module in the RFC.
        RecipientKeyIdentifier().subtype(implicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('subjectAltKeyIdentifier',
        SubjectKeyIdentifier().subtype(implicitTag=tag.Tag(
            tag.tagClassContext, tag.tagFormatSimple, 2)))
)


# The Prefer Binary Inside SMIMECapabilities attribute

id_cap = _OID(id_smime, 11)

id_cap_preferBinaryInside = _OID(id_cap, 1)


# CMS Attribute Map

_cmsAttributesMapUpdate = {
    smimeCapabilities: SMIMECapabilities(),
    id_aa_encrypKeyPref: SMIMEEncryptionKeyPreference(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)


# SMIMECapabilities Attribute Map
#
# Do not include OIDs in the dictionary when the parameters are absent.

_smimeCapabilityMapUpdate = {
    rc2CBC: SMIMECapabilitiesParametersForRC2CBC(),
}

smimeCapabilityMap.update(_smimeCapabilityMapUpdate)
