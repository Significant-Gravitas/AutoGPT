#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley with assistance from asn1ate v.0.6.0.
# Modified by Russ Housley to add a map for use with opentypes.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Update to Enhanced Security Services for S/MIME
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc5035.txt
#

from pyasn1.codec.der.encoder import encode as der_encode

from pyasn1.type import namedtype
from pyasn1.type import univ

from pyasn1_modules import rfc2634
from pyasn1_modules import rfc4055
from pyasn1_modules import rfc5652
from pyasn1_modules import rfc5280

ContentType = rfc5652.ContentType

IssuerAndSerialNumber = rfc5652.IssuerAndSerialNumber

SubjectKeyIdentifier = rfc5652.SubjectKeyIdentifier

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier

PolicyInformation = rfc5280.PolicyInformation

GeneralNames = rfc5280.GeneralNames

CertificateSerialNumber = rfc5280.CertificateSerialNumber


# Signing Certificate Attribute V1 and V2

id_aa_signingCertificate = rfc2634.id_aa_signingCertificate

id_aa_signingCertificateV2 = univ.ObjectIdentifier('1.2.840.113549.1.9.16.2.47')

Hash = rfc2634.Hash

IssuerSerial = rfc2634.IssuerSerial

ESSCertID = rfc2634.ESSCertID

SigningCertificate = rfc2634.SigningCertificate


sha256AlgId = AlgorithmIdentifier()
sha256AlgId['algorithm'] = rfc4055.id_sha256
# A non-schema object for sha256AlgId['parameters'] as absent
sha256AlgId['parameters'] = der_encode(univ.OctetString(''))


class ESSCertIDv2(univ.Sequence):
    pass

ESSCertIDv2.componentType = namedtype.NamedTypes(
    namedtype.DefaultedNamedType('hashAlgorithm', sha256AlgId),
    namedtype.NamedType('certHash', Hash()),
    namedtype.OptionalNamedType('issuerSerial', IssuerSerial())
)


class SigningCertificateV2(univ.Sequence):
    pass

SigningCertificateV2.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certs', univ.SequenceOf(
        componentType=ESSCertIDv2())),
    namedtype.OptionalNamedType('policies', univ.SequenceOf(
        componentType=PolicyInformation()))
)


# Mail List Expansion History Attribute

id_aa_mlExpandHistory = rfc2634.id_aa_mlExpandHistory

ub_ml_expansion_history = rfc2634.ub_ml_expansion_history

EntityIdentifier = rfc2634.EntityIdentifier

MLReceiptPolicy = rfc2634.MLReceiptPolicy

MLData = rfc2634.MLData

MLExpansionHistory = rfc2634.MLExpansionHistory


# ESS Security Label Attribute

id_aa_securityLabel = rfc2634.id_aa_securityLabel

ub_privacy_mark_length = rfc2634.ub_privacy_mark_length

ub_security_categories = rfc2634.ub_security_categories

ub_integer_options = rfc2634.ub_integer_options

ESSPrivacyMark = rfc2634.ESSPrivacyMark

SecurityClassification = rfc2634.SecurityClassification

SecurityPolicyIdentifier = rfc2634.SecurityPolicyIdentifier

SecurityCategory = rfc2634.SecurityCategory

SecurityCategories = rfc2634.SecurityCategories

ESSSecurityLabel = rfc2634.ESSSecurityLabel


# Equivalent Labels Attribute

id_aa_equivalentLabels = rfc2634.id_aa_equivalentLabels

EquivalentLabels = rfc2634.EquivalentLabels


# Content Identifier Attribute

id_aa_contentIdentifier = rfc2634.id_aa_contentIdentifier

ContentIdentifier = rfc2634.ContentIdentifier


# Content Reference Attribute

id_aa_contentReference = rfc2634.id_aa_contentReference

ContentReference = rfc2634.ContentReference


# Message Signature Digest Attribute

id_aa_msgSigDigest = rfc2634.id_aa_msgSigDigest

MsgSigDigest = rfc2634.MsgSigDigest


# Content Hints Attribute

id_aa_contentHint = rfc2634.id_aa_contentHint

ContentHints = rfc2634.ContentHints


# Receipt Request Attribute

AllOrFirstTier = rfc2634.AllOrFirstTier

ReceiptsFrom = rfc2634.ReceiptsFrom

id_aa_receiptRequest = rfc2634.id_aa_receiptRequest

ub_receiptsTo = rfc2634.ub_receiptsTo

ReceiptRequest = rfc2634.ReceiptRequest


# Receipt Content Type

ESSVersion = rfc2634.ESSVersion

id_ct_receipt = rfc2634.id_ct_receipt

Receipt = rfc2634.Receipt

ub_receiptsTo = rfc2634.ub_receiptsTo

ReceiptRequest = rfc2634.ReceiptRequest


# Map of Attribute Type to the Attribute structure is added to the
# ones that are in rfc5652.py

_cmsAttributesMapUpdate = {
    id_aa_signingCertificateV2: SigningCertificateV2(),
}

rfc5652.cmsAttributesMap.update(_cmsAttributesMapUpdate)


# Map of Content Type OIDs to Content Types is added to the
# ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_ct_receipt: Receipt(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)
