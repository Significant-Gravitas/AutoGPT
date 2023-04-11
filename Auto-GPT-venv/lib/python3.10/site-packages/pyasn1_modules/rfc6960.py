#
# This file is part of pyasn1-modules software.
#
# Created by Russ Housley.
#
# Copyright (c) 2019, Vigil Security, LLC
# License: http://snmplabs.com/pyasn1/license.html
#
# Online Certificate Status Protocol (OCSP)
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6960.txt
#

from pyasn1.type import univ, char, namedtype, namedval, tag, constraint, useful

from pyasn1_modules import rfc2560
from pyasn1_modules import rfc5280

MAX = float('inf')


# Imports from RFC 5280

AlgorithmIdentifier = rfc5280.AlgorithmIdentifier
AuthorityInfoAccessSyntax = rfc5280.AuthorityInfoAccessSyntax
Certificate = rfc5280.Certificate
CertificateSerialNumber = rfc5280.CertificateSerialNumber
CRLReason = rfc5280.CRLReason
Extensions = rfc5280.Extensions
GeneralName = rfc5280.GeneralName
Name = rfc5280.Name

id_kp = rfc5280.id_kp

id_ad_ocsp = rfc5280.id_ad_ocsp


# Imports from the original OCSP module in RFC 2560

AcceptableResponses = rfc2560.AcceptableResponses
ArchiveCutoff = rfc2560.ArchiveCutoff
CertStatus = rfc2560.CertStatus
KeyHash = rfc2560.KeyHash
OCSPResponse = rfc2560.OCSPResponse
OCSPResponseStatus = rfc2560.OCSPResponseStatus
ResponseBytes = rfc2560.ResponseBytes
RevokedInfo = rfc2560.RevokedInfo
UnknownInfo = rfc2560.UnknownInfo
Version = rfc2560.Version

id_kp_OCSPSigning = rfc2560.id_kp_OCSPSigning

id_pkix_ocsp = rfc2560.id_pkix_ocsp
id_pkix_ocsp_archive_cutoff = rfc2560.id_pkix_ocsp_archive_cutoff
id_pkix_ocsp_basic = rfc2560.id_pkix_ocsp_basic
id_pkix_ocsp_crl = rfc2560.id_pkix_ocsp_crl
id_pkix_ocsp_nocheck = rfc2560.id_pkix_ocsp_nocheck
id_pkix_ocsp_nonce = rfc2560.id_pkix_ocsp_nonce
id_pkix_ocsp_response = rfc2560.id_pkix_ocsp_response
id_pkix_ocsp_service_locator = rfc2560.id_pkix_ocsp_service_locator


# Additional object identifiers

id_pkix_ocsp_pref_sig_algs = id_pkix_ocsp + (8, )
id_pkix_ocsp_extended_revoke = id_pkix_ocsp + (9, )


# Updated structures (mostly to improve openTypes support)

class CertID(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('hashAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('issuerNameHash', univ.OctetString()),
        namedtype.NamedType('issuerKeyHash', univ.OctetString()),
        namedtype.NamedType('serialNumber', CertificateSerialNumber())
    )


class SingleResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certID', CertID()),
        namedtype.NamedType('certStatus', CertStatus()),
        namedtype.NamedType('thisUpdate', useful.GeneralizedTime()),
        namedtype.OptionalNamedType('nextUpdate', useful.GeneralizedTime().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('singleExtensions', Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class ResponderID(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('byName', Name().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('byKey', KeyHash().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class ResponseData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('version', Version('v1').subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('responderID', ResponderID()),
        namedtype.NamedType('producedAt', useful.GeneralizedTime()),
        namedtype.NamedType('responses', univ.SequenceOf(
            componentType=SingleResponse())),
        namedtype.OptionalNamedType('responseExtensions', Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class BasicOCSPResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsResponseData', ResponseData()),
        namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('signature', univ.BitString()),
        namedtype.OptionalNamedType('certs', univ.SequenceOf(
            componentType=Certificate()).subtype(explicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class Request(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('reqCert', CertID()),
        namedtype.OptionalNamedType('singleRequestExtensions', Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class Signature(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('signatureAlgorithm', AlgorithmIdentifier()),
        namedtype.NamedType('signature', univ.BitString()),
        namedtype.OptionalNamedType('certs', univ.SequenceOf(
            componentType=Certificate()).subtype(explicitTag=tag.Tag(
                tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class TBSRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('version', Version('v1').subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('requestorName', GeneralName().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('requestList', univ.SequenceOf(
            componentType=Request())),
        namedtype.OptionalNamedType('requestExtensions', Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class OCSPRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsRequest', TBSRequest()),
        namedtype.OptionalNamedType('optionalSignature', Signature().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


# Previously omitted structure

class ServiceLocator(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('issuer', Name()),
        namedtype.NamedType('locator', AuthorityInfoAccessSyntax())
    )


# Additional structures

class CrlID(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('crlUrl', char.IA5String().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('crlNum', univ.Integer().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.OptionalNamedType('crlTime', useful.GeneralizedTime().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class PreferredSignatureAlgorithm(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('sigIdentifier', AlgorithmIdentifier()),
        namedtype.OptionalNamedType('certIdentifier', AlgorithmIdentifier())
    )


class PreferredSignatureAlgorithms(univ.SequenceOf):
    componentType = PreferredSignatureAlgorithm()



# Response Type OID to Response Map

ocspResponseMap = {
    id_pkix_ocsp_basic: BasicOCSPResponse(),
}


# Map of Extension OIDs to Extensions added to the ones
# that are in rfc5280.py

_certificateExtensionsMapUpdate = {
    # Certificate Extension
    id_pkix_ocsp_nocheck: univ.Null(""),
    # OCSP Request Extensions
    id_pkix_ocsp_nonce: univ.OctetString(),
    id_pkix_ocsp_response: AcceptableResponses(),
    id_pkix_ocsp_service_locator: ServiceLocator(),
    id_pkix_ocsp_pref_sig_algs: PreferredSignatureAlgorithms(),
    # OCSP Response Extensions
    id_pkix_ocsp_crl: CrlID(),
    id_pkix_ocsp_archive_cutoff: ArchiveCutoff(),
    id_pkix_ocsp_extended_revoke: univ.Null(""),
}

rfc5280.certificateExtensionsMap.update(_certificateExtensionsMapUpdate)
