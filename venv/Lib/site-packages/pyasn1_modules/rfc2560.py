#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# OCSP request/response syntax
#
# Derived from a minimal OCSP library (RFC2560) code written by
# Bud P. Bruegger <bud@ancitel.it>
# Copyright: Ancitel, S.p.a,  Rome, Italy
# License: BSD
#

#
# current limitations:
# * request and response works only for a single certificate
# * only some values are parsed out of the response
# * the request does't set a nonce nor signature
# * there is no signature validation of the response
# * dates are left as strings in GeneralizedTime format -- datetime.datetime
# would be nicer
#
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc2459


# Start of OCSP module definitions

# This should be in directory Authentication Framework (X.509) module

class CRLReason(univ.Enumerated):
    namedValues = namedval.NamedValues(
        ('unspecified', 0),
        ('keyCompromise', 1),
        ('cACompromise', 2),
        ('affiliationChanged', 3),
        ('superseded', 4),
        ('cessationOfOperation', 5),
        ('certificateHold', 6),
        ('removeFromCRL', 8),
        ('privilegeWithdrawn', 9),
        ('aACompromise', 10)
    )


# end of directory Authentication Framework (X.509) module

# This should be in PKIX Certificate Extensions module

class GeneralName(univ.OctetString):
    pass


# end of PKIX Certificate Extensions module

id_kp_OCSPSigning = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 3, 9))
id_pkix_ocsp = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1))
id_pkix_ocsp_basic = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 1))
id_pkix_ocsp_nonce = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 2))
id_pkix_ocsp_crl = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 3))
id_pkix_ocsp_response = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 4))
id_pkix_ocsp_nocheck = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 5))
id_pkix_ocsp_archive_cutoff = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 6))
id_pkix_ocsp_service_locator = univ.ObjectIdentifier((1, 3, 6, 1, 5, 5, 7, 48, 1, 7))


class AcceptableResponses(univ.SequenceOf):
    componentType = univ.ObjectIdentifier()


class ArchiveCutoff(useful.GeneralizedTime):
    pass


class UnknownInfo(univ.Null):
    pass


class RevokedInfo(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('revocationTime', useful.GeneralizedTime()),
        namedtype.OptionalNamedType('revocationReason', CRLReason().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class CertID(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('hashAlgorithm', rfc2459.AlgorithmIdentifier()),
        namedtype.NamedType('issuerNameHash', univ.OctetString()),
        namedtype.NamedType('issuerKeyHash', univ.OctetString()),
        namedtype.NamedType('serialNumber', rfc2459.CertificateSerialNumber())
    )


class CertStatus(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('good',
                            univ.Null().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('revoked',
                            RevokedInfo().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('unknown',
                            UnknownInfo().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class SingleResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certID', CertID()),
        namedtype.NamedType('certStatus', CertStatus()),
        namedtype.NamedType('thisUpdate', useful.GeneralizedTime()),
        namedtype.OptionalNamedType('nextUpdate', useful.GeneralizedTime().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('singleExtensions', rfc2459.Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class KeyHash(univ.OctetString):
    pass


class ResponderID(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('byName',
                            rfc2459.Name().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('byKey',
                            KeyHash().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class Version(univ.Integer):
    namedValues = namedval.NamedValues(('v1', 0))


class ResponseData(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('version', Version('v1').subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('responderID', ResponderID()),
        namedtype.NamedType('producedAt', useful.GeneralizedTime()),
        namedtype.NamedType('responses', univ.SequenceOf(componentType=SingleResponse())),
        namedtype.OptionalNamedType('responseExtensions', rfc2459.Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1)))
    )


class BasicOCSPResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsResponseData', ResponseData()),
        namedtype.NamedType('signatureAlgorithm', rfc2459.AlgorithmIdentifier()),
        namedtype.NamedType('signature', univ.BitString()),
        namedtype.OptionalNamedType('certs', univ.SequenceOf(componentType=rfc2459.Certificate()).subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class ResponseBytes(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('responseType', univ.ObjectIdentifier()),
        namedtype.NamedType('response', univ.OctetString())
    )


class OCSPResponseStatus(univ.Enumerated):
    namedValues = namedval.NamedValues(
        ('successful', 0),
        ('malformedRequest', 1),
        ('internalError', 2),
        ('tryLater', 3),
        ('undefinedStatus', 4),  # should never occur
        ('sigRequired', 5),
        ('unauthorized', 6)
    )


class OCSPResponse(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('responseStatus', OCSPResponseStatus()),
        namedtype.OptionalNamedType('responseBytes', ResponseBytes().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class Request(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('reqCert', CertID()),
        namedtype.OptionalNamedType('singleRequestExtensions', rfc2459.Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class Signature(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('signatureAlgorithm', rfc2459.AlgorithmIdentifier()),
        namedtype.NamedType('signature', univ.BitString()),
        namedtype.OptionalNamedType('certs', univ.SequenceOf(componentType=rfc2459.Certificate()).subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )


class TBSRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.DefaultedNamedType('version', Version('v1').subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('requestorName', GeneralName().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('requestList', univ.SequenceOf(componentType=Request())),
        namedtype.OptionalNamedType('requestExtensions', rfc2459.Extensions().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class OCSPRequest(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('tbsRequest', TBSRequest()),
        namedtype.OptionalNamedType('optionalSignature', Signature().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
    )
