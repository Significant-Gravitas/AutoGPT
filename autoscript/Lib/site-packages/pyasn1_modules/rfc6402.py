# coding: utf-8
#
# This file is part of pyasn1-modules software.
#
# Created by Stanisław Pitucha with asn1ate tool.
# Modified by Russ Housley to add a maps for CMC Control Attributes
#   and CMC Content Types for use with opentypes.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# Certificate Management over CMS (CMC) Updates
#
# ASN.1 source from:
# https://www.rfc-editor.org/rfc/rfc6402.txt
#
from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import opentype
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc4211
from pyasn1_modules import rfc5280
from pyasn1_modules import rfc5652

MAX = float('inf')


def _buildOid(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


# Since CMS Attributes and CMC Controls both use 'attrType', one map is used 
cmcControlAttributesMap = rfc5652.cmsAttributesMap


class ChangeSubjectName(univ.Sequence):
    pass


ChangeSubjectName.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('subject', rfc5280.Name()),
    namedtype.OptionalNamedType('subjectAlt', rfc5280.GeneralNames())
)


class AttributeValue(univ.Any):
    pass


class CMCStatus(univ.Integer):
    pass


CMCStatus.namedValues = namedval.NamedValues(
    ('success', 0),
    ('failed', 2),
    ('pending', 3),
    ('noSupport', 4),
    ('confirmRequired', 5),
    ('popRequired', 6),
    ('partial', 7)
)


class PendInfo(univ.Sequence):
    pass


PendInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pendToken', univ.OctetString()),
    namedtype.NamedType('pendTime', useful.GeneralizedTime())
)

bodyIdMax = univ.Integer(4294967295)


class BodyPartID(univ.Integer):
    pass


BodyPartID.subtypeSpec = constraint.ValueRangeConstraint(0, bodyIdMax)


class BodyPartPath(univ.SequenceOf):
    pass


BodyPartPath.componentType = BodyPartID()
BodyPartPath.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class BodyPartReference(univ.Choice):
    pass


BodyPartReference.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartID', BodyPartID()),
    namedtype.NamedType('bodyPartPath', BodyPartPath())
)


class CMCFailInfo(univ.Integer):
    pass


CMCFailInfo.namedValues = namedval.NamedValues(
    ('badAlg', 0),
    ('badMessageCheck', 1),
    ('badRequest', 2),
    ('badTime', 3),
    ('badCertId', 4),
    ('unsupportedExt', 5),
    ('mustArchiveKeys', 6),
    ('badIdentity', 7),
    ('popRequired', 8),
    ('popFailed', 9),
    ('noKeyReuse', 10),
    ('internalCAError', 11),
    ('tryLater', 12),
    ('authDataFail', 13)
)


class CMCStatusInfoV2(univ.Sequence):
    pass


CMCStatusInfoV2.componentType = namedtype.NamedTypes(
    namedtype.NamedType('cMCStatus', CMCStatus()),
    namedtype.NamedType('bodyList', univ.SequenceOf(componentType=BodyPartReference())),
    namedtype.OptionalNamedType('statusString', char.UTF8String()),
    namedtype.OptionalNamedType(
        'otherInfo', univ.Choice(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType('failInfo', CMCFailInfo()),
                namedtype.NamedType('pendInfo', PendInfo()),
                namedtype.NamedType(
                    'extendedFailInfo', univ.Sequence(
                    componentType=namedtype.NamedTypes(
                        namedtype.NamedType('failInfoOID', univ.ObjectIdentifier()),
                        namedtype.NamedType('failInfoValue', AttributeValue()))
                    )
                )
            )
        )
    )
)


class GetCRL(univ.Sequence):
    pass


GetCRL.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerName', rfc5280.Name()),
    namedtype.OptionalNamedType('cRLName', rfc5280.GeneralName()),
    namedtype.OptionalNamedType('time', useful.GeneralizedTime()),
    namedtype.OptionalNamedType('reasons', rfc5280.ReasonFlags())
)

id_pkix = _buildOid(1, 3, 6, 1, 5, 5, 7)

id_cmc = _buildOid(id_pkix, 7)

id_cmc_batchResponses = _buildOid(id_cmc, 29)

id_cmc_popLinkWitness = _buildOid(id_cmc, 23)


class PopLinkWitnessV2(univ.Sequence):
    pass


PopLinkWitnessV2.componentType = namedtype.NamedTypes(
    namedtype.NamedType('keyGenAlgorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('macAlgorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('witness', univ.OctetString())
)

id_cmc_popLinkWitnessV2 = _buildOid(id_cmc, 33)

id_cmc_identityProofV2 = _buildOid(id_cmc, 34)

id_cmc_revokeRequest = _buildOid(id_cmc, 17)

id_cmc_recipientNonce = _buildOid(id_cmc, 7)


class ControlsProcessed(univ.Sequence):
    pass


ControlsProcessed.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyList', univ.SequenceOf(componentType=BodyPartReference()))
)


class CertificationRequest(univ.Sequence):
    pass


CertificationRequest.componentType = namedtype.NamedTypes(
    namedtype.NamedType(
        'certificationRequestInfo', univ.Sequence(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType('version', univ.Integer()),
                namedtype.NamedType('subject', rfc5280.Name()),
                namedtype.NamedType(
                    'subjectPublicKeyInfo', univ.Sequence(
                        componentType=namedtype.NamedTypes(
                            namedtype.NamedType('algorithm', rfc5280.AlgorithmIdentifier()),
                            namedtype.NamedType('subjectPublicKey', univ.BitString())
                        )
                    )
                ),
                namedtype.NamedType(
                    'attributes', univ.SetOf(
                        componentType=rfc5652.Attribute()).subtype(
                        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))
                )
            )
        )
    ),
    namedtype.NamedType('signatureAlgorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('signature', univ.BitString())
)


class TaggedCertificationRequest(univ.Sequence):
    pass


TaggedCertificationRequest.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartID', BodyPartID()),
    namedtype.NamedType('certificationRequest', CertificationRequest())
)


class TaggedRequest(univ.Choice):
    pass


TaggedRequest.componentType = namedtype.NamedTypes(
    namedtype.NamedType('tcr', TaggedCertificationRequest().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('crm',
                        rfc4211.CertReqMsg().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('orm', univ.Sequence(componentType=namedtype.NamedTypes(
        namedtype.NamedType('bodyPartID', BodyPartID()),
        namedtype.NamedType('requestMessageType', univ.ObjectIdentifier()),
        namedtype.NamedType('requestMessageValue', univ.Any())
    ))
                        .subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)))
)

id_cmc_popLinkRandom = _buildOid(id_cmc, 22)

id_cmc_statusInfo = _buildOid(id_cmc, 1)

id_cmc_trustedAnchors = _buildOid(id_cmc, 26)

id_cmc_transactionId = _buildOid(id_cmc, 5)

id_cmc_encryptedPOP = _buildOid(id_cmc, 9)


class PublishTrustAnchors(univ.Sequence):
    pass


PublishTrustAnchors.componentType = namedtype.NamedTypes(
    namedtype.NamedType('seqNumber', univ.Integer()),
    namedtype.NamedType('hashAlgorithm', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('anchorHashes', univ.SequenceOf(componentType=univ.OctetString()))
)


class RevokeRequest(univ.Sequence):
    pass


RevokeRequest.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerName', rfc5280.Name()),
    namedtype.NamedType('serialNumber', univ.Integer()),
    namedtype.NamedType('reason', rfc5280.CRLReason()),
    namedtype.OptionalNamedType('invalidityDate', useful.GeneralizedTime()),
    namedtype.OptionalNamedType('passphrase', univ.OctetString()),
    namedtype.OptionalNamedType('comment', char.UTF8String())
)

id_cmc_senderNonce = _buildOid(id_cmc, 6)

id_cmc_authData = _buildOid(id_cmc, 27)


class TaggedContentInfo(univ.Sequence):
    pass


TaggedContentInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartID', BodyPartID()),
    namedtype.NamedType('contentInfo', rfc5652.ContentInfo())
)


class IdentifyProofV2(univ.Sequence):
    pass


IdentifyProofV2.componentType = namedtype.NamedTypes(
    namedtype.NamedType('proofAlgID', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('macAlgId', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('witness', univ.OctetString())
)


class CMCPublicationInfo(univ.Sequence):
    pass


CMCPublicationInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('hashAlg', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('certHashes', univ.SequenceOf(componentType=univ.OctetString())),
    namedtype.NamedType('pubInfo', rfc4211.PKIPublicationInfo())
)

id_kp_cmcCA = _buildOid(rfc5280.id_kp, 27)

id_cmc_confirmCertAcceptance = _buildOid(id_cmc, 24)

id_cmc_raIdentityWitness = _buildOid(id_cmc, 35)

id_ExtensionReq = _buildOid(1, 2, 840, 113549, 1, 9, 14)

id_cct = _buildOid(id_pkix, 12)

id_cct_PKIData = _buildOid(id_cct, 2)

id_kp_cmcRA = _buildOid(rfc5280.id_kp, 28)


class CMCStatusInfo(univ.Sequence):
    pass


CMCStatusInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('cMCStatus', CMCStatus()),
    namedtype.NamedType('bodyList', univ.SequenceOf(componentType=BodyPartID())),
    namedtype.OptionalNamedType('statusString', char.UTF8String()),
    namedtype.OptionalNamedType(
        'otherInfo', univ.Choice(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType('failInfo', CMCFailInfo()),
                namedtype.NamedType('pendInfo', PendInfo())
            )
        )
    )
)


class DecryptedPOP(univ.Sequence):
    pass


DecryptedPOP.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartID', BodyPartID()),
    namedtype.NamedType('thePOPAlgID', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('thePOP', univ.OctetString())
)

id_cmc_addExtensions = _buildOid(id_cmc, 8)

id_cmc_modCertTemplate = _buildOid(id_cmc, 31)


class TaggedAttribute(univ.Sequence):
    pass


TaggedAttribute.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartID', BodyPartID()),
    namedtype.NamedType('attrType', univ.ObjectIdentifier()),
    namedtype.NamedType('attrValues', univ.SetOf(componentType=AttributeValue()),
        openType=opentype.OpenType('attrType', cmcControlAttributesMap)
    )
)


class OtherMsg(univ.Sequence):
    pass


OtherMsg.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartID', BodyPartID()),
    namedtype.NamedType('otherMsgType', univ.ObjectIdentifier()),
    namedtype.NamedType('otherMsgValue', univ.Any())
)


class PKIData(univ.Sequence):
    pass


PKIData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('controlSequence', univ.SequenceOf(componentType=TaggedAttribute())),
    namedtype.NamedType('reqSequence', univ.SequenceOf(componentType=TaggedRequest())),
    namedtype.NamedType('cmsSequence', univ.SequenceOf(componentType=TaggedContentInfo())),
    namedtype.NamedType('otherMsgSequence', univ.SequenceOf(componentType=OtherMsg()))
)


class BodyPartList(univ.SequenceOf):
    pass


BodyPartList.componentType = BodyPartID()
BodyPartList.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

id_cmc_responseBody = _buildOid(id_cmc, 37)


class AuthPublish(BodyPartID):
    pass


class CMCUnsignedData(univ.Sequence):
    pass


CMCUnsignedData.componentType = namedtype.NamedTypes(
    namedtype.NamedType('bodyPartPath', BodyPartPath()),
    namedtype.NamedType('identifier', univ.ObjectIdentifier()),
    namedtype.NamedType('content', univ.Any())
)


class CMCCertId(rfc5652.IssuerAndSerialNumber):
    pass


class PKIResponse(univ.Sequence):
    pass


PKIResponse.componentType = namedtype.NamedTypes(
    namedtype.NamedType('controlSequence', univ.SequenceOf(componentType=TaggedAttribute())),
    namedtype.NamedType('cmsSequence', univ.SequenceOf(componentType=TaggedContentInfo())),
    namedtype.NamedType('otherMsgSequence', univ.SequenceOf(componentType=OtherMsg()))
)


class ResponseBody(PKIResponse):
    pass


id_cmc_statusInfoV2 = _buildOid(id_cmc, 25)

id_cmc_lraPOPWitness = _buildOid(id_cmc, 11)


class ModCertTemplate(univ.Sequence):
    pass


ModCertTemplate.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pkiDataReference', BodyPartPath()),
    namedtype.NamedType('certReferences', BodyPartList()),
    namedtype.DefaultedNamedType('replace', univ.Boolean().subtype(value=1)),
    namedtype.NamedType('certTemplate', rfc4211.CertTemplate())
)

id_cmc_regInfo = _buildOid(id_cmc, 18)

id_cmc_identityProof = _buildOid(id_cmc, 3)


class ExtensionReq(univ.SequenceOf):
    pass


ExtensionReq.componentType = rfc5280.Extension()
ExtensionReq.sizeSpec = constraint.ValueSizeConstraint(1, MAX)

id_kp_cmcArchive = _buildOid(rfc5280.id_kp, 28)

id_cmc_publishCert = _buildOid(id_cmc, 30)

id_cmc_dataReturn = _buildOid(id_cmc, 4)


class LraPopWitness(univ.Sequence):
    pass


LraPopWitness.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pkiDataBodyid', BodyPartID()),
    namedtype.NamedType('bodyIds', univ.SequenceOf(componentType=BodyPartID()))
)

id_aa = _buildOid(1, 2, 840, 113549, 1, 9, 16, 2)

id_aa_cmc_unsignedData = _buildOid(id_aa, 34)

id_cmc_getCert = _buildOid(id_cmc, 15)

id_cmc_batchRequests = _buildOid(id_cmc, 28)

id_cmc_decryptedPOP = _buildOid(id_cmc, 10)

id_cmc_responseInfo = _buildOid(id_cmc, 19)

id_cmc_changeSubjectName = _buildOid(id_cmc, 36)


class GetCert(univ.Sequence):
    pass


GetCert.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuerName', rfc5280.GeneralName()),
    namedtype.NamedType('serialNumber', univ.Integer())
)

id_cmc_identification = _buildOid(id_cmc, 2)

id_cmc_queryPending = _buildOid(id_cmc, 21)


class AddExtensions(univ.Sequence):
    pass


AddExtensions.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pkiDataReference', BodyPartID()),
    namedtype.NamedType('certReferences', univ.SequenceOf(componentType=BodyPartID())),
    namedtype.NamedType('extensions', univ.SequenceOf(componentType=rfc5280.Extension()))
)


class EncryptedPOP(univ.Sequence):
    pass


EncryptedPOP.componentType = namedtype.NamedTypes(
    namedtype.NamedType('request', TaggedRequest()),
    namedtype.NamedType('cms', rfc5652.ContentInfo()),
    namedtype.NamedType('thePOPAlgID', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('witnessAlgID', rfc5280.AlgorithmIdentifier()),
    namedtype.NamedType('witness', univ.OctetString())
)

id_cmc_getCRL = _buildOid(id_cmc, 16)

id_cct_PKIResponse = _buildOid(id_cct, 3)

id_cmc_controlProcessed = _buildOid(id_cmc, 32)


class NoSignatureValue(univ.OctetString):
    pass


id_ad_cmc = _buildOid(rfc5280.id_ad, 12)

id_alg_noSignature = _buildOid(id_pkix, 6, 2)


# Map of CMC Control OIDs to CMC Control Attributes

_cmcControlAttributesMapUpdate = {
    id_cmc_statusInfo: CMCStatusInfo(),
    id_cmc_statusInfoV2: CMCStatusInfoV2(),
    id_cmc_identification: char.UTF8String(),
    id_cmc_identityProof: univ.OctetString(),
    id_cmc_identityProofV2: IdentifyProofV2(),
    id_cmc_dataReturn: univ.OctetString(),
    id_cmc_transactionId: univ.Integer(),
    id_cmc_senderNonce: univ.OctetString(),
    id_cmc_recipientNonce: univ.OctetString(),
    id_cmc_addExtensions: AddExtensions(),
    id_cmc_encryptedPOP: EncryptedPOP(),
    id_cmc_decryptedPOP: DecryptedPOP(),
    id_cmc_lraPOPWitness: LraPopWitness(),
    id_cmc_getCert: GetCert(),
    id_cmc_getCRL: GetCRL(),
    id_cmc_revokeRequest: RevokeRequest(),
    id_cmc_regInfo: univ.OctetString(),
    id_cmc_responseInfo: univ.OctetString(),
    id_cmc_queryPending: univ.OctetString(),
    id_cmc_popLinkRandom: univ.OctetString(),
    id_cmc_popLinkWitness: univ.OctetString(),
    id_cmc_popLinkWitnessV2: PopLinkWitnessV2(),
    id_cmc_confirmCertAcceptance: CMCCertId(),
    id_cmc_trustedAnchors: PublishTrustAnchors(),
    id_cmc_authData: AuthPublish(),
    id_cmc_batchRequests: BodyPartList(),
    id_cmc_batchResponses: BodyPartList(),
    id_cmc_publishCert: CMCPublicationInfo(),
    id_cmc_modCertTemplate: ModCertTemplate(),
    id_cmc_controlProcessed: ControlsProcessed(),
    id_ExtensionReq: ExtensionReq(),
}

cmcControlAttributesMap.update(_cmcControlAttributesMapUpdate)


# Map of CMC Content Type OIDs to CMC Content Types are added to
# the ones that are in rfc5652.py

_cmsContentTypesMapUpdate = {
    id_cct_PKIData: PKIData(),
    id_cct_PKIResponse: PKIResponse(),
}

rfc5652.cmsContentTypesMap.update(_cmsContentTypesMapUpdate)

