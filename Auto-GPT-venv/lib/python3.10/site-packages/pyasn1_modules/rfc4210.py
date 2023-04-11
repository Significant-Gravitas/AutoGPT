#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# Certificate Management Protocol structures as per RFC4210
#
# Based on Alex Railean's work
#
from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

from pyasn1_modules import rfc2314
from pyasn1_modules import rfc2459
from pyasn1_modules import rfc2511

MAX = float('inf')


class KeyIdentifier(univ.OctetString):
    pass


class CMPCertificate(rfc2459.Certificate):
    pass


class OOBCert(CMPCertificate):
    pass


class CertAnnContent(CMPCertificate):
    pass


class PKIFreeText(univ.SequenceOf):
    """
    PKIFreeText ::= SEQUENCE SIZE (1..MAX) OF UTF8String
    """
    componentType = char.UTF8String()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


class PollRepContent(univ.SequenceOf):
    """
         PollRepContent ::= SEQUENCE OF SEQUENCE {
         certReqId              INTEGER,
         checkAfter             INTEGER,  -- time in seconds
         reason                 PKIFreeText OPTIONAL
     }
    """

    class CertReq(univ.Sequence):
        componentType = namedtype.NamedTypes(
            namedtype.NamedType('certReqId', univ.Integer()),
            namedtype.NamedType('checkAfter', univ.Integer()),
            namedtype.OptionalNamedType('reason', PKIFreeText())
        )

    componentType = CertReq()


class PollReqContent(univ.SequenceOf):
    """
         PollReqContent ::= SEQUENCE OF SEQUENCE {
         certReqId              INTEGER
     }

    """

    class CertReq(univ.Sequence):
        componentType = namedtype.NamedTypes(
            namedtype.NamedType('certReqId', univ.Integer())
        )

    componentType = CertReq()


class InfoTypeAndValue(univ.Sequence):
    """
    InfoTypeAndValue ::= SEQUENCE {
     infoType               OBJECT IDENTIFIER,
     infoValue              ANY DEFINED BY infoType  OPTIONAL
    }"""
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('infoType', univ.ObjectIdentifier()),
        namedtype.OptionalNamedType('infoValue', univ.Any())
    )


class GenRepContent(univ.SequenceOf):
    componentType = InfoTypeAndValue()


class GenMsgContent(univ.SequenceOf):
    componentType = InfoTypeAndValue()


class PKIConfirmContent(univ.Null):
    pass


class CRLAnnContent(univ.SequenceOf):
    componentType = rfc2459.CertificateList()


class CAKeyUpdAnnContent(univ.Sequence):
    """
    CAKeyUpdAnnContent ::= SEQUENCE {
         oldWithNew   CMPCertificate,
         newWithOld   CMPCertificate,
         newWithNew   CMPCertificate
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('oldWithNew', CMPCertificate()),
        namedtype.NamedType('newWithOld', CMPCertificate()),
        namedtype.NamedType('newWithNew', CMPCertificate())
    )


class RevDetails(univ.Sequence):
    """
    RevDetails ::= SEQUENCE {
         certDetails         CertTemplate,
         crlEntryDetails     Extensions       OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certDetails', rfc2511.CertTemplate()),
        namedtype.OptionalNamedType('crlEntryDetails', rfc2459.Extensions())
    )


class RevReqContent(univ.SequenceOf):
    componentType = RevDetails()


class CertOrEncCert(univ.Choice):
    """
     CertOrEncCert ::= CHOICE {
         certificate     [0] CMPCertificate,
         encryptedCert   [1] EncryptedValue
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certificate', CMPCertificate().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.NamedType('encryptedCert', rfc2511.EncryptedValue().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class CertifiedKeyPair(univ.Sequence):
    """
    CertifiedKeyPair ::= SEQUENCE {
         certOrEncCert       CertOrEncCert,
         privateKey      [0] EncryptedValue      OPTIONAL,
         publicationInfo [1] PKIPublicationInfo  OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certOrEncCert', CertOrEncCert()),
        namedtype.OptionalNamedType('privateKey', rfc2511.EncryptedValue().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
        namedtype.OptionalNamedType('publicationInfo', rfc2511.PKIPublicationInfo().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
    )


class POPODecKeyRespContent(univ.SequenceOf):
    componentType = univ.Integer()


class Challenge(univ.Sequence):
    """
    Challenge ::= SEQUENCE {
         owf                 AlgorithmIdentifier  OPTIONAL,
         witness             OCTET STRING,
         challenge           OCTET STRING
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType('owf', rfc2459.AlgorithmIdentifier()),
        namedtype.NamedType('witness', univ.OctetString()),
        namedtype.NamedType('challenge', univ.OctetString())
    )


class PKIStatus(univ.Integer):
    """
    PKIStatus ::= INTEGER {
         accepted                (0),
         grantedWithMods        (1),
         rejection              (2),
         waiting                (3),
         revocationWarning      (4),
         revocationNotification (5),
         keyUpdateWarning       (6)
     }
    """
    namedValues = namedval.NamedValues(
        ('accepted', 0),
        ('grantedWithMods', 1),
        ('rejection', 2),
        ('waiting', 3),
        ('revocationWarning', 4),
        ('revocationNotification', 5),
        ('keyUpdateWarning', 6)
    )


class PKIFailureInfo(univ.BitString):
    """
    PKIFailureInfo ::= BIT STRING {
         badAlg              (0),
         badMessageCheck     (1),
         badRequest          (2),
         badTime             (3),
         badCertId           (4),
         badDataFormat       (5),
         wrongAuthority      (6),
         incorrectData       (7),
         missingTimeStamp    (8),
         badPOP              (9),
         certRevoked         (10),
         certConfirmed       (11),
         wrongIntegrity      (12),
         badRecipientNonce   (13),
         timeNotAvailable    (14),
         unacceptedPolicy    (15),
         unacceptedExtension (16),
         addInfoNotAvailable (17),
         badSenderNonce      (18),
         badCertTemplate     (19),
         signerNotTrusted    (20),
         transactionIdInUse  (21),
         unsupportedVersion  (22),
         notAuthorized       (23),
         systemUnavail       (24),
         systemFailure       (25),
         duplicateCertReq    (26)
    """
    namedValues = namedval.NamedValues(
        ('badAlg', 0),
        ('badMessageCheck', 1),
        ('badRequest', 2),
        ('badTime', 3),
        ('badCertId', 4),
        ('badDataFormat', 5),
        ('wrongAuthority', 6),
        ('incorrectData', 7),
        ('missingTimeStamp', 8),
        ('badPOP', 9),
        ('certRevoked', 10),
        ('certConfirmed', 11),
        ('wrongIntegrity', 12),
        ('badRecipientNonce', 13),
        ('timeNotAvailable', 14),
        ('unacceptedPolicy', 15),
        ('unacceptedExtension', 16),
        ('addInfoNotAvailable', 17),
        ('badSenderNonce', 18),
        ('badCertTemplate', 19),
        ('signerNotTrusted', 20),
        ('transactionIdInUse', 21),
        ('unsupportedVersion', 22),
        ('notAuthorized', 23),
        ('systemUnavail', 24),
        ('systemFailure', 25),
        ('duplicateCertReq', 26)
    )


class PKIStatusInfo(univ.Sequence):
    """
    PKIStatusInfo ::= SEQUENCE {
         status        PKIStatus,
         statusString  PKIFreeText     OPTIONAL,
         failInfo      PKIFailureInfo  OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('status', PKIStatus()),
        namedtype.OptionalNamedType('statusString', PKIFreeText()),
        namedtype.OptionalNamedType('failInfo', PKIFailureInfo())
    )


class ErrorMsgContent(univ.Sequence):
    """
    ErrorMsgContent ::= SEQUENCE {
         pKIStatusInfo          PKIStatusInfo,
         errorCode              INTEGER           OPTIONAL,
         -- implementation-specific error codes
         errorDetails           PKIFreeText       OPTIONAL
         -- implementation-specific error details
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('pKIStatusInfo', PKIStatusInfo()),
        namedtype.OptionalNamedType('errorCode', univ.Integer()),
        namedtype.OptionalNamedType('errorDetails', PKIFreeText())
    )


class CertStatus(univ.Sequence):
    """
    CertStatus ::= SEQUENCE {
        certHash    OCTET STRING,
        certReqId   INTEGER,
        statusInfo  PKIStatusInfo OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certHash', univ.OctetString()),
        namedtype.NamedType('certReqId', univ.Integer()),
        namedtype.OptionalNamedType('statusInfo', PKIStatusInfo())
    )


class CertConfirmContent(univ.SequenceOf):
    componentType = CertStatus()


class RevAnnContent(univ.Sequence):
    """
    RevAnnContent ::= SEQUENCE {
         status              PKIStatus,
         certId              CertId,
         willBeRevokedAt     GeneralizedTime,
         badSinceDate        GeneralizedTime,
         crlDetails          Extensions  OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('status', PKIStatus()),
        namedtype.NamedType('certId', rfc2511.CertId()),
        namedtype.NamedType('willBeRevokedAt', useful.GeneralizedTime()),
        namedtype.NamedType('badSinceDate', useful.GeneralizedTime()),
        namedtype.OptionalNamedType('crlDetails', rfc2459.Extensions())
    )


class RevRepContent(univ.Sequence):
    """
    RevRepContent ::= SEQUENCE {
         status       SEQUENCE SIZE (1..MAX) OF PKIStatusInfo,
         revCerts [0] SEQUENCE SIZE (1..MAX) OF CertId
                                             OPTIONAL,
         crls     [1] SEQUENCE SIZE (1..MAX) OF CertificateList
                                             OPTIONAL
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType(
            'status', univ.SequenceOf(
                componentType=PKIStatusInfo(),
                sizeSpec=constraint.ValueSizeConstraint(1, MAX)
            )
        ),
        namedtype.OptionalNamedType(
            'revCerts', univ.SequenceOf(componentType=rfc2511.CertId()).subtype(
                sizeSpec=constraint.ValueSizeConstraint(1, MAX),
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)
            )
        ),
        namedtype.OptionalNamedType(
            'crls', univ.SequenceOf(componentType=rfc2459.CertificateList()).subtype(
                sizeSpec=constraint.ValueSizeConstraint(1, MAX),
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)
            )
        )
    )


class KeyRecRepContent(univ.Sequence):
    """
    KeyRecRepContent ::= SEQUENCE {
         status                  PKIStatusInfo,
         newSigCert          [0] CMPCertificate OPTIONAL,
         caCerts             [1] SEQUENCE SIZE (1..MAX) OF
                                             CMPCertificate OPTIONAL,
         keyPairHist         [2] SEQUENCE SIZE (1..MAX) OF
                                             CertifiedKeyPair OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('status', PKIStatusInfo()),
        namedtype.OptionalNamedType(
            'newSigCert', CMPCertificate().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)
            )
        ),
        namedtype.OptionalNamedType(
            'caCerts', univ.SequenceOf(componentType=CMPCertificate()).subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1),
                sizeSpec=constraint.ValueSizeConstraint(1, MAX)
            )
        ),
        namedtype.OptionalNamedType('keyPairHist', univ.SequenceOf(componentType=CertifiedKeyPair()).subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2),
            sizeSpec=constraint.ValueSizeConstraint(1, MAX))
        )
    )


class CertResponse(univ.Sequence):
    """
    CertResponse ::= SEQUENCE {
         certReqId           INTEGER,
         status              PKIStatusInfo,
         certifiedKeyPair    CertifiedKeyPair    OPTIONAL,
         rspInfo             OCTET STRING        OPTIONAL
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('certReqId', univ.Integer()),
        namedtype.NamedType('status', PKIStatusInfo()),
        namedtype.OptionalNamedType('certifiedKeyPair', CertifiedKeyPair()),
        namedtype.OptionalNamedType('rspInfo', univ.OctetString())
    )


class CertRepMessage(univ.Sequence):
    """
    CertRepMessage ::= SEQUENCE {
         caPubs       [1] SEQUENCE SIZE (1..MAX) OF CMPCertificate
                          OPTIONAL,
         response         SEQUENCE OF CertResponse
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType(
            'caPubs', univ.SequenceOf(
                componentType=CMPCertificate()
            ).subtype(sizeSpec=constraint.ValueSizeConstraint(1, MAX),
                      explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))
        ),
        namedtype.NamedType('response', univ.SequenceOf(componentType=CertResponse()))
    )


class POPODecKeyChallContent(univ.SequenceOf):
    componentType = Challenge()


class OOBCertHash(univ.Sequence):
    """
    OOBCertHash ::= SEQUENCE {
         hashAlg     [0] AlgorithmIdentifier     OPTIONAL,
         certId      [1] CertId                  OPTIONAL,
         hashVal         BIT STRING
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.OptionalNamedType(
            'hashAlg', rfc2459.AlgorithmIdentifier().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))
        ),
        namedtype.OptionalNamedType(
            'certId', rfc2511.CertId().subtype(explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))
        ),
        namedtype.NamedType('hashVal', univ.BitString())
    )


# pyasn1 does not naturally handle recursive definitions, thus this hack:
# NestedMessageContent ::= PKIMessages
class NestedMessageContent(univ.SequenceOf):
    """
    NestedMessageContent ::= PKIMessages
    """
    componentType = univ.Any()


class DHBMParameter(univ.Sequence):
    """
    DHBMParameter ::= SEQUENCE {
         owf                 AlgorithmIdentifier,
         -- AlgId for a One-Way Function (SHA-1 recommended)
         mac                 AlgorithmIdentifier
         -- the MAC AlgId (e.g., DES-MAC, Triple-DES-MAC [PKCS11],
     }   -- or HMAC [RFC2104, RFC2202])
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('owf', rfc2459.AlgorithmIdentifier()),
        namedtype.NamedType('mac', rfc2459.AlgorithmIdentifier())
    )


id_DHBasedMac = univ.ObjectIdentifier('1.2.840.113533.7.66.30')


class PBMParameter(univ.Sequence):
    """
    PBMParameter ::= SEQUENCE {
         salt                OCTET STRING,
         owf                 AlgorithmIdentifier,
         iterationCount      INTEGER,
         mac                 AlgorithmIdentifier
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType(
            'salt', univ.OctetString().subtype(subtypeSpec=constraint.ValueSizeConstraint(0, 128))
        ),
        namedtype.NamedType('owf', rfc2459.AlgorithmIdentifier()),
        namedtype.NamedType('iterationCount', univ.Integer()),
        namedtype.NamedType('mac', rfc2459.AlgorithmIdentifier())
    )


id_PasswordBasedMac = univ.ObjectIdentifier('1.2.840.113533.7.66.13')


class PKIProtection(univ.BitString):
    pass


# pyasn1 does not naturally handle recursive definitions, thus this hack:
# NestedMessageContent ::= PKIMessages
nestedMessageContent = NestedMessageContent().subtype(
    explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 20))


class PKIBody(univ.Choice):
    """
    PKIBody ::= CHOICE {       -- message-specific body elements
         ir       [0]  CertReqMessages,        --Initialization Request
         ip       [1]  CertRepMessage,         --Initialization Response
         cr       [2]  CertReqMessages,        --Certification Request
         cp       [3]  CertRepMessage,         --Certification Response
         p10cr    [4]  CertificationRequest,   --imported from [PKCS10]
         popdecc  [5]  POPODecKeyChallContent, --pop Challenge
         popdecr  [6]  POPODecKeyRespContent,  --pop Response
         kur      [7]  CertReqMessages,        --Key Update Request
         kup      [8]  CertRepMessage,         --Key Update Response
         krr      [9]  CertReqMessages,        --Key Recovery Request
         krp      [10] KeyRecRepContent,       --Key Recovery Response
         rr       [11] RevReqContent,          --Revocation Request
         rp       [12] RevRepContent,          --Revocation Response
         ccr      [13] CertReqMessages,        --Cross-Cert. Request
         ccp      [14] CertRepMessage,         --Cross-Cert. Response
         ckuann   [15] CAKeyUpdAnnContent,     --CA Key Update Ann.
         cann     [16] CertAnnContent,         --Certificate Ann.
         rann     [17] RevAnnContent,          --Revocation Ann.
         crlann   [18] CRLAnnContent,          --CRL Announcement
         pkiconf  [19] PKIConfirmContent,      --Confirmation
         nested   [20] NestedMessageContent,   --Nested Message
         genm     [21] GenMsgContent,          --General Message
         genp     [22] GenRepContent,          --General Response
         error    [23] ErrorMsgContent,        --Error Message
         certConf [24] CertConfirmContent,     --Certificate confirm
         pollReq  [25] PollReqContent,         --Polling request
         pollRep  [26] PollRepContent          --Polling response

    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType(
            'ir', rfc2511.CertReqMessages().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)
            )
        ),
        namedtype.NamedType(
            'ip', CertRepMessage().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)
            )
        ),
        namedtype.NamedType(
            'cr', rfc2511.CertReqMessages().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)
            )
        ),
        namedtype.NamedType(
            'cp', CertRepMessage().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3)
            )
        ),
        namedtype.NamedType(
            'p10cr', rfc2314.CertificationRequest().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4)
            )
        ),
        namedtype.NamedType(
            'popdecc', POPODecKeyChallContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 5)
            )
        ),
        namedtype.NamedType(
            'popdecr', POPODecKeyRespContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 6)
            )
        ),
        namedtype.NamedType(
            'kur', rfc2511.CertReqMessages().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 7)
            )
        ),
        namedtype.NamedType(
            'kup', CertRepMessage().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 8)
            )
        ),
        namedtype.NamedType(
            'krr', rfc2511.CertReqMessages().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 9)
            )
        ),
        namedtype.NamedType(
            'krp', KeyRecRepContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 10)
            )
        ),
        namedtype.NamedType(
            'rr', RevReqContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 11)
            )
        ),
        namedtype.NamedType(
            'rp', RevRepContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 12)
            )
        ),
        namedtype.NamedType(
            'ccr', rfc2511.CertReqMessages().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 13)
            )
        ),
        namedtype.NamedType(
            'ccp', CertRepMessage().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 14)
            )
        ),
        namedtype.NamedType(
            'ckuann', CAKeyUpdAnnContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 15)
            )
        ),
        namedtype.NamedType(
            'cann', CertAnnContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 16)
            )
        ),
        namedtype.NamedType(
            'rann', RevAnnContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 17)
            )
        ),
        namedtype.NamedType(
            'crlann', CRLAnnContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 18)
            )
        ),
        namedtype.NamedType(
            'pkiconf', PKIConfirmContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 19)
            )
        ),
        namedtype.NamedType(
            'nested', nestedMessageContent
        ),
        #        namedtype.NamedType('nested', NestedMessageContent().subtype(
        #            explicitTag=tag.Tag(tag.tagClassContext,tag.tagFormatConstructed,20)
        #            )
        #        ),
        namedtype.NamedType(
            'genm', GenMsgContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 21)
            )
        ),
        namedtype.NamedType(
            'gen', GenRepContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 22)
            )
        ),
        namedtype.NamedType(
            'error', ErrorMsgContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 23)
            )
        ),
        namedtype.NamedType(
            'certConf', CertConfirmContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 24)
            )
        ),
        namedtype.NamedType(
            'pollReq', PollReqContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 25)
            )
        ),
        namedtype.NamedType(
            'pollRep', PollRepContent().subtype(
                explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 26)
            )
        )
    )


class PKIHeader(univ.Sequence):
    """
    PKIHeader ::= SEQUENCE {
    pvno                INTEGER     { cmp1999(1), cmp2000(2) },
    sender              GeneralName,
    recipient           GeneralName,
    messageTime     [0] GeneralizedTime         OPTIONAL,
    protectionAlg   [1] AlgorithmIdentifier     OPTIONAL,
    senderKID       [2] KeyIdentifier           OPTIONAL,
    recipKID        [3] KeyIdentifier           OPTIONAL,
    transactionID   [4] OCTET STRING            OPTIONAL,
    senderNonce     [5] OCTET STRING            OPTIONAL,
    recipNonce      [6] OCTET STRING            OPTIONAL,
    freeText        [7] PKIFreeText             OPTIONAL,
    generalInfo     [8] SEQUENCE SIZE (1..MAX) OF
                     InfoTypeAndValue     OPTIONAL
    }

    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType(
            'pvno', univ.Integer(
                namedValues=namedval.NamedValues(('cmp1999', 1), ('cmp2000', 2))
            )
        ),
        namedtype.NamedType('sender', rfc2459.GeneralName()),
        namedtype.NamedType('recipient', rfc2459.GeneralName()),
        namedtype.OptionalNamedType('messageTime', useful.GeneralizedTime().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('protectionAlg', rfc2459.AlgorithmIdentifier().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
        namedtype.OptionalNamedType('senderKID', rfc2459.KeyIdentifier().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
        namedtype.OptionalNamedType('recipKID', rfc2459.KeyIdentifier().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
        namedtype.OptionalNamedType('transactionID', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))),
        namedtype.OptionalNamedType('senderNonce', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 5))),
        namedtype.OptionalNamedType('recipNonce', univ.OctetString().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 6))),
        namedtype.OptionalNamedType('freeText', PKIFreeText().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 7))),
        namedtype.OptionalNamedType('generalInfo',
                                    univ.SequenceOf(
                                        componentType=InfoTypeAndValue().subtype(
                                            sizeSpec=constraint.ValueSizeConstraint(1, MAX)
                                        )
                                    ).subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 8))
        )
    )


class ProtectedPart(univ.Sequence):
    """
     ProtectedPart ::= SEQUENCE {
         header    PKIHeader,
         body      PKIBody
     }
    """
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('header', PKIHeader()),
        namedtype.NamedType('infoValue', PKIBody())
    )


class PKIMessage(univ.Sequence):
    """
    PKIMessage ::= SEQUENCE {
    header           PKIHeader,
    body             PKIBody,
    protection   [0] PKIProtection OPTIONAL,
    extraCerts   [1] SEQUENCE SIZE (1..MAX) OF CMPCertificate
                  OPTIONAL
     }"""
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('header', PKIHeader()),
        namedtype.NamedType('body', PKIBody()),
        namedtype.OptionalNamedType('protection', PKIProtection().subtype(
            explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.OptionalNamedType('extraCerts',
                                    univ.SequenceOf(
                                        componentType=CMPCertificate()
                                    ).subtype(
                                        sizeSpec=constraint.ValueSizeConstraint(1, MAX),
                                        explicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)
                                    )
                                    )
    )


class PKIMessages(univ.SequenceOf):
    """
    PKIMessages ::= SEQUENCE SIZE (1..MAX) OF PKIMessage
    """
    componentType = PKIMessage()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(1, MAX)


# pyasn1 does not naturally handle recursive definitions, thus this hack:
# NestedMessageContent ::= PKIMessages
NestedMessageContent._componentType = PKIMessages()
nestedMessageContent._componentType = PKIMessages()
