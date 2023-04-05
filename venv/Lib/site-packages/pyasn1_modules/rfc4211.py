# coding: utf-8
#
# This file is part of pyasn1-modules software.
#
# Created by Stanisław Pitucha with asn1ate tool.
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# Internet X.509 Public Key Infrastructure Certificate Request
# Message Format (CRMF)
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc4211.txt
#
from pyasn1.type import char
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc3280
from pyasn1_modules import rfc3852

MAX = float('inf')


def _buildOid(*components):
    output = []
    for x in tuple(components):
        if isinstance(x, univ.ObjectIdentifier):
            output.extend(list(x))
        else:
            output.append(int(x))

    return univ.ObjectIdentifier(output)


id_pkix = _buildOid(1, 3, 6, 1, 5, 5, 7)

id_pkip = _buildOid(id_pkix, 5)

id_regCtrl = _buildOid(id_pkip, 1)


class SinglePubInfo(univ.Sequence):
    pass


SinglePubInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('pubMethod', univ.Integer(
        namedValues=namedval.NamedValues(('dontCare', 0), ('x500', 1), ('web', 2), ('ldap', 3)))),
    namedtype.OptionalNamedType('pubLocation', rfc3280.GeneralName())
)


class UTF8Pairs(char.UTF8String):
    pass


class PKMACValue(univ.Sequence):
    pass


PKMACValue.componentType = namedtype.NamedTypes(
    namedtype.NamedType('algId', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('value', univ.BitString())
)


class POPOSigningKeyInput(univ.Sequence):
    pass


POPOSigningKeyInput.componentType = namedtype.NamedTypes(
    namedtype.NamedType(
        'authInfo', univ.Choice(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType(
                    'sender', rfc3280.GeneralName().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))
                ),
                namedtype.NamedType(
                    'publicKeyMAC', PKMACValue()
                )
            )
        )
    ),
    namedtype.NamedType('publicKey', rfc3280.SubjectPublicKeyInfo())
)


class POPOSigningKey(univ.Sequence):
    pass


POPOSigningKey.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('poposkInput', POPOSigningKeyInput().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('algorithmIdentifier', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('signature', univ.BitString())
)


class Attributes(univ.SetOf):
    pass


Attributes.componentType = rfc3280.Attribute()


class PrivateKeyInfo(univ.Sequence):
    pass


PrivateKeyInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('version', univ.Integer()),
    namedtype.NamedType('privateKeyAlgorithm', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('privateKey', univ.OctetString()),
    namedtype.OptionalNamedType('attributes',
                                Attributes().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class EncryptedValue(univ.Sequence):
    pass


EncryptedValue.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('intendedAlg', rfc3280.AlgorithmIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('symmAlg', rfc3280.AlgorithmIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('encSymmKey', univ.BitString().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('keyAlg', rfc3280.AlgorithmIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 3))),
    namedtype.OptionalNamedType('valueHint', univ.OctetString().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4))),
    namedtype.NamedType('encValue', univ.BitString())
)


class EncryptedKey(univ.Choice):
    pass


EncryptedKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('encryptedValue', EncryptedValue()),
    namedtype.NamedType('envelopedData', rfc3852.EnvelopedData().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0)))
)


class KeyGenParameters(univ.OctetString):
    pass


class PKIArchiveOptions(univ.Choice):
    pass


PKIArchiveOptions.componentType = namedtype.NamedTypes(
    namedtype.NamedType('encryptedPrivKey',
                        EncryptedKey().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.NamedType('keyGenParameters',
                        KeyGenParameters().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('archiveRemGenPrivKey',
                        univ.Boolean().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
)

id_regCtrl_authenticator = _buildOid(id_regCtrl, 2)

id_regInfo = _buildOid(id_pkip, 2)

id_regInfo_certReq = _buildOid(id_regInfo, 2)


class ProtocolEncrKey(rfc3280.SubjectPublicKeyInfo):
    pass


class Authenticator(char.UTF8String):
    pass


class SubsequentMessage(univ.Integer):
    pass


SubsequentMessage.namedValues = namedval.NamedValues(
    ('encrCert', 0),
    ('challengeResp', 1)
)


class AttributeTypeAndValue(univ.Sequence):
    pass


AttributeTypeAndValue.componentType = namedtype.NamedTypes(
    namedtype.NamedType('type', univ.ObjectIdentifier()),
    namedtype.NamedType('value', univ.Any())
)


class POPOPrivKey(univ.Choice):
    pass


POPOPrivKey.componentType = namedtype.NamedTypes(
    namedtype.NamedType('thisMessage',
                        univ.BitString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('subsequentMessage',
                        SubsequentMessage().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.NamedType('dhMAC',
                        univ.BitString().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.NamedType('agreeMAC',
                        PKMACValue().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))),
    namedtype.NamedType('encryptedKey', rfc3852.EnvelopedData().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 4)))
)


class ProofOfPossession(univ.Choice):
    pass


ProofOfPossession.componentType = namedtype.NamedTypes(
    namedtype.NamedType('raVerified',
                        univ.Null().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.NamedType('signature', POPOSigningKey().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1))),
    namedtype.NamedType('keyEncipherment',
                        POPOPrivKey().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2))),
    namedtype.NamedType('keyAgreement',
                        POPOPrivKey().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3)))
)


class OptionalValidity(univ.Sequence):
    pass


OptionalValidity.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('notBefore', rfc3280.Time().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0))),
    namedtype.OptionalNamedType('notAfter', rfc3280.Time().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)))
)


class CertTemplate(univ.Sequence):
    pass


CertTemplate.componentType = namedtype.NamedTypes(
    namedtype.OptionalNamedType('version', rfc3280.Version().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
    namedtype.OptionalNamedType('serialNumber', univ.Integer().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
    namedtype.OptionalNamedType('signingAlg', rfc3280.AlgorithmIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2))),
    namedtype.OptionalNamedType('issuer', rfc3280.Name().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3))),
    namedtype.OptionalNamedType('validity', OptionalValidity().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 4))),
    namedtype.OptionalNamedType('subject', rfc3280.Name().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 5))),
    namedtype.OptionalNamedType('publicKey', rfc3280.SubjectPublicKeyInfo().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 6))),
    namedtype.OptionalNamedType('issuerUID', rfc3280.UniqueIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 7))),
    namedtype.OptionalNamedType('subjectUID', rfc3280.UniqueIdentifier().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 8))),
    namedtype.OptionalNamedType('extensions', rfc3280.Extensions().subtype(
        implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 9)))
)


class Controls(univ.SequenceOf):
    pass


Controls.componentType = AttributeTypeAndValue()
Controls.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class CertRequest(univ.Sequence):
    pass


CertRequest.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certReqId', univ.Integer()),
    namedtype.NamedType('certTemplate', CertTemplate()),
    namedtype.OptionalNamedType('controls', Controls())
)


class CertReqMsg(univ.Sequence):
    pass


CertReqMsg.componentType = namedtype.NamedTypes(
    namedtype.NamedType('certReq', CertRequest()),
    namedtype.OptionalNamedType('popo', ProofOfPossession()),
    namedtype.OptionalNamedType('regInfo', univ.SequenceOf(componentType=AttributeTypeAndValue()))
)


class CertReqMessages(univ.SequenceOf):
    pass


CertReqMessages.componentType = CertReqMsg()
CertReqMessages.sizeSpec = constraint.ValueSizeConstraint(1, MAX)


class CertReq(CertRequest):
    pass


id_regCtrl_pkiPublicationInfo = _buildOid(id_regCtrl, 3)


class CertId(univ.Sequence):
    pass


CertId.componentType = namedtype.NamedTypes(
    namedtype.NamedType('issuer', rfc3280.GeneralName()),
    namedtype.NamedType('serialNumber', univ.Integer())
)


class OldCertId(CertId):
    pass


class PKIPublicationInfo(univ.Sequence):
    pass


PKIPublicationInfo.componentType = namedtype.NamedTypes(
    namedtype.NamedType('action',
                        univ.Integer(namedValues=namedval.NamedValues(('dontPublish', 0), ('pleasePublish', 1)))),
    namedtype.OptionalNamedType('pubInfos', univ.SequenceOf(componentType=SinglePubInfo()))
)


class EncKeyWithID(univ.Sequence):
    pass


EncKeyWithID.componentType = namedtype.NamedTypes(
    namedtype.NamedType('privateKey', PrivateKeyInfo()),
    namedtype.OptionalNamedType(
        'identifier', univ.Choice(
            componentType=namedtype.NamedTypes(
                namedtype.NamedType('string', char.UTF8String()),
                namedtype.NamedType('generalName', rfc3280.GeneralName())
            )
        )
    )
)

id_regCtrl_protocolEncrKey = _buildOid(id_regCtrl, 6)

id_regCtrl_oldCertID = _buildOid(id_regCtrl, 5)

id_smime = _buildOid(1, 2, 840, 113549, 1, 9, 16)


class PBMParameter(univ.Sequence):
    pass


PBMParameter.componentType = namedtype.NamedTypes(
    namedtype.NamedType('salt', univ.OctetString()),
    namedtype.NamedType('owf', rfc3280.AlgorithmIdentifier()),
    namedtype.NamedType('iterationCount', univ.Integer()),
    namedtype.NamedType('mac', rfc3280.AlgorithmIdentifier())
)

id_regCtrl_regToken = _buildOid(id_regCtrl, 1)

id_regCtrl_pkiArchiveOptions = _buildOid(id_regCtrl, 4)

id_regInfo_utf8Pairs = _buildOid(id_regInfo, 1)

id_ct = _buildOid(id_smime, 1)

id_ct_encKeyWithID = _buildOid(id_ct, 21)


class RegToken(char.UTF8String):
    pass
