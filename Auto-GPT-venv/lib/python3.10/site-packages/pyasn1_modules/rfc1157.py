#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# SNMPv1 message syntax
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc1157.txt
#
# Sample captures from:
# http://wiki.wireshark.org/SampleCaptures/
#
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc1155


class Version(univ.Integer):
    namedValues = namedval.NamedValues(
        ('version-1', 0)
    )
    defaultValue = 0


class Community(univ.OctetString):
    pass


class RequestID(univ.Integer):
    pass


class ErrorStatus(univ.Integer):
    namedValues = namedval.NamedValues(
        ('noError', 0),
        ('tooBig', 1),
        ('noSuchName', 2),
        ('badValue', 3),
        ('readOnly', 4),
        ('genErr', 5)
    )


class ErrorIndex(univ.Integer):
    pass


class VarBind(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('name', rfc1155.ObjectName()),
        namedtype.NamedType('value', rfc1155.ObjectSyntax())
    )


class VarBindList(univ.SequenceOf):
    componentType = VarBind()


class _RequestBase(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('request-id', RequestID()),
        namedtype.NamedType('error-status', ErrorStatus()),
        namedtype.NamedType('error-index', ErrorIndex()),
        namedtype.NamedType('variable-bindings', VarBindList())
    )


class GetRequestPDU(_RequestBase):
    tagSet = _RequestBase.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)
    )


class GetNextRequestPDU(_RequestBase):
    tagSet = _RequestBase.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)
    )


class GetResponsePDU(_RequestBase):
    tagSet = _RequestBase.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)
    )


class SetRequestPDU(_RequestBase):
    tagSet = _RequestBase.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3)
    )


class TrapPDU(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('enterprise', univ.ObjectIdentifier()),
        namedtype.NamedType('agent-addr', rfc1155.NetworkAddress()),
        namedtype.NamedType('generic-trap', univ.Integer().clone(
            namedValues=namedval.NamedValues(('coldStart', 0), ('warmStart', 1), ('linkDown', 2), ('linkUp', 3),
                                             ('authenticationFailure', 4), ('egpNeighborLoss', 5),
                                             ('enterpriseSpecific', 6)))),
        namedtype.NamedType('specific-trap', univ.Integer()),
        namedtype.NamedType('time-stamp', rfc1155.TimeTicks()),
        namedtype.NamedType('variable-bindings', VarBindList())
    )


class Pdus(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('get-request', GetRequestPDU()),
        namedtype.NamedType('get-next-request', GetNextRequestPDU()),
        namedtype.NamedType('get-response', GetResponsePDU()),
        namedtype.NamedType('set-request', SetRequestPDU()),
        namedtype.NamedType('trap', TrapPDU())
    )


class Message(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('version', Version()),
        namedtype.NamedType('community', Community()),
        namedtype.NamedType('data', Pdus())
    )
