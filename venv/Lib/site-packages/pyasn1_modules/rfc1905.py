#
# This file is part of pyasn1-modules software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# SNMPv2c PDU syntax
#
# ASN.1 source from:
# http://www.ietf.org/rfc/rfc1905.txt
#
from pyasn1.type import constraint
from pyasn1.type import namedtype
from pyasn1.type import namedval
from pyasn1.type import tag
from pyasn1.type import univ

from pyasn1_modules import rfc1902

max_bindings = rfc1902.Integer(2147483647)


class _BindValue(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('value', rfc1902.ObjectSyntax()),
        namedtype.NamedType('unSpecified', univ.Null()),
        namedtype.NamedType('noSuchObject',
                            univ.Null().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 0))),
        namedtype.NamedType('noSuchInstance',
                            univ.Null().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 1))),
        namedtype.NamedType('endOfMibView',
                            univ.Null().subtype(implicitTag=tag.Tag(tag.tagClassContext, tag.tagFormatSimple, 2)))
    )


class VarBind(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('name', rfc1902.ObjectName()),
        namedtype.NamedType('', _BindValue())
    )


class VarBindList(univ.SequenceOf):
    componentType = VarBind()
    sizeSpec = univ.SequenceOf.sizeSpec + constraint.ValueSizeConstraint(
        0, max_bindings
    )


class PDU(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('request-id', rfc1902.Integer32()),
        namedtype.NamedType('error-status', univ.Integer(
            namedValues=namedval.NamedValues(('noError', 0), ('tooBig', 1), ('noSuchName', 2), ('badValue', 3),
                                             ('readOnly', 4), ('genErr', 5), ('noAccess', 6), ('wrongType', 7),
                                             ('wrongLength', 8), ('wrongEncoding', 9), ('wrongValue', 10),
                                             ('noCreation', 11), ('inconsistentValue', 12), ('resourceUnavailable', 13),
                                             ('commitFailed', 14), ('undoFailed', 15), ('authorizationError', 16),
                                             ('notWritable', 17), ('inconsistentName', 18)))),
        namedtype.NamedType('error-index',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, max_bindings))),
        namedtype.NamedType('variable-bindings', VarBindList())
    )


class BulkPDU(univ.Sequence):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('request-id', rfc1902.Integer32()),
        namedtype.NamedType('non-repeaters',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, max_bindings))),
        namedtype.NamedType('max-repetitions',
                            univ.Integer().subtype(subtypeSpec=constraint.ValueRangeConstraint(0, max_bindings))),
        namedtype.NamedType('variable-bindings', VarBindList())
    )


class GetRequestPDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 0)
    )


class GetNextRequestPDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 1)
    )


class ResponsePDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 2)
    )


class SetRequestPDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 3)
    )


class GetBulkRequestPDU(BulkPDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 5)
    )


class InformRequestPDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 6)
    )


class SNMPv2TrapPDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 7)
    )


class ReportPDU(PDU):
    tagSet = PDU.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassContext, tag.tagFormatConstructed, 8)
    )


class PDUs(univ.Choice):
    componentType = namedtype.NamedTypes(
        namedtype.NamedType('get-request', GetRequestPDU()),
        namedtype.NamedType('get-next-request', GetNextRequestPDU()),
        namedtype.NamedType('get-bulk-request', GetBulkRequestPDU()),
        namedtype.NamedType('response', ResponsePDU()),
        namedtype.NamedType('set-request', SetRequestPDU()),
        namedtype.NamedType('inform-request', InformRequestPDU()),
        namedtype.NamedType('snmpV2-trap', SNMPv2TrapPDU()),
        namedtype.NamedType('report', ReportPDU())
    )
