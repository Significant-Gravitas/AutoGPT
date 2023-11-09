#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from pyasn1 import debug
from pyasn1 import error
from pyasn1.type import base
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ
from pyasn1.type import useful

__all__ = ['decode']

LOG = debug.registerLoggee(__name__, flags=debug.DEBUG_DECODER)


class AbstractScalarDecoder(object):
    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        return asn1Spec.clone(pyObject)


class BitStringDecoder(AbstractScalarDecoder):
    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        return asn1Spec.clone(univ.BitString.fromBinaryString(pyObject))


class SequenceOrSetDecoder(object):
    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        asn1Value = asn1Spec.clone()

        componentsTypes = asn1Spec.componentType

        for field in asn1Value:
            if field in pyObject:
                asn1Value[field] = decodeFun(pyObject[field], componentsTypes[field].asn1Object, **options)

        return asn1Value


class SequenceOfOrSetOfDecoder(object):
    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        asn1Value = asn1Spec.clone()

        for pyValue in pyObject:
            asn1Value.append(decodeFun(pyValue, asn1Spec.componentType), **options)

        return asn1Value


class ChoiceDecoder(object):
    def __call__(self, pyObject, asn1Spec, decodeFun=None, **options):
        asn1Value = asn1Spec.clone()

        componentsTypes = asn1Spec.componentType

        for field in pyObject:
            if field in componentsTypes:
                asn1Value[field] = decodeFun(pyObject[field], componentsTypes[field].asn1Object, **options)
                break

        return asn1Value


tagMap = {
    univ.Integer.tagSet: AbstractScalarDecoder(),
    univ.Boolean.tagSet: AbstractScalarDecoder(),
    univ.BitString.tagSet: BitStringDecoder(),
    univ.OctetString.tagSet: AbstractScalarDecoder(),
    univ.Null.tagSet: AbstractScalarDecoder(),
    univ.ObjectIdentifier.tagSet: AbstractScalarDecoder(),
    univ.Enumerated.tagSet: AbstractScalarDecoder(),
    univ.Real.tagSet: AbstractScalarDecoder(),
    univ.Sequence.tagSet: SequenceOrSetDecoder(),  # conflicts with SequenceOf
    univ.Set.tagSet: SequenceOrSetDecoder(),  # conflicts with SetOf
    univ.Choice.tagSet: ChoiceDecoder(),  # conflicts with Any
    # character string types
    char.UTF8String.tagSet: AbstractScalarDecoder(),
    char.NumericString.tagSet: AbstractScalarDecoder(),
    char.PrintableString.tagSet: AbstractScalarDecoder(),
    char.TeletexString.tagSet: AbstractScalarDecoder(),
    char.VideotexString.tagSet: AbstractScalarDecoder(),
    char.IA5String.tagSet: AbstractScalarDecoder(),
    char.GraphicString.tagSet: AbstractScalarDecoder(),
    char.VisibleString.tagSet: AbstractScalarDecoder(),
    char.GeneralString.tagSet: AbstractScalarDecoder(),
    char.UniversalString.tagSet: AbstractScalarDecoder(),
    char.BMPString.tagSet: AbstractScalarDecoder(),
    # useful types
    useful.ObjectDescriptor.tagSet: AbstractScalarDecoder(),
    useful.GeneralizedTime.tagSet: AbstractScalarDecoder(),
    useful.UTCTime.tagSet: AbstractScalarDecoder()
}

# Put in ambiguous & non-ambiguous types for faster codec lookup
typeMap = {
    univ.Integer.typeId: AbstractScalarDecoder(),
    univ.Boolean.typeId: AbstractScalarDecoder(),
    univ.BitString.typeId: BitStringDecoder(),
    univ.OctetString.typeId: AbstractScalarDecoder(),
    univ.Null.typeId: AbstractScalarDecoder(),
    univ.ObjectIdentifier.typeId: AbstractScalarDecoder(),
    univ.Enumerated.typeId: AbstractScalarDecoder(),
    univ.Real.typeId: AbstractScalarDecoder(),
    # ambiguous base types
    univ.Set.typeId: SequenceOrSetDecoder(),
    univ.SetOf.typeId: SequenceOfOrSetOfDecoder(),
    univ.Sequence.typeId: SequenceOrSetDecoder(),
    univ.SequenceOf.typeId: SequenceOfOrSetOfDecoder(),
    univ.Choice.typeId: ChoiceDecoder(),
    univ.Any.typeId: AbstractScalarDecoder(),
    # character string types
    char.UTF8String.typeId: AbstractScalarDecoder(),
    char.NumericString.typeId: AbstractScalarDecoder(),
    char.PrintableString.typeId: AbstractScalarDecoder(),
    char.TeletexString.typeId: AbstractScalarDecoder(),
    char.VideotexString.typeId: AbstractScalarDecoder(),
    char.IA5String.typeId: AbstractScalarDecoder(),
    char.GraphicString.typeId: AbstractScalarDecoder(),
    char.VisibleString.typeId: AbstractScalarDecoder(),
    char.GeneralString.typeId: AbstractScalarDecoder(),
    char.UniversalString.typeId: AbstractScalarDecoder(),
    char.BMPString.typeId: AbstractScalarDecoder(),
    # useful types
    useful.ObjectDescriptor.typeId: AbstractScalarDecoder(),
    useful.GeneralizedTime.typeId: AbstractScalarDecoder(),
    useful.UTCTime.typeId: AbstractScalarDecoder()
}


class Decoder(object):

    # noinspection PyDefaultArgument
    def __init__(self, tagMap, typeMap):
        self.__tagMap = tagMap
        self.__typeMap = typeMap

    def __call__(self, pyObject, asn1Spec, **options):

        if LOG:
            debug.scope.push(type(pyObject).__name__)
            LOG('decoder called at scope %s, working with type %s' % (debug.scope, type(pyObject).__name__))

        if asn1Spec is None or not isinstance(asn1Spec, base.Asn1Item):
            raise error.PyAsn1Error('asn1Spec is not valid (should be an instance of an ASN.1 Item, not %s)' % asn1Spec.__class__.__name__)

        try:
            valueDecoder = self.__typeMap[asn1Spec.typeId]

        except KeyError:
            # use base type for codec lookup to recover untagged types
            baseTagSet = tag.TagSet(asn1Spec.tagSet.baseTag, asn1Spec.tagSet.baseTag)

            try:
                valueDecoder = self.__tagMap[baseTagSet]
            except KeyError:
                raise error.PyAsn1Error('Unknown ASN.1 tag %s' % asn1Spec.tagSet)

        if LOG:
            LOG('calling decoder %s on Python type %s <%s>' % (type(valueDecoder).__name__, type(pyObject).__name__, repr(pyObject)))

        value = valueDecoder(pyObject, asn1Spec, self, **options)

        if LOG:
            LOG('decoder %s produced ASN.1 type %s <%s>' % (type(valueDecoder).__name__, type(value).__name__, repr(value)))
            debug.scope.pop()

        return value


#: Turns Python objects of built-in types into ASN.1 objects.
#:
#: Takes Python objects of built-in types and turns them into a tree of
#: ASN.1 objects (e.g. :py:class:`~pyasn1.type.base.PyAsn1Item` derivative) which
#: may be a scalar or an arbitrary nested structure.
#:
#: Parameters
#: ----------
#: pyObject: :py:class:`object`
#:     A scalar or nested Python objects
#:
#: Keyword Args
#: ------------
#: asn1Spec: any pyasn1 type object e.g. :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
#:     A pyasn1 type object to act as a template guiding the decoder. It is required
#:     for successful interpretation of Python objects mapping into their ASN.1
#:     representations.
#:
#: Returns
#: -------
#: : :py:class:`~pyasn1.type.base.PyAsn1Item` derivative
#:     A scalar or constructed pyasn1 object
#:
#: Raises
#: ------
#: ~pyasn1.error.PyAsn1Error
#:     On decoding errors
#:
#: Examples
#: --------
#: Decode native Python object into ASN.1 objects with ASN.1 schema
#:
#: .. code-block:: pycon
#:
#:    >>> seq = SequenceOf(componentType=Integer())
#:    >>> s, _ = decode([1, 2, 3], asn1Spec=seq)
#:    >>> str(s)
#:    SequenceOf:
#:     1 2 3
#:
decode = Decoder(tagMap, typeMap)
