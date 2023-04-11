#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
import sys

from pyasn1 import error
from pyasn1.type import tag
from pyasn1.type import univ

__all__ = ['NumericString', 'PrintableString', 'TeletexString', 'T61String', 'VideotexString',
           'IA5String', 'GraphicString', 'VisibleString', 'ISO646String',
           'GeneralString', 'UniversalString', 'BMPString', 'UTF8String']

NoValue = univ.NoValue
noValue = univ.noValue


class AbstractCharacterString(univ.OctetString):
    """Creates |ASN.1| schema or value object.

    |ASN.1| class is based on :class:`~pyasn1.type.base.SimpleAsn1Type`,
    its objects are immutable and duck-type Python 2 :class:`str` or Python 3
    :class:`bytes`. When used in octet-stream context, |ASN.1| type assumes
    "|encoding|" encoding.

    Keyword Args
    ------------
    value: :class:`unicode`, :class:`str`, :class:`bytes` or |ASN.1| object
        :class:`unicode` object (Python 2) or :class:`str` (Python 3),
        alternatively :class:`str` (Python 2) or :class:`bytes` (Python 3)
        representing octet-stream of serialised unicode string
        (note `encoding` parameter) or |ASN.1| class instance.
        If `value` is not given, schema object will be created.

    tagSet: :py:class:`~pyasn1.type.tag.TagSet`
        Object representing non-default ASN.1 tag(s)

    subtypeSpec: :py:class:`~pyasn1.type.constraint.ConstraintsIntersection`
        Object representing non-default ASN.1 subtype constraint(s). Constraints
        verification for |ASN.1| type occurs automatically on object
        instantiation.

    encoding: :py:class:`str`
        Unicode codec ID to encode/decode :class:`unicode` (Python 2) or
        :class:`str` (Python 3) the payload when |ASN.1| object is used
        in octet-stream context.

    Raises
    ------
    ~pyasn1.error.ValueConstraintError, ~pyasn1.error.PyAsn1Error
        On constraint violation or bad initializer.
    """

    if sys.version_info[0] <= 2:
        def __str__(self):
            try:
                # `str` is Py2 text representation
                return self._value.encode(self.encoding)

            except UnicodeEncodeError:
                exc = sys.exc_info()[1]
                raise error.PyAsn1UnicodeEncodeError(
                    "Can't encode string '%s' with codec "
                    "%s" % (self._value, self.encoding), exc
                )

        def __unicode__(self):
            return unicode(self._value)

        def prettyIn(self, value):
            try:
                if isinstance(value, unicode):
                    return value
                elif isinstance(value, str):
                    return value.decode(self.encoding)
                elif isinstance(value, (tuple, list)):
                    return self.prettyIn(''.join([chr(x) for x in value]))
                elif isinstance(value, univ.OctetString):
                    return value.asOctets().decode(self.encoding)
                else:
                    return unicode(value)

            except (UnicodeDecodeError, LookupError):
                exc = sys.exc_info()[1]
                raise error.PyAsn1UnicodeDecodeError(
                    "Can't decode string '%s' with codec "
                    "%s" % (value, self.encoding), exc
                )

        def asOctets(self, padding=True):
            return str(self)

        def asNumbers(self, padding=True):
            return tuple([ord(x) for x in str(self)])

    else:
        def __str__(self):
            # `unicode` is Py3 text representation
            return str(self._value)

        def __bytes__(self):
            try:
                return self._value.encode(self.encoding)
            except UnicodeEncodeError:
                exc = sys.exc_info()[1]
                raise error.PyAsn1UnicodeEncodeError(
                    "Can't encode string '%s' with codec "
                    "%s" % (self._value, self.encoding), exc
                )

        def prettyIn(self, value):
            try:
                if isinstance(value, str):
                    return value
                elif isinstance(value, bytes):
                    return value.decode(self.encoding)
                elif isinstance(value, (tuple, list)):
                    return self.prettyIn(bytes(value))
                elif isinstance(value, univ.OctetString):
                    return value.asOctets().decode(self.encoding)
                else:
                    return str(value)

            except (UnicodeDecodeError, LookupError):
                exc = sys.exc_info()[1]
                raise error.PyAsn1UnicodeDecodeError(
                    "Can't decode string '%s' with codec "
                    "%s" % (value, self.encoding), exc
                )

        def asOctets(self, padding=True):
            return bytes(self)

        def asNumbers(self, padding=True):
            return tuple(bytes(self))

    #
    # See OctetString.prettyPrint() for the explanation
    #

    def prettyOut(self, value):
        return value

    def prettyPrint(self, scope=0):
        # first see if subclass has its own .prettyOut()
        value = self.prettyOut(self._value)

        if value is not self._value:
            return value

        return AbstractCharacterString.__str__(self)

    def __reversed__(self):
        return reversed(self._value)


class NumericString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 18)
    )
    encoding = 'us-ascii'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class PrintableString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 19)
    )
    encoding = 'us-ascii'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class TeletexString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 20)
    )
    encoding = 'iso-8859-1'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class T61String(TeletexString):
    __doc__ = TeletexString.__doc__

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class VideotexString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 21)
    )
    encoding = 'iso-8859-1'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class IA5String(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 22)
    )
    encoding = 'us-ascii'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class GraphicString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 25)
    )
    encoding = 'iso-8859-1'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class VisibleString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 26)
    )
    encoding = 'us-ascii'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class ISO646String(VisibleString):
    __doc__ = VisibleString.__doc__

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()

class GeneralString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 27)
    )
    encoding = 'iso-8859-1'

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class UniversalString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 28)
    )
    encoding = "utf-32-be"

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class BMPString(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 30)
    )
    encoding = "utf-16-be"

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()


class UTF8String(AbstractCharacterString):
    __doc__ = AbstractCharacterString.__doc__

    #: Set (on class, not on instance) or return a
    #: :py:class:`~pyasn1.type.tag.TagSet` object representing ASN.1 tag(s)
    #: associated with |ASN.1| type.
    tagSet = AbstractCharacterString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 12)
    )
    encoding = "utf-8"

    # Optimization for faster codec lookup
    typeId = AbstractCharacterString.getTypeId()
