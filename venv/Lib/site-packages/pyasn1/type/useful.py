#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
import datetime

from pyasn1 import error
from pyasn1.compat import dateandtime
from pyasn1.compat import string
from pyasn1.type import char
from pyasn1.type import tag
from pyasn1.type import univ

__all__ = ['ObjectDescriptor', 'GeneralizedTime', 'UTCTime']

NoValue = univ.NoValue
noValue = univ.noValue


class ObjectDescriptor(char.GraphicString):
    __doc__ = char.GraphicString.__doc__

    #: Default :py:class:`~pyasn1.type.tag.TagSet` object for |ASN.1| objects
    tagSet = char.GraphicString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 7)
    )

    # Optimization for faster codec lookup
    typeId = char.GraphicString.getTypeId()


class TimeMixIn(object):

    _yearsDigits = 4
    _hasSubsecond = False
    _optionalMinutes = False
    _shortTZ = False

    class FixedOffset(datetime.tzinfo):
        """Fixed offset in minutes east from UTC."""

        # defaulted arguments required
        # https: // docs.python.org / 2.3 / lib / datetime - tzinfo.html
        def __init__(self, offset=0, name='UTC'):
            self.__offset = datetime.timedelta(minutes=offset)
            self.__name = name

        def utcoffset(self, dt):
            return self.__offset

        def tzname(self, dt):
            return self.__name

        def dst(self, dt):
            return datetime.timedelta(0)

    UTC = FixedOffset()

    @property
    def asDateTime(self):
        """Create :py:class:`datetime.datetime` object from a |ASN.1| object.

        Returns
        -------
        :
            new instance of :py:class:`datetime.datetime` object
        """
        text = str(self)
        if text.endswith('Z'):
            tzinfo = TimeMixIn.UTC
            text = text[:-1]

        elif '-' in text or '+' in text:
            if '+' in text:
                text, plusminus, tz = string.partition(text, '+')
            else:
                text, plusminus, tz = string.partition(text, '-')

            if self._shortTZ and len(tz) == 2:
                tz += '00'

            if len(tz) != 4:
                raise error.PyAsn1Error('malformed time zone offset %s' % tz)

            try:
                minutes = int(tz[:2]) * 60 + int(tz[2:])
                if plusminus == '-':
                    minutes *= -1

            except ValueError:
                raise error.PyAsn1Error('unknown time specification %s' % self)

            tzinfo = TimeMixIn.FixedOffset(minutes, '?')

        else:
            tzinfo = None

        if '.' in text or ',' in text:
            if '.' in text:
                text, _, ms = string.partition(text, '.')
            else:
                text, _, ms = string.partition(text, ',')

            try:
                ms = int(ms) * 1000

            except ValueError:
                raise error.PyAsn1Error('bad sub-second time specification %s' % self)

        else:
            ms = 0

        if self._optionalMinutes and len(text) - self._yearsDigits == 6:
            text += '0000'
        elif len(text) - self._yearsDigits == 8:
            text += '00'

        try:
            dt = dateandtime.strptime(text, self._yearsDigits == 4 and '%Y%m%d%H%M%S' or '%y%m%d%H%M%S')

        except ValueError:
            raise error.PyAsn1Error('malformed datetime format %s' % self)

        return dt.replace(microsecond=ms, tzinfo=tzinfo)

    @classmethod
    def fromDateTime(cls, dt):
        """Create |ASN.1| object from a :py:class:`datetime.datetime` object.

        Parameters
        ----------
        dt: :py:class:`datetime.datetime` object
            The `datetime.datetime` object to initialize the |ASN.1| object
            from

        Returns
        -------
        :
            new instance of |ASN.1| value
        """
        text = dt.strftime(cls._yearsDigits == 4 and '%Y%m%d%H%M%S' or '%y%m%d%H%M%S')
        if cls._hasSubsecond:
            text += '.%d' % (dt.microsecond // 1000)

        if dt.utcoffset():
            seconds = dt.utcoffset().seconds
            if seconds < 0:
                text += '-'
            else:
                text += '+'
            text += '%.2d%.2d' % (seconds // 3600, seconds % 3600)
        else:
            text += 'Z'

        return cls(text)


class GeneralizedTime(char.VisibleString, TimeMixIn):
    __doc__ = char.VisibleString.__doc__

    #: Default :py:class:`~pyasn1.type.tag.TagSet` object for |ASN.1| objects
    tagSet = char.VisibleString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 24)
    )

    # Optimization for faster codec lookup
    typeId = char.VideotexString.getTypeId()

    _yearsDigits = 4
    _hasSubsecond = True
    _optionalMinutes = True
    _shortTZ = True


class UTCTime(char.VisibleString, TimeMixIn):
    __doc__ = char.VisibleString.__doc__

    #: Default :py:class:`~pyasn1.type.tag.TagSet` object for |ASN.1| objects
    tagSet = char.VisibleString.tagSet.tagImplicitly(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 23)
    )

    # Optimization for faster codec lookup
    typeId = char.VideotexString.getTypeId()

    _yearsDigits = 2
    _hasSubsecond = False
    _optionalMinutes = False
    _shortTZ = False
