#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from pyasn1.type import base
from pyasn1.type import tag

__all__ = ['endOfOctets']


class EndOfOctets(base.SimpleAsn1Type):
    defaultValue = 0
    tagSet = tag.initTagSet(
        tag.Tag(tag.tagClassUniversal, tag.tagFormatSimple, 0x00)
    )

    _instance = None

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = object.__new__(cls, *args, **kwargs)

        return cls._instance


endOfOctets = EndOfOctets()
