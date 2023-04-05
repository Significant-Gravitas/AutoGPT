#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
from sys import version_info

__all__ = ['callable']


if (2, 7) < version_info[:2] < (3, 2):
    import collections

    def callable(x):
        return isinstance(x, collections.Callable)

else:

    callable = callable
