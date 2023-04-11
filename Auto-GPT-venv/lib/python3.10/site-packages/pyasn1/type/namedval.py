#
# This file is part of pyasn1 software.
#
# Copyright (c) 2005-2019, Ilya Etingof <etingof@gmail.com>
# License: http://snmplabs.com/pyasn1/license.html
#
# ASN.1 named integers
#
from pyasn1 import error

__all__ = ['NamedValues']


class NamedValues(object):
    """Create named values object.

    The |NamedValues| object represents a collection of string names
    associated with numeric IDs. These objects are used for giving
    names to otherwise numerical values.

    |NamedValues| objects are immutable and duck-type Python
    :class:`dict` object mapping ID to name and vice-versa.

    Parameters
    ----------
    *args: variable number of two-element :py:class:`tuple`

        name: :py:class:`str`
            Value label

        value: :py:class:`int`
            Numeric value

    Keyword Args
    ------------
    name: :py:class:`str`
        Value label

    value: :py:class:`int`
        Numeric value

    Examples
    --------

    .. code-block:: pycon

        >>> nv = NamedValues('a', 'b', ('c', 0), d=1)
        >>> nv
        >>> {'c': 0, 'd': 1, 'a': 2, 'b': 3}
        >>> nv[0]
        'c'
        >>> nv['a']
        2
    """
    def __init__(self, *args, **kwargs):
        self.__names = {}
        self.__numbers = {}

        anonymousNames = []

        for namedValue in args:
            if isinstance(namedValue, (tuple, list)):
                try:
                    name, number = namedValue

                except ValueError:
                    raise error.PyAsn1Error('Not a proper attribute-value pair %r' % (namedValue,))

            else:
                anonymousNames.append(namedValue)
                continue

            if name in self.__names:
                raise error.PyAsn1Error('Duplicate name %s' % (name,))

            if number in self.__numbers:
                raise error.PyAsn1Error('Duplicate number  %s=%s' % (name, number))

            self.__names[name] = number
            self.__numbers[number] = name

        for name, number in kwargs.items():
            if name in self.__names:
                raise error.PyAsn1Error('Duplicate name %s' % (name,))

            if number in self.__numbers:
                raise error.PyAsn1Error('Duplicate number  %s=%s' % (name, number))

            self.__names[name] = number
            self.__numbers[number] = name

        if anonymousNames:

            number = self.__numbers and max(self.__numbers) + 1 or 0

            for name in anonymousNames:

                if name in self.__names:
                    raise error.PyAsn1Error('Duplicate name %s' % (name,))

                self.__names[name] = number
                self.__numbers[number] = name

                number += 1

    def __repr__(self):
        representation = ', '.join(['%s=%d' % x for x in self.items()])

        if len(representation) > 64:
            representation = representation[:32] + '...' + representation[-32:]

        return '<%s object, enums %s>' % (
            self.__class__.__name__, representation)

    def __eq__(self, other):
        return dict(self) == other

    def __ne__(self, other):
        return dict(self) != other

    def __lt__(self, other):
        return dict(self) < other

    def __le__(self, other):
        return dict(self) <= other

    def __gt__(self, other):
        return dict(self) > other

    def __ge__(self, other):
        return dict(self) >= other

    def __hash__(self):
        return hash(self.items())

    # Python dict protocol (read-only)

    def __getitem__(self, key):
        try:
            return self.__numbers[key]

        except KeyError:
            return self.__names[key]

    def __len__(self):
        return len(self.__names)

    def __contains__(self, key):
        return key in self.__names or key in self.__numbers

    def __iter__(self):
        return iter(self.__names)

    def values(self):
        return iter(self.__numbers)

    def keys(self):
        return iter(self.__names)

    def items(self):
        for name in self.__names:
            yield name, self.__names[name]

    # support merging

    def __add__(self, namedValues):
        return self.__class__(*tuple(self.items()) + tuple(namedValues.items()))

    # XXX clone/subtype?

    def clone(self, *args, **kwargs):
        new = self.__class__(*args, **kwargs)
        return self + new

    # legacy protocol

    def getName(self, value):
        if value in self.__numbers:
            return self.__numbers[value]

    def getValue(self, name):
        if name in self.__names:
            return self.__names[name]

    def getValues(self, *names):
        try:
            return [self.__names[name] for name in names]

        except KeyError:
            raise error.PyAsn1Error(
                'Unknown bit identifier(s): %s' % (set(names).difference(self.__names),)
            )
