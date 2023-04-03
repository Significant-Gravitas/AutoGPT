"""Drop-in replacement for collections.AttributedDict by Raymond Hettinger

http://code.activestate.com/recipes/576693/

"""
try:
    from collections.abc import MutableMapping as DictMixin
    py_level = 3
except ImportError:
    # noinspection PyUnresolvedReferences
    from collections import UserDict
    DictMixin = UserDict.DictMixinn
    py_level = 2

# Modified from original to support Python 2.4, see
# http://code.google.com/p/dirtyjson/issues/detail?id=53
try:
    # noinspection PyStatementEffect
    all
except NameError:
    # noinspection PyShadowingBuiltins
    def all(seq):
        for elem in seq:
            if not elem:
                return False
        return True


class AttributedDict(dict, DictMixin):
    def __init__(self):
        super(AttributedDict, self).__init__()
        self.__attributes = {}
        self.clear()

    # noinspection PyAttributeOutsideInit
    def clear(self):
        self.__end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.__map = {}                 # key --> [key, prev, next]
        dict.clear(self)

    def add_with_attributes(self, key, value, attributes):
        self.__setitem__(key, value)
        self.__attributes[key] = attributes

    def attributes(self, key):
        return self.__attributes.get(key)

    def __setitem__(self, key, value):
        if key not in self:
            end = self.__end
            curr = end[1]
            curr[2] = end[1] = self.__map[key] = [key, curr, end]
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        key, prev, next_entry = self.__map.pop(key)
        prev[2] = next_entry
        next_entry[1] = prev

    def __iter__(self):
        end = self.__end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.__end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def popitem(self, last=True):
        if not self:
            raise KeyError('dictionary is empty')
        # Modified from original to support Python 2.4, see
        # http://code.google.com/p/dirtyjson/issues/detail?id=53
        if py_level == 2:
            if last:
                # noinspection PyUnresolvedReferences
                key = reversed(self).next()
            else:
                # noinspection PyUnresolvedReferences
                key = iter(self).next()
        else:
            if last:
                key = next(reversed(self))
            else:
                key = next(iter(self))
        value = self.pop(key)
        return key, value

    def __reduce__(self):
        items = [[k, self[k]] for k in self]
        tmp = self.__map, self.__end
        del self.__map, self.__end
        inst_dict = vars(self).copy()
        self.__map, self.__end = tmp
        if inst_dict:
            return self.__class__, (items,), inst_dict
        return self.__class__, (items,)

    def keys(self):
        return list(self)

    setdefault = DictMixin.setdefault
    update = DictMixin.update
    pop = DictMixin.pop
    values = DictMixin.values
    items = DictMixin.items

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%s)' % (self.__class__.__name__, list(self.items()))

    def copy(self):
        return self.__class__(self)

    # noinspection PyMethodOverriding
    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        if isinstance(other, AttributedDict):
            return len(self) == len(other) and all(
                p == q for p, q in zip(self.items(), other.items()))
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other


class AttributedList(list):
    def __init__(self):
        super(AttributedList, self).__init__()
        self.__attributes = []

    def append(self, value, attributes=None):
        super(AttributedList, self).append(value)
        self.__attributes.append(attributes)

    def attributes(self, index):
        return self.__attributes[index]
