import sys
import types
from collections.abc import MutableSequence


cdef class FrozenList:

    if sys.version_info >= (3, 9):
        __class_getitem__ = classmethod(types.GenericAlias)
    else:
        @classmethod
        def __class_getitem__(cls):
            return cls

    cdef readonly bint frozen
    cdef list _items

    def __init__(self, items=None):
        self.frozen = False
        if items is not None:
            items = list(items)
        else:
            items = []
        self._items = items

    cdef object _check_frozen(self):
        if self.frozen:
            raise RuntimeError("Cannot modify frozen list.")

    cdef inline object _fast_len(self):
        return len(self._items)

    def freeze(self):
        self.frozen = True

    def __getitem__(self, index):
        return self._items[index]

    def __setitem__(self, index, value):
        self._check_frozen()
        self._items[index] = value

    def __delitem__(self, index):
        self._check_frozen()
        del self._items[index]

    def __len__(self):
        return self._fast_len()

    def __iter__(self):
        return self._items.__iter__()

    def __reversed__(self):
        return self._items.__reversed__()

    def __richcmp__(self, other, op):
        if op == 0:  # <
            return list(self) < other
        if op == 1:  # <=
            return list(self) <= other
        if op == 2:  # ==
            return list(self) == other
        if op == 3:  # !=
            return list(self) != other
        if op == 4:  # >
            return list(self) > other
        if op == 5:  # =>
            return list(self) >= other

    def insert(self, pos, item):
        self._check_frozen()
        self._items.insert(pos, item)

    def __contains__(self, item):
        return item in self._items

    def __iadd__(self, items):
        self._check_frozen()
        self._items += list(items)
        return self

    def index(self, item):
        return self._items.index(item)

    def remove(self, item):
        self._check_frozen()
        self._items.remove(item)

    def clear(self):
        self._check_frozen()
        self._items.clear()

    def extend(self, items):
        self._check_frozen()
        self._items += list(items)

    def reverse(self):
        self._check_frozen()
        self._items.reverse()

    def pop(self, index=-1):
        self._check_frozen()
        return self._items.pop(index)

    def append(self, item):
        self._check_frozen()
        return self._items.append(item)

    def count(self, item):
        return self._items.count(item)

    def __repr__(self):
        return '<FrozenList(frozen={}, {!r})>'.format(self.frozen,
                                                      self._items)

    def __hash__(self):
        if self.frozen:
            return hash(tuple(self._items))
        else:
            raise RuntimeError("Cannot hash unfrozen list.")


MutableSequence.register(FrozenList)
