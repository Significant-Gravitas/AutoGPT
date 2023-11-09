import sys
import types
from array import array
from collections import abc

from ._abc import MultiMapping, MutableMultiMapping

_marker = object()

if sys.version_info >= (3, 9):
    GenericAlias = types.GenericAlias
else:
    def GenericAlias(cls):
        return cls


class istr(str):

    """Case insensitive str."""

    __is_istr__ = True


upstr = istr  # for relaxing backward compatibility problems


def getversion(md):
    if not isinstance(md, _Base):
        raise TypeError("Parameter should be multidict or proxy")
    return md._impl._version


_version = array("Q", [0])


class _Impl:
    __slots__ = ("_items", "_version")

    def __init__(self):
        self._items = []
        self.incr_version()

    def incr_version(self):
        global _version
        v = _version
        v[0] += 1
        self._version = v[0]

    if sys.implementation.name != "pypy":

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._items)


class _Base:
    def _title(self, key):
        return key

    def getall(self, key, default=_marker):
        """Return a list of all values matching the key."""
        identity = self._title(key)
        res = [v for i, k, v in self._impl._items if i == identity]
        if res:
            return res
        if not res and default is not _marker:
            return default
        raise KeyError("Key not found: %r" % key)

    def getone(self, key, default=_marker):
        """Get first value matching the key.

        Raises KeyError if the key is not found and no default is provided.
        """
        identity = self._title(key)
        for i, k, v in self._impl._items:
            if i == identity:
                return v
        if default is not _marker:
            return default
        raise KeyError("Key not found: %r" % key)

    # Mapping interface #

    def __getitem__(self, key):
        return self.getone(key)

    def get(self, key, default=None):
        """Get first value matching the key.

        If the key is not found, returns the default (or None if no default is provided)
        """
        return self.getone(key, default)

    def __iter__(self):
        return iter(self.keys())

    def __len__(self):
        return len(self._impl._items)

    def keys(self):
        """Return a new view of the dictionary's keys."""
        return _KeysView(self._impl)

    def items(self):
        """Return a new view of the dictionary's items *(key, value) pairs)."""
        return _ItemsView(self._impl)

    def values(self):
        """Return a new view of the dictionary's values."""
        return _ValuesView(self._impl)

    def __eq__(self, other):
        if not isinstance(other, abc.Mapping):
            return NotImplemented
        if isinstance(other, _Base):
            lft = self._impl._items
            rht = other._impl._items
            if len(lft) != len(rht):
                return False
            for (i1, k2, v1), (i2, k2, v2) in zip(lft, rht):
                if i1 != i2 or v1 != v2:
                    return False
            return True
        if len(self._impl._items) != len(other):
            return False
        for k, v in self.items():
            nv = other.get(k, _marker)
            if v != nv:
                return False
        return True

    def __contains__(self, key):
        identity = self._title(key)
        for i, k, v in self._impl._items:
            if i == identity:
                return True
        return False

    def __repr__(self):
        body = ", ".join("'{}': {!r}".format(k, v) for k, v in self.items())
        return "<{}({})>".format(self.__class__.__name__, body)

    __class_getitem__ = classmethod(GenericAlias)


class MultiDictProxy(_Base, MultiMapping):
    """Read-only proxy for MultiDict instance."""

    def __init__(self, arg):
        if not isinstance(arg, (MultiDict, MultiDictProxy)):
            raise TypeError(
                "ctor requires MultiDict or MultiDictProxy instance"
                ", not {}".format(type(arg))
            )

        self._impl = arg._impl

    def __reduce__(self):
        raise TypeError("can't pickle {} objects".format(self.__class__.__name__))

    def copy(self):
        """Return a copy of itself."""
        return MultiDict(self.items())


class CIMultiDictProxy(MultiDictProxy):
    """Read-only proxy for CIMultiDict instance."""

    def __init__(self, arg):
        if not isinstance(arg, (CIMultiDict, CIMultiDictProxy)):
            raise TypeError(
                "ctor requires CIMultiDict or CIMultiDictProxy instance"
                ", not {}".format(type(arg))
            )

        self._impl = arg._impl

    def _title(self, key):
        return key.title()

    def copy(self):
        """Return a copy of itself."""
        return CIMultiDict(self.items())


class MultiDict(_Base, MutableMultiMapping):
    """Dictionary with the support for duplicate keys."""

    def __init__(self, *args, **kwargs):
        self._impl = _Impl()

        self._extend(args, kwargs, self.__class__.__name__, self._extend_items)

    if sys.implementation.name != "pypy":

        def __sizeof__(self):
            return object.__sizeof__(self) + sys.getsizeof(self._impl)

    def __reduce__(self):
        return (self.__class__, (list(self.items()),))

    def _title(self, key):
        return key

    def _key(self, key):
        if isinstance(key, str):
            return key
        else:
            raise TypeError(
                "MultiDict keys should be either str " "or subclasses of str"
            )

    def add(self, key, value):
        identity = self._title(key)
        self._impl._items.append((identity, self._key(key), value))
        self._impl.incr_version()

    def copy(self):
        """Return a copy of itself."""
        cls = self.__class__
        return cls(self.items())

    __copy__ = copy

    def extend(self, *args, **kwargs):
        """Extend current MultiDict with more values.

        This method must be used instead of update.
        """
        self._extend(args, kwargs, "extend", self._extend_items)

    def _extend(self, args, kwargs, name, method):
        if len(args) > 1:
            raise TypeError(
                "{} takes at most 1 positional argument"
                " ({} given)".format(name, len(args))
            )
        if args:
            arg = args[0]
            if isinstance(args[0], (MultiDict, MultiDictProxy)) and not kwargs:
                items = arg._impl._items
            else:
                if hasattr(arg, "items"):
                    arg = arg.items()
                if kwargs:
                    arg = list(arg)
                    arg.extend(list(kwargs.items()))
                items = []
                for item in arg:
                    if not len(item) == 2:
                        raise TypeError(
                            "{} takes either dict or list of (key, value) "
                            "tuples".format(name)
                        )
                    items.append((self._title(item[0]), self._key(item[0]), item[1]))

            method(items)
        else:
            method(
                [
                    (self._title(key), self._key(key), value)
                    for key, value in kwargs.items()
                ]
            )

    def _extend_items(self, items):
        for identity, key, value in items:
            self.add(key, value)

    def clear(self):
        """Remove all items from MultiDict."""
        self._impl._items.clear()
        self._impl.incr_version()

    # Mapping interface #

    def __setitem__(self, key, value):
        self._replace(key, value)

    def __delitem__(self, key):
        identity = self._title(key)
        items = self._impl._items
        found = False
        for i in range(len(items) - 1, -1, -1):
            if items[i][0] == identity:
                del items[i]
                found = True
        if not found:
            raise KeyError(key)
        else:
            self._impl.incr_version()

    def setdefault(self, key, default=None):
        """Return value for key, set value to default if key is not present."""
        identity = self._title(key)
        for i, k, v in self._impl._items:
            if i == identity:
                return v
        self.add(key, default)
        return default

    def popone(self, key, default=_marker):
        """Remove specified key and return the corresponding value.

        If key is not found, d is returned if given, otherwise
        KeyError is raised.

        """
        identity = self._title(key)
        for i in range(len(self._impl._items)):
            if self._impl._items[i][0] == identity:
                value = self._impl._items[i][2]
                del self._impl._items[i]
                self._impl.incr_version()
                return value
        if default is _marker:
            raise KeyError(key)
        else:
            return default

    pop = popone  # type: ignore

    def popall(self, key, default=_marker):
        """Remove all occurrences of key and return the list of corresponding
        values.

        If key is not found, default is returned if given, otherwise
        KeyError is raised.

        """
        found = False
        identity = self._title(key)
        ret = []
        for i in range(len(self._impl._items) - 1, -1, -1):
            item = self._impl._items[i]
            if item[0] == identity:
                ret.append(item[2])
                del self._impl._items[i]
                self._impl.incr_version()
                found = True
        if not found:
            if default is _marker:
                raise KeyError(key)
            else:
                return default
        else:
            ret.reverse()
            return ret

    def popitem(self):
        """Remove and return an arbitrary (key, value) pair."""
        if self._impl._items:
            i = self._impl._items.pop(0)
            self._impl.incr_version()
            return i[1], i[2]
        else:
            raise KeyError("empty multidict")

    def update(self, *args, **kwargs):
        """Update the dictionary from *other*, overwriting existing keys."""
        self._extend(args, kwargs, "update", self._update_items)

    def _update_items(self, items):
        if not items:
            return
        used_keys = {}
        for identity, key, value in items:
            start = used_keys.get(identity, 0)
            for i in range(start, len(self._impl._items)):
                item = self._impl._items[i]
                if item[0] == identity:
                    used_keys[identity] = i + 1
                    self._impl._items[i] = (identity, key, value)
                    break
            else:
                self._impl._items.append((identity, key, value))
                used_keys[identity] = len(self._impl._items)

        # drop tails
        i = 0
        while i < len(self._impl._items):
            item = self._impl._items[i]
            identity = item[0]
            pos = used_keys.get(identity)
            if pos is None:
                i += 1
                continue
            if i >= pos:
                del self._impl._items[i]
            else:
                i += 1

        self._impl.incr_version()

    def _replace(self, key, value):
        key = self._key(key)
        identity = self._title(key)
        items = self._impl._items

        for i in range(len(items)):
            item = items[i]
            if item[0] == identity:
                items[i] = (identity, key, value)
                # i points to last found item
                rgt = i
                self._impl.incr_version()
                break
        else:
            self._impl._items.append((identity, key, value))
            self._impl.incr_version()
            return

        # remove all tail items
        i = rgt + 1
        while i < len(items):
            item = items[i]
            if item[0] == identity:
                del items[i]
            else:
                i += 1


class CIMultiDict(MultiDict):
    """Dictionary with the support for duplicate case-insensitive keys."""

    def _title(self, key):
        return key.title()


class _Iter:
    __slots__ = ("_size", "_iter")

    def __init__(self, size, iterator):
        self._size = size
        self._iter = iterator

    def __iter__(self):
        return self

    def __next__(self):
        return next(self._iter)

    def __length_hint__(self):
        return self._size


class _ViewBase:
    def __init__(self, impl):
        self._impl = impl

    def __len__(self):
        return len(self._impl._items)


class _ItemsView(_ViewBase, abc.ItemsView):
    def __contains__(self, item):
        assert isinstance(item, tuple) or isinstance(item, list)
        assert len(item) == 2
        for i, k, v in self._impl._items:
            if item[0] == k and item[1] == v:
                return True
        return False

    def __iter__(self):
        return _Iter(len(self), self._iter(self._impl._version))

    def _iter(self, version):
        for i, k, v in self._impl._items:
            if version != self._impl._version:
                raise RuntimeError("Dictionary changed during iteration")
            yield k, v

    def __repr__(self):
        lst = []
        for item in self._impl._items:
            lst.append("{!r}: {!r}".format(item[1], item[2]))
        body = ", ".join(lst)
        return "{}({})".format(self.__class__.__name__, body)


class _ValuesView(_ViewBase, abc.ValuesView):
    def __contains__(self, value):
        for item in self._impl._items:
            if item[2] == value:
                return True
        return False

    def __iter__(self):
        return _Iter(len(self), self._iter(self._impl._version))

    def _iter(self, version):
        for item in self._impl._items:
            if version != self._impl._version:
                raise RuntimeError("Dictionary changed during iteration")
            yield item[2]

    def __repr__(self):
        lst = []
        for item in self._impl._items:
            lst.append("{!r}".format(item[2]))
        body = ", ".join(lst)
        return "{}({})".format(self.__class__.__name__, body)


class _KeysView(_ViewBase, abc.KeysView):
    def __contains__(self, key):
        for item in self._impl._items:
            if item[1] == key:
                return True
        return False

    def __iter__(self):
        return _Iter(len(self), self._iter(self._impl._version))

    def _iter(self, version):
        for item in self._impl._items:
            if version != self._impl._version:
                raise RuntimeError("Dictionary changed during iteration")
            yield item[1]

    def __repr__(self):
        lst = []
        for item in self._impl._items:
            lst.append("{!r}".format(item[1]))
        body = ", ".join(lst)
        return "{}({})".format(self.__class__.__name__, body)
