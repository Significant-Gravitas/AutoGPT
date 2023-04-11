# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

from typing import Any

import collections.abc

from dns._immutable_ctx import immutable


@immutable
class Dict(collections.abc.Mapping):  # lgtm[py/missing-equals]
    def __init__(self, dictionary: Any, no_copy: bool = False):
        """Make an immutable dictionary from the specified dictionary.

        If *no_copy* is `True`, then *dictionary* will be wrapped instead
        of copied.  Only set this if you are sure there will be no external
        references to the dictionary.
        """
        if no_copy and isinstance(dictionary, dict):
            self._odict = dictionary
        else:
            self._odict = dict(dictionary)
        self._hash = None

    def __getitem__(self, key):
        return self._odict.__getitem__(key)

    def __hash__(self):  # pylint: disable=invalid-hash-returned
        if self._hash is None:
            h = 0
            for key in sorted(self._odict.keys()):
                h ^= hash(key)
            object.__setattr__(self, "_hash", h)
        # this does return an int, but pylint doesn't figure that out
        return self._hash

    def __len__(self):
        return len(self._odict)

    def __iter__(self):
        return iter(self._odict)


def constify(o: Any) -> Any:
    """
    Convert mutable types to immutable types.
    """
    if isinstance(o, bytearray):
        return bytes(o)
    if isinstance(o, tuple):
        try:
            hash(o)
            return o
        except Exception:
            return tuple(constify(elt) for elt in o)
    if isinstance(o, list):
        return tuple(constify(elt) for elt in o)
    if isinstance(o, dict):
        cdict = dict()
        for k, v in o.items():
            cdict[k] = constify(v)
        return Dict(cdict, True)
    return o
