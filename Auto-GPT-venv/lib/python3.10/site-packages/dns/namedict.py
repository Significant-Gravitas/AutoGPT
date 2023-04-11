# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2017 Nominum, Inc.
# Copyright (C) 2016 Coresec Systems AB
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND CORESEC SYSTEMS AB DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL CORESEC
# SYSTEMS AB BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT, OR
# CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS
# OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

"""DNS name dictionary"""

# pylint seems to be confused about this one!
from collections.abc import MutableMapping  # pylint: disable=no-name-in-module

import dns.name


class NameDict(MutableMapping):
    """A dictionary whose keys are dns.name.Name objects.

    In addition to being like a regular Python dictionary, this
    dictionary can also get the deepest match for a given key.
    """

    __slots__ = ["max_depth", "max_depth_items", "__store"]

    def __init__(self, *args, **kwargs):
        super().__init__()
        self.__store = dict()
        #: the maximum depth of the keys that have ever been added
        self.max_depth = 0
        #: the number of items of maximum depth
        self.max_depth_items = 0
        self.update(dict(*args, **kwargs))

    def __update_max_depth(self, key):
        if len(key) == self.max_depth:
            self.max_depth_items = self.max_depth_items + 1
        elif len(key) > self.max_depth:
            self.max_depth = len(key)
            self.max_depth_items = 1

    def __getitem__(self, key):
        return self.__store[key]

    def __setitem__(self, key, value):
        if not isinstance(key, dns.name.Name):
            raise ValueError("NameDict key must be a name")
        self.__store[key] = value
        self.__update_max_depth(key)

    def __delitem__(self, key):
        self.__store.pop(key)
        if len(key) == self.max_depth:
            self.max_depth_items = self.max_depth_items - 1
        if self.max_depth_items == 0:
            self.max_depth = 0
            for k in self.__store:
                self.__update_max_depth(k)

    def __iter__(self):
        return iter(self.__store)

    def __len__(self):
        return len(self.__store)

    def has_key(self, key):
        return key in self.__store

    def get_deepest_match(self, name):
        """Find the deepest match to *name* in the dictionary.

        The deepest match is the longest name in the dictionary which is
        a superdomain of *name*.  Note that *superdomain* includes matching
        *name* itself.

        *name*, a ``dns.name.Name``, the name to find.

        Returns a ``(key, value)`` where *key* is the deepest
        ``dns.name.Name``, and *value* is the value associated with *key*.
        """

        depth = len(name)
        if depth > self.max_depth:
            depth = self.max_depth
        for i in range(-depth, 0):
            n = dns.name.Name(name[i:])
            if n in self:
                return (n, self[n])
        v = self[dns.name.empty]
        return (dns.name.empty, v)
