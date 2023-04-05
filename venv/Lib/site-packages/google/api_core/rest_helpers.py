# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for rest transports."""

import functools
import operator


def flatten_query_params(obj, strict=False):
    """Flatten a dict into a list of (name,value) tuples.

    The result is suitable for setting query params on an http request.

    .. code-block:: python

        >>> obj = {'a':
        ...         {'b':
        ...           {'c': ['x', 'y', 'z']} },
        ...      'd': 'uvw',
        ...      'e': True, }
        >>> flatten_query_params(obj, strict=True)
        [('a.b.c', 'x'), ('a.b.c', 'y'), ('a.b.c', 'z'), ('d', 'uvw'), ('e', 'true')]

    Note that, as described in
    https://github.com/googleapis/googleapis/blob/48d9fb8c8e287c472af500221c6450ecd45d7d39/google/api/http.proto#L117,
    repeated fields (i.e. list-valued fields) may only contain primitive types (not lists or dicts).
    This is enforced in this function.

    Args:
      obj: a possibly nested dictionary (from json), or None
      strict: a bool, defaulting to False, to enforce that all values in the
              result tuples be strings and, if boolean, lower-cased.

    Returns: a list of tuples, with each tuple having a (possibly) multi-part name
      and a scalar value.

    Raises:
      TypeError if obj is not a dict or None
      ValueError if obj contains a list of non-primitive values.
    """

    if obj is not None and not isinstance(obj, dict):
        raise TypeError("flatten_query_params must be called with dict object")

    return _flatten(obj, key_path=[], strict=strict)


def _flatten(obj, key_path, strict=False):
    if obj is None:
        return []
    if isinstance(obj, dict):
        return _flatten_dict(obj, key_path=key_path, strict=strict)
    if isinstance(obj, list):
        return _flatten_list(obj, key_path=key_path, strict=strict)
    return _flatten_value(obj, key_path=key_path, strict=strict)


def _is_primitive_value(obj):
    if obj is None:
        return False

    if isinstance(obj, (list, dict)):
        raise ValueError("query params may not contain repeated dicts or lists")

    return True


def _flatten_value(obj, key_path, strict=False):
    return [(".".join(key_path), _canonicalize(obj, strict=strict))]


def _flatten_dict(obj, key_path, strict=False):
    items = (
        _flatten(value, key_path=key_path + [key], strict=strict)
        for key, value in obj.items()
    )
    return functools.reduce(operator.concat, items, [])


def _flatten_list(elems, key_path, strict=False):
    # Only lists of scalar values are supported.
    # The name (key_path) is repeated for each value.
    items = (
        _flatten_value(elem, key_path=key_path, strict=strict)
        for elem in elems
        if _is_primitive_value(elem)
    )
    return functools.reduce(operator.concat, items, [])


def _canonicalize(obj, strict=False):
    if strict:
        value = str(obj)
        if isinstance(obj, bool):
            value = value.lower()
        return value
    return obj
