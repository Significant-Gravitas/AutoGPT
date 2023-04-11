# Protocol Buffers - Google's data interchange format
# Copyright 2008 Google Inc.  All rights reserved.
# https://developers.google.com/protocol-buffers/
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are
# met:
#
#     * Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
# copyright notice, this list of conditions and the following disclaimer
# in the documentation and/or other materials provided with the
# distribution.
#     * Neither the name of Google Inc. nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
# OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
# SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
# DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
# THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Contains container classes to represent different protocol buffer types.

This file defines container classes which represent categories of protocol
buffer field types which need extra maintenance. Currently these categories
are:

-   Repeated scalar fields - These are all repeated fields which aren't
    composite (e.g. they are of simple types like int32, string, etc).
-   Repeated composite fields - Repeated fields which are composite. This
    includes groups and nested messages.
"""

import collections.abc
import copy
import pickle
from typing import (
    Any,
    Iterable,
    Iterator,
    List,
    MutableMapping,
    MutableSequence,
    NoReturn,
    Optional,
    Sequence,
    TypeVar,
    Union,
    overload,
)


_T = TypeVar('_T')
_K = TypeVar('_K')
_V = TypeVar('_V')


class BaseContainer(Sequence[_T]):
  """Base container class."""

  # Minimizes memory usage and disallows assignment to other attributes.
  __slots__ = ['_message_listener', '_values']

  def __init__(self, message_listener: Any) -> None:
    """
    Args:
      message_listener: A MessageListener implementation.
        The RepeatedScalarFieldContainer will call this object's
        Modified() method when it is modified.
    """
    self._message_listener = message_listener
    self._values = []

  @overload
  def __getitem__(self, key: int) -> _T:
    ...

  @overload
  def __getitem__(self, key: slice) -> List[_T]:
    ...

  def __getitem__(self, key):
    """Retrieves item by the specified key."""
    return self._values[key]

  def __len__(self) -> int:
    """Returns the number of elements in the container."""
    return len(self._values)

  def __ne__(self, other: Any) -> bool:
    """Checks if another instance isn't equal to this one."""
    # The concrete classes should define __eq__.
    return not self == other

  __hash__ = None

  def __repr__(self) -> str:
    return repr(self._values)

  def sort(self, *args, **kwargs) -> None:
    # Continue to support the old sort_function keyword argument.
    # This is expected to be a rare occurrence, so use LBYL to avoid
    # the overhead of actually catching KeyError.
    if 'sort_function' in kwargs:
      kwargs['cmp'] = kwargs.pop('sort_function')
    self._values.sort(*args, **kwargs)

  def reverse(self) -> None:
    self._values.reverse()


# TODO(slebedev): Remove this. BaseContainer does *not* conform to
# MutableSequence, only its subclasses do.
collections.abc.MutableSequence.register(BaseContainer)


class RepeatedScalarFieldContainer(BaseContainer[_T], MutableSequence[_T]):
  """Simple, type-checked, list-like container for holding repeated scalars."""

  # Disallows assignment to other attributes.
  __slots__ = ['_type_checker']

  def __init__(
      self,
      message_listener: Any,
      type_checker: Any,
  ) -> None:
    """Args:

      message_listener: A MessageListener implementation. The
      RepeatedScalarFieldContainer will call this object's Modified() method
      when it is modified.
      type_checker: A type_checkers.ValueChecker instance to run on elements
      inserted into this container.
    """
    super().__init__(message_listener)
    self._type_checker = type_checker

  def append(self, value: _T) -> None:
    """Appends an item to the list. Similar to list.append()."""
    self._values.append(self._type_checker.CheckValue(value))
    if not self._message_listener.dirty:
      self._message_listener.Modified()

  def insert(self, key: int, value: _T) -> None:
    """Inserts the item at the specified position. Similar to list.insert()."""
    self._values.insert(key, self._type_checker.CheckValue(value))
    if not self._message_listener.dirty:
      self._message_listener.Modified()

  def extend(self, elem_seq: Iterable[_T]) -> None:
    """Extends by appending the given iterable. Similar to list.extend()."""
    if elem_seq is None:
      return
    try:
      elem_seq_iter = iter(elem_seq)
    except TypeError:
      if not elem_seq:
        # silently ignore falsy inputs :-/.
        # TODO(ptucker): Deprecate this behavior. b/18413862
        return
      raise

    new_values = [self._type_checker.CheckValue(elem) for elem in elem_seq_iter]
    if new_values:
      self._values.extend(new_values)
    self._message_listener.Modified()

  def MergeFrom(
      self,
      other: Union['RepeatedScalarFieldContainer[_T]', Iterable[_T]],
  ) -> None:
    """Appends the contents of another repeated field of the same type to this
    one. We do not check the types of the individual fields.
    """
    self._values.extend(other)
    self._message_listener.Modified()

  def remove(self, elem: _T):
    """Removes an item from the list. Similar to list.remove()."""
    self._values.remove(elem)
    self._message_listener.Modified()

  def pop(self, key: Optional[int] = -1) -> _T:
    """Removes and returns an item at a given index. Similar to list.pop()."""
    value = self._values[key]
    self.__delitem__(key)
    return value

  @overload
  def __setitem__(self, key: int, value: _T) -> None:
    ...

  @overload
  def __setitem__(self, key: slice, value: Iterable[_T]) -> None:
    ...

  def __setitem__(self, key, value) -> None:
    """Sets the item on the specified position."""
    if isinstance(key, slice):
      if key.step is not None:
        raise ValueError('Extended slices not supported')
      self._values[key] = map(self._type_checker.CheckValue, value)
      self._message_listener.Modified()
    else:
      self._values[key] = self._type_checker.CheckValue(value)
      self._message_listener.Modified()

  def __delitem__(self, key: Union[int, slice]) -> None:
    """Deletes the item at the specified position."""
    del self._values[key]
    self._message_listener.Modified()

  def __eq__(self, other: Any) -> bool:
    """Compares the current instance with another one."""
    if self is other:
      return True
    # Special case for the same type which should be common and fast.
    if isinstance(other, self.__class__):
      return other._values == self._values
    # We are presumably comparing against some other sequence type.
    return other == self._values

  def __deepcopy__(
      self,
      unused_memo: Any = None,
  ) -> 'RepeatedScalarFieldContainer[_T]':
    clone = RepeatedScalarFieldContainer(
        copy.deepcopy(self._message_listener), self._type_checker)
    clone.MergeFrom(self)
    return clone

  def __reduce__(self, **kwargs) -> NoReturn:
    raise pickle.PickleError(
        "Can't pickle repeated scalar fields, convert to list first")


# TODO(slebedev): Constrain T to be a subtype of Message.
class RepeatedCompositeFieldContainer(BaseContainer[_T], MutableSequence[_T]):
  """Simple, list-like container for holding repeated composite fields."""

  # Disallows assignment to other attributes.
  __slots__ = ['_message_descriptor']

  def __init__(self, message_listener: Any, message_descriptor: Any) -> None:
    """
    Note that we pass in a descriptor instead of the generated directly,
    since at the time we construct a _RepeatedCompositeFieldContainer we
    haven't yet necessarily initialized the type that will be contained in the
    container.

    Args:
      message_listener: A MessageListener implementation.
        The RepeatedCompositeFieldContainer will call this object's
        Modified() method when it is modified.
      message_descriptor: A Descriptor instance describing the protocol type
        that should be present in this container.  We'll use the
        _concrete_class field of this descriptor when the client calls add().
    """
    super().__init__(message_listener)
    self._message_descriptor = message_descriptor

  def add(self, **kwargs: Any) -> _T:
    """Adds a new element at the end of the list and returns it. Keyword
    arguments may be used to initialize the element.
    """
    new_element = self._message_descriptor._concrete_class(**kwargs)
    new_element._SetListener(self._message_listener)
    self._values.append(new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()
    return new_element

  def append(self, value: _T) -> None:
    """Appends one element by copying the message."""
    new_element = self._message_descriptor._concrete_class()
    new_element._SetListener(self._message_listener)
    new_element.CopyFrom(value)
    self._values.append(new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()

  def insert(self, key: int, value: _T) -> None:
    """Inserts the item at the specified position by copying."""
    new_element = self._message_descriptor._concrete_class()
    new_element._SetListener(self._message_listener)
    new_element.CopyFrom(value)
    self._values.insert(key, new_element)
    if not self._message_listener.dirty:
      self._message_listener.Modified()

  def extend(self, elem_seq: Iterable[_T]) -> None:
    """Extends by appending the given sequence of elements of the same type

    as this one, copying each individual message.
    """
    message_class = self._message_descriptor._concrete_class
    listener = self._message_listener
    values = self._values
    for message in elem_seq:
      new_element = message_class()
      new_element._SetListener(listener)
      new_element.MergeFrom(message)
      values.append(new_element)
    listener.Modified()

  def MergeFrom(
      self,
      other: Union['RepeatedCompositeFieldContainer[_T]', Iterable[_T]],
  ) -> None:
    """Appends the contents of another repeated field of the same type to this
    one, copying each individual message.
    """
    self.extend(other)

  def remove(self, elem: _T) -> None:
    """Removes an item from the list. Similar to list.remove()."""
    self._values.remove(elem)
    self._message_listener.Modified()

  def pop(self, key: Optional[int] = -1) -> _T:
    """Removes and returns an item at a given index. Similar to list.pop()."""
    value = self._values[key]
    self.__delitem__(key)
    return value

  @overload
  def __setitem__(self, key: int, value: _T) -> None:
    ...

  @overload
  def __setitem__(self, key: slice, value: Iterable[_T]) -> None:
    ...

  def __setitem__(self, key, value):
    # This method is implemented to make RepeatedCompositeFieldContainer
    # structurally compatible with typing.MutableSequence. It is
    # otherwise unsupported and will always raise an error.
    raise TypeError(
        f'{self.__class__.__name__} object does not support item assignment')

  def __delitem__(self, key: Union[int, slice]) -> None:
    """Deletes the item at the specified position."""
    del self._values[key]
    self._message_listener.Modified()

  def __eq__(self, other: Any) -> bool:
    """Compares the current instance with another one."""
    if self is other:
      return True
    if not isinstance(other, self.__class__):
      raise TypeError('Can only compare repeated composite fields against '
                      'other repeated composite fields.')
    return self._values == other._values


class ScalarMap(MutableMapping[_K, _V]):
  """Simple, type-checked, dict-like container for holding repeated scalars."""

  # Disallows assignment to other attributes.
  __slots__ = ['_key_checker', '_value_checker', '_values', '_message_listener',
               '_entry_descriptor']

  def __init__(
      self,
      message_listener: Any,
      key_checker: Any,
      value_checker: Any,
      entry_descriptor: Any,
  ) -> None:
    """
    Args:
      message_listener: A MessageListener implementation.
        The ScalarMap will call this object's Modified() method when it
        is modified.
      key_checker: A type_checkers.ValueChecker instance to run on keys
        inserted into this container.
      value_checker: A type_checkers.ValueChecker instance to run on values
        inserted into this container.
      entry_descriptor: The MessageDescriptor of a map entry: key and value.
    """
    self._message_listener = message_listener
    self._key_checker = key_checker
    self._value_checker = value_checker
    self._entry_descriptor = entry_descriptor
    self._values = {}

  def __getitem__(self, key: _K) -> _V:
    try:
      return self._values[key]
    except KeyError:
      key = self._key_checker.CheckValue(key)
      val = self._value_checker.DefaultValue()
      self._values[key] = val
      return val

  def __contains__(self, item: _K) -> bool:
    # We check the key's type to match the strong-typing flavor of the API.
    # Also this makes it easier to match the behavior of the C++ implementation.
    self._key_checker.CheckValue(item)
    return item in self._values

  @overload
  def get(self, key: _K) -> Optional[_V]:
    ...

  @overload
  def get(self, key: _K, default: _T) -> Union[_V, _T]:
    ...

  # We need to override this explicitly, because our defaultdict-like behavior
  # will make the default implementation (from our base class) always insert
  # the key.
  def get(self, key, default=None):
    if key in self:
      return self[key]
    else:
      return default

  def __setitem__(self, key: _K, value: _V) -> _T:
    checked_key = self._key_checker.CheckValue(key)
    checked_value = self._value_checker.CheckValue(value)
    self._values[checked_key] = checked_value
    self._message_listener.Modified()

  def __delitem__(self, key: _K) -> None:
    del self._values[key]
    self._message_listener.Modified()

  def __len__(self) -> int:
    return len(self._values)

  def __iter__(self) -> Iterator[_K]:
    return iter(self._values)

  def __repr__(self) -> str:
    return repr(self._values)

  def MergeFrom(self, other: 'ScalarMap[_K, _V]') -> None:
    self._values.update(other._values)
    self._message_listener.Modified()

  def InvalidateIterators(self) -> None:
    # It appears that the only way to reliably invalidate iterators to
    # self._values is to ensure that its size changes.
    original = self._values
    self._values = original.copy()
    original[None] = None

  # This is defined in the abstract base, but we can do it much more cheaply.
  def clear(self) -> None:
    self._values.clear()
    self._message_listener.Modified()

  def GetEntryClass(self) -> Any:
    return self._entry_descriptor._concrete_class


class MessageMap(MutableMapping[_K, _V]):
  """Simple, type-checked, dict-like container for with submessage values."""

  # Disallows assignment to other attributes.
  __slots__ = ['_key_checker', '_values', '_message_listener',
               '_message_descriptor', '_entry_descriptor']

  def __init__(
      self,
      message_listener: Any,
      message_descriptor: Any,
      key_checker: Any,
      entry_descriptor: Any,
  ) -> None:
    """
    Args:
      message_listener: A MessageListener implementation.
        The ScalarMap will call this object's Modified() method when it
        is modified.
      key_checker: A type_checkers.ValueChecker instance to run on keys
        inserted into this container.
      value_checker: A type_checkers.ValueChecker instance to run on values
        inserted into this container.
      entry_descriptor: The MessageDescriptor of a map entry: key and value.
    """
    self._message_listener = message_listener
    self._message_descriptor = message_descriptor
    self._key_checker = key_checker
    self._entry_descriptor = entry_descriptor
    self._values = {}

  def __getitem__(self, key: _K) -> _V:
    key = self._key_checker.CheckValue(key)
    try:
      return self._values[key]
    except KeyError:
      new_element = self._message_descriptor._concrete_class()
      new_element._SetListener(self._message_listener)
      self._values[key] = new_element
      self._message_listener.Modified()
      return new_element

  def get_or_create(self, key: _K) -> _V:
    """get_or_create() is an alias for getitem (ie. map[key]).

    Args:
      key: The key to get or create in the map.

    This is useful in cases where you want to be explicit that the call is
    mutating the map.  This can avoid lint errors for statements like this
    that otherwise would appear to be pointless statements:

      msg.my_map[key]
    """
    return self[key]

  @overload
  def get(self, key: _K) -> Optional[_V]:
    ...

  @overload
  def get(self, key: _K, default: _T) -> Union[_V, _T]:
    ...

  # We need to override this explicitly, because our defaultdict-like behavior
  # will make the default implementation (from our base class) always insert
  # the key.
  def get(self, key, default=None):
    if key in self:
      return self[key]
    else:
      return default

  def __contains__(self, item: _K) -> bool:
    item = self._key_checker.CheckValue(item)
    return item in self._values

  def __setitem__(self, key: _K, value: _V) -> NoReturn:
    raise ValueError('May not set values directly, call my_map[key].foo = 5')

  def __delitem__(self, key: _K) -> None:
    key = self._key_checker.CheckValue(key)
    del self._values[key]
    self._message_listener.Modified()

  def __len__(self) -> int:
    return len(self._values)

  def __iter__(self) -> Iterator[_K]:
    return iter(self._values)

  def __repr__(self) -> str:
    return repr(self._values)

  def MergeFrom(self, other: 'MessageMap[_K, _V]') -> None:
    # pylint: disable=protected-access
    for key in other._values:
      # According to documentation: "When parsing from the wire or when merging,
      # if there are duplicate map keys the last key seen is used".
      if key in self:
        del self[key]
      self[key].CopyFrom(other[key])
    # self._message_listener.Modified() not required here, because
    # mutations to submessages already propagate.

  def InvalidateIterators(self) -> None:
    # It appears that the only way to reliably invalidate iterators to
    # self._values is to ensure that its size changes.
    original = self._values
    self._values = original.copy()
    original[None] = None

  # This is defined in the abstract base, but we can do it much more cheaply.
  def clear(self) -> None:
    self._values.clear()
    self._message_listener.Modified()

  def GetEntryClass(self) -> Any:
    return self._entry_descriptor._concrete_class


class _UnknownField:
  """A parsed unknown field."""

  # Disallows assignment to other attributes.
  __slots__ = ['_field_number', '_wire_type', '_data']

  def __init__(self, field_number, wire_type, data):
    self._field_number = field_number
    self._wire_type = wire_type
    self._data = data
    return

  def __lt__(self, other):
    # pylint: disable=protected-access
    return self._field_number < other._field_number

  def __eq__(self, other):
    if self is other:
      return True
    # pylint: disable=protected-access
    return (self._field_number == other._field_number and
            self._wire_type == other._wire_type and
            self._data == other._data)


class UnknownFieldRef:  # pylint: disable=missing-class-docstring

  def __init__(self, parent, index):
    self._parent = parent
    self._index = index

  def _check_valid(self):
    if not self._parent:
      raise ValueError('UnknownField does not exist. '
                       'The parent message might be cleared.')
    if self._index >= len(self._parent):
      raise ValueError('UnknownField does not exist. '
                       'The parent message might be cleared.')

  @property
  def field_number(self):
    self._check_valid()
    # pylint: disable=protected-access
    return self._parent._internal_get(self._index)._field_number

  @property
  def wire_type(self):
    self._check_valid()
    # pylint: disable=protected-access
    return self._parent._internal_get(self._index)._wire_type

  @property
  def data(self):
    self._check_valid()
    # pylint: disable=protected-access
    return self._parent._internal_get(self._index)._data


class UnknownFieldSet:
  """UnknownField container"""

  # Disallows assignment to other attributes.
  __slots__ = ['_values']

  def __init__(self):
    self._values = []

  def __getitem__(self, index):
    if self._values is None:
      raise ValueError('UnknownFields does not exist. '
                       'The parent message might be cleared.')
    size = len(self._values)
    if index < 0:
      index += size
    if index < 0 or index >= size:
      raise IndexError('index %d out of range'.index)

    return UnknownFieldRef(self, index)

  def _internal_get(self, index):
    return self._values[index]

  def __len__(self):
    if self._values is None:
      raise ValueError('UnknownFields does not exist. '
                       'The parent message might be cleared.')
    return len(self._values)

  def _add(self, field_number, wire_type, data):
    unknown_field = _UnknownField(field_number, wire_type, data)
    self._values.append(unknown_field)
    return unknown_field

  def __iter__(self):
    for i in range(len(self)):
      yield UnknownFieldRef(self, i)

  def _extend(self, other):
    if other is None:
      return
    # pylint: disable=protected-access
    self._values.extend(other._values)

  def __eq__(self, other):
    if self is other:
      return True
    # Sort unknown fields because their order shouldn't
    # affect equality test.
    values = list(self._values)
    if other is None:
      return not values
    values.sort()
    # pylint: disable=protected-access
    other_values = sorted(other._values)
    return values == other_values

  def _clear(self):
    for value in self._values:
      # pylint: disable=protected-access
      if isinstance(value._data, UnknownFieldSet):
        value._data._clear()  # pylint: disable=protected-access
    self._values = None
