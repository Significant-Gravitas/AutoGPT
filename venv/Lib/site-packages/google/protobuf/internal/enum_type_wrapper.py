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

"""A simple wrapper around enum types to expose utility functions.

Instances are created as properties with the same name as the enum they wrap
on proto classes.  For usage, see:
  reflection_test.py
"""

__author__ = 'rabsatt@google.com (Kevin Rabsatt)'


class EnumTypeWrapper(object):
  """A utility for finding the names of enum values."""

  DESCRIPTOR = None

  # This is a type alias, which mypy typing stubs can type as
  # a genericized parameter constrained to an int, allowing subclasses
  # to be typed with more constraint in .pyi stubs
  # Eg.
  # def MyGeneratedEnum(Message):
  #   ValueType = NewType('ValueType', int)
  #   def Name(self, number: MyGeneratedEnum.ValueType) -> str
  ValueType = int

  def __init__(self, enum_type):
    """Inits EnumTypeWrapper with an EnumDescriptor."""
    self._enum_type = enum_type
    self.DESCRIPTOR = enum_type  # pylint: disable=invalid-name

  def Name(self, number):  # pylint: disable=invalid-name
    """Returns a string containing the name of an enum value."""
    try:
      return self._enum_type.values_by_number[number].name
    except KeyError:
      pass  # fall out to break exception chaining

    if not isinstance(number, int):
      raise TypeError(
          'Enum value for {} must be an int, but got {} {!r}.'.format(
              self._enum_type.name, type(number), number))
    else:
      # repr here to handle the odd case when you pass in a boolean.
      raise ValueError('Enum {} has no name defined for value {!r}'.format(
          self._enum_type.name, number))

  def Value(self, name):  # pylint: disable=invalid-name
    """Returns the value corresponding to the given enum name."""
    try:
      return self._enum_type.values_by_name[name].number
    except KeyError:
      pass  # fall out to break exception chaining
    raise ValueError('Enum {} has no value defined for name {!r}'.format(
        self._enum_type.name, name))

  def keys(self):
    """Return a list of the string names in the enum.

    Returns:
      A list of strs, in the order they were defined in the .proto file.
    """

    return [value_descriptor.name
            for value_descriptor in self._enum_type.values]

  def values(self):
    """Return a list of the integer values in the enum.

    Returns:
      A list of ints, in the order they were defined in the .proto file.
    """

    return [value_descriptor.number
            for value_descriptor in self._enum_type.values]

  def items(self):
    """Return a list of the (name, value) pairs of the enum.

    Returns:
      A list of (str, int) pairs, in the order they were defined
      in the .proto file.
    """
    return [(value_descriptor.name, value_descriptor.number)
            for value_descriptor in self._enum_type.values]

  def __getattr__(self, name):
    """Returns the value corresponding to the given enum name."""
    try:
      return super(
          EnumTypeWrapper,
          self).__getattribute__('_enum_type').values_by_name[name].number
    except KeyError:
      pass  # fall out to break exception chaining
    raise AttributeError('Enum {} has no value defined for name {!r}'.format(
        self._enum_type.name, name))
