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

"""Code for encoding protocol message primitives.

Contains the logic for encoding every logical protocol field type
into one of the 5 physical wire types.

This code is designed to push the Python interpreter's performance to the
limits.

The basic idea is that at startup time, for every field (i.e. every
FieldDescriptor) we construct two functions:  a "sizer" and an "encoder".  The
sizer takes a value of this field's type and computes its byte size.  The
encoder takes a writer function and a value.  It encodes the value into byte
strings and invokes the writer function to write those strings.  Typically the
writer function is the write() method of a BytesIO.

We try to do as much work as possible when constructing the writer and the
sizer rather than when calling them.  In particular:
* We copy any needed global functions to local variables, so that we do not need
  to do costly global table lookups at runtime.
* Similarly, we try to do any attribute lookups at startup time if possible.
* Every field's tag is encoded to bytes at startup, since it can't change at
  runtime.
* Whatever component of the field size we can compute at startup, we do.
* We *avoid* sharing code if doing so would make the code slower and not sharing
  does not burden us too much.  For example, encoders for repeated fields do
  not just call the encoders for singular fields in a loop because this would
  add an extra function call overhead for every loop iteration; instead, we
  manually inline the single-value encoder into the loop.
* If a Python function lacks a return statement, Python actually generates
  instructions to pop the result of the last statement off the stack, push
  None onto the stack, and then return that.  If we really don't care what
  value is returned, then we can save two instructions by returning the
  result of the last statement.  It looks funny but it helps.
* We assume that type and bounds checking has happened at a higher level.
"""

__author__ = 'kenton@google.com (Kenton Varda)'

import struct

from google.protobuf.internal import wire_format


# This will overflow and thus become IEEE-754 "infinity".  We would use
# "float('inf')" but it doesn't work on Windows pre-Python-2.6.
_POS_INF = 1e10000
_NEG_INF = -_POS_INF


def _VarintSize(value):
  """Compute the size of a varint value."""
  if value <= 0x7f: return 1
  if value <= 0x3fff: return 2
  if value <= 0x1fffff: return 3
  if value <= 0xfffffff: return 4
  if value <= 0x7ffffffff: return 5
  if value <= 0x3ffffffffff: return 6
  if value <= 0x1ffffffffffff: return 7
  if value <= 0xffffffffffffff: return 8
  if value <= 0x7fffffffffffffff: return 9
  return 10


def _SignedVarintSize(value):
  """Compute the size of a signed varint value."""
  if value < 0: return 10
  if value <= 0x7f: return 1
  if value <= 0x3fff: return 2
  if value <= 0x1fffff: return 3
  if value <= 0xfffffff: return 4
  if value <= 0x7ffffffff: return 5
  if value <= 0x3ffffffffff: return 6
  if value <= 0x1ffffffffffff: return 7
  if value <= 0xffffffffffffff: return 8
  if value <= 0x7fffffffffffffff: return 9
  return 10


def _TagSize(field_number):
  """Returns the number of bytes required to serialize a tag with this field
  number."""
  # Just pass in type 0, since the type won't affect the tag+type size.
  return _VarintSize(wire_format.PackTag(field_number, 0))


# --------------------------------------------------------------------
# In this section we define some generic sizers.  Each of these functions
# takes parameters specific to a particular field type, e.g. int32 or fixed64.
# It returns another function which in turn takes parameters specific to a
# particular field, e.g. the field number and whether it is repeated or packed.
# Look at the next section to see how these are used.


def _SimpleSizer(compute_value_size):
  """A sizer which uses the function compute_value_size to compute the size of
  each value.  Typically compute_value_size is _VarintSize."""

  def SpecificSizer(field_number, is_repeated, is_packed):
    tag_size = _TagSize(field_number)
    if is_packed:
      local_VarintSize = _VarintSize
      def PackedFieldSize(value):
        result = 0
        for element in value:
          result += compute_value_size(element)
        return result + local_VarintSize(result) + tag_size
      return PackedFieldSize
    elif is_repeated:
      def RepeatedFieldSize(value):
        result = tag_size * len(value)
        for element in value:
          result += compute_value_size(element)
        return result
      return RepeatedFieldSize
    else:
      def FieldSize(value):
        return tag_size + compute_value_size(value)
      return FieldSize

  return SpecificSizer


def _ModifiedSizer(compute_value_size, modify_value):
  """Like SimpleSizer, but modify_value is invoked on each value before it is
  passed to compute_value_size.  modify_value is typically ZigZagEncode."""

  def SpecificSizer(field_number, is_repeated, is_packed):
    tag_size = _TagSize(field_number)
    if is_packed:
      local_VarintSize = _VarintSize
      def PackedFieldSize(value):
        result = 0
        for element in value:
          result += compute_value_size(modify_value(element))
        return result + local_VarintSize(result) + tag_size
      return PackedFieldSize
    elif is_repeated:
      def RepeatedFieldSize(value):
        result = tag_size * len(value)
        for element in value:
          result += compute_value_size(modify_value(element))
        return result
      return RepeatedFieldSize
    else:
      def FieldSize(value):
        return tag_size + compute_value_size(modify_value(value))
      return FieldSize

  return SpecificSizer


def _FixedSizer(value_size):
  """Like _SimpleSizer except for a fixed-size field.  The input is the size
  of one value."""

  def SpecificSizer(field_number, is_repeated, is_packed):
    tag_size = _TagSize(field_number)
    if is_packed:
      local_VarintSize = _VarintSize
      def PackedFieldSize(value):
        result = len(value) * value_size
        return result + local_VarintSize(result) + tag_size
      return PackedFieldSize
    elif is_repeated:
      element_size = value_size + tag_size
      def RepeatedFieldSize(value):
        return len(value) * element_size
      return RepeatedFieldSize
    else:
      field_size = value_size + tag_size
      def FieldSize(value):
        return field_size
      return FieldSize

  return SpecificSizer


# ====================================================================
# Here we declare a sizer constructor for each field type.  Each "sizer
# constructor" is a function that takes (field_number, is_repeated, is_packed)
# as parameters and returns a sizer, which in turn takes a field value as
# a parameter and returns its encoded size.


Int32Sizer = Int64Sizer = EnumSizer = _SimpleSizer(_SignedVarintSize)

UInt32Sizer = UInt64Sizer = _SimpleSizer(_VarintSize)

SInt32Sizer = SInt64Sizer = _ModifiedSizer(
    _SignedVarintSize, wire_format.ZigZagEncode)

Fixed32Sizer = SFixed32Sizer = FloatSizer  = _FixedSizer(4)
Fixed64Sizer = SFixed64Sizer = DoubleSizer = _FixedSizer(8)

BoolSizer = _FixedSizer(1)


def StringSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a string field."""

  tag_size = _TagSize(field_number)
  local_VarintSize = _VarintSize
  local_len = len
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        l = local_len(element.encode('utf-8'))
        result += local_VarintSize(l) + l
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      l = local_len(value.encode('utf-8'))
      return tag_size + local_VarintSize(l) + l
    return FieldSize


def BytesSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a bytes field."""

  tag_size = _TagSize(field_number)
  local_VarintSize = _VarintSize
  local_len = len
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        l = local_len(element)
        result += local_VarintSize(l) + l
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      l = local_len(value)
      return tag_size + local_VarintSize(l) + l
    return FieldSize


def GroupSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a group field."""

  tag_size = _TagSize(field_number) * 2
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        result += element.ByteSize()
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      return tag_size + value.ByteSize()
    return FieldSize


def MessageSizer(field_number, is_repeated, is_packed):
  """Returns a sizer for a message field."""

  tag_size = _TagSize(field_number)
  local_VarintSize = _VarintSize
  assert not is_packed
  if is_repeated:
    def RepeatedFieldSize(value):
      result = tag_size * len(value)
      for element in value:
        l = element.ByteSize()
        result += local_VarintSize(l) + l
      return result
    return RepeatedFieldSize
  else:
    def FieldSize(value):
      l = value.ByteSize()
      return tag_size + local_VarintSize(l) + l
    return FieldSize


# --------------------------------------------------------------------
# MessageSet is special: it needs custom logic to compute its size properly.


def MessageSetItemSizer(field_number):
  """Returns a sizer for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
  static_size = (_TagSize(1) * 2 + _TagSize(2) + _VarintSize(field_number) +
                 _TagSize(3))
  local_VarintSize = _VarintSize

  def FieldSize(value):
    l = value.ByteSize()
    return static_size + local_VarintSize(l) + l

  return FieldSize


# --------------------------------------------------------------------
# Map is special: it needs custom logic to compute its size properly.


def MapSizer(field_descriptor, is_message_map):
  """Returns a sizer for a map field."""

  # Can't look at field_descriptor.message_type._concrete_class because it may
  # not have been initialized yet.
  message_type = field_descriptor.message_type
  message_sizer = MessageSizer(field_descriptor.number, False, False)

  def FieldSize(map_value):
    total = 0
    for key in map_value:
      value = map_value[key]
      # It's wasteful to create the messages and throw them away one second
      # later since we'll do the same for the actual encode.  But there's not an
      # obvious way to avoid this within the current design without tons of code
      # duplication. For message map, value.ByteSize() should be called to
      # update the status.
      entry_msg = message_type._concrete_class(key=key, value=value)
      total += message_sizer(entry_msg)
      if is_message_map:
        value.ByteSize()
    return total

  return FieldSize

# ====================================================================
# Encoders!


def _VarintEncoder():
  """Return an encoder for a basic varint value (does not include tag)."""

  local_int2byte = struct.Struct('>B').pack

  def EncodeVarint(write, value, unused_deterministic=None):
    bits = value & 0x7f
    value >>= 7
    while value:
      write(local_int2byte(0x80|bits))
      bits = value & 0x7f
      value >>= 7
    return write(local_int2byte(bits))

  return EncodeVarint


def _SignedVarintEncoder():
  """Return an encoder for a basic signed varint value (does not include
  tag)."""

  local_int2byte = struct.Struct('>B').pack

  def EncodeSignedVarint(write, value, unused_deterministic=None):
    if value < 0:
      value += (1 << 64)
    bits = value & 0x7f
    value >>= 7
    while value:
      write(local_int2byte(0x80|bits))
      bits = value & 0x7f
      value >>= 7
    return write(local_int2byte(bits))

  return EncodeSignedVarint


_EncodeVarint = _VarintEncoder()
_EncodeSignedVarint = _SignedVarintEncoder()


def _VarintBytes(value):
  """Encode the given integer as a varint and return the bytes.  This is only
  called at startup time so it doesn't need to be fast."""

  pieces = []
  _EncodeVarint(pieces.append, value, True)
  return b"".join(pieces)


def TagBytes(field_number, wire_type):
  """Encode the given tag and return the bytes.  Only called at startup."""

  return bytes(_VarintBytes(wire_format.PackTag(field_number, wire_type)))

# --------------------------------------------------------------------
# As with sizers (see above), we have a number of common encoder
# implementations.


def _SimpleEncoder(wire_type, encode_value, compute_value_size):
  """Return a constructor for an encoder for fields of a particular type.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      encode_value:  A function which encodes an individual value, e.g.
        _EncodeVarint().
      compute_value_size:  A function which computes the size of an individual
        value, e.g. _VarintSize().
  """

  def SpecificEncoder(field_number, is_repeated, is_packed):
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        size = 0
        for element in value:
          size += compute_value_size(element)
        local_EncodeVarint(write, size, deterministic)
        for element in value:
          encode_value(write, element, deterministic)
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, deterministic):
        for element in value:
          write(tag_bytes)
          encode_value(write, element, deterministic)
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, deterministic):
        write(tag_bytes)
        return encode_value(write, value, deterministic)
      return EncodeField

  return SpecificEncoder


def _ModifiedEncoder(wire_type, encode_value, compute_value_size, modify_value):
  """Like SimpleEncoder but additionally invokes modify_value on every value
  before passing it to encode_value.  Usually modify_value is ZigZagEncode."""

  def SpecificEncoder(field_number, is_repeated, is_packed):
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        size = 0
        for element in value:
          size += compute_value_size(modify_value(element))
        local_EncodeVarint(write, size, deterministic)
        for element in value:
          encode_value(write, modify_value(element), deterministic)
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, deterministic):
        for element in value:
          write(tag_bytes)
          encode_value(write, modify_value(element), deterministic)
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, deterministic):
        write(tag_bytes)
        return encode_value(write, modify_value(value), deterministic)
      return EncodeField

  return SpecificEncoder


def _StructPackEncoder(wire_type, format):
  """Return a constructor for an encoder for a fixed-width field.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """

  value_size = struct.calcsize(format)

  def SpecificEncoder(field_number, is_repeated, is_packed):
    local_struct_pack = struct.pack
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        local_EncodeVarint(write, len(value) * value_size, deterministic)
        for element in value:
          write(local_struct_pack(format, element))
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, unused_deterministic=None):
        for element in value:
          write(tag_bytes)
          write(local_struct_pack(format, element))
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, unused_deterministic=None):
        write(tag_bytes)
        return write(local_struct_pack(format, value))
      return EncodeField

  return SpecificEncoder


def _FloatingPointEncoder(wire_type, format):
  """Return a constructor for an encoder for float fields.

  This is like StructPackEncoder, but catches errors that may be due to
  passing non-finite floating-point values to struct.pack, and makes a
  second attempt to encode those values.

  Args:
      wire_type:  The field's wire type, for encoding tags.
      format:  The format string to pass to struct.pack().
  """

  value_size = struct.calcsize(format)
  if value_size == 4:
    def EncodeNonFiniteOrRaise(write, value):
      # Remember that the serialized form uses little-endian byte order.
      if value == _POS_INF:
        write(b'\x00\x00\x80\x7F')
      elif value == _NEG_INF:
        write(b'\x00\x00\x80\xFF')
      elif value != value:           # NaN
        write(b'\x00\x00\xC0\x7F')
      else:
        raise
  elif value_size == 8:
    def EncodeNonFiniteOrRaise(write, value):
      if value == _POS_INF:
        write(b'\x00\x00\x00\x00\x00\x00\xF0\x7F')
      elif value == _NEG_INF:
        write(b'\x00\x00\x00\x00\x00\x00\xF0\xFF')
      elif value != value:                         # NaN
        write(b'\x00\x00\x00\x00\x00\x00\xF8\x7F')
      else:
        raise
  else:
    raise ValueError('Can\'t encode floating-point values that are '
                     '%d bytes long (only 4 or 8)' % value_size)

  def SpecificEncoder(field_number, is_repeated, is_packed):
    local_struct_pack = struct.pack
    if is_packed:
      tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
      local_EncodeVarint = _EncodeVarint
      def EncodePackedField(write, value, deterministic):
        write(tag_bytes)
        local_EncodeVarint(write, len(value) * value_size, deterministic)
        for element in value:
          # This try/except block is going to be faster than any code that
          # we could write to check whether element is finite.
          try:
            write(local_struct_pack(format, element))
          except SystemError:
            EncodeNonFiniteOrRaise(write, element)
      return EncodePackedField
    elif is_repeated:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeRepeatedField(write, value, unused_deterministic=None):
        for element in value:
          write(tag_bytes)
          try:
            write(local_struct_pack(format, element))
          except SystemError:
            EncodeNonFiniteOrRaise(write, element)
      return EncodeRepeatedField
    else:
      tag_bytes = TagBytes(field_number, wire_type)
      def EncodeField(write, value, unused_deterministic=None):
        write(tag_bytes)
        try:
          write(local_struct_pack(format, value))
        except SystemError:
          EncodeNonFiniteOrRaise(write, value)
      return EncodeField

  return SpecificEncoder


# ====================================================================
# Here we declare an encoder constructor for each field type.  These work
# very similarly to sizer constructors, described earlier.


Int32Encoder = Int64Encoder = EnumEncoder = _SimpleEncoder(
    wire_format.WIRETYPE_VARINT, _EncodeSignedVarint, _SignedVarintSize)

UInt32Encoder = UInt64Encoder = _SimpleEncoder(
    wire_format.WIRETYPE_VARINT, _EncodeVarint, _VarintSize)

SInt32Encoder = SInt64Encoder = _ModifiedEncoder(
    wire_format.WIRETYPE_VARINT, _EncodeVarint, _VarintSize,
    wire_format.ZigZagEncode)

# Note that Python conveniently guarantees that when using the '<' prefix on
# formats, they will also have the same size across all platforms (as opposed
# to without the prefix, where their sizes depend on the C compiler's basic
# type sizes).
Fixed32Encoder  = _StructPackEncoder(wire_format.WIRETYPE_FIXED32, '<I')
Fixed64Encoder  = _StructPackEncoder(wire_format.WIRETYPE_FIXED64, '<Q')
SFixed32Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED32, '<i')
SFixed64Encoder = _StructPackEncoder(wire_format.WIRETYPE_FIXED64, '<q')
FloatEncoder    = _FloatingPointEncoder(wire_format.WIRETYPE_FIXED32, '<f')
DoubleEncoder   = _FloatingPointEncoder(wire_format.WIRETYPE_FIXED64, '<d')


def BoolEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a boolean field."""

  false_byte = b'\x00'
  true_byte = b'\x01'
  if is_packed:
    tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
    local_EncodeVarint = _EncodeVarint
    def EncodePackedField(write, value, deterministic):
      write(tag_bytes)
      local_EncodeVarint(write, len(value), deterministic)
      for element in value:
        if element:
          write(true_byte)
        else:
          write(false_byte)
    return EncodePackedField
  elif is_repeated:
    tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_VARINT)
    def EncodeRepeatedField(write, value, unused_deterministic=None):
      for element in value:
        write(tag_bytes)
        if element:
          write(true_byte)
        else:
          write(false_byte)
    return EncodeRepeatedField
  else:
    tag_bytes = TagBytes(field_number, wire_format.WIRETYPE_VARINT)
    def EncodeField(write, value, unused_deterministic=None):
      write(tag_bytes)
      if value:
        return write(true_byte)
      return write(false_byte)
    return EncodeField


def StringEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a string field."""

  tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
  local_EncodeVarint = _EncodeVarint
  local_len = len
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        encoded = element.encode('utf-8')
        write(tag)
        local_EncodeVarint(write, local_len(encoded), deterministic)
        write(encoded)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      encoded = value.encode('utf-8')
      write(tag)
      local_EncodeVarint(write, local_len(encoded), deterministic)
      return write(encoded)
    return EncodeField


def BytesEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a bytes field."""

  tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
  local_EncodeVarint = _EncodeVarint
  local_len = len
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        write(tag)
        local_EncodeVarint(write, local_len(element), deterministic)
        write(element)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      write(tag)
      local_EncodeVarint(write, local_len(value), deterministic)
      return write(value)
    return EncodeField


def GroupEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a group field."""

  start_tag = TagBytes(field_number, wire_format.WIRETYPE_START_GROUP)
  end_tag = TagBytes(field_number, wire_format.WIRETYPE_END_GROUP)
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        write(start_tag)
        element._InternalSerialize(write, deterministic)
        write(end_tag)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      write(start_tag)
      value._InternalSerialize(write, deterministic)
      return write(end_tag)
    return EncodeField


def MessageEncoder(field_number, is_repeated, is_packed):
  """Returns an encoder for a message field."""

  tag = TagBytes(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED)
  local_EncodeVarint = _EncodeVarint
  assert not is_packed
  if is_repeated:
    def EncodeRepeatedField(write, value, deterministic):
      for element in value:
        write(tag)
        local_EncodeVarint(write, element.ByteSize(), deterministic)
        element._InternalSerialize(write, deterministic)
    return EncodeRepeatedField
  else:
    def EncodeField(write, value, deterministic):
      write(tag)
      local_EncodeVarint(write, value.ByteSize(), deterministic)
      return value._InternalSerialize(write, deterministic)
    return EncodeField


# --------------------------------------------------------------------
# As before, MessageSet is special.


def MessageSetItemEncoder(field_number):
  """Encoder for extensions of MessageSet.

  The message set message looks like this:
    message MessageSet {
      repeated group Item = 1 {
        required int32 type_id = 2;
        required string message = 3;
      }
    }
  """
  start_bytes = b"".join([
      TagBytes(1, wire_format.WIRETYPE_START_GROUP),
      TagBytes(2, wire_format.WIRETYPE_VARINT),
      _VarintBytes(field_number),
      TagBytes(3, wire_format.WIRETYPE_LENGTH_DELIMITED)])
  end_bytes = TagBytes(1, wire_format.WIRETYPE_END_GROUP)
  local_EncodeVarint = _EncodeVarint

  def EncodeField(write, value, deterministic):
    write(start_bytes)
    local_EncodeVarint(write, value.ByteSize(), deterministic)
    value._InternalSerialize(write, deterministic)
    return write(end_bytes)

  return EncodeField


# --------------------------------------------------------------------
# As before, Map is special.


def MapEncoder(field_descriptor):
  """Encoder for extensions of MessageSet.

  Maps always have a wire format like this:
    message MapEntry {
      key_type key = 1;
      value_type value = 2;
    }
    repeated MapEntry map = N;
  """
  # Can't look at field_descriptor.message_type._concrete_class because it may
  # not have been initialized yet.
  message_type = field_descriptor.message_type
  encode_message = MessageEncoder(field_descriptor.number, False, False)

  def EncodeField(write, value, deterministic):
    value_keys = sorted(value.keys()) if deterministic else value
    for key in value_keys:
      entry_msg = message_type._concrete_class(key=key, value=value[key])
      encode_message(write, entry_msg, deterministic)

  return EncodeField
