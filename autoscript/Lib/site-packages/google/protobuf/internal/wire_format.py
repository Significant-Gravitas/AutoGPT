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

"""Constants and static functions to support protocol buffer wire format."""

__author__ = 'robinson@google.com (Will Robinson)'

import struct
from google.protobuf import descriptor
from google.protobuf import message


TAG_TYPE_BITS = 3  # Number of bits used to hold type info in a proto tag.
TAG_TYPE_MASK = (1 << TAG_TYPE_BITS) - 1  # 0x7

# These numbers identify the wire type of a protocol buffer value.
# We use the least-significant TAG_TYPE_BITS bits of the varint-encoded
# tag-and-type to store one of these WIRETYPE_* constants.
# These values must match WireType enum in //google/protobuf/wire_format.h.
WIRETYPE_VARINT = 0
WIRETYPE_FIXED64 = 1
WIRETYPE_LENGTH_DELIMITED = 2
WIRETYPE_START_GROUP = 3
WIRETYPE_END_GROUP = 4
WIRETYPE_FIXED32 = 5
_WIRETYPE_MAX = 5


# Bounds for various integer types.
INT32_MAX = int((1 << 31) - 1)
INT32_MIN = int(-(1 << 31))
UINT32_MAX = (1 << 32) - 1

INT64_MAX = (1 << 63) - 1
INT64_MIN = -(1 << 63)
UINT64_MAX = (1 << 64) - 1

# "struct" format strings that will encode/decode the specified formats.
FORMAT_UINT32_LITTLE_ENDIAN = '<I'
FORMAT_UINT64_LITTLE_ENDIAN = '<Q'
FORMAT_FLOAT_LITTLE_ENDIAN = '<f'
FORMAT_DOUBLE_LITTLE_ENDIAN = '<d'


# We'll have to provide alternate implementations of AppendLittleEndian*() on
# any architectures where these checks fail.
if struct.calcsize(FORMAT_UINT32_LITTLE_ENDIAN) != 4:
  raise AssertionError('Format "I" is not a 32-bit number.')
if struct.calcsize(FORMAT_UINT64_LITTLE_ENDIAN) != 8:
  raise AssertionError('Format "Q" is not a 64-bit number.')


def PackTag(field_number, wire_type):
  """Returns an unsigned 32-bit integer that encodes the field number and
  wire type information in standard protocol message wire format.

  Args:
    field_number: Expected to be an integer in the range [1, 1 << 29)
    wire_type: One of the WIRETYPE_* constants.
  """
  if not 0 <= wire_type <= _WIRETYPE_MAX:
    raise message.EncodeError('Unknown wire type: %d' % wire_type)
  return (field_number << TAG_TYPE_BITS) | wire_type


def UnpackTag(tag):
  """The inverse of PackTag().  Given an unsigned 32-bit number,
  returns a (field_number, wire_type) tuple.
  """
  return (tag >> TAG_TYPE_BITS), (tag & TAG_TYPE_MASK)


def ZigZagEncode(value):
  """ZigZag Transform:  Encodes signed integers so that they can be
  effectively used with varint encoding.  See wire_format.h for
  more details.
  """
  if value >= 0:
    return value << 1
  return (value << 1) ^ (~0)


def ZigZagDecode(value):
  """Inverse of ZigZagEncode()."""
  if not value & 0x1:
    return value >> 1
  return (value >> 1) ^ (~0)



# The *ByteSize() functions below return the number of bytes required to
# serialize "field number + type" information and then serialize the value.


def Int32ByteSize(field_number, int32):
  return Int64ByteSize(field_number, int32)


def Int32ByteSizeNoTag(int32):
  return _VarUInt64ByteSizeNoTag(0xffffffffffffffff & int32)


def Int64ByteSize(field_number, int64):
  # Have to convert to uint before calling UInt64ByteSize().
  return UInt64ByteSize(field_number, 0xffffffffffffffff & int64)


def UInt32ByteSize(field_number, uint32):
  return UInt64ByteSize(field_number, uint32)


def UInt64ByteSize(field_number, uint64):
  return TagByteSize(field_number) + _VarUInt64ByteSizeNoTag(uint64)


def SInt32ByteSize(field_number, int32):
  return UInt32ByteSize(field_number, ZigZagEncode(int32))


def SInt64ByteSize(field_number, int64):
  return UInt64ByteSize(field_number, ZigZagEncode(int64))


def Fixed32ByteSize(field_number, fixed32):
  return TagByteSize(field_number) + 4


def Fixed64ByteSize(field_number, fixed64):
  return TagByteSize(field_number) + 8


def SFixed32ByteSize(field_number, sfixed32):
  return TagByteSize(field_number) + 4


def SFixed64ByteSize(field_number, sfixed64):
  return TagByteSize(field_number) + 8


def FloatByteSize(field_number, flt):
  return TagByteSize(field_number) + 4


def DoubleByteSize(field_number, double):
  return TagByteSize(field_number) + 8


def BoolByteSize(field_number, b):
  return TagByteSize(field_number) + 1


def EnumByteSize(field_number, enum):
  return UInt32ByteSize(field_number, enum)


def StringByteSize(field_number, string):
  return BytesByteSize(field_number, string.encode('utf-8'))


def BytesByteSize(field_number, b):
  return (TagByteSize(field_number)
          + _VarUInt64ByteSizeNoTag(len(b))
          + len(b))


def GroupByteSize(field_number, message):
  return (2 * TagByteSize(field_number)  # START and END group.
          + message.ByteSize())


def MessageByteSize(field_number, message):
  return (TagByteSize(field_number)
          + _VarUInt64ByteSizeNoTag(message.ByteSize())
          + message.ByteSize())


def MessageSetItemByteSize(field_number, msg):
  # First compute the sizes of the tags.
  # There are 2 tags for the beginning and ending of the repeated group, that
  # is field number 1, one with field number 2 (type_id) and one with field
  # number 3 (message).
  total_size = (2 * TagByteSize(1) + TagByteSize(2) + TagByteSize(3))

  # Add the number of bytes for type_id.
  total_size += _VarUInt64ByteSizeNoTag(field_number)

  message_size = msg.ByteSize()

  # The number of bytes for encoding the length of the message.
  total_size += _VarUInt64ByteSizeNoTag(message_size)

  # The size of the message.
  total_size += message_size
  return total_size


def TagByteSize(field_number):
  """Returns the bytes required to serialize a tag with this field number."""
  # Just pass in type 0, since the type won't affect the tag+type size.
  return _VarUInt64ByteSizeNoTag(PackTag(field_number, 0))


# Private helper function for the *ByteSize() functions above.

def _VarUInt64ByteSizeNoTag(uint64):
  """Returns the number of bytes required to serialize a single varint
  using boundary value comparisons. (unrolled loop optimization -WPierce)
  uint64 must be unsigned.
  """
  if uint64 <= 0x7f: return 1
  if uint64 <= 0x3fff: return 2
  if uint64 <= 0x1fffff: return 3
  if uint64 <= 0xfffffff: return 4
  if uint64 <= 0x7ffffffff: return 5
  if uint64 <= 0x3ffffffffff: return 6
  if uint64 <= 0x1ffffffffffff: return 7
  if uint64 <= 0xffffffffffffff: return 8
  if uint64 <= 0x7fffffffffffffff: return 9
  if uint64 > UINT64_MAX:
    raise message.EncodeError('Value out of range: %d' % uint64)
  return 10


NON_PACKABLE_TYPES = (
  descriptor.FieldDescriptor.TYPE_STRING,
  descriptor.FieldDescriptor.TYPE_GROUP,
  descriptor.FieldDescriptor.TYPE_MESSAGE,
  descriptor.FieldDescriptor.TYPE_BYTES
)


def IsTypePackable(field_type):
  """Return true iff packable = true is valid for fields of this type.

  Args:
    field_type: a FieldDescriptor::Type value.

  Returns:
    True iff fields of this type are packable.
  """
  return field_type not in NON_PACKABLE_TYPES
