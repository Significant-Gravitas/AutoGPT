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

"""Contains Unknown Fields APIs.

Simple usage example:
  unknown_field_set = UnknownFieldSet(message)
  for unknown_field in unknown_field_set:
    wire_type = unknown_field.wire_type
    field_number = unknown_field.field_number
    data = unknown_field.data
"""


from google.protobuf.internal import api_implementation

if api_implementation._c_module is not None:  # pylint: disable=protected-access
  UnknownFieldSet = api_implementation._c_module.UnknownFieldSet  # pylint: disable=protected-access
else:
  from google.protobuf.internal import decoder  # pylint: disable=g-import-not-at-top
  from google.protobuf.internal import wire_format  # pylint: disable=g-import-not-at-top

  class UnknownField:
    """A parsed unknown field."""

    # Disallows assignment to other attributes.
    __slots__ = ['_field_number', '_wire_type', '_data']

    def __init__(self, field_number, wire_type, data):
      self._field_number = field_number
      self._wire_type = wire_type
      self._data = data
      return

    @property
    def field_number(self):
      return self._field_number

    @property
    def wire_type(self):
      return self._wire_type

    @property
    def data(self):
      return self._data

  class UnknownFieldSet:
    """UnknownField container."""

    # Disallows assignment to other attributes.
    __slots__ = ['_values']

    def __init__(self, msg):

      def InternalAdd(field_number, wire_type, data):
        unknown_field = UnknownField(field_number, wire_type, data)
        self._values.append(unknown_field)

      self._values = []
      msg_des = msg.DESCRIPTOR
      # pylint: disable=protected-access
      unknown_fields = msg._unknown_fields
      if (msg_des.has_options and
          msg_des.GetOptions().message_set_wire_format):
        local_decoder = decoder.UnknownMessageSetItemDecoder()
        for _, buffer in unknown_fields:
          (field_number, data) = local_decoder(memoryview(buffer))
          InternalAdd(field_number, wire_format.WIRETYPE_LENGTH_DELIMITED, data)
      else:
        for tag_bytes, buffer in unknown_fields:
          # pylint: disable=protected-access
          (tag, _) = decoder._DecodeVarint(tag_bytes, 0)
          field_number, wire_type = wire_format.UnpackTag(tag)
          if field_number == 0:
            raise RuntimeError('Field number 0 is illegal.')
          (data, _) = decoder._DecodeUnknownField(
              memoryview(buffer), 0, wire_type)
          InternalAdd(field_number, wire_type, data)

    def __getitem__(self, index):
      size = len(self._values)
      if index < 0:
        index += size
      if index < 0 or index >= size:
        raise IndexError('index %d out of range'.index)

      return self._values[index]

    def __len__(self):
      return len(self._values)

    def __iter__(self):
      return iter(self._values)
