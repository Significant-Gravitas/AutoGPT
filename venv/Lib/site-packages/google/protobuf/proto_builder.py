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

"""Dynamic Protobuf class creator."""

from collections import OrderedDict
import hashlib
import os

from google.protobuf import descriptor_pb2
from google.protobuf import descriptor
from google.protobuf import descriptor_pool
from google.protobuf import message_factory


def _GetMessageFromFactory(pool, full_name):
  """Get a proto class from the MessageFactory by name.

  Args:
    pool: a descriptor pool.
    full_name: str, the fully qualified name of the proto type.
  Returns:
    A class, for the type identified by full_name.
  Raises:
    KeyError, if the proto is not found in the factory's descriptor pool.
  """
  proto_descriptor = pool.FindMessageTypeByName(full_name)
  proto_cls = message_factory.GetMessageClass(proto_descriptor)
  return proto_cls


def MakeSimpleProtoClass(fields, full_name=None, pool=None):
  """Create a Protobuf class whose fields are basic types.

  Note: this doesn't validate field names!

  Args:
    fields: dict of {name: field_type} mappings for each field in the proto. If
        this is an OrderedDict the order will be maintained, otherwise the
        fields will be sorted by name.
    full_name: optional str, the fully-qualified name of the proto type.
    pool: optional DescriptorPool instance.
  Returns:
    a class, the new protobuf class with a FileDescriptor.
  """
  pool_instance = pool or descriptor_pool.DescriptorPool()
  if full_name is not None:
    try:
      proto_cls = _GetMessageFromFactory(pool_instance, full_name)
      return proto_cls
    except KeyError:
      # The factory's DescriptorPool doesn't know about this class yet.
      pass

  # Get a list of (name, field_type) tuples from the fields dict. If fields was
  # an OrderedDict we keep the order, but otherwise we sort the field to ensure
  # consistent ordering.
  field_items = fields.items()
  if not isinstance(fields, OrderedDict):
    field_items = sorted(field_items)

  # Use a consistent file name that is unlikely to conflict with any imported
  # proto files.
  fields_hash = hashlib.sha1()
  for f_name, f_type in field_items:
    fields_hash.update(f_name.encode('utf-8'))
    fields_hash.update(str(f_type).encode('utf-8'))
  proto_file_name = fields_hash.hexdigest() + '.proto'

  # If the proto is anonymous, use the same hash to name it.
  if full_name is None:
    full_name = ('net.proto2.python.public.proto_builder.AnonymousProto_' +
                 fields_hash.hexdigest())
    try:
      proto_cls = _GetMessageFromFactory(pool_instance, full_name)
      return proto_cls
    except KeyError:
      # The factory's DescriptorPool doesn't know about this class yet.
      pass

  # This is the first time we see this proto: add a new descriptor to the pool.
  pool_instance.Add(
      _MakeFileDescriptorProto(proto_file_name, full_name, field_items))
  return _GetMessageFromFactory(pool_instance, full_name)


def _MakeFileDescriptorProto(proto_file_name, full_name, field_items):
  """Populate FileDescriptorProto for MessageFactory's DescriptorPool."""
  package, name = full_name.rsplit('.', 1)
  file_proto = descriptor_pb2.FileDescriptorProto()
  file_proto.name = os.path.join(package.replace('.', '/'), proto_file_name)
  file_proto.package = package
  desc_proto = file_proto.message_type.add()
  desc_proto.name = name
  for f_number, (f_name, f_type) in enumerate(field_items, 1):
    field_proto = desc_proto.field.add()
    field_proto.name = f_name
    # # If the number falls in the reserved range, reassign it to the correct
    # # number after the range.
    if f_number >= descriptor.FieldDescriptor.FIRST_RESERVED_FIELD_NUMBER:
      f_number += (
          descriptor.FieldDescriptor.LAST_RESERVED_FIELD_NUMBER -
          descriptor.FieldDescriptor.FIRST_RESERVED_FIELD_NUMBER + 1)
    field_proto.number = f_number
    field_proto.label = descriptor_pb2.FieldDescriptorProto.LABEL_OPTIONAL
    field_proto.type = f_type
  return file_proto
