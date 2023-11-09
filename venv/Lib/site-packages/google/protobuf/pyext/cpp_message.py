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

"""Protocol message implementation hooks for C++ implementation.

Contains helper functions used to create protocol message classes from
Descriptor objects at runtime backed by the protocol buffer C++ API.
"""

__author__ = 'tibell@google.com (Johan Tibell)'

from google.protobuf.internal import api_implementation


# pylint: disable=protected-access
_message = api_implementation._c_module
# TODO(jieluo): Remove this import after fix api_implementation
if _message is None:
  from google.protobuf.pyext import _message


class GeneratedProtocolMessageType(_message.MessageMeta):

  """Metaclass for protocol message classes created at runtime from Descriptors.

  The protocol compiler currently uses this metaclass to create protocol
  message classes at runtime.  Clients can also manually create their own
  classes at runtime, as in this example:

  mydescriptor = Descriptor(.....)
  factory = symbol_database.Default()
  factory.pool.AddDescriptor(mydescriptor)
  MyProtoClass = factory.GetPrototype(mydescriptor)
  myproto_instance = MyProtoClass()
  myproto.foo_field = 23
  ...

  The above example will not work for nested types. If you wish to include them,
  use reflection.MakeClass() instead of manually instantiating the class in
  order to create the appropriate class structure.
  """

  # Must be consistent with the protocol-compiler code in
  # proto2/compiler/internal/generator.*.
  _DESCRIPTOR_KEY = 'DESCRIPTOR'
