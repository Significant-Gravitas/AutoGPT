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

# This code is meant to work on Python 2.4 and above only.

"""Contains a metaclass and helper functions used to create
protocol message classes from Descriptor objects at runtime.

Recall that a metaclass is the "type" of a class.
(A class is to a metaclass what an instance is to a class.)

In this case, we use the GeneratedProtocolMessageType metaclass
to inject all the useful functionality into the classes
output by the protocol compiler at compile-time.

The upshot of all this is that the real implementation
details for ALL pure-Python protocol buffers are *here in
this file*.
"""

__author__ = 'robinson@google.com (Will Robinson)'


from google.protobuf import message_factory
from google.protobuf import symbol_database

# The type of all Message classes.
# Part of the public interface, but normally only used by message factories.
GeneratedProtocolMessageType = message_factory._GENERATED_PROTOCOL_MESSAGE_TYPE

MESSAGE_CLASS_CACHE = {}


# Deprecated. Please NEVER use reflection.ParseMessage().
def ParseMessage(descriptor, byte_str):
  """Generate a new Message instance from this Descriptor and a byte string.

  DEPRECATED: ParseMessage is deprecated because it is using MakeClass().
  Please use MessageFactory.GetPrototype() instead.

  Args:
    descriptor: Protobuf Descriptor object
    byte_str: Serialized protocol buffer byte string

  Returns:
    Newly created protobuf Message object.
  """
  result_class = MakeClass(descriptor)
  new_msg = result_class()
  new_msg.ParseFromString(byte_str)
  return new_msg


# Deprecated. Please NEVER use reflection.MakeClass().
def MakeClass(descriptor):
  """Construct a class object for a protobuf described by descriptor.

  DEPRECATED: use MessageFactory.GetPrototype() instead.

  Args:
    descriptor: A descriptor.Descriptor object describing the protobuf.
  Returns:
    The Message class object described by the descriptor.
  """
  # Original implementation leads to duplicate message classes, which won't play
  # well with extensions. Message factory info is also missing.
  # Redirect to message_factory.
  return message_factory.GetMessageClass(descriptor)
