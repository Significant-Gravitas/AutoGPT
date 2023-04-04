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

"""Contains _ExtensionDict class to represent extensions.
"""

from google.protobuf.internal import type_checkers
from google.protobuf.descriptor import FieldDescriptor


def _VerifyExtensionHandle(message, extension_handle):
  """Verify that the given extension handle is valid."""

  if not isinstance(extension_handle, FieldDescriptor):
    raise KeyError('HasExtension() expects an extension handle, got: %s' %
                   extension_handle)

  if not extension_handle.is_extension:
    raise KeyError('"%s" is not an extension.' % extension_handle.full_name)

  if not extension_handle.containing_type:
    raise KeyError('"%s" is missing a containing_type.'
                   % extension_handle.full_name)

  if extension_handle.containing_type is not message.DESCRIPTOR:
    raise KeyError('Extension "%s" extends message type "%s", but this '
                   'message is of type "%s".' %
                   (extension_handle.full_name,
                    extension_handle.containing_type.full_name,
                    message.DESCRIPTOR.full_name))


# TODO(robinson): Unify error handling of "unknown extension" crap.
# TODO(robinson): Support iteritems()-style iteration over all
# extensions with the "has" bits turned on?
class _ExtensionDict(object):

  """Dict-like container for Extension fields on proto instances.

  Note that in all cases we expect extension handles to be
  FieldDescriptors.
  """

  def __init__(self, extended_message):
    """
    Args:
      extended_message: Message instance for which we are the Extensions dict.
    """
    self._extended_message = extended_message

  def __getitem__(self, extension_handle):
    """Returns the current value of the given extension handle."""

    _VerifyExtensionHandle(self._extended_message, extension_handle)

    result = self._extended_message._fields.get(extension_handle)
    if result is not None:
      return result

    if extension_handle.label == FieldDescriptor.LABEL_REPEATED:
      result = extension_handle._default_constructor(self._extended_message)
    elif extension_handle.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
      message_type = extension_handle.message_type
      if not hasattr(message_type, '_concrete_class'):
        # pylint: disable=g-import-not-at-top
        from google.protobuf import message_factory
        message_factory.GetMessageClass(message_type)
      assert getattr(extension_handle.message_type, '_concrete_class', None), (
          'Uninitialized concrete class found for field %r (message type %r)'
          % (extension_handle.full_name,
             extension_handle.message_type.full_name))
      result = extension_handle.message_type._concrete_class()
      try:
        result._SetListener(self._extended_message._listener_for_children)
      except ReferenceError:
        pass
    else:
      # Singular scalar -- just return the default without inserting into the
      # dict.
      return extension_handle.default_value

    # Atomically check if another thread has preempted us and, if not, swap
    # in the new object we just created.  If someone has preempted us, we
    # take that object and discard ours.
    # WARNING:  We are relying on setdefault() being atomic.  This is true
    #   in CPython but we haven't investigated others.  This warning appears
    #   in several other locations in this file.
    result = self._extended_message._fields.setdefault(
        extension_handle, result)

    return result

  def __eq__(self, other):
    if not isinstance(other, self.__class__):
      return False

    my_fields = self._extended_message.ListFields()
    other_fields = other._extended_message.ListFields()

    # Get rid of non-extension fields.
    my_fields = [field for field in my_fields if field.is_extension]
    other_fields = [field for field in other_fields if field.is_extension]

    return my_fields == other_fields

  def __ne__(self, other):
    return not self == other

  def __len__(self):
    fields = self._extended_message.ListFields()
    # Get rid of non-extension fields.
    extension_fields = [field for field in fields if field[0].is_extension]
    return len(extension_fields)

  def __hash__(self):
    raise TypeError('unhashable object')

  # Note that this is only meaningful for non-repeated, scalar extension
  # fields.  Note also that we may have to call _Modified() when we do
  # successfully set a field this way, to set any necessary "has" bits in the
  # ancestors of the extended message.
  def __setitem__(self, extension_handle, value):
    """If extension_handle specifies a non-repeated, scalar extension
    field, sets the value of that field.
    """

    _VerifyExtensionHandle(self._extended_message, extension_handle)

    if (extension_handle.label == FieldDescriptor.LABEL_REPEATED or
        extension_handle.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE):
      raise TypeError(
          'Cannot assign to extension "%s" because it is a repeated or '
          'composite type.' % extension_handle.full_name)

    # It's slightly wasteful to lookup the type checker each time,
    # but we expect this to be a vanishingly uncommon case anyway.
    type_checker = type_checkers.GetTypeChecker(extension_handle)
    # pylint: disable=protected-access
    self._extended_message._fields[extension_handle] = (
        type_checker.CheckValue(value))
    self._extended_message._Modified()

  def __delitem__(self, extension_handle):
    self._extended_message.ClearExtension(extension_handle)

  def _FindExtensionByName(self, name):
    """Tries to find a known extension with the specified name.

    Args:
      name: Extension full name.

    Returns:
      Extension field descriptor.
    """
    return self._extended_message._extensions_by_name.get(name, None)

  def _FindExtensionByNumber(self, number):
    """Tries to find a known extension with the field number.

    Args:
      number: Extension field number.

    Returns:
      Extension field descriptor.
    """
    return self._extended_message._extensions_by_number.get(number, None)

  def __iter__(self):
    # Return a generator over the populated extension fields
    return (f[0] for f in self._extended_message.ListFields()
            if f[0].is_extension)

  def __contains__(self, extension_handle):
    _VerifyExtensionHandle(self._extended_message, extension_handle)

    if extension_handle not in self._extended_message._fields:
      return False

    if extension_handle.label == FieldDescriptor.LABEL_REPEATED:
      return bool(self._extended_message._fields.get(extension_handle))

    if extension_handle.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
      value = self._extended_message._fields.get(extension_handle)
      # pylint: disable=protected-access
      return value is not None and value._is_present_in_parent

    return True
