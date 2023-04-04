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
#
# TODO(robinson): Helpers for verbose, common checks like seeing if a
# descriptor's cpp_type is CPPTYPE_MESSAGE.

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

from io import BytesIO
import struct
import sys
import weakref

# We use "as" to avoid name collisions with variables.
from google.protobuf.internal import api_implementation
from google.protobuf.internal import containers
from google.protobuf.internal import decoder
from google.protobuf.internal import encoder
from google.protobuf.internal import enum_type_wrapper
from google.protobuf.internal import extension_dict
from google.protobuf.internal import message_listener as message_listener_mod
from google.protobuf.internal import type_checkers
from google.protobuf.internal import well_known_types
from google.protobuf.internal import wire_format
from google.protobuf import descriptor as descriptor_mod
from google.protobuf import message as message_mod
from google.protobuf import text_format

_FieldDescriptor = descriptor_mod.FieldDescriptor
_AnyFullTypeName = 'google.protobuf.Any'
_ExtensionDict = extension_dict._ExtensionDict

class GeneratedProtocolMessageType(type):

  """Metaclass for protocol message classes created at runtime from Descriptors.

  We add implementations for all methods described in the Message class.  We
  also create properties to allow getting/setting all fields in the protocol
  message.  Finally, we create slots to prevent users from accidentally
  "setting" nonexistent fields in the protocol message, which then wouldn't get
  serialized / deserialized properly.

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
  """

  # Must be consistent with the protocol-compiler code in
  # proto2/compiler/internal/generator.*.
  _DESCRIPTOR_KEY = 'DESCRIPTOR'

  def __new__(cls, name, bases, dictionary):
    """Custom allocation for runtime-generated class types.

    We override __new__ because this is apparently the only place
    where we can meaningfully set __slots__ on the class we're creating(?).
    (The interplay between metaclasses and slots is not very well-documented).

    Args:
      name: Name of the class (ignored, but required by the
        metaclass protocol).
      bases: Base classes of the class we're constructing.
        (Should be message.Message).  We ignore this field, but
        it's required by the metaclass protocol
      dictionary: The class dictionary of the class we're
        constructing.  dictionary[_DESCRIPTOR_KEY] must contain
        a Descriptor object describing this protocol message
        type.

    Returns:
      Newly-allocated class.

    Raises:
      RuntimeError: Generated code only work with python cpp extension.
    """
    descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]

    if isinstance(descriptor, str):
      raise RuntimeError('The generated code only work with python cpp '
                         'extension, but it is using pure python runtime.')

    # If a concrete class already exists for this descriptor, don't try to
    # create another.  Doing so will break any messages that already exist with
    # the existing class.
    #
    # The C++ implementation appears to have its own internal `PyMessageFactory`
    # to achieve similar results.
    #
    # This most commonly happens in `text_format.py` when using descriptors from
    # a custom pool; it calls symbol_database.Global().getPrototype() on a
    # descriptor which already has an existing concrete class.
    new_class = getattr(descriptor, '_concrete_class', None)
    if new_class:
      return new_class

    if descriptor.full_name in well_known_types.WKTBASES:
      bases += (well_known_types.WKTBASES[descriptor.full_name],)
    _AddClassAttributesForNestedExtensions(descriptor, dictionary)
    _AddSlots(descriptor, dictionary)

    superclass = super(GeneratedProtocolMessageType, cls)
    new_class = superclass.__new__(cls, name, bases, dictionary)
    return new_class

  def __init__(cls, name, bases, dictionary):
    """Here we perform the majority of our work on the class.
    We add enum getters, an __init__ method, implementations
    of all Message methods, and properties for all fields
    in the protocol type.

    Args:
      name: Name of the class (ignored, but required by the
        metaclass protocol).
      bases: Base classes of the class we're constructing.
        (Should be message.Message).  We ignore this field, but
        it's required by the metaclass protocol
      dictionary: The class dictionary of the class we're
        constructing.  dictionary[_DESCRIPTOR_KEY] must contain
        a Descriptor object describing this protocol message
        type.
    """
    descriptor = dictionary[GeneratedProtocolMessageType._DESCRIPTOR_KEY]

    # If this is an _existing_ class looked up via `_concrete_class` in the
    # __new__ method above, then we don't need to re-initialize anything.
    existing_class = getattr(descriptor, '_concrete_class', None)
    if existing_class:
      assert existing_class is cls, (
          'Duplicate `GeneratedProtocolMessageType` created for descriptor %r'
          % (descriptor.full_name))
      return

    cls._decoders_by_tag = {}
    if (descriptor.has_options and
        descriptor.GetOptions().message_set_wire_format):
      cls._decoders_by_tag[decoder.MESSAGE_SET_ITEM_TAG] = (
          decoder.MessageSetItemDecoder(descriptor), None)

    # Attach stuff to each FieldDescriptor for quick lookup later on.
    for field in descriptor.fields:
      _AttachFieldHelpers(cls, field)

    descriptor._concrete_class = cls  # pylint: disable=protected-access
    _AddEnumValues(descriptor, cls)
    _AddInitMethod(descriptor, cls)
    _AddPropertiesForFields(descriptor, cls)
    _AddPropertiesForExtensions(descriptor, cls)
    _AddStaticMethods(cls)
    _AddMessageMethods(descriptor, cls)
    _AddPrivateHelperMethods(descriptor, cls)

    superclass = super(GeneratedProtocolMessageType, cls)
    superclass.__init__(name, bases, dictionary)


# Stateless helpers for GeneratedProtocolMessageType below.
# Outside clients should not access these directly.
#
# I opted not to make any of these methods on the metaclass, to make it more
# clear that I'm not really using any state there and to keep clients from
# thinking that they have direct access to these construction helpers.


def _PropertyName(proto_field_name):
  """Returns the name of the public property attribute which
  clients can use to get and (in some cases) set the value
  of a protocol message field.

  Args:
    proto_field_name: The protocol message field name, exactly
      as it appears (or would appear) in a .proto file.
  """
  # TODO(robinson): Escape Python keywords (e.g., yield), and test this support.
  # nnorwitz makes my day by writing:
  # """
  # FYI.  See the keyword module in the stdlib. This could be as simple as:
  #
  # if keyword.iskeyword(proto_field_name):
  #   return proto_field_name + "_"
  # return proto_field_name
  # """
  # Kenton says:  The above is a BAD IDEA.  People rely on being able to use
  #   getattr() and setattr() to reflectively manipulate field values.  If we
  #   rename the properties, then every such user has to also make sure to apply
  #   the same transformation.  Note that currently if you name a field "yield",
  #   you can still access it just fine using getattr/setattr -- it's not even
  #   that cumbersome to do so.
  # TODO(kenton):  Remove this method entirely if/when everyone agrees with my
  #   position.
  return proto_field_name


def _AddSlots(message_descriptor, dictionary):
  """Adds a __slots__ entry to dictionary, containing the names of all valid
  attributes for this message type.

  Args:
    message_descriptor: A Descriptor instance describing this message type.
    dictionary: Class dictionary to which we'll add a '__slots__' entry.
  """
  dictionary['__slots__'] = ['_cached_byte_size',
                             '_cached_byte_size_dirty',
                             '_fields',
                             '_unknown_fields',
                             '_unknown_field_set',
                             '_is_present_in_parent',
                             '_listener',
                             '_listener_for_children',
                             '__weakref__',
                             '_oneofs']


def _IsMessageSetExtension(field):
  return (field.is_extension and
          field.containing_type.has_options and
          field.containing_type.GetOptions().message_set_wire_format and
          field.type == _FieldDescriptor.TYPE_MESSAGE and
          field.label == _FieldDescriptor.LABEL_OPTIONAL)


def _IsMapField(field):
  return (field.type == _FieldDescriptor.TYPE_MESSAGE and
          field.message_type.has_options and
          field.message_type.GetOptions().map_entry)


def _IsMessageMapField(field):
  value_type = field.message_type.fields_by_name['value']
  return value_type.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE


def _AttachFieldHelpers(cls, field_descriptor):
  is_repeated = (field_descriptor.label == _FieldDescriptor.LABEL_REPEATED)
  is_map_entry = _IsMapField(field_descriptor)
  is_packed = field_descriptor.is_packed

  if is_map_entry:
    field_encoder = encoder.MapEncoder(field_descriptor)
    sizer = encoder.MapSizer(field_descriptor,
                             _IsMessageMapField(field_descriptor))
  elif _IsMessageSetExtension(field_descriptor):
    field_encoder = encoder.MessageSetItemEncoder(field_descriptor.number)
    sizer = encoder.MessageSetItemSizer(field_descriptor.number)
  else:
    field_encoder = type_checkers.TYPE_TO_ENCODER[field_descriptor.type](
        field_descriptor.number, is_repeated, is_packed)
    sizer = type_checkers.TYPE_TO_SIZER[field_descriptor.type](
        field_descriptor.number, is_repeated, is_packed)

  field_descriptor._encoder = field_encoder
  field_descriptor._sizer = sizer
  field_descriptor._default_constructor = _DefaultValueConstructorForField(
      field_descriptor)

  def AddDecoder(wiretype, is_packed):
    tag_bytes = encoder.TagBytes(field_descriptor.number, wiretype)
    decode_type = field_descriptor.type
    if (decode_type == _FieldDescriptor.TYPE_ENUM and
        not field_descriptor.enum_type.is_closed):
      decode_type = _FieldDescriptor.TYPE_INT32

    oneof_descriptor = None
    if field_descriptor.containing_oneof is not None:
      oneof_descriptor = field_descriptor

    if is_map_entry:
      is_message_map = _IsMessageMapField(field_descriptor)

      field_decoder = decoder.MapDecoder(
          field_descriptor, _GetInitializeDefaultForMap(field_descriptor),
          is_message_map)
    elif decode_type == _FieldDescriptor.TYPE_STRING:
      field_decoder = decoder.StringDecoder(
          field_descriptor.number, is_repeated, is_packed,
          field_descriptor, field_descriptor._default_constructor,
          not field_descriptor.has_presence)
    elif field_descriptor.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
      field_decoder = type_checkers.TYPE_TO_DECODER[decode_type](
          field_descriptor.number, is_repeated, is_packed,
          field_descriptor, field_descriptor._default_constructor)
    else:
      field_decoder = type_checkers.TYPE_TO_DECODER[decode_type](
          field_descriptor.number, is_repeated, is_packed,
          # pylint: disable=protected-access
          field_descriptor, field_descriptor._default_constructor,
          not field_descriptor.has_presence)

    cls._decoders_by_tag[tag_bytes] = (field_decoder, oneof_descriptor)

  AddDecoder(type_checkers.FIELD_TYPE_TO_WIRE_TYPE[field_descriptor.type],
             False)

  if is_repeated and wire_format.IsTypePackable(field_descriptor.type):
    # To support wire compatibility of adding packed = true, add a decoder for
    # packed values regardless of the field's options.
    AddDecoder(wire_format.WIRETYPE_LENGTH_DELIMITED, True)


def _AddClassAttributesForNestedExtensions(descriptor, dictionary):
  extensions = descriptor.extensions_by_name
  for extension_name, extension_field in extensions.items():
    assert extension_name not in dictionary
    dictionary[extension_name] = extension_field


def _AddEnumValues(descriptor, cls):
  """Sets class-level attributes for all enum fields defined in this message.

  Also exporting a class-level object that can name enum values.

  Args:
    descriptor: Descriptor object for this message type.
    cls: Class we're constructing for this message type.
  """
  for enum_type in descriptor.enum_types:
    setattr(cls, enum_type.name, enum_type_wrapper.EnumTypeWrapper(enum_type))
    for enum_value in enum_type.values:
      setattr(cls, enum_value.name, enum_value.number)


def _GetInitializeDefaultForMap(field):
  if field.label != _FieldDescriptor.LABEL_REPEATED:
    raise ValueError('map_entry set on non-repeated field %s' % (
        field.name))
  fields_by_name = field.message_type.fields_by_name
  key_checker = type_checkers.GetTypeChecker(fields_by_name['key'])

  value_field = fields_by_name['value']
  if _IsMessageMapField(field):
    def MakeMessageMapDefault(message):
      return containers.MessageMap(
          message._listener_for_children, value_field.message_type, key_checker,
          field.message_type)
    return MakeMessageMapDefault
  else:
    value_checker = type_checkers.GetTypeChecker(value_field)
    def MakePrimitiveMapDefault(message):
      return containers.ScalarMap(
          message._listener_for_children, key_checker, value_checker,
          field.message_type)
    return MakePrimitiveMapDefault

def _DefaultValueConstructorForField(field):
  """Returns a function which returns a default value for a field.

  Args:
    field: FieldDescriptor object for this field.

  The returned function has one argument:
    message: Message instance containing this field, or a weakref proxy
      of same.

  That function in turn returns a default value for this field.  The default
    value may refer back to |message| via a weak reference.
  """

  if _IsMapField(field):
    return _GetInitializeDefaultForMap(field)

  if field.label == _FieldDescriptor.LABEL_REPEATED:
    if field.has_default_value and field.default_value != []:
      raise ValueError('Repeated field default value not empty list: %s' % (
          field.default_value))
    if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
      # We can't look at _concrete_class yet since it might not have
      # been set.  (Depends on order in which we initialize the classes).
      message_type = field.message_type
      def MakeRepeatedMessageDefault(message):
        return containers.RepeatedCompositeFieldContainer(
            message._listener_for_children, field.message_type)
      return MakeRepeatedMessageDefault
    else:
      type_checker = type_checkers.GetTypeChecker(field)
      def MakeRepeatedScalarDefault(message):
        return containers.RepeatedScalarFieldContainer(
            message._listener_for_children, type_checker)
      return MakeRepeatedScalarDefault

  if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
    # _concrete_class may not yet be initialized.
    message_type = field.message_type
    def MakeSubMessageDefault(message):
      assert getattr(message_type, '_concrete_class', None), (
          'Uninitialized concrete class found for field %r (message type %r)'
          % (field.full_name, message_type.full_name))
      result = message_type._concrete_class()
      result._SetListener(
          _OneofListener(message, field)
          if field.containing_oneof is not None
          else message._listener_for_children)
      return result
    return MakeSubMessageDefault

  def MakeScalarDefault(message):
    # TODO(protobuf-team): This may be broken since there may not be
    # default_value.  Combine with has_default_value somehow.
    return field.default_value
  return MakeScalarDefault


def _ReraiseTypeErrorWithFieldName(message_name, field_name):
  """Re-raise the currently-handled TypeError with the field name added."""
  exc = sys.exc_info()[1]
  if len(exc.args) == 1 and type(exc) is TypeError:
    # simple TypeError; add field name to exception message
    exc = TypeError('%s for field %s.%s' % (str(exc), message_name, field_name))

  # re-raise possibly-amended exception with original traceback:
  raise exc.with_traceback(sys.exc_info()[2])


def _AddInitMethod(message_descriptor, cls):
  """Adds an __init__ method to cls."""

  def _GetIntegerEnumValue(enum_type, value):
    """Convert a string or integer enum value to an integer.

    If the value is a string, it is converted to the enum value in
    enum_type with the same name.  If the value is not a string, it's
    returned as-is.  (No conversion or bounds-checking is done.)
    """
    if isinstance(value, str):
      try:
        return enum_type.values_by_name[value].number
      except KeyError:
        raise ValueError('Enum type %s: unknown label "%s"' % (
            enum_type.full_name, value))
    return value

  def init(self, **kwargs):
    self._cached_byte_size = 0
    self._cached_byte_size_dirty = len(kwargs) > 0
    self._fields = {}
    # Contains a mapping from oneof field descriptors to the descriptor
    # of the currently set field in that oneof field.
    self._oneofs = {}

    # _unknown_fields is () when empty for efficiency, and will be turned into
    # a list if fields are added.
    self._unknown_fields = ()
    # _unknown_field_set is None when empty for efficiency, and will be
    # turned into UnknownFieldSet struct if fields are added.
    self._unknown_field_set = None      # pylint: disable=protected-access
    self._is_present_in_parent = False
    self._listener = message_listener_mod.NullMessageListener()
    self._listener_for_children = _Listener(self)
    for field_name, field_value in kwargs.items():
      field = _GetFieldByName(message_descriptor, field_name)
      if field is None:
        raise TypeError('%s() got an unexpected keyword argument "%s"' %
                        (message_descriptor.name, field_name))
      if field_value is None:
        # field=None is the same as no field at all.
        continue
      if field.label == _FieldDescriptor.LABEL_REPEATED:
        copy = field._default_constructor(self)
        if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:  # Composite
          if _IsMapField(field):
            if _IsMessageMapField(field):
              for key in field_value:
                copy[key].MergeFrom(field_value[key])
            else:
              copy.update(field_value)
          else:
            for val in field_value:
              if isinstance(val, dict):
                copy.add(**val)
              else:
                copy.add().MergeFrom(val)
        else:  # Scalar
          if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
            field_value = [_GetIntegerEnumValue(field.enum_type, val)
                           for val in field_value]
          copy.extend(field_value)
        self._fields[field] = copy
      elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        copy = field._default_constructor(self)
        new_val = field_value
        if isinstance(field_value, dict):
          new_val = field.message_type._concrete_class(**field_value)
        try:
          copy.MergeFrom(new_val)
        except TypeError:
          _ReraiseTypeErrorWithFieldName(message_descriptor.name, field_name)
        self._fields[field] = copy
      else:
        if field.cpp_type == _FieldDescriptor.CPPTYPE_ENUM:
          field_value = _GetIntegerEnumValue(field.enum_type, field_value)
        try:
          setattr(self, field_name, field_value)
        except TypeError:
          _ReraiseTypeErrorWithFieldName(message_descriptor.name, field_name)

  init.__module__ = None
  init.__doc__ = None
  cls.__init__ = init


def _GetFieldByName(message_descriptor, field_name):
  """Returns a field descriptor by field name.

  Args:
    message_descriptor: A Descriptor describing all fields in message.
    field_name: The name of the field to retrieve.
  Returns:
    The field descriptor associated with the field name.
  """
  try:
    return message_descriptor.fields_by_name[field_name]
  except KeyError:
    raise ValueError('Protocol message %s has no "%s" field.' %
                     (message_descriptor.name, field_name))


def _AddPropertiesForFields(descriptor, cls):
  """Adds properties for all fields in this protocol message type."""
  for field in descriptor.fields:
    _AddPropertiesForField(field, cls)

  if descriptor.is_extendable:
    # _ExtensionDict is just an adaptor with no state so we allocate a new one
    # every time it is accessed.
    cls.Extensions = property(lambda self: _ExtensionDict(self))


def _AddPropertiesForField(field, cls):
  """Adds a public property for a protocol message field.
  Clients can use this property to get and (in the case
  of non-repeated scalar fields) directly set the value
  of a protocol message field.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
  # Catch it if we add other types that we should
  # handle specially here.
  assert _FieldDescriptor.MAX_CPPTYPE == 10

  constant_name = field.name.upper() + '_FIELD_NUMBER'
  setattr(cls, constant_name, field.number)

  if field.label == _FieldDescriptor.LABEL_REPEATED:
    _AddPropertiesForRepeatedField(field, cls)
  elif field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
    _AddPropertiesForNonRepeatedCompositeField(field, cls)
  else:
    _AddPropertiesForNonRepeatedScalarField(field, cls)


class _FieldProperty(property):
  __slots__ = ('DESCRIPTOR',)

  def __init__(self, descriptor, getter, setter, doc):
    property.__init__(self, getter, setter, doc=doc)
    self.DESCRIPTOR = descriptor


def _AddPropertiesForRepeatedField(field, cls):
  """Adds a public property for a "repeated" protocol message field.  Clients
  can use this property to get the value of the field, which will be either a
  RepeatedScalarFieldContainer or RepeatedCompositeFieldContainer (see
  below).

  Note that when clients add values to these containers, we perform
  type-checking in the case of repeated scalar fields, and we also set any
  necessary "has" bits as a side-effect.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
  proto_field_name = field.name
  property_name = _PropertyName(proto_field_name)

  def getter(self):
    field_value = self._fields.get(field)
    if field_value is None:
      # Construct a new object to represent this field.
      field_value = field._default_constructor(self)

      # Atomically check if another thread has preempted us and, if not, swap
      # in the new object we just created.  If someone has preempted us, we
      # take that object and discard ours.
      # WARNING:  We are relying on setdefault() being atomic.  This is true
      #   in CPython but we haven't investigated others.  This warning appears
      #   in several other locations in this file.
      field_value = self._fields.setdefault(field, field_value)
    return field_value
  getter.__module__ = None
  getter.__doc__ = 'Getter for %s.' % proto_field_name

  # We define a setter just so we can throw an exception with a more
  # helpful error message.
  def setter(self, new_value):
    raise AttributeError('Assignment not allowed to repeated field '
                         '"%s" in protocol message object.' % proto_field_name)

  doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
  setattr(cls, property_name, _FieldProperty(field, getter, setter, doc=doc))


def _AddPropertiesForNonRepeatedScalarField(field, cls):
  """Adds a public property for a nonrepeated, scalar protocol message field.
  Clients can use this property to get and directly set the value of the field.
  Note that when the client sets the value of a field by using this property,
  all necessary "has" bits are set as a side-effect, and we also perform
  type-checking.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
  proto_field_name = field.name
  property_name = _PropertyName(proto_field_name)
  type_checker = type_checkers.GetTypeChecker(field)
  default_value = field.default_value

  def getter(self):
    # TODO(protobuf-team): This may be broken since there may not be
    # default_value.  Combine with has_default_value somehow.
    return self._fields.get(field, default_value)
  getter.__module__ = None
  getter.__doc__ = 'Getter for %s.' % proto_field_name

  def field_setter(self, new_value):
    # pylint: disable=protected-access
    # Testing the value for truthiness captures all of the proto3 defaults
    # (0, 0.0, enum 0, and False).
    try:
      new_value = type_checker.CheckValue(new_value)
    except TypeError as e:
      raise TypeError(
          'Cannot set %s to %.1024r: %s' % (field.full_name, new_value, e))
    if not field.has_presence and not new_value:
      self._fields.pop(field, None)
    else:
      self._fields[field] = new_value
    # Check _cached_byte_size_dirty inline to improve performance, since scalar
    # setters are called frequently.
    if not self._cached_byte_size_dirty:
      self._Modified()

  if field.containing_oneof:
    def setter(self, new_value):
      field_setter(self, new_value)
      self._UpdateOneofState(field)
  else:
    setter = field_setter

  setter.__module__ = None
  setter.__doc__ = 'Setter for %s.' % proto_field_name

  # Add a property to encapsulate the getter/setter.
  doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
  setattr(cls, property_name, _FieldProperty(field, getter, setter, doc=doc))


def _AddPropertiesForNonRepeatedCompositeField(field, cls):
  """Adds a public property for a nonrepeated, composite protocol message field.
  A composite field is a "group" or "message" field.

  Clients can use this property to get the value of the field, but cannot
  assign to the property directly.

  Args:
    field: A FieldDescriptor for this field.
    cls: The class we're constructing.
  """
  # TODO(robinson): Remove duplication with similar method
  # for non-repeated scalars.
  proto_field_name = field.name
  property_name = _PropertyName(proto_field_name)

  def getter(self):
    field_value = self._fields.get(field)
    if field_value is None:
      # Construct a new object to represent this field.
      field_value = field._default_constructor(self)

      # Atomically check if another thread has preempted us and, if not, swap
      # in the new object we just created.  If someone has preempted us, we
      # take that object and discard ours.
      # WARNING:  We are relying on setdefault() being atomic.  This is true
      #   in CPython but we haven't investigated others.  This warning appears
      #   in several other locations in this file.
      field_value = self._fields.setdefault(field, field_value)
    return field_value
  getter.__module__ = None
  getter.__doc__ = 'Getter for %s.' % proto_field_name

  # We define a setter just so we can throw an exception with a more
  # helpful error message.
  def setter(self, new_value):
    raise AttributeError('Assignment not allowed to composite field '
                         '"%s" in protocol message object.' % proto_field_name)

  # Add a property to encapsulate the getter.
  doc = 'Magic attribute generated for "%s" proto field.' % proto_field_name
  setattr(cls, property_name, _FieldProperty(field, getter, setter, doc=doc))


def _AddPropertiesForExtensions(descriptor, cls):
  """Adds properties for all fields in this protocol message type."""
  extensions = descriptor.extensions_by_name
  for extension_name, extension_field in extensions.items():
    constant_name = extension_name.upper() + '_FIELD_NUMBER'
    setattr(cls, constant_name, extension_field.number)

  # TODO(amauryfa): Migrate all users of these attributes to functions like
  #   pool.FindExtensionByNumber(descriptor).
  if descriptor.file is not None:
    # TODO(amauryfa): Use cls.MESSAGE_FACTORY.pool when available.
    pool = descriptor.file.pool
    cls._extensions_by_number = pool._extensions_by_number[descriptor]
    cls._extensions_by_name = pool._extensions_by_name[descriptor]

def _AddStaticMethods(cls):
  # TODO(robinson): This probably needs to be thread-safe(?)
  def RegisterExtension(field_descriptor):
    field_descriptor.containing_type = cls.DESCRIPTOR
    # TODO(amauryfa): Use cls.MESSAGE_FACTORY.pool when available.
    # pylint: disable=protected-access
    cls.DESCRIPTOR.file.pool._AddExtensionDescriptor(field_descriptor)
    _AttachFieldHelpers(cls, field_descriptor)
  cls.RegisterExtension = staticmethod(RegisterExtension)

  def FromString(s):
    message = cls()
    message.MergeFromString(s)
    return message
  cls.FromString = staticmethod(FromString)


def _IsPresent(item):
  """Given a (FieldDescriptor, value) tuple from _fields, return true if the
  value should be included in the list returned by ListFields()."""

  if item[0].label == _FieldDescriptor.LABEL_REPEATED:
    return bool(item[1])
  elif item[0].cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
    return item[1]._is_present_in_parent
  else:
    return True


def _AddListFieldsMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""

  def ListFields(self):
    all_fields = [item for item in self._fields.items() if _IsPresent(item)]
    all_fields.sort(key = lambda item: item[0].number)
    return all_fields

  cls.ListFields = ListFields


def _AddHasFieldMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""

  hassable_fields = {}
  for field in message_descriptor.fields:
    if field.label == _FieldDescriptor.LABEL_REPEATED:
      continue
    # For proto3, only submessages and fields inside a oneof have presence.
    if not field.has_presence:
      continue
    hassable_fields[field.name] = field

  # Has methods are supported for oneof descriptors.
  for oneof in message_descriptor.oneofs:
    hassable_fields[oneof.name] = oneof

  def HasField(self, field_name):
    try:
      field = hassable_fields[field_name]
    except KeyError as exc:
      raise ValueError('Protocol message %s has no non-repeated field "%s" '
                       'nor has presence is not available for this field.' % (
                           message_descriptor.full_name, field_name)) from exc

    if isinstance(field, descriptor_mod.OneofDescriptor):
      try:
        return HasField(self, self._oneofs[field].name)
      except KeyError:
        return False
    else:
      if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        value = self._fields.get(field)
        return value is not None and value._is_present_in_parent
      else:
        return field in self._fields

  cls.HasField = HasField


def _AddClearFieldMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""
  def ClearField(self, field_name):
    try:
      field = message_descriptor.fields_by_name[field_name]
    except KeyError:
      try:
        field = message_descriptor.oneofs_by_name[field_name]
        if field in self._oneofs:
          field = self._oneofs[field]
        else:
          return
      except KeyError:
        raise ValueError('Protocol message %s has no "%s" field.' %
                         (message_descriptor.name, field_name))

    if field in self._fields:
      # To match the C++ implementation, we need to invalidate iterators
      # for map fields when ClearField() happens.
      if hasattr(self._fields[field], 'InvalidateIterators'):
        self._fields[field].InvalidateIterators()

      # Note:  If the field is a sub-message, its listener will still point
      #   at us.  That's fine, because the worst than can happen is that it
      #   will call _Modified() and invalidate our byte size.  Big deal.
      del self._fields[field]

      if self._oneofs.get(field.containing_oneof, None) is field:
        del self._oneofs[field.containing_oneof]

    # Always call _Modified() -- even if nothing was changed, this is
    # a mutating method, and thus calling it should cause the field to become
    # present in the parent message.
    self._Modified()

  cls.ClearField = ClearField


def _AddClearExtensionMethod(cls):
  """Helper for _AddMessageMethods()."""
  def ClearExtension(self, field_descriptor):
    extension_dict._VerifyExtensionHandle(self, field_descriptor)

    # Similar to ClearField(), above.
    if field_descriptor in self._fields:
      del self._fields[field_descriptor]
    self._Modified()
  cls.ClearExtension = ClearExtension


def _AddHasExtensionMethod(cls):
  """Helper for _AddMessageMethods()."""
  def HasExtension(self, field_descriptor):
    extension_dict._VerifyExtensionHandle(self, field_descriptor)
    if field_descriptor.label == _FieldDescriptor.LABEL_REPEATED:
      raise KeyError('"%s" is repeated.' % field_descriptor.full_name)

    if field_descriptor.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
      value = self._fields.get(field_descriptor)
      return value is not None and value._is_present_in_parent
    else:
      return field_descriptor in self._fields
  cls.HasExtension = HasExtension

def _InternalUnpackAny(msg):
  """Unpacks Any message and returns the unpacked message.

  This internal method is different from public Any Unpack method which takes
  the target message as argument. _InternalUnpackAny method does not have
  target message type and need to find the message type in descriptor pool.

  Args:
    msg: An Any message to be unpacked.

  Returns:
    The unpacked message.
  """
  # TODO(amauryfa): Don't use the factory of generated messages.
  # To make Any work with custom factories, use the message factory of the
  # parent message.
  # pylint: disable=g-import-not-at-top
  from google.protobuf import symbol_database
  factory = symbol_database.Default()

  type_url = msg.type_url

  if not type_url:
    return None

  # TODO(haberman): For now we just strip the hostname.  Better logic will be
  # required.
  type_name = type_url.split('/')[-1]
  descriptor = factory.pool.FindMessageTypeByName(type_name)

  if descriptor is None:
    return None

  message_class = factory.GetPrototype(descriptor)
  message = message_class()

  message.ParseFromString(msg.value)
  return message


def _AddEqualsMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""
  def __eq__(self, other):
    if (not isinstance(other, message_mod.Message) or
        other.DESCRIPTOR != self.DESCRIPTOR):
      return False

    if self is other:
      return True

    if self.DESCRIPTOR.full_name == _AnyFullTypeName:
      any_a = _InternalUnpackAny(self)
      any_b = _InternalUnpackAny(other)
      if any_a and any_b:
        return any_a == any_b

    if not self.ListFields() == other.ListFields():
      return False

    # TODO(jieluo): Fix UnknownFieldSet to consider MessageSet extensions,
    # then use it for the comparison.
    unknown_fields = list(self._unknown_fields)
    unknown_fields.sort()
    other_unknown_fields = list(other._unknown_fields)
    other_unknown_fields.sort()
    return unknown_fields == other_unknown_fields

  cls.__eq__ = __eq__


def _AddStrMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""
  def __str__(self):
    return text_format.MessageToString(self)
  cls.__str__ = __str__


def _AddReprMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""
  def __repr__(self):
    return text_format.MessageToString(self)
  cls.__repr__ = __repr__


def _AddUnicodeMethod(unused_message_descriptor, cls):
  """Helper for _AddMessageMethods()."""

  def __unicode__(self):
    return text_format.MessageToString(self, as_utf8=True).decode('utf-8')
  cls.__unicode__ = __unicode__


def _BytesForNonRepeatedElement(value, field_number, field_type):
  """Returns the number of bytes needed to serialize a non-repeated element.
  The returned byte count includes space for tag information and any
  other additional space associated with serializing value.

  Args:
    value: Value we're serializing.
    field_number: Field number of this value.  (Since the field number
      is stored as part of a varint-encoded tag, this has an impact
      on the total bytes required to serialize the value).
    field_type: The type of the field.  One of the TYPE_* constants
      within FieldDescriptor.
  """
  try:
    fn = type_checkers.TYPE_TO_BYTE_SIZE_FN[field_type]
    return fn(field_number, value)
  except KeyError:
    raise message_mod.EncodeError('Unrecognized field type: %d' % field_type)


def _AddByteSizeMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""

  def ByteSize(self):
    if not self._cached_byte_size_dirty:
      return self._cached_byte_size

    size = 0
    descriptor = self.DESCRIPTOR
    if descriptor.GetOptions().map_entry:
      # Fields of map entry should always be serialized.
      size = descriptor.fields_by_name['key']._sizer(self.key)
      size += descriptor.fields_by_name['value']._sizer(self.value)
    else:
      for field_descriptor, field_value in self.ListFields():
        size += field_descriptor._sizer(field_value)
      for tag_bytes, value_bytes in self._unknown_fields:
        size += len(tag_bytes) + len(value_bytes)

    self._cached_byte_size = size
    self._cached_byte_size_dirty = False
    self._listener_for_children.dirty = False
    return size

  cls.ByteSize = ByteSize


def _AddSerializeToStringMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""

  def SerializeToString(self, **kwargs):
    # Check if the message has all of its required fields set.
    if not self.IsInitialized():
      raise message_mod.EncodeError(
          'Message %s is missing required fields: %s' % (
          self.DESCRIPTOR.full_name, ','.join(self.FindInitializationErrors())))
    return self.SerializePartialToString(**kwargs)
  cls.SerializeToString = SerializeToString


def _AddSerializePartialToStringMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""

  def SerializePartialToString(self, **kwargs):
    out = BytesIO()
    self._InternalSerialize(out.write, **kwargs)
    return out.getvalue()
  cls.SerializePartialToString = SerializePartialToString

  def InternalSerialize(self, write_bytes, deterministic=None):
    if deterministic is None:
      deterministic = (
          api_implementation.IsPythonDefaultSerializationDeterministic())
    else:
      deterministic = bool(deterministic)

    descriptor = self.DESCRIPTOR
    if descriptor.GetOptions().map_entry:
      # Fields of map entry should always be serialized.
      descriptor.fields_by_name['key']._encoder(
          write_bytes, self.key, deterministic)
      descriptor.fields_by_name['value']._encoder(
          write_bytes, self.value, deterministic)
    else:
      for field_descriptor, field_value in self.ListFields():
        field_descriptor._encoder(write_bytes, field_value, deterministic)
      for tag_bytes, value_bytes in self._unknown_fields:
        write_bytes(tag_bytes)
        write_bytes(value_bytes)
  cls._InternalSerialize = InternalSerialize


def _AddMergeFromStringMethod(message_descriptor, cls):
  """Helper for _AddMessageMethods()."""
  def MergeFromString(self, serialized):
    serialized = memoryview(serialized)
    length = len(serialized)
    try:
      if self._InternalParse(serialized, 0, length) != length:
        # The only reason _InternalParse would return early is if it
        # encountered an end-group tag.
        raise message_mod.DecodeError('Unexpected end-group tag.')
    except (IndexError, TypeError):
      # Now ord(buf[p:p+1]) == ord('') gets TypeError.
      raise message_mod.DecodeError('Truncated message.')
    except struct.error as e:
      raise message_mod.DecodeError(e)
    return length   # Return this for legacy reasons.
  cls.MergeFromString = MergeFromString

  local_ReadTag = decoder.ReadTag
  local_SkipField = decoder.SkipField
  decoders_by_tag = cls._decoders_by_tag

  def InternalParse(self, buffer, pos, end):
    """Create a message from serialized bytes.

    Args:
      self: Message, instance of the proto message object.
      buffer: memoryview of the serialized data.
      pos: int, position to start in the serialized data.
      end: int, end position of the serialized data.

    Returns:
      Message object.
    """
    # Guard against internal misuse, since this function is called internally
    # quite extensively, and its easy to accidentally pass bytes.
    assert isinstance(buffer, memoryview)
    self._Modified()
    field_dict = self._fields
    # pylint: disable=protected-access
    unknown_field_set = self._unknown_field_set
    while pos != end:
      (tag_bytes, new_pos) = local_ReadTag(buffer, pos)
      field_decoder, field_desc = decoders_by_tag.get(tag_bytes, (None, None))
      if field_decoder is None:
        if not self._unknown_fields:   # pylint: disable=protected-access
          self._unknown_fields = []    # pylint: disable=protected-access
        if unknown_field_set is None:
          # pylint: disable=protected-access
          self._unknown_field_set = containers.UnknownFieldSet()
          # pylint: disable=protected-access
          unknown_field_set = self._unknown_field_set
        # pylint: disable=protected-access
        (tag, _) = decoder._DecodeVarint(tag_bytes, 0)
        field_number, wire_type = wire_format.UnpackTag(tag)
        if field_number == 0:
          raise message_mod.DecodeError('Field number 0 is illegal.')
        # TODO(jieluo): remove old_pos.
        old_pos = new_pos
        (data, new_pos) = decoder._DecodeUnknownField(
            buffer, new_pos, wire_type)  # pylint: disable=protected-access
        if new_pos == -1:
          return pos
        # pylint: disable=protected-access
        unknown_field_set._add(field_number, wire_type, data)
        # TODO(jieluo): remove _unknown_fields.
        new_pos = local_SkipField(buffer, old_pos, end, tag_bytes)
        if new_pos == -1:
          return pos
        self._unknown_fields.append(
            (tag_bytes, buffer[old_pos:new_pos].tobytes()))
        pos = new_pos
      else:
        pos = field_decoder(buffer, new_pos, end, self, field_dict)
        if field_desc:
          self._UpdateOneofState(field_desc)
    return pos
  cls._InternalParse = InternalParse


def _AddIsInitializedMethod(message_descriptor, cls):
  """Adds the IsInitialized and FindInitializationError methods to the
  protocol message class."""

  required_fields = [field for field in message_descriptor.fields
                           if field.label == _FieldDescriptor.LABEL_REQUIRED]

  def IsInitialized(self, errors=None):
    """Checks if all required fields of a message are set.

    Args:
      errors:  A list which, if provided, will be populated with the field
               paths of all missing required fields.

    Returns:
      True iff the specified message has all required fields set.
    """

    # Performance is critical so we avoid HasField() and ListFields().

    for field in required_fields:
      if (field not in self._fields or
          (field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE and
           not self._fields[field]._is_present_in_parent)):
        if errors is not None:
          errors.extend(self.FindInitializationErrors())
        return False

    for field, value in list(self._fields.items()):  # dict can change size!
      if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        if field.label == _FieldDescriptor.LABEL_REPEATED:
          if (field.message_type.has_options and
              field.message_type.GetOptions().map_entry):
            continue
          for element in value:
            if not element.IsInitialized():
              if errors is not None:
                errors.extend(self.FindInitializationErrors())
              return False
        elif value._is_present_in_parent and not value.IsInitialized():
          if errors is not None:
            errors.extend(self.FindInitializationErrors())
          return False

    return True

  cls.IsInitialized = IsInitialized

  def FindInitializationErrors(self):
    """Finds required fields which are not initialized.

    Returns:
      A list of strings.  Each string is a path to an uninitialized field from
      the top-level message, e.g. "foo.bar[5].baz".
    """

    errors = []  # simplify things

    for field in required_fields:
      if not self.HasField(field.name):
        errors.append(field.name)

    for field, value in self.ListFields():
      if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
        if field.is_extension:
          name = '(%s)' % field.full_name
        else:
          name = field.name

        if _IsMapField(field):
          if _IsMessageMapField(field):
            for key in value:
              element = value[key]
              prefix = '%s[%s].' % (name, key)
              sub_errors = element.FindInitializationErrors()
              errors += [prefix + error for error in sub_errors]
          else:
            # ScalarMaps can't have any initialization errors.
            pass
        elif field.label == _FieldDescriptor.LABEL_REPEATED:
          for i in range(len(value)):
            element = value[i]
            prefix = '%s[%d].' % (name, i)
            sub_errors = element.FindInitializationErrors()
            errors += [prefix + error for error in sub_errors]
        else:
          prefix = name + '.'
          sub_errors = value.FindInitializationErrors()
          errors += [prefix + error for error in sub_errors]

    return errors

  cls.FindInitializationErrors = FindInitializationErrors


def _FullyQualifiedClassName(klass):
  module = klass.__module__
  name = getattr(klass, '__qualname__', klass.__name__)
  if module in (None, 'builtins', '__builtin__'):
    return name
  return module + '.' + name


def _AddMergeFromMethod(cls):
  LABEL_REPEATED = _FieldDescriptor.LABEL_REPEATED
  CPPTYPE_MESSAGE = _FieldDescriptor.CPPTYPE_MESSAGE

  def MergeFrom(self, msg):
    if not isinstance(msg, cls):
      raise TypeError(
          'Parameter to MergeFrom() must be instance of same class: '
          'expected %s got %s.' % (_FullyQualifiedClassName(cls),
                                   _FullyQualifiedClassName(msg.__class__)))

    assert msg is not self
    self._Modified()

    fields = self._fields

    for field, value in msg._fields.items():
      if field.label == LABEL_REPEATED:
        field_value = fields.get(field)
        if field_value is None:
          # Construct a new object to represent this field.
          field_value = field._default_constructor(self)
          fields[field] = field_value
        field_value.MergeFrom(value)
      elif field.cpp_type == CPPTYPE_MESSAGE:
        if value._is_present_in_parent:
          field_value = fields.get(field)
          if field_value is None:
            # Construct a new object to represent this field.
            field_value = field._default_constructor(self)
            fields[field] = field_value
          field_value.MergeFrom(value)
      else:
        self._fields[field] = value
        if field.containing_oneof:
          self._UpdateOneofState(field)

    if msg._unknown_fields:
      if not self._unknown_fields:
        self._unknown_fields = []
      self._unknown_fields.extend(msg._unknown_fields)
      # pylint: disable=protected-access
      if self._unknown_field_set is None:
        self._unknown_field_set = containers.UnknownFieldSet()
      self._unknown_field_set._extend(msg._unknown_field_set)

  cls.MergeFrom = MergeFrom


def _AddWhichOneofMethod(message_descriptor, cls):
  def WhichOneof(self, oneof_name):
    """Returns the name of the currently set field inside a oneof, or None."""
    try:
      field = message_descriptor.oneofs_by_name[oneof_name]
    except KeyError:
      raise ValueError(
          'Protocol message has no oneof "%s" field.' % oneof_name)

    nested_field = self._oneofs.get(field, None)
    if nested_field is not None and self.HasField(nested_field.name):
      return nested_field.name
    else:
      return None

  cls.WhichOneof = WhichOneof


def _Clear(self):
  # Clear fields.
  self._fields = {}
  self._unknown_fields = ()
  # pylint: disable=protected-access
  if self._unknown_field_set is not None:
    self._unknown_field_set._clear()
    self._unknown_field_set = None

  self._oneofs = {}
  self._Modified()


def _UnknownFields(self):
  if self._unknown_field_set is None:  # pylint: disable=protected-access
    # pylint: disable=protected-access
    self._unknown_field_set = containers.UnknownFieldSet()
  return self._unknown_field_set    # pylint: disable=protected-access


def _DiscardUnknownFields(self):
  self._unknown_fields = []
  self._unknown_field_set = None      # pylint: disable=protected-access
  for field, value in self.ListFields():
    if field.cpp_type == _FieldDescriptor.CPPTYPE_MESSAGE:
      if _IsMapField(field):
        if _IsMessageMapField(field):
          for key in value:
            value[key].DiscardUnknownFields()
      elif field.label == _FieldDescriptor.LABEL_REPEATED:
        for sub_message in value:
          sub_message.DiscardUnknownFields()
      else:
        value.DiscardUnknownFields()


def _SetListener(self, listener):
  if listener is None:
    self._listener = message_listener_mod.NullMessageListener()
  else:
    self._listener = listener


def _AddMessageMethods(message_descriptor, cls):
  """Adds implementations of all Message methods to cls."""
  _AddListFieldsMethod(message_descriptor, cls)
  _AddHasFieldMethod(message_descriptor, cls)
  _AddClearFieldMethod(message_descriptor, cls)
  if message_descriptor.is_extendable:
    _AddClearExtensionMethod(cls)
    _AddHasExtensionMethod(cls)
  _AddEqualsMethod(message_descriptor, cls)
  _AddStrMethod(message_descriptor, cls)
  _AddReprMethod(message_descriptor, cls)
  _AddUnicodeMethod(message_descriptor, cls)
  _AddByteSizeMethod(message_descriptor, cls)
  _AddSerializeToStringMethod(message_descriptor, cls)
  _AddSerializePartialToStringMethod(message_descriptor, cls)
  _AddMergeFromStringMethod(message_descriptor, cls)
  _AddIsInitializedMethod(message_descriptor, cls)
  _AddMergeFromMethod(cls)
  _AddWhichOneofMethod(message_descriptor, cls)
  # Adds methods which do not depend on cls.
  cls.Clear = _Clear
  cls.UnknownFields = _UnknownFields
  cls.DiscardUnknownFields = _DiscardUnknownFields
  cls._SetListener = _SetListener


def _AddPrivateHelperMethods(message_descriptor, cls):
  """Adds implementation of private helper methods to cls."""

  def Modified(self):
    """Sets the _cached_byte_size_dirty bit to true,
    and propagates this to our listener iff this was a state change.
    """

    # Note:  Some callers check _cached_byte_size_dirty before calling
    #   _Modified() as an extra optimization.  So, if this method is ever
    #   changed such that it does stuff even when _cached_byte_size_dirty is
    #   already true, the callers need to be updated.
    if not self._cached_byte_size_dirty:
      self._cached_byte_size_dirty = True
      self._listener_for_children.dirty = True
      self._is_present_in_parent = True
      self._listener.Modified()

  def _UpdateOneofState(self, field):
    """Sets field as the active field in its containing oneof.

    Will also delete currently active field in the oneof, if it is different
    from the argument. Does not mark the message as modified.
    """
    other_field = self._oneofs.setdefault(field.containing_oneof, field)
    if other_field is not field:
      del self._fields[other_field]
      self._oneofs[field.containing_oneof] = field

  cls._Modified = Modified
  cls.SetInParent = Modified
  cls._UpdateOneofState = _UpdateOneofState


class _Listener(object):

  """MessageListener implementation that a parent message registers with its
  child message.

  In order to support semantics like:

    foo.bar.baz.moo = 23
    assert foo.HasField('bar')

  ...child objects must have back references to their parents.
  This helper class is at the heart of this support.
  """

  def __init__(self, parent_message):
    """Args:
      parent_message: The message whose _Modified() method we should call when
        we receive Modified() messages.
    """
    # This listener establishes a back reference from a child (contained) object
    # to its parent (containing) object.  We make this a weak reference to avoid
    # creating cyclic garbage when the client finishes with the 'parent' object
    # in the tree.
    if isinstance(parent_message, weakref.ProxyType):
      self._parent_message_weakref = parent_message
    else:
      self._parent_message_weakref = weakref.proxy(parent_message)

    # As an optimization, we also indicate directly on the listener whether
    # or not the parent message is dirty.  This way we can avoid traversing
    # up the tree in the common case.
    self.dirty = False

  def Modified(self):
    if self.dirty:
      return
    try:
      # Propagate the signal to our parents iff this is the first field set.
      self._parent_message_weakref._Modified()
    except ReferenceError:
      # We can get here if a client has kept a reference to a child object,
      # and is now setting a field on it, but the child's parent has been
      # garbage-collected.  This is not an error.
      pass


class _OneofListener(_Listener):
  """Special listener implementation for setting composite oneof fields."""

  def __init__(self, parent_message, field):
    """Args:
      parent_message: The message whose _Modified() method we should call when
        we receive Modified() messages.
      field: The descriptor of the field being set in the parent message.
    """
    super(_OneofListener, self).__init__(parent_message)
    self._field = field

  def Modified(self):
    """Also updates the state of the containing oneof in the parent message."""
    try:
      self._parent_message_weakref._UpdateOneofState(self._field)
      super(_OneofListener, self).Modified()
    except ReferenceError:
      pass
