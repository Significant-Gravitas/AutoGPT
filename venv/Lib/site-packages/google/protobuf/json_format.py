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

"""Contains routines for printing protocol messages in JSON format.

Simple usage example:

  # Create a proto object and serialize it to a json format string.
  message = my_proto_pb2.MyMessage(foo='bar')
  json_string = json_format.MessageToJson(message)

  # Parse a json format string to proto object.
  message = json_format.Parse(json_string, my_proto_pb2.MyMessage())
"""

__author__ = 'jieluo@google.com (Jie Luo)'


import base64
from collections import OrderedDict
import json
import math
from operator import methodcaller
import re
import sys

from google.protobuf.internal import type_checkers
from google.protobuf import descriptor
from google.protobuf import message_factory
from google.protobuf import symbol_database


_TIMESTAMPFOMAT = '%Y-%m-%dT%H:%M:%S'
_INT_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_INT32,
                        descriptor.FieldDescriptor.CPPTYPE_UINT32,
                        descriptor.FieldDescriptor.CPPTYPE_INT64,
                        descriptor.FieldDescriptor.CPPTYPE_UINT64])
_INT64_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_INT64,
                          descriptor.FieldDescriptor.CPPTYPE_UINT64])
_FLOAT_TYPES = frozenset([descriptor.FieldDescriptor.CPPTYPE_FLOAT,
                          descriptor.FieldDescriptor.CPPTYPE_DOUBLE])
_INFINITY = 'Infinity'
_NEG_INFINITY = '-Infinity'
_NAN = 'NaN'

_UNPAIRED_SURROGATE_PATTERN = re.compile(
    u'[\ud800-\udbff](?![\udc00-\udfff])|(?<![\ud800-\udbff])[\udc00-\udfff]')

_VALID_EXTENSION_NAME = re.compile(r'\[[a-zA-Z0-9\._]*\]$')


class Error(Exception):
  """Top-level module error for json_format."""


class SerializeToJsonError(Error):
  """Thrown if serialization to JSON fails."""


class ParseError(Error):
  """Thrown in case of parsing error."""


def MessageToJson(
    message,
    including_default_value_fields=False,
    preserving_proto_field_name=False,
    indent=2,
    sort_keys=False,
    use_integers_for_enums=False,
    descriptor_pool=None,
    float_precision=None,
    ensure_ascii=True):
  """Converts protobuf message to JSON format.

  Args:
    message: The protocol buffers message instance to serialize.
    including_default_value_fields: If True, singular primitive fields,
        repeated fields, and map fields will always be serialized.  If
        False, only serialize non-empty fields.  Singular message fields
        and oneof fields are not affected by this option.
    preserving_proto_field_name: If True, use the original proto field
        names as defined in the .proto file. If False, convert the field
        names to lowerCamelCase.
    indent: The JSON object will be pretty-printed with this indent level.
        An indent level of 0 or negative will only insert newlines. If the
        indent level is None, no newlines will be inserted.
    sort_keys: If True, then the output will be sorted by field names.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
        default.
    float_precision: If set, use this to specify float field valid digits.
    ensure_ascii: If True, strings with non-ASCII characters are escaped.
        If False, Unicode strings are returned unchanged.

  Returns:
    A string containing the JSON formatted protocol buffer message.
  """
  printer = _Printer(
      including_default_value_fields,
      preserving_proto_field_name,
      use_integers_for_enums,
      descriptor_pool,
      float_precision=float_precision)
  return printer.ToJsonString(message, indent, sort_keys, ensure_ascii)


def MessageToDict(
    message,
    including_default_value_fields=False,
    preserving_proto_field_name=False,
    use_integers_for_enums=False,
    descriptor_pool=None,
    float_precision=None):
  """Converts protobuf message to a dictionary.

  When the dictionary is encoded to JSON, it conforms to proto3 JSON spec.

  Args:
    message: The protocol buffers message instance to serialize.
    including_default_value_fields: If True, singular primitive fields,
        repeated fields, and map fields will always be serialized.  If
        False, only serialize non-empty fields.  Singular message fields
        and oneof fields are not affected by this option.
    preserving_proto_field_name: If True, use the original proto field
        names as defined in the .proto file. If False, convert the field
        names to lowerCamelCase.
    use_integers_for_enums: If true, print integers instead of enum names.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
        default.
    float_precision: If set, use this to specify float field valid digits.

  Returns:
    A dict representation of the protocol buffer message.
  """
  printer = _Printer(
      including_default_value_fields,
      preserving_proto_field_name,
      use_integers_for_enums,
      descriptor_pool,
      float_precision=float_precision)
  # pylint: disable=protected-access
  return printer._MessageToJsonObject(message)


def _IsMapEntry(field):
  return (field.type == descriptor.FieldDescriptor.TYPE_MESSAGE and
          field.message_type.has_options and
          field.message_type.GetOptions().map_entry)


class _Printer(object):
  """JSON format printer for protocol message."""

  def __init__(
      self,
      including_default_value_fields=False,
      preserving_proto_field_name=False,
      use_integers_for_enums=False,
      descriptor_pool=None,
      float_precision=None):
    self.including_default_value_fields = including_default_value_fields
    self.preserving_proto_field_name = preserving_proto_field_name
    self.use_integers_for_enums = use_integers_for_enums
    self.descriptor_pool = descriptor_pool
    if float_precision:
      self.float_format = '.{}g'.format(float_precision)
    else:
      self.float_format = None

  def ToJsonString(self, message, indent, sort_keys, ensure_ascii):
    js = self._MessageToJsonObject(message)
    return json.dumps(
        js, indent=indent, sort_keys=sort_keys, ensure_ascii=ensure_ascii)

  def _MessageToJsonObject(self, message):
    """Converts message to an object according to Proto3 JSON Specification."""
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
      return self._WrapperMessageToJsonObject(message)
    if full_name in _WKTJSONMETHODS:
      return methodcaller(_WKTJSONMETHODS[full_name][0], message)(self)
    js = {}
    return self._RegularMessageToJsonObject(message, js)

  def _RegularMessageToJsonObject(self, message, js):
    """Converts normal message according to Proto3 JSON Specification."""
    fields = message.ListFields()

    try:
      for field, value in fields:
        if self.preserving_proto_field_name:
          name = field.name
        else:
          name = field.json_name
        if _IsMapEntry(field):
          # Convert a map field.
          v_field = field.message_type.fields_by_name['value']
          js_map = {}
          for key in value:
            if isinstance(key, bool):
              if key:
                recorded_key = 'true'
              else:
                recorded_key = 'false'
            else:
              recorded_key = str(key)
            js_map[recorded_key] = self._FieldToJsonObject(
                v_field, value[key])
          js[name] = js_map
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          # Convert a repeated field.
          js[name] = [self._FieldToJsonObject(field, k)
                      for k in value]
        elif field.is_extension:
          name = '[%s]' % field.full_name
          js[name] = self._FieldToJsonObject(field, value)
        else:
          js[name] = self._FieldToJsonObject(field, value)

      # Serialize default value if including_default_value_fields is True.
      if self.including_default_value_fields:
        message_descriptor = message.DESCRIPTOR
        for field in message_descriptor.fields:
          # Singular message fields and oneof fields will not be affected.
          if ((field.label != descriptor.FieldDescriptor.LABEL_REPEATED and
               field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE) or
              field.containing_oneof):
            continue
          if self.preserving_proto_field_name:
            name = field.name
          else:
            name = field.json_name
          if name in js:
            # Skip the field which has been serialized already.
            continue
          if _IsMapEntry(field):
            js[name] = {}
          elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
            js[name] = []
          else:
            js[name] = self._FieldToJsonObject(field, field.default_value)

    except ValueError as e:
      raise SerializeToJsonError(
          'Failed to serialize {0} field: {1}.'.format(field.name, e)) from e

    return js

  def _FieldToJsonObject(self, field, value):
    """Converts field value according to Proto3 JSON Specification."""
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
      return self._MessageToJsonObject(value)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
      if self.use_integers_for_enums:
        return value
      if field.enum_type.full_name == 'google.protobuf.NullValue':
        return None
      enum_value = field.enum_type.values_by_number.get(value, None)
      if enum_value is not None:
        return enum_value.name
      else:
        if field.enum_type.is_closed:
          raise SerializeToJsonError('Enum field contains an integer value '
                                     'which can not mapped to an enum value.')
        else:
          return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
      if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        # Use base64 Data encoding for bytes
        return base64.b64encode(value).decode('utf-8')
      else:
        return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
      return bool(value)
    elif field.cpp_type in _INT64_TYPES:
      return str(value)
    elif field.cpp_type in _FLOAT_TYPES:
      if math.isinf(value):
        if value < 0.0:
          return _NEG_INFINITY
        else:
          return _INFINITY
      if math.isnan(value):
        return _NAN
      if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
        if self.float_format:
          return float(format(value, self.float_format))
        else:
          return type_checkers.ToShortestFloat(value)

    return value

  def _AnyMessageToJsonObject(self, message):
    """Converts Any message according to Proto3 JSON Specification."""
    if not message.ListFields():
      return {}
    # Must print @type first, use OrderedDict instead of {}
    js = OrderedDict()
    type_url = message.type_url
    js['@type'] = type_url
    sub_message = _CreateMessageFromTypeUrl(type_url, self.descriptor_pool)
    sub_message.ParseFromString(message.value)
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
      js['value'] = self._WrapperMessageToJsonObject(sub_message)
      return js
    if full_name in _WKTJSONMETHODS:
      js['value'] = methodcaller(_WKTJSONMETHODS[full_name][0],
                                 sub_message)(self)
      return js
    return self._RegularMessageToJsonObject(sub_message, js)

  def _GenericMessageToJsonObject(self, message):
    """Converts message according to Proto3 JSON Specification."""
    # Duration, Timestamp and FieldMask have ToJsonString method to do the
    # convert. Users can also call the method directly.
    return message.ToJsonString()

  def _ValueMessageToJsonObject(self, message):
    """Converts Value message according to Proto3 JSON Specification."""
    which = message.WhichOneof('kind')
    # If the Value message is not set treat as null_value when serialize
    # to JSON. The parse back result will be different from original message.
    if which is None or which == 'null_value':
      return None
    if which == 'list_value':
      return self._ListValueMessageToJsonObject(message.list_value)
    if which == 'number_value':
      value = message.number_value
      if math.isinf(value):
        raise ValueError('Fail to serialize Infinity for Value.number_value, '
                         'which would parse as string_value')
      if math.isnan(value):
        raise ValueError('Fail to serialize NaN for Value.number_value, '
                         'which would parse as string_value')
    else:
      value = getattr(message, which)
    oneof_descriptor = message.DESCRIPTOR.fields_by_name[which]
    return self._FieldToJsonObject(oneof_descriptor, value)

  def _ListValueMessageToJsonObject(self, message):
    """Converts ListValue message according to Proto3 JSON Specification."""
    return [self._ValueMessageToJsonObject(value)
            for value in message.values]

  def _StructMessageToJsonObject(self, message):
    """Converts Struct message according to Proto3 JSON Specification."""
    fields = message.fields
    ret = {}
    for key in fields:
      ret[key] = self._ValueMessageToJsonObject(fields[key])
    return ret

  def _WrapperMessageToJsonObject(self, message):
    return self._FieldToJsonObject(
        message.DESCRIPTOR.fields_by_name['value'], message.value)


def _IsWrapperMessage(message_descriptor):
  return message_descriptor.file.name == 'google/protobuf/wrappers.proto'


def _DuplicateChecker(js):
  result = {}
  for name, value in js:
    if name in result:
      raise ParseError('Failed to load JSON: duplicate key {0}.'.format(name))
    result[name] = value
  return result


def _CreateMessageFromTypeUrl(type_url, descriptor_pool):
  """Creates a message from a type URL."""
  db = symbol_database.Default()
  pool = db.pool if descriptor_pool is None else descriptor_pool
  type_name = type_url.split('/')[-1]
  try:
    message_descriptor = pool.FindMessageTypeByName(type_name)
  except KeyError as e:
    raise TypeError(
        'Can not find message descriptor by type_url: {0}'.format(type_url)
      ) from e
  message_class = message_factory.GetMessageClass(message_descriptor)
  return message_class()


def Parse(text,
          message,
          ignore_unknown_fields=False,
          descriptor_pool=None,
          max_recursion_depth=100):
  """Parses a JSON representation of a protocol message into a message.

  Args:
    text: Message JSON representation.
    message: A protocol buffer message to merge into.
    ignore_unknown_fields: If True, do not raise errors for unknown fields.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    max_recursion_depth: max recursion depth of JSON message to be
      deserialized. JSON messages over this depth will fail to be
      deserialized. Default value is 100.

  Returns:
    The same message passed as argument.

  Raises::
    ParseError: On JSON parsing problems.
  """
  if not isinstance(text, str):
    text = text.decode('utf-8')
  try:
    js = json.loads(text, object_pairs_hook=_DuplicateChecker)
  except ValueError as e:
    raise ParseError('Failed to load JSON: {0}.'.format(str(e))) from e
  return ParseDict(js, message, ignore_unknown_fields, descriptor_pool,
                   max_recursion_depth)


def ParseDict(js_dict,
              message,
              ignore_unknown_fields=False,
              descriptor_pool=None,
              max_recursion_depth=100):
  """Parses a JSON dictionary representation into a message.

  Args:
    js_dict: Dict representation of a JSON message.
    message: A protocol buffer message to merge into.
    ignore_unknown_fields: If True, do not raise errors for unknown fields.
    descriptor_pool: A Descriptor Pool for resolving types. If None use the
      default.
    max_recursion_depth: max recursion depth of JSON message to be
      deserialized. JSON messages over this depth will fail to be
      deserialized. Default value is 100.

  Returns:
    The same message passed as argument.
  """
  parser = _Parser(ignore_unknown_fields, descriptor_pool, max_recursion_depth)
  parser.ConvertMessage(js_dict, message, '')
  return message


_INT_OR_FLOAT = (int, float)


class _Parser(object):
  """JSON format parser for protocol message."""

  def __init__(self, ignore_unknown_fields, descriptor_pool,
               max_recursion_depth):
    self.ignore_unknown_fields = ignore_unknown_fields
    self.descriptor_pool = descriptor_pool
    self.max_recursion_depth = max_recursion_depth
    self.recursion_depth = 0

  def ConvertMessage(self, value, message, path):
    """Convert a JSON object into a message.

    Args:
      value: A JSON object.
      message: A WKT or regular protocol message to record the data.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of convert problems.
    """
    self.recursion_depth += 1
    if self.recursion_depth > self.max_recursion_depth:
      raise ParseError('Message too deep. Max recursion depth is {0}'.format(
          self.max_recursion_depth))
    message_descriptor = message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if not path:
      path = message_descriptor.name
    if _IsWrapperMessage(message_descriptor):
      self._ConvertWrapperMessage(value, message, path)
    elif full_name in _WKTJSONMETHODS:
      methodcaller(_WKTJSONMETHODS[full_name][1], value, message, path)(self)
    else:
      self._ConvertFieldValuePair(value, message, path)
    self.recursion_depth -= 1

  def _ConvertFieldValuePair(self, js, message, path):
    """Convert field value pairs into regular message.

    Args:
      js: A JSON object to convert the field value pairs.
      message: A regular protocol message to record the data.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of problems converting.
    """
    names = []
    message_descriptor = message.DESCRIPTOR
    fields_by_json_name = dict((f.json_name, f)
                               for f in message_descriptor.fields)
    for name in js:
      try:
        field = fields_by_json_name.get(name, None)
        if not field:
          field = message_descriptor.fields_by_name.get(name, None)
        if not field and _VALID_EXTENSION_NAME.match(name):
          if not message_descriptor.is_extendable:
            raise ParseError(
                'Message type {0} does not have extensions at {1}'.format(
                    message_descriptor.full_name, path))
          identifier = name[1:-1]  # strip [] brackets
          # pylint: disable=protected-access
          field = message.Extensions._FindExtensionByName(identifier)
          # pylint: enable=protected-access
          if not field:
            # Try looking for extension by the message type name, dropping the
            # field name following the final . separator in full_name.
            identifier = '.'.join(identifier.split('.')[:-1])
            # pylint: disable=protected-access
            field = message.Extensions._FindExtensionByName(identifier)
            # pylint: enable=protected-access
        if not field:
          if self.ignore_unknown_fields:
            continue
          raise ParseError(
              ('Message type "{0}" has no field named "{1}" at "{2}".\n'
               ' Available Fields(except extensions): "{3}"').format(
                   message_descriptor.full_name, name, path,
                   [f.json_name for f in message_descriptor.fields]))
        if name in names:
          raise ParseError('Message type "{0}" should not have multiple '
                           '"{1}" fields at "{2}".'.format(
                               message.DESCRIPTOR.full_name, name, path))
        names.append(name)
        value = js[name]
        # Check no other oneof field is parsed.
        if field.containing_oneof is not None and value is not None:
          oneof_name = field.containing_oneof.name
          if oneof_name in names:
            raise ParseError('Message type "{0}" should not have multiple '
                             '"{1}" oneof fields at "{2}".'.format(
                                 message.DESCRIPTOR.full_name, oneof_name,
                                 path))
          names.append(oneof_name)

        if value is None:
          if (field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE
              and field.message_type.full_name == 'google.protobuf.Value'):
            sub_message = getattr(message, field.name)
            sub_message.null_value = 0
          elif (field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM
                and field.enum_type.full_name == 'google.protobuf.NullValue'):
            setattr(message, field.name, 0)
          else:
            message.ClearField(field.name)
          continue

        # Parse field value.
        if _IsMapEntry(field):
          message.ClearField(field.name)
          self._ConvertMapFieldValue(value, message, field,
                                     '{0}.{1}'.format(path, name))
        elif field.label == descriptor.FieldDescriptor.LABEL_REPEATED:
          message.ClearField(field.name)
          if not isinstance(value, list):
            raise ParseError('repeated field {0} must be in [] which is '
                             '{1} at {2}'.format(name, value, path))
          if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
            # Repeated message field.
            for index, item in enumerate(value):
              sub_message = getattr(message, field.name).add()
              # None is a null_value in Value.
              if (item is None and
                  sub_message.DESCRIPTOR.full_name != 'google.protobuf.Value'):
                raise ParseError('null is not allowed to be used as an element'
                                 ' in a repeated field at {0}.{1}[{2}]'.format(
                                     path, name, index))
              self.ConvertMessage(item, sub_message,
                                  '{0}.{1}[{2}]'.format(path, name, index))
          else:
            # Repeated scalar field.
            for index, item in enumerate(value):
              if item is None:
                raise ParseError('null is not allowed to be used as an element'
                                 ' in a repeated field at {0}.{1}[{2}]'.format(
                                     path, name, index))
              getattr(message, field.name).append(
                  _ConvertScalarFieldValue(
                      item, field, '{0}.{1}[{2}]'.format(path, name, index)))
        elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
          if field.is_extension:
            sub_message = message.Extensions[field]
          else:
            sub_message = getattr(message, field.name)
          sub_message.SetInParent()
          self.ConvertMessage(value, sub_message, '{0}.{1}'.format(path, name))
        else:
          if field.is_extension:
            message.Extensions[field] = _ConvertScalarFieldValue(
                value, field, '{0}.{1}'.format(path, name))
          else:
            setattr(
                message, field.name,
                _ConvertScalarFieldValue(value, field,
                                         '{0}.{1}'.format(path, name)))
      except ParseError as e:
        if field and field.containing_oneof is None:
          raise ParseError(
            'Failed to parse {0} field: {1}.'.format(name, e)
          ) from e
        else:
          raise ParseError(str(e)) from e
      except ValueError as e:
        raise ParseError(
          'Failed to parse {0} field: {1}.'.format(name, e)
        ) from e
      except TypeError as e:
        raise ParseError(
          'Failed to parse {0} field: {1}.'.format(name, e)
        ) from e

  def _ConvertAnyMessage(self, value, message, path):
    """Convert a JSON representation into Any message."""
    if isinstance(value, dict) and not value:
      return
    try:
      type_url = value['@type']
    except KeyError as e:
      raise ParseError(
        '@type is missing when parsing any message at {0}'.format(path)
      ) from e

    try:
      sub_message = _CreateMessageFromTypeUrl(type_url, self.descriptor_pool)
    except TypeError as e:
      raise ParseError('{0} at {1}'.format(e, path)) from e
    message_descriptor = sub_message.DESCRIPTOR
    full_name = message_descriptor.full_name
    if _IsWrapperMessage(message_descriptor):
      self._ConvertWrapperMessage(value['value'], sub_message,
                                  '{0}.value'.format(path))
    elif full_name in _WKTJSONMETHODS:
      methodcaller(_WKTJSONMETHODS[full_name][1], value['value'], sub_message,
                   '{0}.value'.format(path))(
                       self)
    else:
      del value['@type']
      self._ConvertFieldValuePair(value, sub_message, path)
      value['@type'] = type_url
    # Sets Any message
    message.value = sub_message.SerializeToString()
    message.type_url = type_url

  def _ConvertGenericMessage(self, value, message, path):
    """Convert a JSON representation into message with FromJsonString."""
    # Duration, Timestamp, FieldMask have a FromJsonString method to do the
    # conversion. Users can also call the method directly.
    try:
      message.FromJsonString(value)
    except ValueError as e:
      raise ParseError('{0} at {1}'.format(e, path)) from e

  def _ConvertValueMessage(self, value, message, path):
    """Convert a JSON representation into Value message."""
    if isinstance(value, dict):
      self._ConvertStructMessage(value, message.struct_value, path)
    elif isinstance(value, list):
      self._ConvertListValueMessage(value, message.list_value, path)
    elif value is None:
      message.null_value = 0
    elif isinstance(value, bool):
      message.bool_value = value
    elif isinstance(value, str):
      message.string_value = value
    elif isinstance(value, _INT_OR_FLOAT):
      message.number_value = value
    else:
      raise ParseError('Value {0} has unexpected type {1} at {2}'.format(
          value, type(value), path))

  def _ConvertListValueMessage(self, value, message, path):
    """Convert a JSON representation into ListValue message."""
    if not isinstance(value, list):
      raise ParseError('ListValue must be in [] which is {0} at {1}'.format(
          value, path))
    message.ClearField('values')
    for index, item in enumerate(value):
      self._ConvertValueMessage(item, message.values.add(),
                                '{0}[{1}]'.format(path, index))

  def _ConvertStructMessage(self, value, message, path):
    """Convert a JSON representation into Struct message."""
    if not isinstance(value, dict):
      raise ParseError('Struct must be in a dict which is {0} at {1}'.format(
          value, path))
    # Clear will mark the struct as modified so it will be created even if
    # there are no values.
    message.Clear()
    for key in value:
      self._ConvertValueMessage(value[key], message.fields[key],
                                '{0}.{1}'.format(path, key))
    return

  def _ConvertWrapperMessage(self, value, message, path):
    """Convert a JSON representation into Wrapper message."""
    field = message.DESCRIPTOR.fields_by_name['value']
    setattr(
        message, 'value',
        _ConvertScalarFieldValue(value, field, path='{0}.value'.format(path)))

  def _ConvertMapFieldValue(self, value, message, field, path):
    """Convert map field value for a message map field.

    Args:
      value: A JSON object to convert the map field value.
      message: A protocol message to record the converted data.
      field: The descriptor of the map field to be converted.
      path: parent path to log parse error info.

    Raises:
      ParseError: In case of convert problems.
    """
    if not isinstance(value, dict):
      raise ParseError(
          'Map field {0} must be in a dict which is {1} at {2}'.format(
              field.name, value, path))
    key_field = field.message_type.fields_by_name['key']
    value_field = field.message_type.fields_by_name['value']
    for key in value:
      key_value = _ConvertScalarFieldValue(key, key_field,
                                           '{0}.key'.format(path), True)
      if value_field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_MESSAGE:
        self.ConvertMessage(value[key],
                            getattr(message, field.name)[key_value],
                            '{0}[{1}]'.format(path, key_value))
      else:
        getattr(message, field.name)[key_value] = _ConvertScalarFieldValue(
            value[key], value_field, path='{0}[{1}]'.format(path, key_value))


def _ConvertScalarFieldValue(value, field, path, require_str=False):
  """Convert a single scalar field value.

  Args:
    value: A scalar value to convert the scalar field value.
    field: The descriptor of the field to convert.
    path: parent path to log parse error info.
    require_str: If True, the field value must be a str.

  Returns:
    The converted scalar field value

  Raises:
    ParseError: In case of convert problems.
  """
  try:
    if field.cpp_type in _INT_TYPES:
      return _ConvertInteger(value)
    elif field.cpp_type in _FLOAT_TYPES:
      return _ConvertFloat(value, field)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_BOOL:
      return _ConvertBool(value, require_str)
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_STRING:
      if field.type == descriptor.FieldDescriptor.TYPE_BYTES:
        if isinstance(value, str):
          encoded = value.encode('utf-8')
        else:
          encoded = value
        # Add extra padding '='
        padded_value = encoded + b'=' * (4 - len(encoded) % 4)
        return base64.urlsafe_b64decode(padded_value)
      else:
        # Checking for unpaired surrogates appears to be unreliable,
        # depending on the specific Python version, so we check manually.
        if _UNPAIRED_SURROGATE_PATTERN.search(value):
          raise ParseError('Unpaired surrogate')
        return value
    elif field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_ENUM:
      # Convert an enum value.
      enum_value = field.enum_type.values_by_name.get(value, None)
      if enum_value is None:
        try:
          number = int(value)
          enum_value = field.enum_type.values_by_number.get(number, None)
        except ValueError as e:
          raise ParseError('Invalid enum value {0} for enum type {1}'.format(
              value, field.enum_type.full_name)) from e
        if enum_value is None:
          if field.enum_type.is_closed:
            raise ParseError('Invalid enum value {0} for enum type {1}'.format(
                value, field.enum_type.full_name))
          else:
            return number
      return enum_value.number
  except ParseError as e:
    raise ParseError('{0} at {1}'.format(e, path)) from e


def _ConvertInteger(value):
  """Convert an integer.

  Args:
    value: A scalar value to convert.

  Returns:
    The integer value.

  Raises:
    ParseError: If an integer couldn't be consumed.
  """
  if isinstance(value, float) and not value.is_integer():
    raise ParseError('Couldn\'t parse integer: {0}'.format(value))

  if isinstance(value, str) and value.find(' ') != -1:
    raise ParseError('Couldn\'t parse integer: "{0}"'.format(value))

  if isinstance(value, bool):
    raise ParseError('Bool value {0} is not acceptable for '
                     'integer field'.format(value))

  return int(value)


def _ConvertFloat(value, field):
  """Convert an floating point number."""
  if isinstance(value, float):
    if math.isnan(value):
      raise ParseError('Couldn\'t parse NaN, use quoted "NaN" instead')
    if math.isinf(value):
      if value > 0:
        raise ParseError('Couldn\'t parse Infinity or value too large, '
                         'use quoted "Infinity" instead')
      else:
        raise ParseError('Couldn\'t parse -Infinity or value too small, '
                         'use quoted "-Infinity" instead')
    if field.cpp_type == descriptor.FieldDescriptor.CPPTYPE_FLOAT:
      # pylint: disable=protected-access
      if value > type_checkers._FLOAT_MAX:
        raise ParseError('Float value too large')
      # pylint: disable=protected-access
      if value < type_checkers._FLOAT_MIN:
        raise ParseError('Float value too small')
  if value == 'nan':
    raise ParseError('Couldn\'t parse float "nan", use "NaN" instead')
  try:
    # Assume Python compatible syntax.
    return float(value)
  except ValueError as e:
    # Check alternative spellings.
    if value == _NEG_INFINITY:
      return float('-inf')
    elif value == _INFINITY:
      return float('inf')
    elif value == _NAN:
      return float('nan')
    else:
      raise ParseError('Couldn\'t parse float: {0}'.format(value)) from e


def _ConvertBool(value, require_str):
  """Convert a boolean value.

  Args:
    value: A scalar value to convert.
    require_str: If True, value must be a str.

  Returns:
    The bool parsed.

  Raises:
    ParseError: If a boolean value couldn't be consumed.
  """
  if require_str:
    if value == 'true':
      return True
    elif value == 'false':
      return False
    else:
      raise ParseError('Expected "true" or "false", not {0}'.format(value))

  if not isinstance(value, bool):
    raise ParseError('Expected true or false without quotes')
  return value

_WKTJSONMETHODS = {
    'google.protobuf.Any': ['_AnyMessageToJsonObject',
                            '_ConvertAnyMessage'],
    'google.protobuf.Duration': ['_GenericMessageToJsonObject',
                                 '_ConvertGenericMessage'],
    'google.protobuf.FieldMask': ['_GenericMessageToJsonObject',
                                  '_ConvertGenericMessage'],
    'google.protobuf.ListValue': ['_ListValueMessageToJsonObject',
                                  '_ConvertListValueMessage'],
    'google.protobuf.Struct': ['_StructMessageToJsonObject',
                               '_ConvertStructMessage'],
    'google.protobuf.Timestamp': ['_GenericMessageToJsonObject',
                                  '_ConvertGenericMessage'],
    'google.protobuf.Value': ['_ValueMessageToJsonObject',
                              '_ConvertValueMessage']
}
