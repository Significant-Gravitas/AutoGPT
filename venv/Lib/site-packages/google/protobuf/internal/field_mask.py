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

"""Contains FieldMask class."""

from google.protobuf.descriptor import FieldDescriptor


class FieldMask(object):
  """Class for FieldMask message type."""

  __slots__ = ()

  def ToJsonString(self):
    """Converts FieldMask to string according to proto3 JSON spec."""
    camelcase_paths = []
    for path in self.paths:
      camelcase_paths.append(_SnakeCaseToCamelCase(path))
    return ','.join(camelcase_paths)

  def FromJsonString(self, value):
    """Converts string to FieldMask according to proto3 JSON spec."""
    if not isinstance(value, str):
      raise ValueError('FieldMask JSON value not a string: {!r}'.format(value))
    self.Clear()
    if value:
      for path in value.split(','):
        self.paths.append(_CamelCaseToSnakeCase(path))

  def IsValidForDescriptor(self, message_descriptor):
    """Checks whether the FieldMask is valid for Message Descriptor."""
    for path in self.paths:
      if not _IsValidPath(message_descriptor, path):
        return False
    return True

  def AllFieldsFromDescriptor(self, message_descriptor):
    """Gets all direct fields of Message Descriptor to FieldMask."""
    self.Clear()
    for field in message_descriptor.fields:
      self.paths.append(field.name)

  def CanonicalFormFromMask(self, mask):
    """Converts a FieldMask to the canonical form.

    Removes paths that are covered by another path. For example,
    "foo.bar" is covered by "foo" and will be removed if "foo"
    is also in the FieldMask. Then sorts all paths in alphabetical order.

    Args:
      mask: The original FieldMask to be converted.
    """
    tree = _FieldMaskTree(mask)
    tree.ToFieldMask(self)

  def Union(self, mask1, mask2):
    """Merges mask1 and mask2 into this FieldMask."""
    _CheckFieldMaskMessage(mask1)
    _CheckFieldMaskMessage(mask2)
    tree = _FieldMaskTree(mask1)
    tree.MergeFromFieldMask(mask2)
    tree.ToFieldMask(self)

  def Intersect(self, mask1, mask2):
    """Intersects mask1 and mask2 into this FieldMask."""
    _CheckFieldMaskMessage(mask1)
    _CheckFieldMaskMessage(mask2)
    tree = _FieldMaskTree(mask1)
    intersection = _FieldMaskTree()
    for path in mask2.paths:
      tree.IntersectPath(path, intersection)
    intersection.ToFieldMask(self)

  def MergeMessage(
      self, source, destination,
      replace_message_field=False, replace_repeated_field=False):
    """Merges fields specified in FieldMask from source to destination.

    Args:
      source: Source message.
      destination: The destination message to be merged into.
      replace_message_field: Replace message field if True. Merge message
          field if False.
      replace_repeated_field: Replace repeated field if True. Append
          elements of repeated field if False.
    """
    tree = _FieldMaskTree(self)
    tree.MergeMessage(
        source, destination, replace_message_field, replace_repeated_field)


def _IsValidPath(message_descriptor, path):
  """Checks whether the path is valid for Message Descriptor."""
  parts = path.split('.')
  last = parts.pop()
  for name in parts:
    field = message_descriptor.fields_by_name.get(name)
    if (field is None or
        field.label == FieldDescriptor.LABEL_REPEATED or
        field.type != FieldDescriptor.TYPE_MESSAGE):
      return False
    message_descriptor = field.message_type
  return last in message_descriptor.fields_by_name


def _CheckFieldMaskMessage(message):
  """Raises ValueError if message is not a FieldMask."""
  message_descriptor = message.DESCRIPTOR
  if (message_descriptor.name != 'FieldMask' or
      message_descriptor.file.name != 'google/protobuf/field_mask.proto'):
    raise ValueError('Message {0} is not a FieldMask.'.format(
        message_descriptor.full_name))


def _SnakeCaseToCamelCase(path_name):
  """Converts a path name from snake_case to camelCase."""
  result = []
  after_underscore = False
  for c in path_name:
    if c.isupper():
      raise ValueError(
          'Fail to print FieldMask to Json string: Path name '
          '{0} must not contain uppercase letters.'.format(path_name))
    if after_underscore:
      if c.islower():
        result.append(c.upper())
        after_underscore = False
      else:
        raise ValueError(
            'Fail to print FieldMask to Json string: The '
            'character after a "_" must be a lowercase letter '
            'in path name {0}.'.format(path_name))
    elif c == '_':
      after_underscore = True
    else:
      result += c

  if after_underscore:
    raise ValueError('Fail to print FieldMask to Json string: Trailing "_" '
                     'in path name {0}.'.format(path_name))
  return ''.join(result)


def _CamelCaseToSnakeCase(path_name):
  """Converts a field name from camelCase to snake_case."""
  result = []
  for c in path_name:
    if c == '_':
      raise ValueError('Fail to parse FieldMask: Path name '
                       '{0} must not contain "_"s.'.format(path_name))
    if c.isupper():
      result += '_'
      result += c.lower()
    else:
      result += c
  return ''.join(result)


class _FieldMaskTree(object):
  """Represents a FieldMask in a tree structure.

  For example, given a FieldMask "foo.bar,foo.baz,bar.baz",
  the FieldMaskTree will be:
      [_root] -+- foo -+- bar
            |       |
            |       +- baz
            |
            +- bar --- baz
  In the tree, each leaf node represents a field path.
  """

  __slots__ = ('_root',)

  def __init__(self, field_mask=None):
    """Initializes the tree by FieldMask."""
    self._root = {}
    if field_mask:
      self.MergeFromFieldMask(field_mask)

  def MergeFromFieldMask(self, field_mask):
    """Merges a FieldMask to the tree."""
    for path in field_mask.paths:
      self.AddPath(path)

  def AddPath(self, path):
    """Adds a field path into the tree.

    If the field path to add is a sub-path of an existing field path
    in the tree (i.e., a leaf node), it means the tree already matches
    the given path so nothing will be added to the tree. If the path
    matches an existing non-leaf node in the tree, that non-leaf node
    will be turned into a leaf node with all its children removed because
    the path matches all the node's children. Otherwise, a new path will
    be added.

    Args:
      path: The field path to add.
    """
    node = self._root
    for name in path.split('.'):
      if name not in node:
        node[name] = {}
      elif not node[name]:
        # Pre-existing empty node implies we already have this entire tree.
        return
      node = node[name]
    # Remove any sub-trees we might have had.
    node.clear()

  def ToFieldMask(self, field_mask):
    """Converts the tree to a FieldMask."""
    field_mask.Clear()
    _AddFieldPaths(self._root, '', field_mask)

  def IntersectPath(self, path, intersection):
    """Calculates the intersection part of a field path with this tree.

    Args:
      path: The field path to calculates.
      intersection: The out tree to record the intersection part.
    """
    node = self._root
    for name in path.split('.'):
      if name not in node:
        return
      elif not node[name]:
        intersection.AddPath(path)
        return
      node = node[name]
    intersection.AddLeafNodes(path, node)

  def AddLeafNodes(self, prefix, node):
    """Adds leaf nodes begin with prefix to this tree."""
    if not node:
      self.AddPath(prefix)
    for name in node:
      child_path = prefix + '.' + name
      self.AddLeafNodes(child_path, node[name])

  def MergeMessage(
      self, source, destination,
      replace_message, replace_repeated):
    """Merge all fields specified by this tree from source to destination."""
    _MergeMessage(
        self._root, source, destination, replace_message, replace_repeated)


def _StrConvert(value):
  """Converts value to str if it is not."""
  # This file is imported by c extension and some methods like ClearField
  # requires string for the field name. py2/py3 has different text
  # type and may use unicode.
  if not isinstance(value, str):
    return value.encode('utf-8')
  return value


def _MergeMessage(
    node, source, destination, replace_message, replace_repeated):
  """Merge all fields specified by a sub-tree from source to destination."""
  source_descriptor = source.DESCRIPTOR
  for name in node:
    child = node[name]
    field = source_descriptor.fields_by_name[name]
    if field is None:
      raise ValueError('Error: Can\'t find field {0} in message {1}.'.format(
          name, source_descriptor.full_name))
    if child:
      # Sub-paths are only allowed for singular message fields.
      if (field.label == FieldDescriptor.LABEL_REPEATED or
          field.cpp_type != FieldDescriptor.CPPTYPE_MESSAGE):
        raise ValueError('Error: Field {0} in message {1} is not a singular '
                         'message field and cannot have sub-fields.'.format(
                             name, source_descriptor.full_name))
      if source.HasField(name):
        _MergeMessage(
            child, getattr(source, name), getattr(destination, name),
            replace_message, replace_repeated)
      continue
    if field.label == FieldDescriptor.LABEL_REPEATED:
      if replace_repeated:
        destination.ClearField(_StrConvert(name))
      repeated_source = getattr(source, name)
      repeated_destination = getattr(destination, name)
      repeated_destination.MergeFrom(repeated_source)
    else:
      if field.cpp_type == FieldDescriptor.CPPTYPE_MESSAGE:
        if replace_message:
          destination.ClearField(_StrConvert(name))
        if source.HasField(name):
          getattr(destination, name).MergeFrom(getattr(source, name))
      else:
        setattr(destination, name, getattr(source, name))


def _AddFieldPaths(node, prefix, field_mask):
  """Adds the field paths descended from node to field_mask."""
  if not node and prefix:
    field_mask.paths.append(prefix)
    return
  for name in sorted(node):
    if prefix:
      child_path = prefix + '.' + name
    else:
      child_path = name
    _AddFieldPaths(node[name], child_path, field_mask)
