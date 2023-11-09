# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Helpers for :mod:`protobuf`."""

import collections
import collections.abc
import copy
import inspect

from google.protobuf import field_mask_pb2
from google.protobuf import message
from google.protobuf import wrappers_pb2


_SENTINEL = object()
_WRAPPER_TYPES = (
    wrappers_pb2.BoolValue,
    wrappers_pb2.BytesValue,
    wrappers_pb2.DoubleValue,
    wrappers_pb2.FloatValue,
    wrappers_pb2.Int32Value,
    wrappers_pb2.Int64Value,
    wrappers_pb2.StringValue,
    wrappers_pb2.UInt32Value,
    wrappers_pb2.UInt64Value,
)


def from_any_pb(pb_type, any_pb):
    """Converts an ``Any`` protobuf to the specified message type.

    Args:
        pb_type (type): the type of the message that any_pb stores an instance
            of.
        any_pb (google.protobuf.any_pb2.Any): the object to be converted.

    Returns:
        pb_type: An instance of the pb_type message.

    Raises:
        TypeError: if the message could not be converted.
    """
    msg = pb_type()

    # Unwrap proto-plus wrapped messages.
    if callable(getattr(pb_type, "pb", None)):
        msg_pb = pb_type.pb(msg)
    else:
        msg_pb = msg

    # Unpack the Any object and populate the protobuf message instance.
    if not any_pb.Unpack(msg_pb):
        raise TypeError(
            "Could not convert {} to {}".format(
                any_pb.__class__.__name__, pb_type.__name__
            )
        )

    # Done; return the message.
    return msg


def check_oneof(**kwargs):
    """Raise ValueError if more than one keyword argument is not ``None``.

    Args:
        kwargs (dict): The keyword arguments sent to the function.

    Raises:
        ValueError: If more than one entry in ``kwargs`` is not ``None``.
    """
    # Sanity check: If no keyword arguments were sent, this is fine.
    if not kwargs:
        return

    not_nones = [val for val in kwargs.values() if val is not None]
    if len(not_nones) > 1:
        raise ValueError(
            "Only one of {fields} should be set.".format(
                fields=", ".join(sorted(kwargs.keys()))
            )
        )


def get_messages(module):
    """Discovers all protobuf Message classes in a given import module.

    Args:
        module (module): A Python module; :func:`dir` will be run against this
            module to find Message subclasses.

    Returns:
        dict[str, google.protobuf.message.Message]: A dictionary with the
            Message class names as keys, and the Message subclasses themselves
            as values.
    """
    answer = collections.OrderedDict()
    for name in dir(module):
        candidate = getattr(module, name)
        if inspect.isclass(candidate) and issubclass(candidate, message.Message):
            answer[name] = candidate
    return answer


def _resolve_subkeys(key, separator="."):
    """Resolve a potentially nested key.

    If the key contains the ``separator`` (e.g. ``.``) then the key will be
    split on the first instance of the subkey::

       >>> _resolve_subkeys('a.b.c')
       ('a', 'b.c')
       >>> _resolve_subkeys('d|e|f', separator='|')
       ('d', 'e|f')

    If not, the subkey will be :data:`None`::

        >>> _resolve_subkeys('foo')
        ('foo', None)

    Args:
        key (str): A string that may or may not contain the separator.
        separator (str): The namespace separator. Defaults to `.`.

    Returns:
        Tuple[str, str]: The key and subkey(s).
    """
    parts = key.split(separator, 1)

    if len(parts) > 1:
        return parts
    else:
        return parts[0], None


def get(msg_or_dict, key, default=_SENTINEL):
    """Retrieve a key's value from a protobuf Message or dictionary.

    Args:
        mdg_or_dict (Union[~google.protobuf.message.Message, Mapping]): the
            object.
        key (str): The key to retrieve from the object.
        default (Any): If the key is not present on the object, and a default
            is set, returns that default instead. A type-appropriate falsy
            default is generally recommended, as protobuf messages almost
            always have default values for unset values and it is not always
            possible to tell the difference between a falsy value and an
            unset one. If no default is set then :class:`KeyError` will be
            raised if the key is not present in the object.

    Returns:
        Any: The return value from the underlying Message or dict.

    Raises:
        KeyError: If the key is not found. Note that, for unset values,
            messages and dictionaries may not have consistent behavior.
        TypeError: If ``msg_or_dict`` is not a Message or Mapping.
    """
    # We may need to get a nested key. Resolve this.
    key, subkey = _resolve_subkeys(key)

    # Attempt to get the value from the two types of objects we know about.
    # If we get something else, complain.
    if isinstance(msg_or_dict, message.Message):
        answer = getattr(msg_or_dict, key, default)
    elif isinstance(msg_or_dict, collections.abc.Mapping):
        answer = msg_or_dict.get(key, default)
    else:
        raise TypeError(
            "get() expected a dict or protobuf message, got {!r}.".format(
                type(msg_or_dict)
            )
        )

    # If the object we got back is our sentinel, raise KeyError; this is
    # a "not found" case.
    if answer is _SENTINEL:
        raise KeyError(key)

    # If a subkey exists, call this method recursively against the answer.
    if subkey is not None and answer is not default:
        return get(answer, subkey, default=default)

    return answer


def _set_field_on_message(msg, key, value):
    """Set helper for protobuf Messages."""
    # Attempt to set the value on the types of objects we know how to deal
    # with.
    if isinstance(value, (collections.abc.MutableSequence, tuple)):
        # Clear the existing repeated protobuf message of any elements
        # currently inside it.
        while getattr(msg, key):
            getattr(msg, key).pop()

        # Write our new elements to the repeated field.
        for item in value:
            if isinstance(item, collections.abc.Mapping):
                getattr(msg, key).add(**item)
            else:
                # protobuf's RepeatedCompositeContainer doesn't support
                # append.
                getattr(msg, key).extend([item])
    elif isinstance(value, collections.abc.Mapping):
        # Assign the dictionary values to the protobuf message.
        for item_key, item_value in value.items():
            set(getattr(msg, key), item_key, item_value)
    elif isinstance(value, message.Message):
        getattr(msg, key).CopyFrom(value)
    else:
        setattr(msg, key, value)


def set(msg_or_dict, key, value):
    """Set a key's value on a protobuf Message or dictionary.

    Args:
        msg_or_dict (Union[~google.protobuf.message.Message, Mapping]): the
            object.
        key (str): The key to set.
        value (Any): The value to set.

    Raises:
        TypeError: If ``msg_or_dict`` is not a Message or dictionary.
    """
    # Sanity check: Is our target object valid?
    if not isinstance(msg_or_dict, (collections.abc.MutableMapping, message.Message)):
        raise TypeError(
            "set() expected a dict or protobuf message, got {!r}.".format(
                type(msg_or_dict)
            )
        )

    # We may be setting a nested key. Resolve this.
    basekey, subkey = _resolve_subkeys(key)

    # If a subkey exists, then get that object and call this method
    # recursively against it using the subkey.
    if subkey is not None:
        if isinstance(msg_or_dict, collections.abc.MutableMapping):
            msg_or_dict.setdefault(basekey, {})
        set(get(msg_or_dict, basekey), subkey, value)
        return

    if isinstance(msg_or_dict, collections.abc.MutableMapping):
        msg_or_dict[key] = value
    else:
        _set_field_on_message(msg_or_dict, key, value)


def setdefault(msg_or_dict, key, value):
    """Set the key on a protobuf Message or dictionary to a given value if the
    current value is falsy.

    Because protobuf Messages do not distinguish between unset values and
    falsy ones particularly well (by design), this method treats any falsy
    value (e.g. 0, empty list) as a target to be overwritten, on both Messages
    and dictionaries.

    Args:
        msg_or_dict (Union[~google.protobuf.message.Message, Mapping]): the
            object.
        key (str): The key on the object in question.
        value (Any): The value to set.

    Raises:
        TypeError: If ``msg_or_dict`` is not a Message or dictionary.
    """
    if not get(msg_or_dict, key, default=None):
        set(msg_or_dict, key, value)


def field_mask(original, modified):
    """Create a field mask by comparing two messages.

    Args:
        original (~google.protobuf.message.Message): the original message.
            If set to None, this field will be interpretted as an empty
            message.
        modified (~google.protobuf.message.Message): the modified message.
            If set to None, this field will be interpretted as an empty
            message.

    Returns:
        google.protobuf.field_mask_pb2.FieldMask: field mask that contains
        the list of field names that have different values between the two
        messages. If the messages are equivalent, then the field mask is empty.

    Raises:
        ValueError: If the ``original`` or ``modified`` are not the same type.
    """
    if original is None and modified is None:
        return field_mask_pb2.FieldMask()

    if original is None and modified is not None:
        original = copy.deepcopy(modified)
        original.Clear()

    if modified is None and original is not None:
        modified = copy.deepcopy(original)
        modified.Clear()

    if type(original) != type(modified):
        raise ValueError(
            "expected that both original and modified should be of the "
            'same type, received "{!r}" and "{!r}".'.format(
                type(original), type(modified)
            )
        )

    return field_mask_pb2.FieldMask(paths=_field_mask_helper(original, modified))


def _field_mask_helper(original, modified, current=""):
    answer = []

    for name in original.DESCRIPTOR.fields_by_name:
        field_path = _get_path(current, name)

        original_val = getattr(original, name)
        modified_val = getattr(modified, name)

        if _is_message(original_val) or _is_message(modified_val):
            if original_val != modified_val:
                # Wrapper types do not need to include the .value part of the
                # path.
                if _is_wrapper(original_val) or _is_wrapper(modified_val):
                    answer.append(field_path)
                elif not modified_val.ListFields():
                    answer.append(field_path)
                else:
                    answer.extend(
                        _field_mask_helper(original_val, modified_val, field_path)
                    )
        else:
            if original_val != modified_val:
                answer.append(field_path)

    return answer


def _get_path(current, name):
    # gapic-generator-python appends underscores to field names
    # that collide with python keywords.
    # `_` is stripped away as it is not possible to
    # natively define a field with a trailing underscore in protobuf.
    # APIs will reject field masks if fields have trailing underscores.
    # See https://github.com/googleapis/python-api-core/issues/227
    name = name.rstrip("_")
    if not current:
        return name
    return "%s.%s" % (current, name)


def _is_message(value):
    return isinstance(value, message.Message)


def _is_wrapper(value):
    return type(value) in _WRAPPER_TYPES
