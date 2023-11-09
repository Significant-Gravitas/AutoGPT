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

"""Contains well known classes.

This files defines well known classes which need extra maintenance including:
  - Any
  - Duration
  - FieldMask
  - Struct
  - Timestamp
"""

__author__ = 'jieluo@google.com (Jie Luo)'

import calendar
import collections.abc
import datetime

from google.protobuf.internal import field_mask

FieldMask = field_mask.FieldMask

_TIMESTAMPFOMAT = '%Y-%m-%dT%H:%M:%S'
_NANOS_PER_SECOND = 1000000000
_NANOS_PER_MILLISECOND = 1000000
_NANOS_PER_MICROSECOND = 1000
_MILLIS_PER_SECOND = 1000
_MICROS_PER_SECOND = 1000000
_SECONDS_PER_DAY = 24 * 3600
_DURATION_SECONDS_MAX = 315576000000


class Any(object):
  """Class for Any Message type."""

  __slots__ = ()

  def Pack(self, msg, type_url_prefix='type.googleapis.com/',
           deterministic=None):
    """Packs the specified message into current Any message."""
    if len(type_url_prefix) < 1 or type_url_prefix[-1] != '/':
      self.type_url = '%s/%s' % (type_url_prefix, msg.DESCRIPTOR.full_name)
    else:
      self.type_url = '%s%s' % (type_url_prefix, msg.DESCRIPTOR.full_name)
    self.value = msg.SerializeToString(deterministic=deterministic)

  def Unpack(self, msg):
    """Unpacks the current Any message into specified message."""
    descriptor = msg.DESCRIPTOR
    if not self.Is(descriptor):
      return False
    msg.ParseFromString(self.value)
    return True

  def TypeName(self):
    """Returns the protobuf type name of the inner message."""
    # Only last part is to be used: b/25630112
    return self.type_url.split('/')[-1]

  def Is(self, descriptor):
    """Checks if this Any represents the given protobuf type."""
    return '/' in self.type_url and self.TypeName() == descriptor.full_name


_EPOCH_DATETIME_NAIVE = datetime.datetime.utcfromtimestamp(0)
_EPOCH_DATETIME_AWARE = datetime.datetime.fromtimestamp(
    0, tz=datetime.timezone.utc)


class Timestamp(object):
  """Class for Timestamp message type."""

  __slots__ = ()

  def ToJsonString(self):
    """Converts Timestamp to RFC 3339 date string format.

    Returns:
      A string converted from timestamp. The string is always Z-normalized
      and uses 3, 6 or 9 fractional digits as required to represent the
      exact time. Example of the return format: '1972-01-01T10:00:20.021Z'
    """
    nanos = self.nanos % _NANOS_PER_SECOND
    total_sec = self.seconds + (self.nanos - nanos) // _NANOS_PER_SECOND
    seconds = total_sec % _SECONDS_PER_DAY
    days = (total_sec - seconds) // _SECONDS_PER_DAY
    dt = datetime.datetime(1970, 1, 1) + datetime.timedelta(days, seconds)

    result = dt.isoformat()
    if (nanos % 1e9) == 0:
      # If there are 0 fractional digits, the fractional
      # point '.' should be omitted when serializing.
      return result + 'Z'
    if (nanos % 1e6) == 0:
      # Serialize 3 fractional digits.
      return result + '.%03dZ' % (nanos / 1e6)
    if (nanos % 1e3) == 0:
      # Serialize 6 fractional digits.
      return result + '.%06dZ' % (nanos / 1e3)
    # Serialize 9 fractional digits.
    return result + '.%09dZ' % nanos

  def FromJsonString(self, value):
    """Parse a RFC 3339 date string format to Timestamp.

    Args:
      value: A date string. Any fractional digits (or none) and any offset are
          accepted as long as they fit into nano-seconds precision.
          Example of accepted format: '1972-01-01T10:00:20.021-05:00'

    Raises:
      ValueError: On parsing problems.
    """
    if not isinstance(value, str):
      raise ValueError('Timestamp JSON value not a string: {!r}'.format(value))
    timezone_offset = value.find('Z')
    if timezone_offset == -1:
      timezone_offset = value.find('+')
    if timezone_offset == -1:
      timezone_offset = value.rfind('-')
    if timezone_offset == -1:
      raise ValueError(
          'Failed to parse timestamp: missing valid timezone offset.')
    time_value = value[0:timezone_offset]
    # Parse datetime and nanos.
    point_position = time_value.find('.')
    if point_position == -1:
      second_value = time_value
      nano_value = ''
    else:
      second_value = time_value[:point_position]
      nano_value = time_value[point_position + 1:]
    if 't' in second_value:
      raise ValueError(
          'time data \'{0}\' does not match format \'%Y-%m-%dT%H:%M:%S\', '
          'lowercase \'t\' is not accepted'.format(second_value))
    date_object = datetime.datetime.strptime(second_value, _TIMESTAMPFOMAT)
    td = date_object - datetime.datetime(1970, 1, 1)
    seconds = td.seconds + td.days * _SECONDS_PER_DAY
    if len(nano_value) > 9:
      raise ValueError(
          'Failed to parse Timestamp: nanos {0} more than '
          '9 fractional digits.'.format(nano_value))
    if nano_value:
      nanos = round(float('0.' + nano_value) * 1e9)
    else:
      nanos = 0
    # Parse timezone offsets.
    if value[timezone_offset] == 'Z':
      if len(value) != timezone_offset + 1:
        raise ValueError('Failed to parse timestamp: invalid trailing'
                         ' data {0}.'.format(value))
    else:
      timezone = value[timezone_offset:]
      pos = timezone.find(':')
      if pos == -1:
        raise ValueError(
            'Invalid timezone offset value: {0}.'.format(timezone))
      if timezone[0] == '+':
        seconds -= (int(timezone[1:pos])*60+int(timezone[pos+1:]))*60
      else:
        seconds += (int(timezone[1:pos])*60+int(timezone[pos+1:]))*60
    # Set seconds and nanos
    self.seconds = int(seconds)
    self.nanos = int(nanos)

  def GetCurrentTime(self):
    """Get the current UTC into Timestamp."""
    self.FromDatetime(datetime.datetime.utcnow())

  def ToNanoseconds(self):
    """Converts Timestamp to nanoseconds since epoch."""
    return self.seconds * _NANOS_PER_SECOND + self.nanos

  def ToMicroseconds(self):
    """Converts Timestamp to microseconds since epoch."""
    return (self.seconds * _MICROS_PER_SECOND +
            self.nanos // _NANOS_PER_MICROSECOND)

  def ToMilliseconds(self):
    """Converts Timestamp to milliseconds since epoch."""
    return (self.seconds * _MILLIS_PER_SECOND +
            self.nanos // _NANOS_PER_MILLISECOND)

  def ToSeconds(self):
    """Converts Timestamp to seconds since epoch."""
    return self.seconds

  def FromNanoseconds(self, nanos):
    """Converts nanoseconds since epoch to Timestamp."""
    self.seconds = nanos // _NANOS_PER_SECOND
    self.nanos = nanos % _NANOS_PER_SECOND

  def FromMicroseconds(self, micros):
    """Converts microseconds since epoch to Timestamp."""
    self.seconds = micros // _MICROS_PER_SECOND
    self.nanos = (micros % _MICROS_PER_SECOND) * _NANOS_PER_MICROSECOND

  def FromMilliseconds(self, millis):
    """Converts milliseconds since epoch to Timestamp."""
    self.seconds = millis // _MILLIS_PER_SECOND
    self.nanos = (millis % _MILLIS_PER_SECOND) * _NANOS_PER_MILLISECOND

  def FromSeconds(self, seconds):
    """Converts seconds since epoch to Timestamp."""
    self.seconds = seconds
    self.nanos = 0

  def ToDatetime(self, tzinfo=None):
    """Converts Timestamp to a datetime.

    Args:
      tzinfo: A datetime.tzinfo subclass; defaults to None.

    Returns:
      If tzinfo is None, returns a timezone-naive UTC datetime (with no timezone
      information, i.e. not aware that it's UTC).

      Otherwise, returns a timezone-aware datetime in the input timezone.
    """
    delta = datetime.timedelta(
        seconds=self.seconds,
        microseconds=_RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND))
    if tzinfo is None:
      return _EPOCH_DATETIME_NAIVE + delta
    else:
      return _EPOCH_DATETIME_AWARE.astimezone(tzinfo) + delta

  def FromDatetime(self, dt):
    """Converts datetime to Timestamp.

    Args:
      dt: A datetime. If it's timezone-naive, it's assumed to be in UTC.
    """
    # Using this guide: http://wiki.python.org/moin/WorkingWithTime
    # And this conversion guide: http://docs.python.org/library/time.html

    # Turn the date parameter into a tuple (struct_time) that can then be
    # manipulated into a long value of seconds.  During the conversion from
    # struct_time to long, the source date in UTC, and so it follows that the
    # correct transformation is calendar.timegm()
    self.seconds = calendar.timegm(dt.utctimetuple())
    self.nanos = dt.microsecond * _NANOS_PER_MICROSECOND


class Duration(object):
  """Class for Duration message type."""

  __slots__ = ()

  def ToJsonString(self):
    """Converts Duration to string format.

    Returns:
      A string converted from self. The string format will contains
      3, 6, or 9 fractional digits depending on the precision required to
      represent the exact Duration value. For example: "1s", "1.010s",
      "1.000000100s", "-3.100s"
    """
    _CheckDurationValid(self.seconds, self.nanos)
    if self.seconds < 0 or self.nanos < 0:
      result = '-'
      seconds = - self.seconds + int((0 - self.nanos) // 1e9)
      nanos = (0 - self.nanos) % 1e9
    else:
      result = ''
      seconds = self.seconds + int(self.nanos // 1e9)
      nanos = self.nanos % 1e9
    result += '%d' % seconds
    if (nanos % 1e9) == 0:
      # If there are 0 fractional digits, the fractional
      # point '.' should be omitted when serializing.
      return result + 's'
    if (nanos % 1e6) == 0:
      # Serialize 3 fractional digits.
      return result + '.%03ds' % (nanos / 1e6)
    if (nanos % 1e3) == 0:
      # Serialize 6 fractional digits.
      return result + '.%06ds' % (nanos / 1e3)
    # Serialize 9 fractional digits.
    return result + '.%09ds' % nanos

  def FromJsonString(self, value):
    """Converts a string to Duration.

    Args:
      value: A string to be converted. The string must end with 's'. Any
          fractional digits (or none) are accepted as long as they fit into
          precision. For example: "1s", "1.01s", "1.0000001s", "-3.100s

    Raises:
      ValueError: On parsing problems.
    """
    if not isinstance(value, str):
      raise ValueError('Duration JSON value not a string: {!r}'.format(value))
    if len(value) < 1 or value[-1] != 's':
      raise ValueError(
          'Duration must end with letter "s": {0}.'.format(value))
    try:
      pos = value.find('.')
      if pos == -1:
        seconds = int(value[:-1])
        nanos = 0
      else:
        seconds = int(value[:pos])
        if value[0] == '-':
          nanos = int(round(float('-0{0}'.format(value[pos: -1])) *1e9))
        else:
          nanos = int(round(float('0{0}'.format(value[pos: -1])) *1e9))
      _CheckDurationValid(seconds, nanos)
      self.seconds = seconds
      self.nanos = nanos
    except ValueError as e:
      raise ValueError(
          'Couldn\'t parse duration: {0} : {1}.'.format(value, e))

  def ToNanoseconds(self):
    """Converts a Duration to nanoseconds."""
    return self.seconds * _NANOS_PER_SECOND + self.nanos

  def ToMicroseconds(self):
    """Converts a Duration to microseconds."""
    micros = _RoundTowardZero(self.nanos, _NANOS_PER_MICROSECOND)
    return self.seconds * _MICROS_PER_SECOND + micros

  def ToMilliseconds(self):
    """Converts a Duration to milliseconds."""
    millis = _RoundTowardZero(self.nanos, _NANOS_PER_MILLISECOND)
    return self.seconds * _MILLIS_PER_SECOND + millis

  def ToSeconds(self):
    """Converts a Duration to seconds."""
    return self.seconds

  def FromNanoseconds(self, nanos):
    """Converts nanoseconds to Duration."""
    self._NormalizeDuration(nanos // _NANOS_PER_SECOND,
                            nanos % _NANOS_PER_SECOND)

  def FromMicroseconds(self, micros):
    """Converts microseconds to Duration."""
    self._NormalizeDuration(
        micros // _MICROS_PER_SECOND,
        (micros % _MICROS_PER_SECOND) * _NANOS_PER_MICROSECOND)

  def FromMilliseconds(self, millis):
    """Converts milliseconds to Duration."""
    self._NormalizeDuration(
        millis // _MILLIS_PER_SECOND,
        (millis % _MILLIS_PER_SECOND) * _NANOS_PER_MILLISECOND)

  def FromSeconds(self, seconds):
    """Converts seconds to Duration."""
    self.seconds = seconds
    self.nanos = 0

  def ToTimedelta(self):
    """Converts Duration to timedelta."""
    return datetime.timedelta(
        seconds=self.seconds, microseconds=_RoundTowardZero(
            self.nanos, _NANOS_PER_MICROSECOND))

  def FromTimedelta(self, td):
    """Converts timedelta to Duration."""
    self._NormalizeDuration(td.seconds + td.days * _SECONDS_PER_DAY,
                            td.microseconds * _NANOS_PER_MICROSECOND)

  def _NormalizeDuration(self, seconds, nanos):
    """Set Duration by seconds and nanos."""
    # Force nanos to be negative if the duration is negative.
    if seconds < 0 and nanos > 0:
      seconds += 1
      nanos -= _NANOS_PER_SECOND
    self.seconds = seconds
    self.nanos = nanos


def _CheckDurationValid(seconds, nanos):
  if seconds < -_DURATION_SECONDS_MAX or seconds > _DURATION_SECONDS_MAX:
    raise ValueError(
        'Duration is not valid: Seconds {0} must be in range '
        '[-315576000000, 315576000000].'.format(seconds))
  if nanos <= -_NANOS_PER_SECOND or nanos >= _NANOS_PER_SECOND:
    raise ValueError(
        'Duration is not valid: Nanos {0} must be in range '
        '[-999999999, 999999999].'.format(nanos))
  if (nanos < 0 and seconds > 0) or (nanos > 0 and seconds < 0):
    raise ValueError(
        'Duration is not valid: Sign mismatch.')


def _RoundTowardZero(value, divider):
  """Truncates the remainder part after division."""
  # For some languages, the sign of the remainder is implementation
  # dependent if any of the operands is negative. Here we enforce
  # "rounded toward zero" semantics. For example, for (-5) / 2 an
  # implementation may give -3 as the result with the remainder being
  # 1. This function ensures we always return -2 (closer to zero).
  result = value // divider
  remainder = value % divider
  if result < 0 and remainder > 0:
    return result + 1
  else:
    return result


def _SetStructValue(struct_value, value):
  if value is None:
    struct_value.null_value = 0
  elif isinstance(value, bool):
    # Note: this check must come before the number check because in Python
    # True and False are also considered numbers.
    struct_value.bool_value = value
  elif isinstance(value, str):
    struct_value.string_value = value
  elif isinstance(value, (int, float)):
    struct_value.number_value = value
  elif isinstance(value, (dict, Struct)):
    struct_value.struct_value.Clear()
    struct_value.struct_value.update(value)
  elif isinstance(value, (list, ListValue)):
    struct_value.list_value.Clear()
    struct_value.list_value.extend(value)
  else:
    raise ValueError('Unexpected type')


def _GetStructValue(struct_value):
  which = struct_value.WhichOneof('kind')
  if which == 'struct_value':
    return struct_value.struct_value
  elif which == 'null_value':
    return None
  elif which == 'number_value':
    return struct_value.number_value
  elif which == 'string_value':
    return struct_value.string_value
  elif which == 'bool_value':
    return struct_value.bool_value
  elif which == 'list_value':
    return struct_value.list_value
  elif which is None:
    raise ValueError('Value not set')


class Struct(object):
  """Class for Struct message type."""

  __slots__ = ()

  def __getitem__(self, key):
    return _GetStructValue(self.fields[key])

  def __contains__(self, item):
    return item in self.fields

  def __setitem__(self, key, value):
    _SetStructValue(self.fields[key], value)

  def __delitem__(self, key):
    del self.fields[key]

  def __len__(self):
    return len(self.fields)

  def __iter__(self):
    return iter(self.fields)

  def keys(self):  # pylint: disable=invalid-name
    return self.fields.keys()

  def values(self):  # pylint: disable=invalid-name
    return [self[key] for key in self]

  def items(self):  # pylint: disable=invalid-name
    return [(key, self[key]) for key in self]

  def get_or_create_list(self, key):
    """Returns a list for this key, creating if it didn't exist already."""
    if not self.fields[key].HasField('list_value'):
      # Clear will mark list_value modified which will indeed create a list.
      self.fields[key].list_value.Clear()
    return self.fields[key].list_value

  def get_or_create_struct(self, key):
    """Returns a struct for this key, creating if it didn't exist already."""
    if not self.fields[key].HasField('struct_value'):
      # Clear will mark struct_value modified which will indeed create a struct.
      self.fields[key].struct_value.Clear()
    return self.fields[key].struct_value

  def update(self, dictionary):  # pylint: disable=invalid-name
    for key, value in dictionary.items():
      _SetStructValue(self.fields[key], value)

collections.abc.MutableMapping.register(Struct)


class ListValue(object):
  """Class for ListValue message type."""

  __slots__ = ()

  def __len__(self):
    return len(self.values)

  def append(self, value):
    _SetStructValue(self.values.add(), value)

  def extend(self, elem_seq):
    for value in elem_seq:
      self.append(value)

  def __getitem__(self, index):
    """Retrieves item by the specified index."""
    return _GetStructValue(self.values.__getitem__(index))

  def __setitem__(self, index, value):
    _SetStructValue(self.values.__getitem__(index), value)

  def __delitem__(self, key):
    del self.values[key]

  def items(self):
    for i in range(len(self)):
      yield self[i]

  def add_struct(self):
    """Appends and returns a struct value as the next value in the list."""
    struct_value = self.values.add().struct_value
    # Clear will mark struct_value modified which will indeed create a struct.
    struct_value.Clear()
    return struct_value

  def add_list(self):
    """Appends and returns a list value as the next value in the list."""
    list_value = self.values.add().list_value
    # Clear will mark list_value modified which will indeed create a list.
    list_value.Clear()
    return list_value

collections.abc.MutableSequence.register(ListValue)


# LINT.IfChange(wktbases)
WKTBASES = {
    'google.protobuf.Any': Any,
    'google.protobuf.Duration': Duration,
    'google.protobuf.FieldMask': FieldMask,
    'google.protobuf.ListValue': ListValue,
    'google.protobuf.Struct': Struct,
    'google.protobuf.Timestamp': Timestamp,
}
# LINT.ThenChange(//depot/google.protobuf/compiler/python/pyi_generator.cc:wktbases)
