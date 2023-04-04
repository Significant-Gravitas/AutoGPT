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

"""Helpers for :mod:`datetime`."""

import calendar
import datetime
import re

from google.protobuf import timestamp_pb2


_UTC_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
_RFC3339_MICROS = "%Y-%m-%dT%H:%M:%S.%fZ"
_RFC3339_NO_FRACTION = "%Y-%m-%dT%H:%M:%S"
# datetime.strptime cannot handle nanosecond precision:  parse w/ regex
_RFC3339_NANOS = re.compile(
    r"""
    (?P<no_fraction>
        \d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}  # YYYY-MM-DDTHH:MM:SS
    )
    (                                        # Optional decimal part
     \.                                      # decimal point
     (?P<nanos>\d{1,9})                      # nanoseconds, maybe truncated
    )?
    Z                                        # Zulu
""",
    re.VERBOSE,
)


def utcnow():
    """A :meth:`datetime.datetime.utcnow()` alias to allow mocking in tests."""
    return datetime.datetime.utcnow()


def to_milliseconds(value):
    """Convert a zone-aware datetime to milliseconds since the unix epoch.

    Args:
        value (datetime.datetime): The datetime to covert.

    Returns:
        int: Milliseconds since the unix epoch.
    """
    micros = to_microseconds(value)
    return micros // 1000


def from_microseconds(value):
    """Convert timestamp in microseconds since the unix epoch to datetime.

    Args:
        value (float): The timestamp to convert, in microseconds.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp in
            UTC.
    """
    return _UTC_EPOCH + datetime.timedelta(microseconds=value)


def to_microseconds(value):
    """Convert a datetime to microseconds since the unix epoch.

    Args:
        value (datetime.datetime): The datetime to covert.

    Returns:
        int: Microseconds since the unix epoch.
    """
    if not value.tzinfo:
        value = value.replace(tzinfo=datetime.timezone.utc)
    # Regardless of what timezone is on the value, convert it to UTC.
    value = value.astimezone(datetime.timezone.utc)
    # Convert the datetime to a microsecond timestamp.
    return int(calendar.timegm(value.timetuple()) * 1e6) + value.microsecond


def from_iso8601_date(value):
    """Convert a ISO8601 date string to a date.

    Args:
        value (str): The ISO8601 date string.

    Returns:
        datetime.date: A date equivalent to the date string.
    """
    return datetime.datetime.strptime(value, "%Y-%m-%d").date()


def from_iso8601_time(value):
    """Convert a zoneless ISO8601 time string to a time.

    Args:
        value (str): The ISO8601 time string.

    Returns:
        datetime.time: A time equivalent to the time string.
    """
    return datetime.datetime.strptime(value, "%H:%M:%S").time()


def from_rfc3339(value):
    """Convert an RFC3339-format timestamp to a native datetime.

    Supported formats include those without fractional seconds, or with
    any fraction up to nanosecond precision.

    .. note::
        Python datetimes do not support nanosecond precision; this function
        therefore truncates such values to microseconds.

    Args:
        value (str): The RFC3339 string to convert.

    Returns:
        datetime.datetime: The datetime object equivalent to the timestamp
        in UTC.

    Raises:
        ValueError: If the timestamp does not match the RFC3339
            regular expression.
    """
    with_nanos = _RFC3339_NANOS.match(value)

    if with_nanos is None:
        raise ValueError(
            "Timestamp: {!r}, does not match pattern: {!r}".format(
                value, _RFC3339_NANOS.pattern
            )
        )

    bare_seconds = datetime.datetime.strptime(
        with_nanos.group("no_fraction"), _RFC3339_NO_FRACTION
    )
    fraction = with_nanos.group("nanos")

    if fraction is None:
        micros = 0
    else:
        scale = 9 - len(fraction)
        nanos = int(fraction) * (10**scale)
        micros = nanos // 1000

    return bare_seconds.replace(microsecond=micros, tzinfo=datetime.timezone.utc)


from_rfc3339_nanos = from_rfc3339  # from_rfc3339_nanos method was deprecated.


def to_rfc3339(value, ignore_zone=True):
    """Convert a datetime to an RFC3339 timestamp string.

    Args:
        value (datetime.datetime):
            The datetime object to be converted to a string.
        ignore_zone (bool): If True, then the timezone (if any) of the
            datetime object is ignored and the datetime is treated as UTC.

    Returns:
        str: The RFC3339 formated string representing the datetime.
    """
    if not ignore_zone and value.tzinfo is not None:
        # Convert to UTC and remove the time zone info.
        value = value.replace(tzinfo=None) - value.utcoffset()

    return value.strftime(_RFC3339_MICROS)


class DatetimeWithNanoseconds(datetime.datetime):
    """Track nanosecond in addition to normal datetime attrs.

    Nanosecond can be passed only as a keyword argument.
    """

    __slots__ = ("_nanosecond",)

    # pylint: disable=arguments-differ
    def __new__(cls, *args, **kw):
        nanos = kw.pop("nanosecond", 0)
        if nanos > 0:
            if "microsecond" in kw:
                raise TypeError("Specify only one of 'microsecond' or 'nanosecond'")
            kw["microsecond"] = nanos // 1000
        inst = datetime.datetime.__new__(cls, *args, **kw)
        inst._nanosecond = nanos or 0
        return inst

    # pylint: disable=arguments-differ

    @property
    def nanosecond(self):
        """Read-only: nanosecond precision."""
        return self._nanosecond

    def rfc3339(self):
        """Return an RFC3339-compliant timestamp.

        Returns:
            (str): Timestamp string according to RFC3339 spec.
        """
        if self._nanosecond == 0:
            return to_rfc3339(self)
        nanos = str(self._nanosecond).rjust(9, "0").rstrip("0")
        return "{}.{}Z".format(self.strftime(_RFC3339_NO_FRACTION), nanos)

    @classmethod
    def from_rfc3339(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (str): RFC3339 stamp, with up to nanosecond precision

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp string

        Raises:
            ValueError: if `stamp` does not match the expected format
        """
        with_nanos = _RFC3339_NANOS.match(stamp)
        if with_nanos is None:
            raise ValueError(
                "Timestamp: {}, does not match pattern: {}".format(
                    stamp, _RFC3339_NANOS.pattern
                )
            )
        bare = datetime.datetime.strptime(
            with_nanos.group("no_fraction"), _RFC3339_NO_FRACTION
        )
        fraction = with_nanos.group("nanos")
        if fraction is None:
            nanos = 0
        else:
            scale = 9 - len(fraction)
            nanos = int(fraction) * (10**scale)
        return cls(
            bare.year,
            bare.month,
            bare.day,
            bare.hour,
            bare.minute,
            bare.second,
            nanosecond=nanos,
            tzinfo=datetime.timezone.utc,
        )

    def timestamp_pb(self):
        """Return a timestamp message.

        Returns:
            (:class:`~google.protobuf.timestamp_pb2.Timestamp`): Timestamp message
        """
        inst = (
            self
            if self.tzinfo is not None
            else self.replace(tzinfo=datetime.timezone.utc)
        )
        delta = inst - _UTC_EPOCH
        seconds = int(delta.total_seconds())
        nanos = self._nanosecond or self.microsecond * 1000
        return timestamp_pb2.Timestamp(seconds=seconds, nanos=nanos)

    @classmethod
    def from_timestamp_pb(cls, stamp):
        """Parse RFC3339-compliant timestamp, preserving nanoseconds.

        Args:
            stamp (:class:`~google.protobuf.timestamp_pb2.Timestamp`): timestamp message

        Returns:
            :class:`DatetimeWithNanoseconds`:
                an instance matching the timestamp message
        """
        microseconds = int(stamp.seconds * 1e6)
        bare = from_microseconds(microseconds)
        return cls(
            bare.year,
            bare.month,
            bare.day,
            bare.hour,
            bare.minute,
            bare.second,
            nanosecond=stamp.nanos,
            tzinfo=datetime.timezone.utc,
        )
