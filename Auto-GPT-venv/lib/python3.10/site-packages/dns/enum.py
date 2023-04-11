# Copyright (C) Dnspython Contributors, see LICENSE for text of ISC license

# Copyright (C) 2003-2017 Nominum, Inc.
#
# Permission to use, copy, modify, and distribute this software and its
# documentation for any purpose with or without fee is hereby granted,
# provided that the above copyright notice and this permission notice
# appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND NOMINUM DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL NOMINUM BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT
# OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

import enum


class IntEnum(enum.IntEnum):
    @classmethod
    def _check_value(cls, value):
        max = cls._maximum()
        if value < 0 or value > max:
            name = cls._short_name()
            raise ValueError(f"{name} must be between >= 0 and <= {max}")

    @classmethod
    def from_text(cls, text):
        text = text.upper()
        try:
            return cls[text]
        except KeyError:
            pass
        value = cls._extra_from_text(text)
        if value:
            return value
        prefix = cls._prefix()
        if text.startswith(prefix) and text[len(prefix) :].isdigit():
            value = int(text[len(prefix) :])
            cls._check_value(value)
            try:
                return cls(value)
            except ValueError:
                return value
        raise cls._unknown_exception_class()

    @classmethod
    def to_text(cls, value):
        cls._check_value(value)
        try:
            text = cls(value).name
        except ValueError:
            text = None
        text = cls._extra_to_text(value, text)
        if text is None:
            text = f"{cls._prefix()}{value}"
        return text

    @classmethod
    def make(cls, value):
        """Convert text or a value into an enumerated type, if possible.

        *value*, the ``int`` or ``str`` to convert.

        Raises a class-specific exception if a ``str`` is provided that
        cannot be converted.

        Raises ``ValueError`` if the value is out of range.

        Returns an enumeration from the calling class corresponding to the
        value, if one is defined, or an ``int`` otherwise.
        """

        if isinstance(value, str):
            return cls.from_text(value)
        cls._check_value(value)
        try:
            return cls(value)
        except ValueError:
            return value

    @classmethod
    def _maximum(cls):
        raise NotImplementedError  # pragma: no cover

    @classmethod
    def _short_name(cls):
        return cls.__name__.lower()

    @classmethod
    def _prefix(cls):
        return ""

    @classmethod
    def _extra_from_text(cls, text):  # pylint: disable=W0613
        return None

    @classmethod
    def _extra_to_text(cls, value, current_text):  # pylint: disable=W0613
        return current_text

    @classmethod
    def _unknown_exception_class(cls):
        return ValueError
