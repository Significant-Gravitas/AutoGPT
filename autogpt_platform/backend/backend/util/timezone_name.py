"""
Time zone name validation and serialization.

This file is adapted from pydantic-extra-types:
https://github.com/pydantic/pydantic-extra-types/blob/main/pydantic_extra_types/timezone_name.py

The MIT License (MIT)

Copyright (c) 2023 Samuel Colvin and other contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Modifications:
- Modified to always use pytz for timezone data to ensure consistency across environments
- Removed zoneinfo support to prevent environment-specific timezone lists
"""

from __future__ import annotations

from typing import Any, Callable, cast

import pytz
from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler
from pydantic_core import PydanticCustomError, core_schema

# Cache the timezones at module level to avoid repeated computation
ALL_TIMEZONES: set[str] = set(pytz.all_timezones)


def get_timezones() -> set[str]:
    """Get timezones from pytz for consistency across all environments."""
    # Return cached timezone set
    return ALL_TIMEZONES


class TimeZoneNameSettings(type):
    def __new__(
        cls, name: str, bases: tuple[type, ...], dct: dict[str, Any], **kwargs: Any
    ) -> type[TimeZoneName]:
        dct["strict"] = kwargs.pop("strict", True)
        return cast("type[TimeZoneName]", super().__new__(cls, name, bases, dct))

    def __init__(
        cls, name: str, bases: tuple[type, ...], dct: dict[str, Any], **kwargs: Any
    ) -> None:
        super().__init__(name, bases, dct)
        cls.strict = kwargs.get("strict", True)


def timezone_name_settings(
    **kwargs: Any,
) -> Callable[[type[TimeZoneName]], type[TimeZoneName]]:
    def wrapper(cls: type[TimeZoneName]) -> type[TimeZoneName]:
        cls.strict = kwargs.get("strict", True)
        return cls

    return wrapper


@timezone_name_settings(strict=True)
class TimeZoneName(str):
    """TimeZoneName is a custom string subclass for validating and serializing timezone names."""

    __slots__: list[str] = []
    allowed_values: set[str] = set(get_timezones())
    allowed_values_list: list[str] = sorted(allowed_values)
    allowed_values_upper_to_correct: dict[str, str] = {
        val.upper(): val for val in allowed_values
    }
    strict: bool

    @classmethod
    def _validate(
        cls, __input_value: str, _: core_schema.ValidationInfo
    ) -> TimeZoneName:
        if __input_value not in cls.allowed_values:
            if not cls.strict:
                upper_value = __input_value.strip().upper()
                if upper_value in cls.allowed_values_upper_to_correct:
                    return cls(cls.allowed_values_upper_to_correct[upper_value])
            raise PydanticCustomError("TimeZoneName", "Invalid timezone name.")
        return cls(__input_value)

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _: type[Any], __: GetCoreSchemaHandler
    ) -> core_schema.AfterValidatorFunctionSchema:
        return core_schema.with_info_after_validator_function(
            cls._validate,
            core_schema.str_schema(min_length=1),
        )

    @classmethod
    def __get_pydantic_json_schema__(
        cls, schema: core_schema.CoreSchema, handler: GetJsonSchemaHandler
    ) -> dict[str, Any]:
        json_schema = handler(schema)
        json_schema.update({"enum": cls.allowed_values_list})
        return json_schema
