import asyncio
import time
from datetime import datetime, timedelta
from typing import Any, Literal, Union
from zoneinfo import ZoneInfo

from pydantic import BaseModel

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField

# Shared timezone literal type for all time/date blocks
TimezoneLiteral = Literal[
    "UTC",  # UTC±00:00
    "Pacific/Honolulu",  # UTC-10:00
    "America/Anchorage",  # UTC-09:00 (Alaska)
    "America/Los_Angeles",  # UTC-08:00 (Pacific)
    "America/Denver",  # UTC-07:00 (Mountain)
    "America/Chicago",  # UTC-06:00 (Central)
    "America/New_York",  # UTC-05:00 (Eastern)
    "America/Caracas",  # UTC-04:00
    "America/Sao_Paulo",  # UTC-03:00
    "America/St_Johns",  # UTC-02:30 (Newfoundland)
    "Atlantic/South_Georgia",  # UTC-02:00
    "Atlantic/Azores",  # UTC-01:00
    "Europe/London",  # UTC+00:00 (GMT/BST)
    "Europe/Paris",  # UTC+01:00 (CET)
    "Europe/Athens",  # UTC+02:00 (EET)
    "Europe/Moscow",  # UTC+03:00
    "Asia/Tehran",  # UTC+03:30 (Iran)
    "Asia/Dubai",  # UTC+04:00
    "Asia/Kabul",  # UTC+04:30 (Afghanistan)
    "Asia/Karachi",  # UTC+05:00 (Pakistan)
    "Asia/Kolkata",  # UTC+05:30 (India)
    "Asia/Kathmandu",  # UTC+05:45 (Nepal)
    "Asia/Dhaka",  # UTC+06:00 (Bangladesh)
    "Asia/Yangon",  # UTC+06:30 (Myanmar)
    "Asia/Bangkok",  # UTC+07:00
    "Asia/Shanghai",  # UTC+08:00 (China)
    "Australia/Eucla",  # UTC+08:45
    "Asia/Tokyo",  # UTC+09:00 (Japan)
    "Australia/Adelaide",  # UTC+09:30
    "Australia/Sydney",  # UTC+10:00
    "Australia/Lord_Howe",  # UTC+10:30
    "Pacific/Noumea",  # UTC+11:00
    "Pacific/Auckland",  # UTC+12:00 (New Zealand)
    "Pacific/Chatham",  # UTC+12:45
    "Pacific/Tongatapu",  # UTC+13:00
    "Pacific/Kiritimati",  # UTC+14:00
    "Etc/GMT-12",  # UTC+12:00
    "Etc/GMT+12",  # UTC-12:00
]


class TimeStrftimeFormat(BaseModel):
    discriminator: Literal["strftime"]
    format: str = "%H:%M:%S"
    timezone: TimezoneLiteral = "UTC"


class TimeISO8601Format(BaseModel):
    discriminator: Literal["iso8601"]
    timezone: TimezoneLiteral = "UTC"
    include_microseconds: bool = False


class GetCurrentTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str = SchemaField(
            description="Trigger any data to output the current time"
        )
        format_type: Union[TimeStrftimeFormat, TimeISO8601Format] = SchemaField(
            discriminator="discriminator",
            description="Format type for time output (strftime with custom format or ISO 8601)",
            default=TimeStrftimeFormat(discriminator="strftime"),
        )

    class Output(BlockSchema):
        time: str = SchemaField(
            description="Current time in the specified format (default: %H:%M:%S)"
        )

    def __init__(self):
        super().__init__(
            id="a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa",
            description="This block outputs the current time.",
            categories={BlockCategory.TEXT},
            input_schema=GetCurrentTimeBlock.Input,
            output_schema=GetCurrentTimeBlock.Output,
            test_input=[
                {"trigger": "Hello"},
                {
                    "trigger": "Hello",
                    "format_type": {
                        "discriminator": "strftime",
                        "format": "%H:%M",
                    },
                },
                {
                    "trigger": "Hello",
                    "format_type": {
                        "discriminator": "iso8601",
                        "timezone": "UTC",
                        "include_microseconds": False,
                    },
                },
            ],
            test_output=[
                ("time", lambda _: time.strftime("%H:%M:%S")),
                ("time", lambda _: time.strftime("%H:%M")),
                (
                    "time",
                    lambda t: "T" in t and ("+" in t or "Z" in t),
                ),  # Check for ISO format with timezone
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if isinstance(input_data.format_type, TimeISO8601Format):
            # ISO 8601 format for time only (extract time portion from full ISO datetime)
            tz = ZoneInfo(input_data.format_type.timezone)
            dt = datetime.now(tz=tz)

            # Get the full ISO format and extract just the time portion with timezone
            if input_data.format_type.include_microseconds:
                full_iso = dt.isoformat()
            else:
                full_iso = dt.isoformat(timespec="seconds")

            # Extract time portion (everything after 'T')
            current_time = full_iso.split("T")[1] if "T" in full_iso else full_iso
            current_time = f"T{current_time}"  # Add T prefix for ISO 8601 time format
        else:  # TimeStrftimeFormat
            tz = ZoneInfo(input_data.format_type.timezone)
            dt = datetime.now(tz=tz)
            current_time = dt.strftime(input_data.format_type.format)
        yield "time", current_time


class DateStrftimeFormat(BaseModel):
    discriminator: Literal["strftime"]
    format: str = "%Y-%m-%d"
    timezone: TimezoneLiteral = "UTC"


class DateISO8601Format(BaseModel):
    discriminator: Literal["iso8601"]
    timezone: TimezoneLiteral = "UTC"


class GetCurrentDateBlock(Block):
    class Input(BlockSchema):
        trigger: str = SchemaField(
            description="Trigger any data to output the current date"
        )
        offset: Union[int, str] = SchemaField(
            title="Days Offset",
            description="Offset in days from the current date",
            default=0,
        )
        format_type: Union[DateStrftimeFormat, DateISO8601Format] = SchemaField(
            discriminator="discriminator",
            description="Format type for date output (strftime with custom format or ISO 8601)",
            default=DateStrftimeFormat(discriminator="strftime"),
        )

    class Output(BlockSchema):
        date: str = SchemaField(
            description="Current date in the specified format (default: YYYY-MM-DD)"
        )

    def __init__(self):
        super().__init__(
            id="b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0b1",
            description="This block outputs the current date with an optional offset.",
            categories={BlockCategory.TEXT},
            input_schema=GetCurrentDateBlock.Input,
            output_schema=GetCurrentDateBlock.Output,
            test_input=[
                {"trigger": "Hello", "offset": "7"},
                {
                    "trigger": "Hello",
                    "offset": "7",
                    "format_type": {
                        "discriminator": "strftime",
                        "format": "%m/%d/%Y",
                    },
                },
                {
                    "trigger": "Hello",
                    "offset": "0",
                    "format_type": {
                        "discriminator": "iso8601",
                        "timezone": "UTC",
                    },
                },
            ],
            test_output=[
                (
                    "date",
                    lambda t: abs(datetime.now() - datetime.strptime(t, "%Y-%m-%d"))
                    < timedelta(days=8),  # 7 days difference + 1 day error margin.
                ),
                (
                    "date",
                    lambda t: abs(datetime.now() - datetime.strptime(t, "%m/%d/%Y"))
                    < timedelta(days=8),
                    # 7 days difference + 1 day error margin.
                ),
                (
                    "date",
                    lambda t: len(t) == 10
                    and t[4] == "-"
                    and t[7] == "-",  # ISO date format YYYY-MM-DD
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            offset = int(input_data.offset)
        except ValueError:
            offset = 0

        if isinstance(input_data.format_type, DateISO8601Format):
            # ISO 8601 format for date only (YYYY-MM-DD)
            tz = ZoneInfo(input_data.format_type.timezone)
            current_date = datetime.now(tz=tz) - timedelta(days=offset)
            # ISO 8601 date format is YYYY-MM-DD
            date_str = current_date.date().isoformat()
        else:  # DateStrftimeFormat
            tz = ZoneInfo(input_data.format_type.timezone)
            current_date = datetime.now(tz=tz) - timedelta(days=offset)
            date_str = current_date.strftime(input_data.format_type.format)

        yield "date", date_str


class StrftimeFormat(BaseModel):
    discriminator: Literal["strftime"]
    format: str = "%Y-%m-%d %H:%M:%S"
    timezone: TimezoneLiteral = "UTC"


class ISO8601Format(BaseModel):
    discriminator: Literal["iso8601"]
    timezone: TimezoneLiteral = "UTC"
    include_microseconds: bool = False


class GetCurrentDateAndTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str = SchemaField(
            description="Trigger any data to output the current date and time"
        )
        format_type: Union[StrftimeFormat, ISO8601Format] = SchemaField(
            discriminator="discriminator",
            description="Format type for date and time output (strftime with custom format or ISO 8601/RFC 3339)",
            default=StrftimeFormat(discriminator="strftime"),
        )

    class Output(BlockSchema):
        date_time: str = SchemaField(
            description="Current date and time in the specified format (default: YYYY-MM-DD HH:MM:SS)"
        )

    def __init__(self):
        super().__init__(
            id="716a67b3-6760-42e7-86dc-18645c6e00fc",
            description="This block outputs the current date and time.",
            categories={BlockCategory.TEXT},
            input_schema=GetCurrentDateAndTimeBlock.Input,
            output_schema=GetCurrentDateAndTimeBlock.Output,
            test_input=[
                {"trigger": "Hello"},
                {
                    "trigger": "Hello",
                    "format_type": {
                        "discriminator": "strftime",
                        "format": "%Y/%m/%d",
                    },
                },
                {
                    "trigger": "Hello",
                    "format_type": {
                        "discriminator": "iso8601",
                        "timezone": "UTC",
                        "include_microseconds": False,
                    },
                },
            ],
            test_output=[
                (
                    "date_time",
                    lambda t: abs(
                        datetime.now(tz=ZoneInfo("UTC"))
                        - datetime.strptime(t + "+00:00", "%Y-%m-%d %H:%M:%S%z")
                    )
                    < timedelta(seconds=10),  # 10 seconds error margin.
                ),
                (
                    "date_time",
                    lambda t: abs(
                        datetime.now().date() - datetime.strptime(t, "%Y/%m/%d").date()
                    )
                    < timedelta(days=1),  # Date format only, no time component
                ),
                (
                    "date_time",
                    lambda t: abs(
                        datetime.now(tz=ZoneInfo("UTC")) - datetime.fromisoformat(t)
                    )
                    < timedelta(seconds=10),  # 10 seconds error margin for ISO format.
                ),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if isinstance(input_data.format_type, ISO8601Format):
            # ISO 8601 format with specified timezone (also RFC3339-compliant)
            tz = ZoneInfo(input_data.format_type.timezone)
            dt = datetime.now(tz=tz)

            # Format with or without microseconds
            if input_data.format_type.include_microseconds:
                current_date_time = dt.isoformat()
            else:
                current_date_time = dt.isoformat(timespec="seconds")
        else:  # StrftimeFormat
            tz = ZoneInfo(input_data.format_type.timezone)
            dt = datetime.now(tz=tz)
            current_date_time = dt.strftime(input_data.format_type.format)
        yield "date_time", current_date_time


class CountdownTimerBlock(Block):
    class Input(BlockSchema):
        input_message: Any = SchemaField(
            advanced=False,
            description="Message to output after the timer finishes",
            default="timer finished",
        )
        seconds: Union[int, str] = SchemaField(
            advanced=False, description="Duration in seconds", default=0
        )
        minutes: Union[int, str] = SchemaField(
            advanced=False, description="Duration in minutes", default=0
        )
        hours: Union[int, str] = SchemaField(
            advanced=False, description="Duration in hours", default=0
        )
        days: Union[int, str] = SchemaField(
            advanced=False, description="Duration in days", default=0
        )
        repeat: int = SchemaField(
            description="Number of times to repeat the timer",
            default=1,
        )

    class Output(BlockSchema):
        output_message: Any = SchemaField(
            description="Message after the timer finishes"
        )

    def __init__(self):
        super().__init__(
            id="d67a9c52-5e4e-11e2-bcfd-0800200c9a71",
            description="This block triggers after a specified duration.",
            categories={BlockCategory.TEXT},
            input_schema=CountdownTimerBlock.Input,
            output_schema=CountdownTimerBlock.Output,
            test_input=[
                {"seconds": 1},
                {"input_message": "Custom message"},
            ],
            test_output=[
                ("output_message", "timer finished"),
                ("output_message", "Custom message"),
            ],
        )

    async def run(self, input_data: Input, **kwargs) -> BlockOutput:
        seconds = int(input_data.seconds)
        minutes = int(input_data.minutes)
        hours = int(input_data.hours)
        days = int(input_data.days)

        total_seconds = seconds + minutes * 60 + hours * 3600 + days * 86400

        for _ in range(input_data.repeat):
            if total_seconds > 0:
                await asyncio.sleep(total_seconds)
            yield "output_message", input_data.input_message
