import time
from datetime import datetime, timedelta
from typing import Any, Union

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class GetCurrentTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str = SchemaField(
            description="Trigger any data to output the current time"
        )
        format: str = SchemaField(
            description="Format of the time to output", default="%H:%M:%S"
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
                {"trigger": "Hello", "format": "%H:%M"},
            ],
            test_output=[
                ("time", lambda _: time.strftime("%H:%M:%S")),
                ("time", lambda _: time.strftime("%H:%M")),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        current_time = time.strftime(input_data.format)
        yield "time", current_time


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
        format: str = SchemaField(
            description="Format of the date to output", default="%Y-%m-%d"
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
                {"trigger": "Hello", "offset": "7", "format": "%m/%d/%Y"},
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
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            offset = int(input_data.offset)
        except ValueError:
            offset = 0
        current_date = datetime.now() - timedelta(days=offset)
        yield "date", current_date.strftime(input_data.format)


class GetCurrentDateAndTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str = SchemaField(
            description="Trigger any data to output the current date and time"
        )
        format: str = SchemaField(
            description="Format of the date and time to output",
            default="%Y-%m-%d %H:%M:%S",
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
            ],
            test_output=[
                (
                    "date_time",
                    lambda t: abs(
                        datetime.now() - datetime.strptime(t, "%Y-%m-%d %H:%M:%S")
                    )
                    < timedelta(seconds=10),  # 10 seconds error margin.
                ),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        current_date_time = time.strftime(input_data.format)
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

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        seconds = int(input_data.seconds)
        minutes = int(input_data.minutes)
        hours = int(input_data.hours)
        days = int(input_data.days)

        total_seconds = seconds + minutes * 60 + hours * 3600 + days * 86400

        for _ in range(input_data.repeat):
            time.sleep(total_seconds)
            yield "output_message", input_data.input_message
