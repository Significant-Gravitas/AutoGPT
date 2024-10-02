import time
from datetime import datetime, timedelta
from typing import Any, Union

from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema


class GetCurrentTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str

    class Output(BlockSchema):
        time: str

    def __init__(self):
        super().__init__(
            id="a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa",
            description="This block outputs the current time.",
            categories={BlockCategory.TEXT},
            input_schema=GetCurrentTimeBlock.Input,
            output_schema=GetCurrentTimeBlock.Output,
            test_input=[
                {"trigger": "Hello", "format": "{time}"},
            ],
            test_output=[
                ("time", lambda _: time.strftime("%H:%M:%S")),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        current_time = time.strftime("%H:%M:%S")
        yield "time", current_time


class GetCurrentDateBlock(Block):
    class Input(BlockSchema):
        trigger: str
        offset: Union[int, str]

    class Output(BlockSchema):
        date: str

    def __init__(self):
        super().__init__(
            id="b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0b1",
            description="This block outputs the current date with an optional offset.",
            categories={BlockCategory.TEXT},
            input_schema=GetCurrentDateBlock.Input,
            output_schema=GetCurrentDateBlock.Output,
            test_input=[
                {"trigger": "Hello", "format": "{date}", "offset": "7"},
            ],
            test_output=[
                (
                    "date",
                    lambda t: abs(datetime.now() - datetime.strptime(t, "%Y-%m-%d"))
                    < timedelta(days=8),  # 7 days difference + 1 day error margin.
                ),
            ],
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        try:
            offset = int(input_data.offset)
        except ValueError:
            offset = 0
        current_date = datetime.now() - timedelta(days=offset)
        yield "date", current_date.strftime("%Y-%m-%d")


class GetCurrentDateAndTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str

    class Output(BlockSchema):
        date_time: str

    def __init__(self):
        super().__init__(
            id="716a67b3-6760-42e7-86dc-18645c6e00fc",
            description="This block outputs the current date and time.",
            categories={BlockCategory.TEXT},
            input_schema=GetCurrentDateAndTimeBlock.Input,
            output_schema=GetCurrentDateAndTimeBlock.Output,
            test_input=[
                {"trigger": "Hello", "format": "{date_time}"},
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
        current_date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        yield "date_time", current_date_time


class CountdownTimerBlock(Block):
    class Input(BlockSchema):
        input_message: Any = "timer finished"
        seconds: Union[int, str] = 0
        minutes: Union[int, str] = 0
        hours: Union[int, str] = 0
        days: Union[int, str] = 0

    class Output(BlockSchema):
        output_message: str

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

        time.sleep(total_seconds)
        yield "output_message", input_data.input_message
