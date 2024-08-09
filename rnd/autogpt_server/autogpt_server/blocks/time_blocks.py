import time
from datetime import datetime, timedelta
from typing import Union

from autogpt_server.data.block import Block, BlockCategory, BlockOutput, BlockSchema


class CurrentTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str

    class Output(BlockSchema):
        time: str

    def __init__(self):
        super().__init__(
            id="a892b8d9-3e4e-4e9c-9c1e-75f8efcf1bfa",
            description="This block outputs the current time.",
            categories={BlockCategory.TEXT},
            input_schema=CurrentTimeBlock.Input,
            output_schema=CurrentTimeBlock.Output,
            test_input=[
                {"trigger": "Hello", "format": "{time}"},
            ],
            test_output=[
                ("time", time.strftime("%H:%M:%S")),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        current_time = time.strftime("%H:%M:%S")
        yield "time", current_time


class CurrentDateBlock(Block):
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
            input_schema=CurrentDateBlock.Input,
            output_schema=CurrentDateBlock.Output,
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

    def run(self, input_data: Input) -> BlockOutput:
        try:
            offset = int(input_data.offset)
        except ValueError:
            offset = 0
        current_date = datetime.now() - timedelta(days=offset)
        yield "date", current_date.strftime("%Y-%m-%d")


class CurrentDateAndTimeBlock(Block):
    class Input(BlockSchema):
        trigger: str

    class Output(BlockSchema):
        date_time: str

    def __init__(self):
        super().__init__(
            id="b29c1b50-5d0e-4d9f-8f9d-1b0e6fcbf0h2",
            description="This block outputs the current date and time.",
            categories={BlockCategory.TEXT},
            input_schema=CurrentDateAndTimeBlock.Input,
            output_schema=CurrentDateAndTimeBlock.Output,
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

    def run(self, input_data: Input) -> BlockOutput:
        current_date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        yield "date_time", current_date_time


class TimerBlock(Block):
    class Input(BlockSchema):
        seconds: Union[int, str] = 0
        minutes: Union[int, str] = 0
        hours: Union[int, str] = 0
        days: Union[int, str] = 0

    class Output(BlockSchema):
        message: str

    def __init__(self):
        super().__init__(
            id="d67a9c52-5e4e-11e2-bcfd-0800200c9a71",
            description="This block triggers after a specified duration.",
            categories={BlockCategory.TEXT},
            input_schema=TimerBlock.Input,
            output_schema=TimerBlock.Output,
            test_input=[
                {"seconds": 1},
            ],
            test_output=[
                ("message", "timer finished"),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:

        seconds = int(input_data.seconds)
        minutes = int(input_data.minutes)
        hours = int(input_data.hours)
        days = int(input_data.days)

        total_seconds = seconds + minutes * 60 + hours * 3600 + days * 86400

        time.sleep(total_seconds)
        yield "message", "timer finished"
