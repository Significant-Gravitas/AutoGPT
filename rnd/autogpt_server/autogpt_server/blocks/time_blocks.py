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
                ("date", (datetime.now() - timedelta(days=7)).strftime("%Y-%m-%d")),
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
                ("date_time", time.strftime("%Y-%m-%d %H:%M:%S")),
            ],
        )

    def run(self, input_data: Input) -> BlockOutput:
        current_date_time = time.strftime("%Y-%m-%d %H:%M:%S")
        yield "date_time", current_date_time
