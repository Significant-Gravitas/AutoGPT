from typing import Any

from autogpt_server.data.block import Block, BlockOutput, BlockSchema


class ObjectParser(Block):
    class Input(BlockSchema):
        object: Any
        field_path: str

    class Output(BlockSchema):
        field_value: Any

    def __init__(self):
        super().__init__(
            id="be45299a-193b-4852-bda4-510883d21814",
            input_schema=ObjectParser.Input,
            output_schema=ObjectParser.Output,
        )

    def run(self, input_data: Input) -> BlockOutput:
        field_path = input_data.field_path.split(".")
        field_value = input_data.object
        for field in field_path:
            if isinstance(field_value, dict) and field in field_value:
                field_value = field_value.get(field)
            elif isinstance(field_value, object) and hasattr(field_value, field):
                field_value = getattr(field_value, field)
            else:
                yield "error", input_data.object
                return

        yield "field_value", field_value
