from backend.data.block import Block, BlockCategory, BlockOutput, BlockSchema
from backend.data.model import SchemaField


class InputValidationBlock(Block):
    """
    TEMPORARY BLOCK FOR TESTING
    """

    class Input(BlockSchema):
        required_field: str = SchemaField(
            description="parent of dependent_field", default="hello"
        )

        required_field_2: str = SchemaField(
            description="parent of dependent_field"
        )

        optional_field: str = SchemaField(
            description="This field is optional", default=""
        )
        dependent_field: str = SchemaField(
            description="This field depends on required_field being set",
            depends_on=["required_field", "required_field_2"],
            default = ""
        )

    class Output(BlockSchema):
        is_valid: bool = SchemaField(description="Whether the input validation passed")

    def __init__(self):
        super().__init__(
            id="a1b2c3d4-e5f6-4a5b-9c8d-7e6f5d4c3b2a",
            description="Validates input fields based on requirements and dependencies",
            categories={BlockCategory.BASIC},
            input_schema=InputValidationBlock.Input,
            output_schema=InputValidationBlock.Output,
        )

    def run(self, input_data: Input, **kwargs) -> BlockOutput:
        if not input_data.required_field or not input_data.required_field_2:
            yield "is_valid", False
            return

        if input_data.dependent_field and (not input_data.required_field or not input_data.required_field_2):
            yield "is_valid", False
            return

        yield "is_valid", True
